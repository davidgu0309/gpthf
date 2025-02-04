from functools import cache
from operator import is_
import os
from pyexpat import model
from re import L, T
from turtle import position

from regex import D
from sympy import O
import torch
from transformers import PretrainedConfig, PreTrainedModel

from typing import Optional
from omegaconf import OmegaConf
from torch.nn import init
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .llama_components import GPTHFCache, GPTHFLlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.configuration_llama import LlamaConfig
from .components import (
    EmbeddingComponent,
    PoolingComponent,
    _init_module,
)
from .thf_utils import generate_block_matrix, last_sentence_indices, fetch_last_sentence_embeddings, create_column_mask, combine_attention_masks, truncate_tensor, prepare_inputs_for_generation
from transformers.cache_utils import DynamicCache, Cache

class GPTHFLlamaConfig(LlamaConfig):
    model_type = "GPTHF_Llama"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        super().__init__(**kwargs)
        for key, value in cfg_arch_container.items():
            setattr(self, key, value)

def construct_gpthf_llama(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = GPTHFLlamaConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.vocab_size = vocab_size
    config.num_labels = downstream_classes if downstream_classes is not None else 2

    if downstream_classes is None:
        model = ScriptableLMForPreTraining(config)
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model


class ScriptableTHFLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = GPTHFLlamaConfig

    def __init__(self, config):
        
        super().__init__(config)
        self.cfg = config

        self.word_embedding = torch.nn.Embedding(
            config.vocab_size, self.cfg.hidden_size, self.cfg.pad_token_id
        )
        self.wlt_encoder = torch.nn.ModuleList([GPTHFLlamaDecoderLayer(self.cfg, i) for i in range(self.cfg.wlt_encoder_num_hidden_layers)])
        self.slt_body = torch.nn.ModuleList([GPTHFLlamaDecoderLayer(self.cfg, i) for i in range(self.cfg.slt_body_num_hidden_layers)])

        if self.cfg.final_norm:
            self.final_norm = LlamaRMSNorm(self.cfg.hidden_size)
        else:
            self.final_norm = torch.nn.Identity()

        self.encoder_mode = self.cfg.encoder_mode
        self.mode = self.cfg.mode
        self.nheads = self.cfg.num_attention_heads
        self.seq_length = self.cfg.max_position_embeddings

    def forward(self, input_ids, attention_mask, sentence_ids=None, position_ids=None):
        sentence_embeddings, sentence_indices, sentence_attention_mask, block_mask, word_triangular_mask, _, _ = self.forward_encoder(input_ids, attention_mask, sentence_ids, is_triangular=True, use_cache=False, position_ids=position_ids)
        y = self.forward_body(sentence_embeddings, attention_mask, sentence_indices, block_mask, word_triangular_mask, position_ids)

        return self.final_norm(y[0]), sentence_attention_mask
    
    def forward_encoder(self, input_ids, attention_mask, sentence_ids=None, is_triangular=False, past_key_value=None, use_cache=False, position_ids=None, cache_position=None):   
        #only slow mode uses the key_padding_mask, it processes all tokens but keeps padding tokens "confined" in their sentence
        B, sent_len = input_ids.shape  
        if position_ids is None:
            position_ids = torch.arange(sent_len, device=input_ids.device).expand(B, -1)

        if self.encoder_mode == 'block':
            block_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            block_mask, sentence_attention_mask, sentence_indices = block_mask.to(self.device), sentence_attention_mask.to(self.device), sentence_indices.to(self.device)
            block_mask = block_mask.unsqueeze(1) # expand to the number of heads
        else:
            _, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            sentence_indices, sentence_attention_mask = sentence_indices.to(self.device), sentence_attention_mask.to(self.device)
            block_mask = attention_mask.unsqueeze(1).expand(-1, sent_len, -1)
            block_mask = ~block_mask.expand(B, -1, -1).bool().unsqueeze(1)

        word_triangular_mask = torch.triu(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool), diagonal=1).unsqueeze(0)
        if is_triangular:
            block_mask = word_triangular_mask | block_mask

        if use_cache:
            model_inputs = prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_value, attention_mask=block_mask, use_cache=True, position_ids=position_ids
            )
            assert model_inputs['cache_position'].max() < block_mask.size(2), "Cache position will be out of bounds"
            block_mask = model_inputs['attention_mask'][:, :, model_inputs['cache_position'], :]

        block_mask = torch.where(block_mask, torch.tensor(-1e9, device=block_mask.device), torch.tensor(0, device=block_mask.device))

        embeddings = self.word_embedding(model_inputs['input_ids']) if use_cache else self.word_embedding(input_ids)
        position_ids = model_inputs['position_ids'] if use_cache else position_ids
        past_key_value = model_inputs['past_key_values'] if use_cache else None
        cache_position = model_inputs['cache_position'] if use_cache else None
        
        for i, layer in enumerate(self.wlt_encoder):
            if use_cache:
                embeddings, past_key_value = layer(embeddings, attention_mask=block_mask, position_ids=position_ids, 
                                               past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position)
            else:
                embeddings = layer(embeddings, attention_mask=block_mask, position_ids=position_ids)[0]     
        return embeddings, sentence_indices, sentence_attention_mask, block_mask, word_triangular_mask, past_key_value, cache_position
    
    def forward_body(self, sentence_embeddings, attention_mask, sentence_indices, block_mask, word_triangular_mask, position_ids=None):
        B, sent_len, _ = sentence_embeddings.shape
        if position_ids is None:
            position_ids = torch.arange(sent_len, device=sentence_embeddings.device).expand(B, -1)

        if self.mode == 'fast':
            sentence_indices2 = last_sentence_indices(sentence_indices, key_padding_mask=attention_mask)
            column_mask = create_column_mask(sentence_indices2, sent_len)
            column_triangular_mask = torch.tril(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool)) & column_mask
            column_diagonal_mask = column_triangular_mask | torch.eye(sent_len, device=column_triangular_mask.device, dtype=torch.bool)
            body_attention_mask = ~column_diagonal_mask
        elif self.mode == 'keep_last_sentence':
            sentence_indices2 = last_sentence_indices(sentence_indices, key_padding_mask=attention_mask)
            column_mask = create_column_mask(sentence_indices2, sent_len)
            column_triangular_mask = torch.tril(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool)) & column_mask
            column_triangular_mask = column_triangular_mask.repeat_interleave(self.nheads, dim=0)
            column_diagonal_mask = column_triangular_mask | ~block_mask
            body_attention_mask = ~column_diagonal_mask
        else:
            body_attention_mask = word_triangular_mask.expand(B, -1, -1)

        body_attention_mask = ~combine_attention_masks(~body_attention_mask, attention_mask).unsqueeze(1)
        #make body_attention_mask False -> 0, True -> -inf
        body_attention_mask = torch.where(body_attention_mask, torch.tensor(-1e9, device=body_attention_mask.device), torch.tensor(0, device=body_attention_mask.device))

        for layer in self.slt_body:
            if type(sentence_embeddings) == tuple:
                sentence_embeddings = sentence_embeddings[0]
            sentence_embeddings = layer(sentence_embeddings, attention_mask=body_attention_mask, position_ids=position_ids)

        y = sentence_embeddings
        return y


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = GPTHFLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config

        self.encoder = ScriptableTHFLM(config)
        #self.guided_attention_weight = self.cfg.guided_attention_weight_init
        self.prediction_head = torch.nn.Identity()

        self.decoder = torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size)
        self.decoder.weight = self.encoder.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init_type,
                self.cfg.init_std,
                self.cfg.hidden_size,
                self.cfg.num_hidden_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        #set padding tokens to -100 here
        if labels is not None:
            labels[labels == 50257] = -100

        outputs, _ = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = outputs.view(-1, outputs.shape[-1])

        # if self.sparse_prediction and labels is not None:
        masked_lm_loss = self._forward_sparse(outputs, labels)

        return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):

        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        labels = labels[mask_positions]

        outputs = self.decoder(self.prediction_head(outputs))
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss
    
    @torch.no_grad()
    def generate(self, idx, attention_mask, sentence_ids, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, use_cache=True):
        assert idx.dim() == 2, "idx must be a LongTensor of shape (b,t)"
        eos_token_id = 50259

        past_key_value_encoder = DynamicCache() if use_cache else None
        past_key_value_body = GPTHFCache() if use_cache else None

        B = idx.size(0)

        self.eval()

        idx = idx.to(self.encoder.device)
        attention_mask = attention_mask.to(self.encoder.device)
        sentence_ids = sentence_ids.to(self.encoder.device)
        POSITION_IDS = torch.arange(self.encoder.seq_length, device=idx.device).expand(B, -1)
        cache_position = None

        for i in range(max_new_tokens):
            idx = idx if idx.size(1) <= self.encoder.seq_length else idx[:, -self.encoder.seq_length:]
            attention_mask = attention_mask if attention_mask.size(1) <= self.encoder.seq_length else attention_mask[:, -self.encoder.seq_length:]
            sentence_ids = sentence_ids if sentence_ids.size(1) <= self.encoder.seq_length else sentence_ids[:, -self.encoder.seq_length:]

            is_eos = idx[:, -1] == eos_token_id if i > 0 else torch.zeros(B, dtype=torch.bool, device=idx.device)
            # forward the model to get the logits for the index in the sequence
            position_ids = POSITION_IDS[:, :idx.size(1)]

            embeddings, sentence_indices, sentence_attention_mask_candidate, _, _, _, cache_position = self.encoder.forward_encoder(idx, attention_mask,
                            sentence_ids, is_triangular=False, past_key_value=past_key_value_encoder, use_cache=use_cache, position_ids=position_ids, cache_position=cache_position)
            if use_cache:
                if i == 0:
                    sentence_attention_mask = sentence_attention_mask_candidate
                else:
                    sentence_attention_mask = torch.cat((sentence_attention_mask, torch.ones(B, 1, device=sentence_attention_mask.device, dtype=torch.bool)), dim=1)
            else:
                sentence_attention_mask = sentence_attention_mask_candidate

            if use_cache:
                if i == 0:
                    sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask)
                else:
                    sentence_embeddings = embeddings
                sentence_position_ids = POSITION_IDS[:, :sentence_embeddings.size(1)]
                sentence_model_inputs = prepare_inputs_for_generation(
                    sentence_embeddings, attention_mask=sentence_attention_mask, past_key_values=past_key_value_body, use_cache=True, position_ids=sentence_position_ids
                )    
                sentence_attention_mask = sentence_model_inputs['attention_mask']
            else:
                sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask)   
                sentence_position_ids = POSITION_IDS[:, :sentence_embeddings.size(1)]
            
            sentence_attention_mask_expanded = sentence_attention_mask.unsqueeze(1).expand(-1, sentence_attention_mask.size(1), -1).unsqueeze(1)
            if use_cache:
                assert sentence_model_inputs['cache_position'].max() < sentence_attention_mask_expanded.size(2), "Cache position will be out of bounds"
                sentence_attention_mask_expanded = sentence_attention_mask_expanded[:, :, sentence_model_inputs['cache_position'], :]
                sentence_embeddings = sentence_model_inputs['input_ids']
                sentence_position_ids = sentence_model_inputs['position_ids']
                past_key_value_body = sentence_model_inputs['past_key_values']
                cache_position = sentence_model_inputs['cache_position']

            sentence_attention_mask_float = torch.where(sentence_attention_mask_expanded, torch.tensor(0, device=sentence_attention_mask.device), torch.tensor(-1e9, device=sentence_attention_mask.device))

            for layer in self.encoder.slt_body:
                if use_cache:
                    sentence_embeddings, past_key_value_body = layer(sentence_embeddings, sentence_attention_mask_float, position_ids=sentence_position_ids,
                                                        past_key_value=past_key_value_body, cache_position=cache_position, is_sentence_end=is_eos.any(), use_cache=use_cache)
                else:
                    sentence_embeddings = layer(sentence_embeddings, sentence_attention_mask_float, position_ids=sentence_position_ids, is_sentence_end=is_eos.any(), use_cache=use_cache)[0]
            
            y = self.encoder.final_norm(sentence_embeddings)
            y = y[torch.arange(B), sentence_attention_mask.sum(dim=1)-1] if sentence_embeddings.size(1) > 1 else y[:, -1, :]
    
            logits = self.decoder(self.prediction_head(y))
            # Temperature scaling and sampling
            logits /= temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(idx_next)), dim=1)
            was_last_eos = (idx[torch.arange(B).to(idx.device), attention_mask.sum(dim=1)-2] == eos_token_id).unsqueeze(1)
            sentence_ids = torch.cat((sentence_ids, sentence_ids.max(dim=1)[0].unsqueeze(1) +  was_last_eos), dim=1)

            if i > 0:
                sentence_attention_mask = sentence_attention_mask[:, :-1]
            if is_eos.any():
                if len(is_eos.shape) == 1:
                    is_eos = is_eos.unsqueeze(1)
                    
                sentence_attention_mask = torch.cat((sentence_attention_mask, is_eos), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_fast(self, idx, attention_mask, sentence_ids, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, use_cache=True):
        # assert self.encoder.encoder_mode == 'block', "Calling generate_fast on full wlt_encoder is likely an error."
        # assert self.encoder.mode == 'fast', "Calling generate_fast on slow body is likely an error."
        # assert idx.dim() == 2, "idx must be a LongTensor of shape (b,t)"

        B, T = idx.shape
        idx, attention_mask, sentence_ids = idx.to(self.encoder.device), attention_mask.to(self.encoder.device), sentence_ids.to(self.encoder.device)
        self.eval()
        sentence_embedding_cache = torch.zeros(B, 256, self.encoder.cfg.hidden_size, device=idx.device)
        num_cached_sentences = torch.zeros(B, dtype=torch.long, device=idx.device)
        
        # Sentence boundaries need to be determined based on sentence_ids and the presence of the end-of-sentence token
        eos_token_id = 50259
        pad_token_id = 50257
        is_eos = idx == eos_token_id

        POSITION_IDS = torch.arange(self.encoder.seq_length, device=idx.device).expand(B, -1)
        cache_position = None

        past_key_value_encoder = DynamicCache() if use_cache else None
        past_key_value_body = GPTHFCache() if use_cache else None

        for i in range(max_new_tokens):
            B, T = idx.shape
            idx = idx if idx.size(1) <= self.encoder.seq_length else idx[:, -self.encoder.seq_length:]
            attention_mask = attention_mask if attention_mask.size(1) <= self.encoder.seq_length else attention_mask[:, -self.encoder.seq_length:]
            sentence_ids = sentence_ids if sentence_ids.size(1) <= self.encoder.seq_length else sentence_ids[:, -self.encoder.seq_length:]

            is_last_eos = idx[:, -1] == eos_token_id if i > 0 else torch.zeros(B, dtype=torch.bool, device=idx.device)

            embeddings = self.encoder.word_embedding(idx)
            position_ids = POSITION_IDS[:, :embeddings.size(1)]
            embeddings, sentence_indices, sentence_attention_mask_candidate, _, _, _, cache_position = self.encoder.forward_encoder(idx, attention_mask,
                             sentence_ids, is_triangular=False, past_key_value=past_key_value_encoder, use_cache=use_cache, position_ids=position_ids, cache_position=cache_position)
            
            if use_cache:
                if i == 0:
                    sentence_attention_mask = sentence_attention_mask_candidate
                else:
                    sentence_attention_mask = torch.cat((sentence_attention_mask, torch.ones(B, 1, device=sentence_attention_mask.device, dtype=torch.bool)), dim=1)
            else:
                sentence_attention_mask = sentence_attention_mask_candidate

            if use_cache:
                if i == 0:
                    sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask)
                else:
                    sentence_embeddings = embeddings
                sentence_position_ids = POSITION_IDS[:, :sentence_embeddings.size(1)]
                sentence_model_inputs = prepare_inputs_for_generation(
                    sentence_embeddings, attention_mask=sentence_attention_mask, past_key_values=past_key_value_body, use_cache=True, position_ids=sentence_position_ids
                )    
            else:
                sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask)  

            sentence_body_input = sentence_embedding_cache.clone().to(self.encoder.device)
            
            num_new_sentences = (sentence_ids.max(dim=1)[0] + 1)
            for b in range(B):
                new_sentence_embeddings_b = sentence_embeddings[b, :num_new_sentences[b]]
                sentence_body_input[b, num_cached_sentences[b]:num_cached_sentences[b]+num_new_sentences[b]] = new_sentence_embeddings_b

            current_sentences = (num_new_sentences + num_cached_sentences)
            max_current_sentences = current_sentences.max()
            sentence_body_input = sentence_body_input[:, :max_current_sentences]
            sentence_cache_candidate = sentence_body_input.clone()
            
            range_tensor = POSITION_IDS[:, :max_current_sentences]
            sentence_attention_mask = range_tensor < current_sentences.unsqueeze(1)
            sentence_attention_mask = sentence_model_inputs['attention_mask'] if use_cache else sentence_attention_mask         
            sentence_attention_mask_expanded = sentence_attention_mask.unsqueeze(1).expand(-1, sentence_body_input.size(1), -1).unsqueeze(1).to(sentence_body_input.device)

            if use_cache:
                assert sentence_model_inputs['cache_position'].max() < sentence_attention_mask_expanded.size(2), "Cache position will be out of bounds"
                sentence_attention_mask_expanded = sentence_attention_mask_expanded[:, :, sentence_model_inputs['cache_position'], :]
            sentence_attention_mask_float = torch.where(sentence_attention_mask_expanded, torch.tensor(0, device=sentence_attention_mask_expanded.device), torch.tensor(-1e9, device=sentence_attention_mask.device))
            sentence_embeddings = sentence_model_inputs['input_ids'] if use_cache else sentence_embeddings

            position_ids = sentence_model_inputs['position_ids'] if use_cache else range_tensor
            past_key_value_body = sentence_model_inputs['past_key_values'] if use_cache else None
            cache_position = sentence_model_inputs['cache_position'] if use_cache else None
    
            for layer in self.encoder.slt_body:
                if use_cache:
                    sentence_embeddings, past_key_value_body = layer(sentence_embeddings, sentence_attention_mask_float, position_ids=position_ids,
                                                        past_key_value=past_key_value_body, cache_position=cache_position, is_sentence_end=is_eos.any(), use_cache=use_cache)
                else:
                    sentence_body_input = layer(sentence_body_input, sentence_attention_mask_float, position_ids=position_ids, is_sentence_end=is_eos.any(), use_cache=use_cache)[0]
            
            y = self.encoder.final_norm(sentence_embeddings) if use_cache else self.encoder.final_norm(sentence_body_input)
            y = y[torch.arange(B), sentence_attention_mask.sum(dim=1)-1] if sentence_embeddings.size(1) > 1 else y[:, -1, :]

            logits = self.decoder(self.prediction_head(y))

            # Temperature scaling and sampling
            logits /= temperature
            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)
                logits[logits < top_values[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            #if num_new_sentences > 1, update cache with all but the last sentence. If predicted token is last sentence, also update last sentence. Truncate idx afterwards
            if i > 0:
                sentence_attention_mask = sentence_attention_mask[:, :-1]
            if is_last_eos.any():
                sentence_attention_mask = torch.cat((sentence_attention_mask, is_eos), dim=1)
            
            if is_eos.any():
                sentence_embedding_cache[:, :max_current_sentences] = sentence_cache_candidate
                sentence_embedding_cache[torch.arange(B), current_sentences-1] = 0
                #idx, attention_mask and sentence_ids should be updated by starting from index of torch.argmax(sentence_ids, dim=1) and updating the las token
                idx, attention_mask, sentence_ids = truncate_tensor(idx, sentence_ids, pad_token_id, past_key_value_encoder=past_key_value_encoder), truncate_tensor(attention_mask, sentence_ids, 0), truncate_tensor(sentence_ids, sentence_ids, -1, normalize=True)
                
            idx = torch.cat((idx, idx_next), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(idx_next)), dim=1)
            is_eos = idx == eos_token_id
            was_last_eos = (idx[torch.arange(B).to(idx.device), attention_mask.sum(dim=1)-2] == eos_token_id).unsqueeze(1)
            sentence_ids = torch.cat((sentence_ids, sentence_ids.max(dim=1)[0].unsqueeze(1) +  was_last_eos), dim=1)
            # number of cached sentences is number of sentence_embedding_cache != 0 
            num_cached_sentences = (sentence_embedding_cache.sum(dim=2) != 0).sum(dim=1)
        return idx

class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = GPTHFLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableTHFLM(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(1024, self.num_labels)
        self.activation = torch.nn.Tanh()
        print(self.pooler)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init_type,
                self.cfg.init_std,
                self.cfg.hidden_size,
                self.cfg.num_hidden_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor]=None, sentence_ids=None, labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        POSITION_IDS = torch.arange(self.encoder.seq_length, device=input_ids.device).expand(B, -1)
        position_ids = POSITION_IDS[:, :T]
        embeddings, sentence_indices, sentence_attention_mask, _, _, _, cache_position = self.encoder.forward_encoder(input_ids, attention_mask,
                            sentence_ids, is_triangular=False, use_cache=False, position_ids=position_ids)

        sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask)   
        sentence_position_ids = POSITION_IDS[:, :sentence_embeddings.size(1)]
        sentence_attention_mask_expanded = sentence_attention_mask.unsqueeze(1).expand(-1, sentence_attention_mask.size(1), -1).unsqueeze(1)
        sentence_attention_mask_float = torch.where(sentence_attention_mask_expanded, torch.tensor(0, device=sentence_attention_mask.device), torch.tensor(-1e9, device=sentence_attention_mask.device))

        for layer in self.encoder.slt_body:
                sentence_embeddings = layer(sentence_embeddings, sentence_attention_mask_float, position_ids=sentence_position_ids)[0]
        
        y = self.encoder.final_norm(sentence_embeddings)
        logits = self.pooler(y, sentence_attention_mask)
        logits = self.head(logits)

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)

