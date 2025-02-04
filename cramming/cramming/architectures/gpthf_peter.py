from operator import is_
import os
from re import L, T

from sympy import O
import torch
from transformers import PretrainedConfig, PreTrainedModel

from typing import Optional
from omegaconf import OmegaConf
from torch.nn import init
import torch.nn.functional as F

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
from .attention import get_attention_mechanism
from .thf_utils import generate_block_matrix, last_sentence_indices, fetch_last_sentence_embeddings, create_column_mask, combine_attention_masks


class GPTHFConfig(PretrainedConfig):
    model_type = "GPTHF_Peter"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_gpthf_peter(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = GPTHFConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes
    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM" or config.arch["objective_layout"] == "CLM":
            model = ScriptableLMForPreTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model

class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        return self.dense(self.self_attention(hidden_states, attention_mask=attention_mask, key_padding_mask=key_padding_mask))
    
class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))
    

class TransformerEncoderLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch, hidden_size):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            hidden_size,
            cfg_arch.intermed_size,                   # experiment if this should scale with wlt_embedding_with, which it currently doesnt
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(self.norm1(states), attention_mask, key_padding_mask)
        states = states + self.dropout(attn_out)
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states

class ScriptableTHFLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = GPTHFConfig

    def __init__(self, config):
        
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)

        self.wlt_encoder = torch.nn.ModuleList([TransformerEncoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_encoder_num_hidden_layers)])
        self.body_dimension = self.cfg.wlt_embedding_width * self.cfg.hidden_size
        self.slt_body = torch.nn.ModuleList([TransformerEncoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.slt_body_num_hidden_layers)])
        self.wlt_decoder = torch.nn.ModuleList([TransformerEncoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_decoder_num_hidden_layers)])

        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = torch.nn.LayerNorm(self.body_dimension, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.encoder_mode = self.cfg.encoder_mode
        self.mode = self.cfg.mode
        self.nheads = self.cfg.attention.num_attention_heads
        self.seq_first = self.wlt_encoder[0].LAYOUT == "[S B H]"
        self.seq_length = self.cfg.embedding.max_seq_length

    def forward(self, input_ids, attention_mask, sentence_ids=None):
        x = self.embedding(input_ids) # (batch_size, sent_len, dim)
        sentence_embeddings, sentence_indices, sentence_attention_mask, block_mask, word_triangular_mask = self.forward_encoder(x, attention_mask, sentence_ids)
        y = self.forward_body(sentence_embeddings, attention_mask, sentence_indices, block_mask, word_triangular_mask)
        return self.final_norm(y)
    
    def forward_encoder(self, embeddings, attention_mask, sentence_ids=None, is_triangular=True):   
        #forward_encoder doesnt actually use the key_padding_mask, it processes all tokens but keeps padding tokens "confined" in their sentence

        B, sent_len, _ = embeddings.shape  
        #print(sent_len)
        if self.encoder_mode == 'block':
            block_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            block_mask, sentence_attention_mask, sentence_indices = block_mask.to(self.device), sentence_attention_mask.to(self.device), sentence_indices.to(self.device)
            block_mask = block_mask.unsqueeze(1).expand(-1, self.nheads, -1, -1).reshape(-1, sent_len, sent_len) # expand to the number of heads
        else:
            _, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            sentence_indices, sentence_attention_mask = sentence_indices.to(self.device), sentence_attention_mask.to(self.device)
            block_mask = torch.zeros(sent_len, sent_len, device=embeddings.device, dtype=torch.bool)
            block_mask = block_mask.unsqueeze(0).expand(B*self.nheads, -1, -1)
        
        if is_triangular:
            word_triangular_mask = torch.triu(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool), diagonal=1).unsqueeze(0)
            block_mask = word_triangular_mask | block_mask
        else:
            word_triangular_mask = None

        if self.seq_first:
            embeddings = embeddings.transpose(0, 1)
            block_mask = block_mask.view(B, self.nheads, sent_len, sent_len)

        for layer in self.wlt_encoder:
            embeddings = layer(embeddings, attention_mask=block_mask)     #process padding tokens as well and only throw them away at a later step

        if self.seq_first:
            embeddings = embeddings.transpose(0, 1)

        return embeddings, sentence_indices, sentence_attention_mask, block_mask, word_triangular_mask
    
    def forward_body(self, sentence_embeddings, attention_mask, sentence_indices, block_mask, word_triangular_mask):
        B, sent_len, _ = sentence_embeddings.shape

        if self.mode == 'fast':
            sentence_indices2 = last_sentence_indices(sentence_indices, key_padding_mask=attention_mask)
            column_mask = create_column_mask(sentence_indices2, sent_len)
            column_triangular_mask = torch.tril(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool)) & column_mask
            column_triangular_mask = column_triangular_mask.repeat_interleave(self.nheads, dim=0)
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
            body_attention_mask = word_triangular_mask.expand(B*self.nheads, -1, -1)

        if self.seq_first:
            sentence_embeddings = sentence_embeddings.transpose(0, 1)
            body_attention_mask = ~combine_attention_masks(~body_attention_mask, attention_mask.repeat_interleave(self.nheads, dim=0))
            body_attention_mask = body_attention_mask.view(B, self.nheads, sent_len, sent_len)


        for layer in self.slt_body:
            sentence_embeddings = layer(sentence_embeddings, attention_mask=body_attention_mask, key_padding_mask=~attention_mask)

        if self.seq_first:
            sentence_embeddings = sentence_embeddings.transpose(0, 1)

        y = sentence_embeddings
        return y


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = GPTHFConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableTHFLM(config)
        #self.guided_attention_weight = self.cfg.guided_attention_weight_init

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        #set padding tokens to -100 here
        labels = labels.masked_fill(labels == 50257, -100)

        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = outputs.view(-1, outputs.shape[-1])

        if self.sparse_prediction and labels is not None:
            masked_lm_loss = self._forward_sparse(outputs, labels)
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

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
    def generate(self, idx, attention_mask, sentence_ids, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        assert idx.dim() == 2, "idx must be a LongTensor of shape (b,t)"
        pad_token_id = 50257
        eos_token_id = 50259

        B = idx.size(0)

        self.eval()

        for _ in range(max_new_tokens):
            idx = idx if idx.size(1) <= self.encoder.seq_length else idx[:, -self.encoder.seq_length:]
            # forward the model to get the logits for the index in the sequence
            idx = idx.to(self.encoder.device)
            attention_mask = attention_mask.to(self.encoder.device)
            sentence_ids = sentence_ids.to(self.encoder.device)

            embeddings = self.encoder.embedding(idx)
            embeddings, sentence_indices, sentence_attention_mask, _, _ = self.encoder.forward_encoder(embeddings, attention_mask, sentence_ids, is_triangular=False)
            sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask)

            if self.encoder.seq_first:
                sentence_embeddings = sentence_embeddings.transpose(0, 1)

            for layer in self.encoder.slt_body:
                sentence_embeddings = layer(sentence_embeddings, key_padding_mask=~sentence_attention_mask)

            if self.encoder.seq_first:
                sentence_embeddings = sentence_embeddings.transpose(0, 1)

            y = self.encoder.final_norm(sentence_embeddings)
            y = y[torch.arange(B), sentence_attention_mask.sum(dim=1)-1]

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
            idx = concatenate_with_padding(idx, idx_next, pad_token_id)
            attention_mask = concatenate_with_padding(attention_mask, torch.ones_like(idx_next), 0)
            was_last_eos = (idx[torch.arange(B).to(idx.device), attention_mask.sum(dim=1)-2] == eos_token_id).unsqueeze(1)
            sentence_ids = concatenate_with_padding(sentence_ids, sentence_ids.max(dim=1)[0].unsqueeze(1) +  was_last_eos , -1)

        return idx
    
    @torch.no_grad()
    def generate_fast(self, idx, attention_mask, sentence_ids, max_new_tokens, temperature=1.0, do_sample=True, top_k=None):
        # assert self.encoder.encoder_mode == 'block', "Calling generate_fast on full wlt_encoder is likely an error."
        # assert self.encoder.mode == 'fast', "Calling generate_fast on slow body is likely an error."
        #assert idx.dim() == 2, "idx must be a LongTensor of shape (b,t)"
        B, T = idx.shape
        self.eval()
        if not hasattr(self, 'sentence_embedding_cache'):
            self.sentence_embedding_cache = torch.zeros(B, 64, self.encoder.cfg.hidden_size, device=idx.device)
            num_cached_sentences = torch.zeros(B, dtype=torch.long, device=idx.device)
        
        # Sentence boundaries need to be determined based on sentence_ids and the presence of the end-of-sentence token
        eos_token_id = 50259
        pad_token_id = 50257
        is_eos = idx == eos_token_id

        # Initialize the sequence processing
        for _ in range(max_new_tokens):
            B, T = idx.shape
            idx = idx if idx.size(1) <= self.encoder.seq_length else idx[:, -self.encoder.seq_length:]
            is_last_sentence = sentence_ids == (num_cached_sentences-1).unsqueeze(1).repeat(1, T)
            # Find the last eos token for each sequence
            last_cached_positions = torch.where(is_eos & is_last_sentence,
                                                torch.arange(T).unsqueeze(0).expand(B, -1), 0).argmax(dim=1)
            
            last_cached_positions = torch.where(num_cached_sentences == 0, (torch.ones(B) * -1), last_cached_positions).int()

            # Prepare the new inputs
            new_input_ids = [idx[b, last_cached_positions[b]+1:] for b in range(B)]
            new_attention_masks = [attention_mask[b, last_cached_positions[b]+1:] for b in range(B)]
            new_sentence_ids = [sentence_ids[b, last_cached_positions[b]+1:] - num_cached_sentences[b] for b in range(B)]
            
            # Pad inputs to align them
            max_length = max(len(ni) for ni in new_input_ids)
            new_length = ((max_length + 7) // 8) * 8  # Round up to the nearest multiple of 8

            input_ids2 = torch.stack([F.pad(ni, (0, new_length - len(ni)), value=50257) for ni in new_input_ids]).to(self.encoder.device)
            attention_mask2 = torch.stack([F.pad(nm, (0, new_length - len(nm)), value=0) for nm in new_attention_masks]).to(self.encoder.device)
            sentence_ids2 = torch.stack([F.pad(ns, (0, new_length - len(ns)), value=-1) for ns in new_sentence_ids]).to(self.encoder.device)
            
            # Get sentence embeddings
            embeddings = self.encoder.embedding(input_ids2)

            embeddings, sentence_indices, sentence_attention_mask, _, _ = self.encoder.forward_encoder(embeddings, attention_mask2, sentence_ids2, is_triangular=False)
            last_sentence_embeddings = fetch_last_sentence_embeddings(embeddings, sentence_indices, self.cfg.wlt_embedding_width, attention_mask2)

            # Update cache by adding last_sentence_embeddings shifted right by num_sentences for each batch element
            num_new_sentences = (sentence_ids2.max(dim=1)[0] + 1).cpu()
            sentence_body_input = self.sentence_embedding_cache.clone().to(self.encoder.device)

            for b in range(B):
                new_sentence_embeddings_b = last_sentence_embeddings[b, :num_new_sentences[b]]
                sentence_body_input[b, num_cached_sentences[b]:num_cached_sentences[b]+num_new_sentences[b]] = new_sentence_embeddings_b
                            
            max_current_sentences = (num_new_sentences + num_cached_sentences).max()
            sentence_body_input = sentence_body_input[:, :max_current_sentences]
            sentence_cache_candidate = sentence_body_input.clone()

            range_tensor = torch.arange(max_current_sentences).expand(B, max_current_sentences)
            sentence_attention_mask = range_tensor < (num_new_sentences + num_cached_sentences).unsqueeze(1)

            if self.encoder.seq_first:
                sentence_body_input = sentence_body_input.transpose(0, 1)
            for layer in self.encoder.slt_body:
                sentence_body_input = layer(sentence_body_input, key_padding_mask=~sentence_attention_mask)

            if self.encoder.seq_first:
                sentence_body_input = sentence_body_input.transpose(0, 1)

            y = self.encoder.final_norm(sentence_body_input)
            y = y[torch.arange(B), (num_new_sentences + num_cached_sentences)-1]

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
            
            # Update sequences
            idx = concatenate_with_padding(idx, idx_next, pad_token_id)
            attention_mask = concatenate_with_padding(attention_mask, torch.ones_like(idx_next), 0)
            #extremely hacky code, basically sentence_ids2.max(dim=1)[0].unsqueeze(1).cpu() denotes the current sentence and is_eos[torch.arange(B), attention_mask.sum(dim=1)-2] looks up if the last token was an eos token
            sentence_ids = concatenate_with_padding(sentence_ids, sentence_ids.max(dim=1)[0].unsqueeze(1).cpu() + is_eos[torch.arange(B), attention_mask.sum(dim=1)-2].unsqueeze(1), -1)
            is_eos = idx == eos_token_id
            # Write all sentence embeddings that ended with a 50259 into the cache
            old_num_cached_sentences = num_cached_sentences.clone()
            num_cached_sentences = is_eos.sum(dim=1)

            for b in range(B):
                self.sentence_embedding_cache[b, :num_cached_sentences[b]] = sentence_cache_candidate[b, :is_eos[b].sum()]
            
        return idx
    
class ScriptableTHFLMFinetune(PreTrainedModel):
    config_class = GPTHFConfig

    def __init__(self, config):
        
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)

        self.wlt_encoder = torch.nn.ModuleList([TransformerEncoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_encoder_num_hidden_layers)])
        self.body_dimension = self.cfg.wlt_embedding_width * self.cfg.hidden_size
        self.slt_body = torch.nn.ModuleList([TransformerEncoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.slt_body_num_hidden_layers)])
        self.wlt_decoder = torch.nn.ModuleList([TransformerEncoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_decoder_num_hidden_layers)])

        #self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = torch.nn.LayerNorm(self.body_dimension, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.encoder_mode = self.cfg.encoder_mode
        self.mode = self.cfg.mode
        self.nheads = self.cfg.attention.num_attention_heads
        self.seq_first = self.wlt_encoder[0].LAYOUT == "[S B H]"

    def forward(self, input_ids, attention_mask, sentence_ids=None):
        B, sent_len = input_ids.shape  
        #print(sent_len)

        if self.encoder_mode == 'block':
            block_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            block_mask, sentence_attention_mask, sentence_indices = block_mask.to(self.device), sentence_attention_mask.to(self.device), sentence_indices.to(self.device)
            block_mask = block_mask.unsqueeze(1).expand(-1, self.nheads, -1, -1).reshape(-1, sent_len, sent_len) # expand to the number of heads
        else:
            _, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            block_mask = torch.tril(torch.ones(sent_len, sent_len, device=input_ids.device, dtype=torch.bool), diagonal=-1)

        x = self.embedding(input_ids) # (batch_size, sent_len, dim)

        word_embeddings = x
        word_triangular_mask = torch.triu(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool), diagonal=1).unsqueeze(0)
        block_triangular_mask = word_triangular_mask | block_mask

        if self.seq_first:
            word_embeddings = word_embeddings.transpose(0, 1)
            block_triangular_mask = block_triangular_mask.view(B, self.nheads, sent_len, sent_len)

        for layer in self.wlt_encoder:
            word_embeddings = layer(word_embeddings, attention_mask=block_triangular_mask)     #process padding tokens as well and only throw them away at a later step

        if self.seq_first:
            word_embeddings = word_embeddings.transpose(0, 1)
        
        sentence_embeddings = word_embeddings

        if self.mode == 'fast':
            sentence_indices2 = last_sentence_indices(sentence_indices, key_padding_mask=attention_mask)
            column_mask = create_column_mask(sentence_indices2, sent_len)
            column_triangular_mask = torch.tril(torch.ones(sent_len, sent_len, device=block_mask.device, dtype=torch.bool)) & column_mask
            column_triangular_mask = column_triangular_mask.repeat_interleave(self.nheads, dim=0)
            column_diagonal_mask = column_triangular_mask | torch.eye(sent_len, device=column_triangular_mask.device, dtype=torch.bool)
            body_attention_mask = ~column_diagonal_mask
        else:
            body_attention_mask = word_triangular_mask.expand(B*self.nheads, -1, -1)
        
        if self.embedding.pos_embedding is not None:
            sentence_embeddings = sentence_embeddings + self.embedding.pos_embedding(sentence_embeddings)

        if self.seq_first:
            sentence_embeddings = sentence_embeddings.transpose(0, 1)
            body_attention_mask = ~combine_attention_masks(~body_attention_mask, attention_mask.repeat_interleave(self.nheads, dim=0))
            body_attention_mask = body_attention_mask.view(B, self.nheads, sent_len, sent_len)


        for layer in self.slt_body:
            sentence_embeddings = layer(sentence_embeddings, attention_mask=body_attention_mask, key_padding_mask=~attention_mask)

        if self.seq_first:
            sentence_embeddings = sentence_embeddings.transpose(0, 1)

        y = fetch_last_sentence_embeddings(sentence_embeddings, sentence_indices, self.cfg.wlt_embedding_width, key_padding_mask=attention_mask)

        return self.final_norm(y), sentence_attention_mask

class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = GPTHFConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableTHFLMFinetune(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)
        self.activation = torch.nn.Tanh()
        self.dual_head = torch.nn.Linear(2*self.num_labels, self.num_labels)


        print(self.pooler)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor]=None, sentence_ids=None, labels: Optional[torch.Tensor] = None):
        logits, logits_mask = self.encoder(input_ids, attention_mask=attention_mask, sentence_ids=sentence_ids)
        logits = self.pooler(logits, logits_mask)
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

