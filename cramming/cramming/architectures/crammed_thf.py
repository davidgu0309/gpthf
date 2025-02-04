from turtle import pos, position
import torch
from transformers import PretrainedConfig, PreTrainedModel

from typing import Optional, final
from omegaconf import OmegaConf
import math
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import pdb
from torch.nn import init

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    _get_upsampling_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
from .attention import get_attention_mechanism
from .thf_utils import fetch_sentence_embeddings, generate_block_matrix, inverse_fetch, thf_pooling_mask

class crammedTHFConfig(PretrainedConfig):
    model_type = "crammedTHF"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_thf(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedTHFConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes
    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining(config)
        #elif config.arch["objective_layout"] == "SCRIPT":
        #    model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model

class CrossAttention(torch.nn.Module):
    """Wrapper around pytorch multi-head attention, allowing to specify a different query than for key and value."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=cfg_attention.dropout_prob,
            batch_first=True,
            bias=False,
            add_bias_kv=cfg_attention.qkv_bias,
        )

        # Do something terrible to patch the fact that the output projection is somewhere else in our code:
        del self.attn.out_proj.weight
        del self.attn.out_proj.bias
        self.attn.out_proj.register_buffer("weight", torch.eye(hidden_size))
        self.attn.out_proj.register_buffer("bias", torch.zeros(hidden_size))
        self.output_dim = hidden_size

    def forward(self, query, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(query=query, key=hidden_states, value=hidden_states, attn_mask=attention_mask, need_weights=False)[0]


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
        return self.dense(self.self_attention(hidden_states, attention_mask, key_padding_mask))

class THFPoolingComponent(torch.nn.Module):
    """Pooling components for encoding and decoding."""
    def __init__(self, hidden_size, cfg_attention, use_bias):
        super().__init__()
        self.self_attention = CrossAttention(hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, query, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.dense(self.self_attention(query=query, hidden_states=hidden_states, attention_mask=attention_mask))
    
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

class THFEnc2BodyLayer(torch.nn.Module):
    def __init__(self, cfg_arch, hidden_size):
        super().__init__()
        self.pool = THFPoolingComponent(hidden_size, cfg_arch.attention, cfg_arch.use_bias)
        self.wlt_embedding_width = cfg_arch.wlt_embedding_width
        self.norm = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)

    def forward(self, query, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        states = self.pool(query, hidden_states, attention_mask) 
        states = self.norm(states + query)
        return states
    
class THFBody2DecLayer(torch.nn.Module):
    def __init__(self, cfg_arch, hidden_size):
        super().__init__()
        self.pool = THFPoolingComponent(hidden_size, cfg_arch.attention, cfg_arch.use_bias)
        self.wlt_embedding_width = cfg_arch.wlt_embedding_width

    def forward(self, query, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        states = self.pool(query, hidden_states, attention_mask) 
        return states 
    
class QueryLayer(torch.nn.Module):
    def __init__(self, cfg_arch, hidden_size):
        super(QueryLayer, self).__init__()
        self.config = OmegaConf.create(cfg_arch)
        self.mixing_weights = nn.Parameter(torch.Tensor(cfg_arch.embedding.max_seq_length, hidden_size))
        #use proper initialization for mixing weightrs
        init.xavier_uniform_(self.mixing_weights, gain=init.calculate_gain('linear') / 2)
        #nn.init.normal_(self.mixing_weights, mean=0.0, std=0.02)
        self.norm = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)

    def forward(self, x):
        # Assuming x is of shape (batch_size, X, D)
        # Aggregate the input vectors (e.g., by averaging)
        aggregated_input = x.mean(dim=1, keepdim=True) # shape: (batch_size, 1, D)
        
        # Expand aggregated input to match mixing weights dimensions
        expanded_input = aggregated_input.expand(-1, self.config.embedding.max_seq_length, -1) # shape: (batch_size, 128, D)
        
        # Step 3: Apply mixing weights
        output = expanded_input * self.mixing_weights # Element-wise multiplication
        output = self.norm(output)
        return output


class TransformerLayer(torch.nn.Module):
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
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask, key_padding_mask))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states

class ScriptableTHFLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedTHFConfig

    def __init__(self, config):
        
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)

        self.wlt_encoder = torch.nn.ModuleList([TransformerLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_encoder_num_hidden_layers)])
        self.body_dimension = self.cfg.wlt_embedding_width * self.cfg.hidden_size
        self.slt_body = torch.nn.ModuleList([TransformerLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.slt_body_num_hidden_layers)])
        self.wlt_decoder = torch.nn.ModuleList([TransformerLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_decoder_num_hidden_layers)])

        #self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.mode = self.cfg.mode
        self.every_n = self.cfg.every_n

        if self.mode == 'avg':
            self.pool = torch.nn.AvgPool1d(self.every_n, stride=self.every_n)

        if hasattr(self.cfg, 'use_thf_pool') and self.cfg.use_thf_pool:
            #q_hidden_dim = self.cfg.hidden_size // 2

            self.thf_enc_to_body = THFEnc2BodyLayer(self.cfg, self.cfg.hidden_size)

            #self.qs = nn.Parameter(torch.Tensor(self.cfg.embedding.max_seq_length, q_hidden_dim))
            self.qs = QueryLayer(self.cfg, self.cfg.hidden_size)
            #nn.init.normal_(self.qs, mean=0.0, std=0.02)

            self.thf_body_to_dec = THFBody2DecLayer(self.cfg, self.cfg.hidden_size)


    def forward(self, input_ids, attention_mask, labels=None, sentence_ids=None):
        batch_size, sent_len = input_ids.shape    
        x = self.embedding(input_ids) # (batch_size, sent_len, dim)

       # pdb.set_trace()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if self.mode == 'regular':
            for layer in self.wlt_encoder:
                x = layer(x, key_padding_mask=~(attention_mask))    

            sent_num = math.ceil(sent_len / self.every_n)
            sentence_embeddings = x[:, ::self.every_n, :]
            sentence_attention_mask = attention_mask[:, ::self.every_n]
            sentence_ids = torch.arange(sent_num, device=self.device).repeat_interleave(self.every_n).unsqueeze(0).expand(batch_size, -1)
        
        elif self.mode == 'avg':
            for layer in self.wlt_encoder:
                x = layer(x, key_padding_mask=~(attention_mask))    

            sent_num = math.ceil(sent_len / self.every_n)
            sentence_embeddings = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
            sentence_attention_mask = attention_mask[:, ::self.every_n]
            sentence_ids = torch.arange(sent_num, device=self.device).repeat_interleave(self.every_n).unsqueeze(0).expand(batch_size, -1)


        elif self.mode == 'sentence':
            word_attention_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            word_attention_mask, sentence_attention_mask, sentence_indices = word_attention_mask.to(self.device), sentence_attention_mask.to(self.device), sentence_indices.to(self.device)
            word_attention_mask = word_attention_mask.unsqueeze(1).expand(-1, self.cfg.attention.num_attention_heads, -1, -1).reshape(-1, sent_len, sent_len) # expand to the number of heads

            for layer in self.wlt_encoder:
                x = layer(x, attention_mask=word_attention_mask)     #process padding tokens as well and only throw them away at a later step
            
            sent_num = sentence_attention_mask.size(-1)
            sentence_embeddings = fetch_sentence_embeddings(x, sentence_indices, self.cfg.wlt_embedding_width)

        else:
            raise ValueError(f"Invalid mode {self.mode} for THF.")
        
        sentence_embeddings = sentence_embeddings + self.embedding.pos_embedding(sentence_embeddings)

        if hasattr(self.cfg, 'use_thf_pool') and self.cfg.use_thf_pool:
            thf_mask = thf_pooling_mask(sentence_ids, self.cfg.wlt_embedding_width)
            thf_mask = thf_mask.unsqueeze(1).expand(-1, self.cfg.attention.num_attention_heads, -1, -1).reshape(-1, self.cfg.wlt_embedding_width*sent_num, sent_len) # expand to the number of heads
            sentence_embeddings = self.thf_enc_to_body(query=sentence_embeddings, hidden_states=x, attention_mask=~thf_mask)

        sentence_attention_mask = sentence_attention_mask.repeat_interleave(self.cfg.wlt_embedding_width, dim=1)
        for layer in self.slt_body:
            sentence_embeddings = layer(sentence_embeddings, key_padding_mask=~(sentence_attention_mask))

        sentence_embeddings = sentence_embeddings.repeat_interleave(self.cfg.wlt_embedding_multiplier, dim=1) # (batch_size, sent_len, dim)

        if hasattr(self.cfg, 'use_thf_pool') and self.cfg.use_thf_pool:
            q = self.qs(sentence_embeddings)
            y = self.thf_body_to_dec(query=q, hidden_states=sentence_embeddings)
        else:
            y = torch.nn.functional.pad(sentence_embeddings, (0, 0, 0, sent_len - sentence_embeddings.size(-2)), mode='constant', value=1.0) #+ position_encoding 

        y = y[:, :sent_len, :]
        y = y + self.embedding.pos_embedding(input_ids) # (batch_size, sent_len, dim)

        if hasattr(self.cfg, 'use_skip') and self.cfg.use_skip:
            y = y + x

        for layer in self.wlt_decoder:
            y = layer(y, key_padding_mask=~(attention_mask))

        return self.final_norm(y)

class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedTHFConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableTHFLM(config)

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
        #num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        labels = labels[mask_positions]
        # torch.masked_select(labels, mask_positions)  # not allowed as a dynamic shape operator

        # indices = torch.arange(mask_positions.shape[0], device=outputs.device)[mask_positions] # not allowed
        #indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh

        # outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        # labels = labels[indices]
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs))
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss

class ScriptableTHFLMFinetune(PreTrainedModel):

    config_class = crammedTHFConfig

    def __init__(self, config):
        
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)

        self.wlt_encoder = torch.nn.ModuleList([TransformerLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_encoder_num_hidden_layers)])
        self.body_dimension = self.cfg.wlt_embedding_width * self.cfg.hidden_size
        self.slt_body = torch.nn.ModuleList([TransformerLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.slt_body_num_hidden_layers)])

        #self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.mode = self.cfg.mode
        self.every_n = self.cfg.every_n

        if self.mode == 'avg':
            self.pool = torch.nn.AvgPool1d(self.every_n, stride=self.every_n)

        if hasattr(self.cfg, 'use_thf_pool') and self.cfg.use_thf_pool:
            #q_hidden_dim = self.cfg.hidden_size // 2

            self.thf_enc_to_body = THFEnc2BodyLayer(self.cfg, self.cfg.hidden_size)

            #self.qs = nn.Parameter(torch.Tensor(self.cfg.embedding.max_seq_length, q_hidden_dim))
            self.qs = QueryLayer(self.cfg, self.cfg.hidden_size)
            #nn.init.normal_(self.qs, mean=0.0, std=0.02)

            self.thf_body_to_dec = THFBody2DecLayer(self.cfg, self.cfg.hidden_size)


    def forward(self, input_ids, attention_mask, labels=None, sentence_ids=None):
        batch_size, sent_len = input_ids.shape    
        x = self.embedding(input_ids) # (batch_size, sent_len, dim)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if self.mode == 'regular':
            for layer in self.wlt_encoder:
                x = layer(x, key_padding_mask=~(attention_mask))    

            sent_num = math.ceil(sent_len / self.every_n)
            sentence_embeddings = x[:, ::self.every_n, :]
            sentence_attention_mask = attention_mask[:, ::self.every_n]
            sentence_ids = torch.arange(sent_num, device=self.device).repeat_interleave(self.every_n).unsqueeze(0).expand(batch_size, -1)
        
        elif self.mode == 'avg':
            for layer in self.wlt_encoder:
                x = layer(x, key_padding_mask=~(attention_mask))    

            sent_num = math.ceil(sent_len / self.every_n)
            sentence_embeddings = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
            sentence_attention_mask = attention_mask[:, ::self.every_n]
            sentence_ids = torch.arange(sent_num, device=self.device).repeat_interleave(self.every_n).unsqueeze(0).expand(batch_size, -1)

        elif self.mode == 'sentence':
            word_attention_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
            word_attention_mask, sentence_attention_mask, sentence_indices = word_attention_mask.to(self.device), sentence_attention_mask.to(self.device), sentence_indices.to(self.device)
            word_attention_mask = word_attention_mask.unsqueeze(1).expand(-1, self.cfg.attention.num_attention_heads, -1, -1).reshape(-1, sent_len, sent_len) # expand to the number of heads

            for layer in self.wlt_encoder:
                x = layer(x, attention_mask=word_attention_mask)     #process padding tokens as well and only throw them away at a later step
            
            sent_num = sentence_attention_mask.size(-1)
            sentence_embeddings = fetch_sentence_embeddings(x, sentence_indices, self.cfg.wlt_embedding_width)

        else:
            raise ValueError(f"Invalid mode {self.mode} for THF.")
        
        sentence_embeddings = sentence_embeddings + self.embedding.pos_embedding(sentence_embeddings)

        if hasattr(self.cfg, 'use_thf_pool') and self.cfg.use_thf_pool:
            thf_mask = thf_pooling_mask(sentence_ids, self.cfg.wlt_embedding_width)
            thf_mask = thf_mask.unsqueeze(1).expand(-1, self.cfg.attention.num_attention_heads, -1, -1).reshape(-1, self.cfg.wlt_embedding_width*sent_num, sent_len) # expand to the number of heads
            sentence_embeddings = self.thf_enc_to_body(query=sentence_embeddings, hidden_states=x, attention_mask=~thf_mask)

        sentence_attention_mask = sentence_attention_mask.repeat_interleave(self.cfg.wlt_embedding_width, dim=1)
        for layer in self.slt_body:
            sentence_embeddings = layer(sentence_embeddings, key_padding_mask=~(sentence_attention_mask))

        return self.final_norm(sentence_embeddings)

class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedTHFConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableTHFLMFinetune(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        logits = self.encoder(input_ids, attention_mask.bool(), **kwargs)
        logits = self.pooler(logits)
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