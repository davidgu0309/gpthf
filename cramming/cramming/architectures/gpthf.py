import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
from .thf_utils import fetch_sentence_embeddings, generate_block_matrix, thf_pooling_mask

class GPTHFConfig(PretrainedConfig):
    model_type = "GPTHF"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_gpthf(cfg_arch, vocab_size, downstream_classes=None):
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
        #raise NotImplementedError("Fine-tuning for generative model not implemented yet.")
        model = ScriptableLMForPreTraining(config)
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

    def forward(self, query, hidden_states, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        return self.attn(query=query, key=hidden_states, value=hidden_states, attn_mask=attention_mask, need_weights=False, is_causal=is_causal)[0]


class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        return self.dense(self.self_attention(hidden_states, attention_mask=attention_mask, key_padding_mask=key_padding_mask, is_causal=is_causal))
    
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

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, is_causal=False):
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask, key_padding_mask, is_causal=is_causal))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states
    
class GatingLayer(torch.nn.Module):
    def __init__(self, hidden_size, cfg_arch):
        super().__init__()
        self.G1 = torch.nn.Linear(hidden_size, hidden_size, bias=cfg_arch.use_bias)
        self.G2 = torch.nn.Linear(hidden_size, hidden_size, bias=cfg_arch.use_bias)
        self.V = torch.nn.Linear(hidden_size, hidden_size, bias=cfg_arch.use_bias)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, query, sentence_embedding):
        return self.sigmoid(self.G1(query) + self.G2(sentence_embedding)) * self.V(sentence_embedding)
    
class GPTHFDecoderLayer(torch.nn.Module):
    """A transformer decoder layer with two modes: gating and cross-attention. cross-attention mode is like a regular transformer layer. Mode gating 
        assumes that the key and element vector will only come from one sentence embedding and therefore 
        replaces the cross-attention layer with a Gating layer, which computes the function o_t = sigmoid(G1 * q_t + G2 * s) * V * s, 
        where G1, G2, V are linear layers and s is the sentence embedding.
    """

    def __init__(self, idx, cfg_arch, hidden_size):
        super().__init__()
        self.cfg = OmegaConf.create(cfg_arch)
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)
        self.norm3 = _get_norm_fn(cfg_arch.norm)(hidden_size, eps=cfg_arch.norm_eps)

        self.attn = AttentionComponent(
            idx,
            hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )

        if cfg_arch.gating:
            self.gating = GatingLayer(hidden_size, cfg_arch)
        else:
            self.cross_attn = CrossAttention(hidden_size, cfg_arch.attention)

        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            hidden_size,
            cfg_arch.intermed_size,               # experiment if this should scale with wlt_embedding_with, which it currently doesnt
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, sentence_embeddings, sentence_indices, sentence_ids, attention_mask: Optional[torch.Tensor] = None):
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask))
        norm_states = self.norm2(states)

        sentence_embeddings = torch.cat([torch.zeros_like(sentence_embeddings[:, 0].unsqueeze(1)), sentence_embeddings], dim=1)

        if hasattr(self, "gating"):
            tmp_mask = sentence_indices == -1
            sentence_indices[tmp_mask] = 0
            states = states + self.dropout(self.gating(query=norm_states, sentence_embedding=sentence_embeddings[sentence_indices]))
        else:
            pooling_mask = thf_pooling_mask(sentence_ids, 1)
            pooling_mask = pooling_mask.repeat_interleave(self.cfg.attention.num_attention_heads, dim=0)
            states = states + self.dropout(self.cross_attn(query=norm_states, hidden_states=sentence_embeddings, attention_mask=~pooling_mask))

        states = states + self.dropout(self.ffn(self.norm3(states)))
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
        self.wlt_decoder = torch.nn.ModuleList([GPTHFDecoderLayer(idx, self.cfg, self.cfg.hidden_size) for idx in range(self.cfg.wlt_decoder_num_hidden_layers)])

        #self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.gating = self.cfg.gating


    def forward(self, input_ids, attention_mask, sentence_ids=None):
        _, sent_len = input_ids.shape    

        block_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids)
        block_mask, sentence_attention_mask, sentence_indices = block_mask.to(self.device), sentence_attention_mask.to(self.device), sentence_indices.to(self.device)
        block_mask = block_mask.unsqueeze(1).expand(-1, self.cfg.attention.num_attention_heads, -1, -1).reshape(-1, sent_len, sent_len) # expand to the number of heads

        position_values = self.embedding.pos_embedding(input_ids)
        #position_embedding = fetch_positional_encoding(position_values, sentence_indices)
        position_embedding = position_values

        x = self.embedding(input_ids, position_embedding) # (batch_size, sent_len, dim)

        if attention_mask is None:
            print("Attention mask is None")

        word_embeddings = x
        for layer in self.wlt_encoder:
            word_embeddings = layer(word_embeddings, attention_mask=block_mask)     #process padding tokens as well and only throw them away at a later step
        
        sentence_embeddings = fetch_sentence_embeddings(word_embeddings, sentence_indices, self.cfg.wlt_embedding_width)
        num_sentences = sentence_embeddings.shape[1]
        
        sentence_embeddings += self.embedding.pos_embedding(sentence_embeddings)

        sentence_triangular_mask = torch.triu(torch.ones(num_sentences, num_sentences, device=sentence_embeddings.device, dtype=torch.bool), diagonal=1)
        word_triangular_mask = torch.triu(torch.ones(sent_len, sent_len, device=sentence_embeddings.device, dtype=torch.bool), diagonal=1)

        for layer in self.slt_body:
            sentence_embeddings = layer(sentence_embeddings, attention_mask=sentence_triangular_mask, key_padding_mask=~(sentence_attention_mask))

        y = x

        block_triangular_mask = word_triangular_mask | block_mask

        #def forward(self, states, sentence_embeddings, sentence_indices, sentence_ids, attention_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):

        for layer in self.wlt_decoder:
            y = layer(y, sentence_embeddings, sentence_indices, sentence_ids, block_triangular_mask)

        return self.final_norm(y)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
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

        return idx



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

