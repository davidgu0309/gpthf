import torch
from transformers import PreTrainedModel
from torch import nn

from .configuration_bert import BERTConfig

from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
import math

class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.pe[:, : input_ids.shape[1], :]


class ScaledSinusoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(torch.tensor([1.0 / embedding_dim**0.5]))

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.scale_factor * self.pe[:, : input_ids.shape[1], :]
    
class CrammingTransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads)

        # GLU replaces the standard feedforward network
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size * 2)  # Notice the *2
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout(src2)

        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        gate, linear = src2.chunk(2, dim=-1)  # Splitting the output for gating
        src2 = torch.sigmoid(gate) * linear  # GLU operation
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout(src2)

        return src


class BERT(PreTrainedModel):
    config_class = BERTConfig

    def __init__(self, config: BERTConfig, device="cuda"):
        super().__init__(config)

        #encoder_layer = PreNormTransformerEncoderLayer(config)
        encoder_layer = nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_dropout_prob, activation='gelu', batch_first=True)

        for module in encoder_layer.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.in_proj_bias = None
                module.out_proj.bias = None
            if isinstance(module, nn.Linear):
                module.bias = None

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, device=device)
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        #Tie weights
        self.lm_head.weight = self.word_embeddings.weight

        #self.lm_head = BertOnlyMLMHead(config)

        # self.position_encoding = torch.zeros((config.max_position_embeddings, config.hidden_size), device=device)
        # X = torch.arange(config.max_position_embeddings, dtype=torch.float32, device=device).reshape(-1, 1) \
		# 	/ torch.pow(10000, torch.arange(0, config.hidden_size, 2, dtype=torch.float32, device=device) / config.hidden_size)
        # self.position_encoding[:, 0::2] = torch.sin(X)
        # self.position_encoding[:, 1::2] = torch.cos(X)
        self.position_embeddings = ScaledSinusoidal(config.hidden_size, config.max_position_embeddings)
        #self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.word_embeddings(input_ids) + self.position_embeddings(input_ids)
        x = self.embedding_layer_norm(x)
        x = self.encoder(x, src_key_padding_mask=~(attention_mask).bool()) # (batch_size, sent_len, dim)
        token_logits = self.lm_head(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(token_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return loss, token_logits
        
        return token_logits
