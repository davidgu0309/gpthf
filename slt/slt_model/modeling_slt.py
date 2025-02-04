from transformers import PreTrainedModel
import torch
from torch import nn
from .configuration_slt import SLTConfig


class SLT(PreTrainedModel):
	config_class = SLTConfig

	def __init__(self, config: SLTConfig):
		super().__init__(config)
		
		encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.num_attention_heads, dim_feedforward=config.hidden_dim, dropout=config.dropout, activation=config.activation, batch_first=True)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

		self.classification_head = nn.Linear(config.dim, config.num_classes, bias=False)

		self.position_encoding = torch.zeros((1, config.max_len, config.dim), device=self.device)
		X = torch.arange(config.max_len, dtype=torch.float32).reshape(-1, 1) \
			/ torch.pow(10000, torch.arange(0, config.dim, 2, dtype=torch.float32) / config.dim)
		self.position_encoding[:, :, 0::2] = torch.sin(X)
		self.position_encoding[:, :, 1::2] = torch.cos(X)

	def forward(self, input_embeddings, attention_mask):
		# input_embeddings: (batch_size, seq_len, dim)
		# attention_mask: (batch_size, seq_len)
		# output: (batch_size, num_classes)
		batch_size, seq_len, dim = input_embeddings.shape
		position_encoding = self.position_encoding[:, :seq_len, :].repeat(batch_size, 1, 1).to(self.device)
		x = position_encoding + input_embeddings
		x = self.encoder(x, src_key_padding_mask=~attention_mask)
		
		classification_logits = self.classification_head(x[:, 0, :])

		return classification_logits
	
	