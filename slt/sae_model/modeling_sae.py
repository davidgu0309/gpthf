from transformers import PreTrainedModel
import torch
from torch import nn
from .configuration_sae import SAEConfig


class SAE(PreTrainedModel):
	config_class = SAEConfig

	def __init__(self, config: SAEConfig):
		super().__init__(config)
		
		wlt_encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.wlt_encoder_num_attention_heads, dim_feedforward=config.wlt_encoder_hidden_dim, dropout=config.dropout, activation=config.activation, batch_first=True)
		self.wlt_encoder = nn.TransformerEncoder(wlt_encoder_layer, num_layers=config.wlt_encoder_num_hidden_layers)

		wlt_decoder_layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.wlt_decoder_num_attention_heads, dim_feedforward=config.wlt_decoder_hidden_dim, dropout=config.dropout, activation=config.activation, batch_first=True)
		self.wlt_decoder = nn.TransformerEncoder(wlt_decoder_layer, num_layers=config.wlt_decoder_num_hidden_layers)

		self.word_embeddings = nn.Embedding(config.vocab_size, config.dim)
		self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

		self.position_encoding = torch.zeros((1, config.max_len, config.dim), device=self.device)
		X = torch.arange(config.max_len, dtype=torch.float32).reshape(-1, 1) \
			/ torch.pow(10000, torch.arange(0, config.dim, 2, dtype=torch.float32) / config.dim)
		self.position_encoding[:, :, 0::2] = torch.sin(X)
		self.position_encoding[:, :, 1::2] = torch.cos(X)

	def forward_encoder(self, input_ids, attention_mask):
		# input_ids: (batch_size, seq_len)
		# attention_mask: (batch_size, seq_len)
		# output: (batch_size, seq_len, dim)
		batch_size, seq_len = input_ids.shape
		position_encoding = self.position_encoding[:, :seq_len, :].repeat(batch_size, 1, 1).to(self.device)
		x = position_encoding + self.word_embeddings(input_ids)
		x = self.wlt_encoder(x, src_key_padding_mask=~attention_mask)

		bottleneck_encoding_out = x[:, 0:self.config.wlt_embedding_width, :]
		return bottleneck_encoding_out

	def forward(self, input_ids, attention_mask):
		# input_ids: (batch_size, seq_len)
		# attention_mask: (batch_size, seq_len)
		# output: (batch_size, seq_len, dim)
		batch_size, seq_len = input_ids.shape
		position_encoding = self.position_encoding[:, :seq_len, :].repeat(batch_size, 1, 1).to(self.device)
		bottleneck_encoding_out = self.forward_encoder(input_ids, attention_mask)

		if self.config.wlt_embedding_multiplier == -1:
			bottleneck_embedding = bottleneck_encoding_out.repeat(1, seq_len // self.config.wlt_embedding_width, 1).to(self.device)
		else:
			bottleneck_embedding = bottleneck_encoding_out.repeat(1, self.config.wlt_embedding_multiplier, 1).to(self.device)
		
		y = position_encoding + torch.nn.functional.pad(bottleneck_embedding, (0, 0, 0, seq_len - bottleneck_embedding.size(-2)), mode='constant', value=1.0)
		y = self.wlt_decoder(y, src_key_padding_mask=~attention_mask)
		
		return self.lm_head(y)
	
	