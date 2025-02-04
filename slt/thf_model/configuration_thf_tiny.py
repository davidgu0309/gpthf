from transformers import PretrainedConfig
import torch


class THFConfig(PretrainedConfig):
	model_type = "thf"

	def __init__(
		self,
		vocab_size = 30522,
		max_len = 128,
		max_body_len = 128,
		dim = 256,
		dropout = 0.1,
		activation="gelu",
		
		wlt_encoder_num_attention_heads = 16,
		wlt_encoder_num_hidden_layers = 2,

		wlt_embedding_width = 1,
		wlt_embedding_multiplier = 4,

		slt_body_num_attention_heads = 16,
		slt_body_num_hidden_layers = 2,

		wlt_decoder_num_attention_heads = 16,
		wlt_decoder_num_hidden_layers = 2,

		mode='regular',
		every_n=1,
		
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_len = max_len
		self.max_body_len = max_body_len
		self.dim = dim
		self.activation = activation
		self.dropout = dropout
		
		self.wlt_encoder_num_attention_heads = wlt_encoder_num_attention_heads
		self.wlt_encoder_hidden_dim = 4*dim
		self.wlt_encoder_num_hidden_layers = wlt_encoder_num_hidden_layers

		self.wlt_embedding_width = wlt_embedding_width
		self.wlt_embedding_multiplier = wlt_embedding_multiplier

		self.slt_body_num_attention_heads = slt_body_num_attention_heads
		self.slt_body_hidden_dim = 4*dim
		self.slt_body_num_hidden_layers = slt_body_num_hidden_layers
		
		self.wlt_decoder_num_attention_heads = wlt_decoder_num_attention_heads
		self.wlt_decoder_hidden_dim = 4*dim
		self.wlt_decoder_num_hidden_layers = wlt_decoder_num_hidden_layers

		self.mode = mode
		self.every_n = every_n
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
		super().__init__(**kwargs)