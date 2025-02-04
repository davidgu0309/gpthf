from transformers import PretrainedConfig


class SLTConfig(PretrainedConfig):
	model_type = "slt"

	def __init__(
		self,
		max_len = 64,
		dim = 768,
		dropout = 0.1,
		activation="gelu",
		
		num_attention_heads = 12,
		hidden_dim = 2048,
		num_hidden_layers = 4,

		num_classes = 4,
		
		**kwargs,
	):
		self.max_len = max_len
		self.dim = dim
		self.activation = activation
		self.dropout = dropout
		
		self.num_attention_heads = num_attention_heads
		self.hidden_dim = hidden_dim
		self.num_hidden_layers = num_hidden_layers

		self.num_classes = num_classes
	
		super().__init__(**kwargs)