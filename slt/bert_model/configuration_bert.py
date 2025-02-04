from transformers import PretrainedConfig
import torch

class BERTConfig(PretrainedConfig):
	model_type = "bert"

	def __init__(
		self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=128,
        classifier_dropout=0,
		type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
	):
		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.hidden_act = hidden_act
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.classifier_dropout = classifier_dropout
		self.num_attention_heads = num_attention_heads
		self.num_hidden_layers = num_hidden_layers
		self.intermediate_size = intermediate_size
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.pad_token_id = pad_token_id
		self.position_embedding_type = position_embedding_type
		self.use_cache = use_cache
		self.type_vocab_size = type_vocab_size
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
		super().__init__(**kwargs)