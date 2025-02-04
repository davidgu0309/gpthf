import os
from re import L, T
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import OPTForCausalLM, OPTConfig


def construct_opt_125m(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = OPTConfig(
        vocab_size=50260,  # Standard for OPT models
        hidden_size=768,   # Example size, change according to the model scale you want
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024
        # Add any other parameters as needed
    )

    model = OPTForCausalLM(config)
    return model

def construct_opt_350m(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = OPTConfig(
        vocab_size=50260,  # Standard for OPT models
        hidden_size=1024,   # Example size, change according to the model scale you want
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=1024
        # Add any other parameters as needed
    )

    model = OPTForCausalLM(config)
    return model