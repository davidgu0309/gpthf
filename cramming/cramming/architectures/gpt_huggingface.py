import os
from re import L, T
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sympy import O

from transformers import GPT2Config, GPT2LMHeadModel

def construct_gpt2_small(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = GPT2Config(
        vocab_size=50260,  
        n_embd=768,   
        n_layer=12,
        n_head=12,
        n_positions=1024
        # Add any other parameters as needed
    )

    model = GPT2LMHeadModel(config)
    return model

def construct_gpt2_standard(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = GPT2Config(
        vocab_size=50260,  
        n_embd=1024,   
        n_layer=24,
        n_head=16,
        n_positions=1024
        # Add any other parameters as needed
    )

    model = GPT2LMHeadModel(config)
    return model