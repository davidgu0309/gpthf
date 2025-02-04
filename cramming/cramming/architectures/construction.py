"""Interface to construct models."""

from .huggingface_interface import construct_huggingface_model
from .funnel_transformers import construct_scriptable_funnel
from .recurrent_transformers import construct_scriptable_recurrent
from .sanity_check import SanityCheckforPreTraining
from .crammed_bert import construct_crammed_bert
from .crammed_thf import construct_crammed_thf
from .gpthf import construct_gpthf
from .opt import construct_opt_125m, construct_opt_350m
from .llama import construct_llama_small, construct_llama_standard
from .gpt_huggingface import construct_gpt2_small, construct_gpt2_standard
from .gpthf_llama import construct_gpthf_llama

import logging
from ..utils import is_main_process

log = logging.getLogger(__name__)


def construct_model(cfg_arch, vocab_size, downstream_classes=None):
    model = None
    if cfg_arch.architectures is not None:
        # attempt to solve locally
        if "ScriptableCrammedBERT" in cfg_arch.architectures:
            model = construct_crammed_bert(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableFunnelLM" in cfg_arch.architectures:
            model = construct_scriptable_funnel(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableRecurrentLM" in cfg_arch.architectures:
            model = construct_scriptable_recurrent(cfg_arch, vocab_size, downstream_classes)
        elif "SanityCheckLM" in cfg_arch.architectures:
            model = SanityCheckforPreTraining(cfg_arch.width, vocab_size)
        elif "ScriptableCrammedTHF" in cfg_arch.architectures:
            model = construct_crammed_thf(cfg_arch, vocab_size, downstream_classes)
        elif "GPTHF" in cfg_arch.architectures:
            model = construct_gpthf(cfg_arch, vocab_size, downstream_classes)
        elif "GPTHF_Peter" in cfg_arch.architectures:
            model = construct_gpthf_peter(cfg_arch, vocab_size, downstream_classes)
        elif "OPT-125m" in cfg_arch.architectures:
            model = construct_opt_125m(cfg_arch, vocab_size, downstream_classes)
        elif "OPT-350m" in cfg_arch.architectures:
            model = construct_opt_350m(cfg_arch, vocab_size, downstream_classes)
        elif "Llama-small" in cfg_arch.architectures:
            model = construct_llama_small(cfg_arch, vocab_size, downstream_classes)
        elif "Llama-standard" in cfg_arch.architectures:
            model = construct_llama_standard(cfg_arch, vocab_size, downstream_classes)
        elif "GPT2_small" in cfg_arch.architectures:
            model = construct_gpt2_small(cfg_arch, vocab_size, downstream_classes)
        elif "GPT2_standard" in cfg_arch.architectures:
            model = construct_gpt2_standard(cfg_arch, vocab_size, downstream_classes) 
        elif "GPTHF_Llama" in cfg_arch.architectures:
            model = construct_gpthf_llama(cfg_arch, vocab_size, downstream_classes)

    if model is not None:  # Return local model arch
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with architecture {cfg_arch.architectures[0]} loaded with {num_params:,} parameters.")
        return model

    try:  # else try on HF
        model = construct_huggingface_model(cfg_arch, vocab_size, downstream_classes)
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with config {cfg_arch} loaded with {num_params:,} parameters.")
        return model
    except Exception as e:
        raise ValueError(f"Invalid model architecture {cfg_arch.architectures} given. Error: {e}")
