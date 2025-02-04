"""Script to evaluate a pretrained model."""

import torch
import hydra

import logging

import cramming
from transformers import AutoTokenizer
from cramming.utils import system_startup, pathfinder
import re

from transformers import GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig

import nltk
nltk.download('punkt')

from profiler import FlopsProfiler
from transformers import AutoModel
import datasets
import os
import time

log = logging.getLogger(__name__)
cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../data')

def load_tiny_openwebtext():
    dataset = datasets.load_dataset("stas/openwebtext-10k", split='train', cache_dir=cache_path)
    return dataset

def load_openwebtext():
    dataset = datasets.load_dataset("Skylion007/openwebtext", split='train', cache_dir=cache_path)
    return dataset

def load_wikipedia():
    dataset = datasets.load_dataset("wikipedia", "20220301.en", split='train', cache_dir=cache_path)
    return dataset

def main_generate_process(cfg, setup):
    """This function controls the central routine."""
    cfg = pathfinder(cfg)
    openwebtext = load_tiny_openwebtext()

    print(openwebtext if openwebtext is not None else "None")

    #if cfg.impl.resume_run_after_preempt:
    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    tokenizer.add_tokens('<|endofsentence|>')

    model = cramming.construct_model(cfg_arch, len(tokenizer), downstream_classes=None)
    model_engine, _, _,_ , _ = cramming.load_backend(model, openwebtext, tokenizer, cfg.train, cfg.impl, setup=setup, is_generate=True)
    model_engine.load_checkpoint(cfg_arch, model_file)
    model = model_engine.model

    # else:
    #     tokenizer = AutoTokenizer.from_pretrained('gpt2')
    #     if tokenizer.pad_token is None:
    #         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     if tokenizer.sep_token is None:
    #         tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    #     tokenizer.add_tokens('<|endofsentence|>')
    #     model = cramming.construct_model(cfg.arch, len(tokenizer))
    #     model_engine, _, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup, is_generate=True)

    # prompt = ["The quick brown fox jumps over the lazy dog. The dog barks at the fox. The fox runs away. The dog chases the fox. The fox escapes. The dog returns home.",
    #           "The cat is sleeping on the windowsill. The sun is shining through the window.",]

    # inputs = tokenize_and_sentenize(tokenizer, prompt)

    generate_and_profile(model, openwebtext, tokenizer, 'GPTHF_Llama', cfg, 'fast', use_cache=False, n_samples=1)
    #generate_and_profile(model, tokenizer, 'GPTHF_Llama', inputs, cfg, 'fast', use_cache=False)
    generate_and_profile(model, openwebtext, tokenizer, 'GPTHF_Llama', cfg, 'slow', use_cache=False, n_samples=1)

    # # gpt = GPT2LMHeadModel.from_pretrained('gpt2')
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.add_tokens('<|endofsentence|>')
    # inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
    # # gpt.resize_token_embeddings(len(tokenizer))  # Adjust the model's embeddings
    # # generate_and_profile(gpt, 'GPT2', inputs, cfg)
    # print(inputs)

    # config = LlamaConfig(
    #     vocab_size=50260,  # Standard for OPT models
    #     hidden_size=768,   # Example size, change according to the model scale you want
    #     num_hidden_layers=12,
    #     num_attention_heads=12,
    #     intermediate_size=3072,
    #     max_position_embeddings=1024
    #     # Add any other parameters as needed
    # )

    # llama = LlamaForCausalLM(config)
    # llama.resize_token_embeddings(len(tokenizer))  # Adjust the model's embeddings
    # generate_and_profile(llama, tokenizer, 'LLAMA', inputs, cfg, use_cache=False)
    # generate_and_profile(llama, tokenizer, 'LLAMA', inputs, cfg, use_cache=True)

def generate_and_profile(model, dataset, tokenizer, model_name, cfg, mode='slow', use_cache=False, n_samples=1000, batch_size=1):
    prof = FlopsProfiler(model)

    model.eval()
    prof.start_profile()

    kwargs = {'max_new_tokens': 500, 'temperature': cfg.temperature, 'do_sample': cfg.do_sample, 'top_k': cfg.top_k, 'use_cache': use_cache}
    dataset_iter = iter(dataset)

    i = 0
    while i < n_samples:
        examples = []
        for _ in range(batch_size):
            example = next(dataset_iter)['text'][20]
            examples.append(example)
            i += 1
            if i >= n_samples:
                break

        inputs = tokenize_and_sentenize(tokenizer, prompt=examples)
        if model_name == 'GPTHF' or model_name == 'GPTHF_Llama':
            args = [inputs['input_ids'], inputs['attention_mask'], inputs['sentence_ids']]
        else:
            args = [inputs['input_ids']]
            kwargs['attention_mask'] = inputs['attention_mask']

        if mode == 'fast':
            output = model.generate_fast(*args, **kwargs)    
        else:
            output = model.generate(*args, **kwargs)

    prof.stop_profile()
    flops = prof.get_total_flops() / 1e12
    macs = prof.get_total_macs() / 1e12
    params = prof.get_total_params() / 1e6
    #prof.print_model_profile(profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None)
    prof.end_profile()

    print(f"Model: {model_name}")
    if mode == 'fast':
        print(f"Fast generation")
    print(f"Avg FLOPs per sample: {(flops/n_samples):.3f}T, Avg MACs per batch: {(macs/n_samples):.3f}T, Params: {params:.3f}M")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    #Now measure runtime, without profiling since it might slow down
    i = 0
    times = []
    while i < n_samples:
        examples = []
        for _ in range(batch_size):
            example = next(dataset_iter)['text']
            examples.append(example)
            i += 1

        inputs = tokenize_and_sentenize(tokenizer, prompt=examples)
        if model_name == 'GPTHF' or model_name == 'GPTHF_Llama':
            args = [inputs['input_ids'], inputs['attention_mask'], inputs['sentence_ids']]
        else:
            args = [inputs['input_ids']]
            kwargs['attention_mask'] = inputs['attention_mask']

        start = time.time()
        if mode == 'fast':
            try:
                output = model.generate_fast(*args, **kwargs)    
            except RuntimeError as e:
                print(inputs['input_ids'].shape)
        else:
            output = model.generate(*args, **kwargs)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / n_samples
    print(f"Avg time per sample: {avg_time:.3f}s")
    print('---------------------------------')


def is_complete_sentence(text):
    # Use nltk to tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Check the last sentence for proper ending punctuation
    if sentences:
        last_sentence = sentences[-1]
        return bool(re.match(r'.*[.!?]$', last_sentence.strip()))
    return False
                        
def tokenize_and_sentenize(tokenizer, prompt, max_size=512):
    if not prompt:
        return {'input_ids': [], 'attention_mask': [], 'sentence_ids': []}
    
    all_input_ids = []
    all_attention_masks = []
    all_sentence_ids = []

    MAX_SIZE = max_size

    for text_entry in prompt:
        sentences = nltk.sent_tokenize(text_entry)
        if not sentences:
            continue

        input_ids = []
        sentence_ids = []
        attention_masks = []

        for idx, sentence in enumerate(sentences):
            tokenized_sentence = tokenizer.encode_plus(sentence, truncation=True, return_attention_mask=True, max_length=MAX_SIZE)
            projected_length = len(input_ids) + len(tokenized_sentence['input_ids']) + 1

            if projected_length > MAX_SIZE:
                remaining_length = MAX_SIZE - len(input_ids)
                truncated_input_ids = tokenized_sentence['input_ids'][:remaining_length]
                truncated_attention_masks = tokenized_sentence['attention_mask'][:remaining_length]

                input_ids.extend(truncated_input_ids)
                sentence_ids.extend([idx] * remaining_length)
                attention_masks.extend(truncated_attention_masks)
                break
            elif projected_length == MAX_SIZE:
                input_ids.extend(tokenized_sentence['input_ids'] + tokenizer.encode('<|endofsentence|>'))
                sentence_ids.extend([idx] * (len(tokenized_sentence['input_ids']) + 1))
                attention_masks.extend(tokenized_sentence['attention_mask'] + [1])
                break
            else:
                if is_complete_sentence(sentence):
                    input_ids.extend(tokenized_sentence['input_ids'] + tokenizer.encode('<|endofsentence|>'))
                    sentence_ids.extend([idx] * (len(tokenized_sentence['input_ids']) + 1))
                    attention_masks.extend(tokenized_sentence['attention_mask'] + [1])
                else:
                    input_ids.extend(tokenized_sentence['input_ids'])
                    sentence_ids.extend([idx] * len(tokenized_sentence['input_ids']))
                    attention_masks.extend(tokenized_sentence['attention_mask'])

        all_input_ids.append(torch.tensor(input_ids))
        all_attention_masks.append(torch.tensor(attention_masks))
        all_sentence_ids.append(torch.tensor(sentence_ids))

    assert len(all_input_ids) == len(all_attention_masks) == len(all_sentence_ids), f"Length mismatch: {len(all_input_ids)}, {len(all_attention_masks)}, {len(all_sentence_ids)}"
    assert len(all_input_ids[0]) == len(all_attention_masks[0]) == len(all_sentence_ids[0]), f"Length mismatch: {len(all_input_ids[0])}, {len(all_attention_masks[0])}, {len(all_sentence_ids[0])}"

    max_length = min(max([len(input_ids) for input_ids in all_input_ids]), max_size)
    for i in range(len(all_input_ids)):
        padding_length = max_length - len(all_input_ids[i])
        all_input_ids[i] = torch.nn.functional.pad(all_input_ids[i], (0, padding_length), value=tokenizer.pad_token_id)
        all_attention_masks[i] = torch.nn.functional.pad(all_attention_masks[i], (0, padding_length), value=0)
        all_sentence_ids[i] = torch.nn.functional.pad(all_sentence_ids[i], (0, padding_length), value=-1)

    return {'input_ids': torch.stack(all_input_ids), 'attention_mask': torch.stack(all_attention_masks), 'sentence_ids': torch.stack(all_sentence_ids)}

        
@hydra.main(config_path="cramming/config", config_name="cfg_generate", version_base="1.1")
def launch(cfg):
    setup, kWh_counter = system_startup(cfg)
    main_generate_process(cfg, setup=setup)

if __name__ == "__main__":
    launch()
