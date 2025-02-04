"""Script to evaluate a pretrained model."""

import torch
import hydra

import logging
from collections import defaultdict

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
import numpy as np
import pandas as pd
from filelock import FileLock


import matplotlib.pyplot as plt

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

def load_thf_model(cfg, setup):
    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    tokenizer.add_tokens('<|endofsentence|>')
    model = cramming.construct_model(cfg_arch, len(tokenizer), downstream_classes=None)
    model_engine, _, _,_ , _ = cramming.load_backend(model, None, tokenizer, cfg.train, cfg.impl, setup=setup, is_generate=True)
    model_engine.load_checkpoint(cfg_arch, model_file)
    model = model_engine.model
    return model, tokenizer

def load_gpt_tokenizer(cfg, setup):
    GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens('<|endofsentence|>')
    return tokenizer
  
CONFIGURATIONS = [(100,100), (100,250), (250,100), (250,250), (20, 500), (500, 20)]

def evaluate_model(model, dataset, tokenizer, model_name, cfg, use_cache=False):
    model.eval()
    for prompt_length, max_new_tokens in CONFIGURATIONS:
        if 'fast' in cfg.name:
            generate_and_profile(model, dataset, tokenizer, model_name, cfg, 'fast', use_cache, n_batches=10, batch_size=cfg.batch_size, prompt_length=prompt_length, max_new_tokens=max_new_tokens)
        generate_and_profile(model, dataset, tokenizer, model_name, cfg, 'slow', use_cache, n_batches=1, batch_size=cfg.batch_size, prompt_length=prompt_length, max_new_tokens=max_new_tokens)
       
def main_generate_process(cfg, setup):
    """This function controls the central routine."""
    cfg = pathfinder(cfg)
    if cfg.dataset == 'openwebtext':
        dataset = load_tiny_openwebtext()
    elif cfg.dataset == 'wikipedia':
        dataset = load_wikipedia()

    print(f"Results for {cfg.dataset}, batch size {cfg.batch_size}")
    OUR_MODELS = ['gpthf-llama-8-4-c512-d1024-48h-slow', 'gpthf-llama-8-4-c512-d1024-48h-fast']

    for model in OUR_MODELS:
        print(f"Results for model: {model}")
        cfg.name = model
        model, tokenizer = load_thf_model(cfg, setup)
        if 'fast' in cfg.name:
            plot_sentence_vs_results(model, dataset, tokenizer, 'GPTHF_Llama', cfg, mode='fast', use_cache=False, n_batches=5, batch_size=cfg.batch_size, prompt_length=100, max_new_tokens=20)
        else:
            plot_sentence_vs_results(model, dataset, tokenizer, 'GPTHF_Llama', cfg, mode='slow', use_cache=False, n_batches=5, batch_size=cfg.batch_size, prompt_length=100, max_new_tokens=20)
    #BASELINES = ['llama-small', 'llama-standard', 'opt-125m', 'opt-350m']

    # for model_name in OUR_MODELS:
    #     print(f'Results for model: {model_name}')
    #     cfg.name = model_name
    #     model, tokenizer = load_thf_model(cfg, setup)
    #     evaluate_model(model, dataset, tokenizer, 'GPTHF_Llama', cfg)

    # for model_name in BASELINES:
    #     print(f'Results for model: {model_name}')
    #     cfg.name = model_name
    #     model, tokenizer = load_thf_model(cfg, setup)
    #     evaluate_model(model, dataset, tokenizer, 'BASELINE', cfg)

def generate_and_profile(model, dataset, tokenizer, model_name, cfg, mode='slow', use_cache=False, n_batches=1, batch_size=1, prompt_length=100, max_new_tokens=100):
    prof = FlopsProfiler(model)

    model.eval()

    is_our_model = model_name == 'GPTHF' or model_name == 'GPTHF_Llama'

    kwargs = {'max_new_tokens': max_new_tokens, 'temperature': cfg.temperature, 'do_sample': cfg.do_sample, 'top_k': cfg.top_k if cfg.do_sample else None}
    dataset_iter = iter(dataset)

    document_lengths = []
    num_sentences = []
    flops = []
    times = []
    sentence_vs_flops = defaultdict(list)
    sentence_vs_times = defaultdict(list)

    for _ in range(n_batches):
        examples = []
        for _ in range(batch_size):
            example = next(dataset_iter)['text']
            examples.append(example)

        if is_our_model:
            inputs = tokenize_and_sentenize(tokenizer, prompt=examples)
            args = [inputs['input_ids'][:, :prompt_length].to(torch.device('cuda')),
                     inputs['attention_mask'][:, :prompt_length].to(torch.device('cuda')), 
                     inputs['sentence_ids'][:, :prompt_length].to(torch.device('cuda'))]            
        else:
            inputs = tokenizer(examples, return_tensors='pt', max_length=512, truncation=True, padding=True).to(torch.device('cuda'))
            args = [inputs['input_ids'][:, :prompt_length]]
            kwargs['attention_mask'] = inputs['attention_mask'][:, :prompt_length].to(torch.device('cuda'))
            document_lengths.append(inputs['input_ids'].shape[1])

        avg_sentences = inputs['sentence_ids'][:, :prompt_length].max(dim=1)[0].float().mean() + 1

        prof.start_profile()
        with torch.no_grad():
            if mode == 'fast':
                output = model.generate_fast(*args, **kwargs)    
            else:
                if is_our_model:
                    output = model.generate(*args, **kwargs)
                else:
                    output = model.generate(generation_config=None, *args, **kwargs)
        prof.stop_profile()
        measured_flops = prof.get_total_flops()
        flops.append(measured_flops)
        sentence_vs_flops[avg_sentences].append(measured_flops)

        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if mode == 'fast':
                output = model.generate_fast(*args, **kwargs)    
            else:
                if is_our_model:
                    output = model.generate(*args, **kwargs)
                else:
                    output = model.generate(generation_config=None, *args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            measured_time = start.elapsed_time(end)
            times.append(measured_time)
            sentence_vs_times[avg_sentences].append(measured_time)

    params = prof.get_total_params() / 1e6
    prof.end_profile()
        
    print(f"Model: {cfg.name}")
    if mode == 'fast':
        print(f"Fast generation")
    n_samples = n_batches * batch_size
    print(f"Result for Prompt length: {prompt_length}, Max new tokens: {max_new_tokens}")
    print(f"Avg FLOPs per sample: {(sum(flops)/(n_samples * 1e12)):.3f}T, Params: {params:.3f}M")
    print(f"Std FLOPs per sample: {np.std(np.array(flops))/1e12:.3f}T")
    print(f"Avg time per sample: {sum(times)/n_samples:.3f}ms, Std time per sample: {np.std(np.array(times))/n_samples:.3f}ms")
    print(f"Median time per sample: {np.median(np.array(times))/n_samples:.3f}ms")
    print("------------------------------------")

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'outputs')


def plot_sentence_vs_results(model, dataset, tokenizer, model_name, cfg, mode='slow', use_cache=False, n_batches=1, batch_size=1, prompt_length=100, max_new_tokens=100):
    #FILE_NAME_TIME = os.path.join(base_path, f'sentence_vs_time_{batch_size}.csv')
    FILE_NAME_FLOPS = os.path.join(base_path, f'sentence_vs_flops_{batch_size}.csv')
    prof = FlopsProfiler(model)
    model.eval()

    kwargs = {'max_new_tokens': max_new_tokens, 'temperature': cfg.temperature, 'do_sample': cfg.do_sample, 'top_k': cfg.top_k if cfg.do_sample else None}
    dataset_iter = iter(dataset)

    sentence_vs_flops = defaultdict(list)
    sentence_vs_times = defaultdict(list)

    prompt_lengths = [50,100,150,200,250,300,350,400,450,500]

    for prompt_length in prompt_lengths:
        for _ in range(n_batches):
            examples = []
            for _ in range(batch_size):
                example = next(dataset_iter)['text']
                examples.append(example)

            inputs = tokenize_and_sentenize(tokenizer, prompt=examples)
            args = [inputs['input_ids'][:, :prompt_length].to(torch.device('cuda')),
                    inputs['attention_mask'][:, :prompt_length].to(torch.device('cuda')), 
                    inputs['sentence_ids'][:, :prompt_length].to(torch.device('cuda'))]            
            avg_sentences = inputs['sentence_ids'][:, :prompt_length].max(dim=1)[0].float().mean() + 1

            prof.start_profile()
            with torch.no_grad():
                if mode == 'fast':
                    output = model.generate_fast(*args, **kwargs)
                else:
                    output = model.generate(*args, **kwargs)
                prof.stop_profile()
                measured_flops = prof.get_total_flops()
                sentence_vs_flops[avg_sentences.item()].append(measured_flops)

            # with torch.no_grad():
            #     start = torch.cuda.Event(enable_timing=True)
            #     end = torch.cuda.Event(enable_timing=True)
            #     start.record() 
            #     if mode == 'fast':
            #         output = model.generate_fast(*args, **kwargs)
            #     else:  
            #         output = model.generate(*args, **kwargs)     
            #     end.record()
            #     torch.cuda.synchronize()
            #     measured_time = start.elapsed_time(end)
            #     sentence_vs_times[avg_sentences.item()].append(measured_time)

    params = prof.get_total_params() / 1e6
    prof.end_profile()
        
 
    # plt.figure()
    # plt.scatter(list(sentence_vs_times.keys()), [np.mean(sentence_vs_times[key]) for key in sentence_vs_times.keys()], color='b', marker='o')  # Using scatter here
    # plt.xlabel("Avg Number of sentences in batch")
    # plt.ylabel("Inference time (ms)")
    # plt.legend()
    # plt.savefig(f'{cfg.base_dir}/{cfg.name}_{mode}_{batch_size}_sentence_vs_time2.png')
    # print(f"Saved plot to {cfg.base_dir}/{cfg.name}/{mode}_{batch_size}_sentence_vs_time2.png")
    # plt.close()

    plt.figure()
    plt.scatter(list(sentence_vs_flops.keys()), [np.mean(sentence_vs_flops[key]) for key in sentence_vs_flops.keys()], color='r', marker='o')  # Using scatter here
    plt.xlabel("Avg Number of sentences in batch")
    plt.ylabel("FLOPs speedup (x)")
    plt.legend()
    plt.savefig(f'{cfg.base_dir}/{cfg.name}_{mode}_{batch_size}_sentence_vs_flops2.png')
    plt.close()

    #save the dictionaries as pandas dataframes, store the keys under one column and the values under another. add a new index column. add a new column with the model name
    # df_time = pd.DataFrame(list(sentence_vs_times.items()), columns=['avg_sentences', 'time'])
    # df_time['model'] = cfg.name
    df_flops = pd.DataFrame(list(sentence_vs_flops.items()), columns=['avg_sentences', 'flops'])
    df_flops['model'] = cfg.name

    #append_to_csv(df_time, FILE_NAME_TIME)
    append_to_csv(df_flops, FILE_NAME_FLOPS)


    
def append_to_csv(df, file_name):
    lock = FileLock(file_name + '.lock')
    with lock:
        header = not os.path.exists(file_name)
        # Open the file in append mode with header conditionally
        df.to_csv(file_name, mode='a', header=header, index=False)


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
