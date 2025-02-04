"""Script to evaluate a pretrained model."""

import torch
import hydra

import logging
from collections import defaultdict

import cramming
from transformers import AutoTokenizer, BertForSequenceClassification
from cramming.utils import system_startup, pathfinder
import re

import nltk
nltk.download('punkt')

from transformers import AutoModel
import datasets
import os
import numpy as np
from profiler import FlopsProfiler

log = logging.getLogger(__name__)
cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../data')
OUR_MODELS = ['ourbert-12-d768', 'ourbert-10-d768', 'ourbert-8-d768', 
              'thf-every2-posenc-12h', 'thf-every4-posenc-12h']
SENTENCE_MODELS = ['thf-sentence-8-4-2-d768', 'thf-sentence-6-4-2-d768', 'thf-sentence-10-4-2-d768', 'thf-sentence-8-6-4-d768']

def load_tiny_openwebtext():
    dataset = datasets.load_dataset("stas/openwebtext-10k", split='train', cache_dir=cache_path)
    return dataset

def load_thf_model(cfg, setup, keep_decoder=False):
    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    model = cramming.construct_model(cfg_arch, len(tokenizer), downstream_classes=None if keep_decoder else 2)
    model_engine, _, _,_ , _ = cramming.load_backend(model, None, tokenizer, cfg.train, cfg.impl, setup=setup, is_generate=True)
    model_engine.load_checkpoint(cfg_arch, model_file)
    model = model_engine.model
    return model, tokenizer
  
def plot_batch_size_vs_inference_speed(cfg, setup):
    batch_sizes = [1, 128]
    dataset = load_tiny_openwebtext()

    for model_name in OUR_MODELS:
        print(f'Results for model: {model_name}')
        cfg.name = model_name
        model, tokenizer = load_thf_model(cfg, setup, keep_decoder=True)
        for batch_size in batch_sizes:
            print(f"Measuring inference speed for batch size {batch_size}...")
            measure_inference_speed(model, dataset, tokenizer, 'THF', n_batches=50, batch_size=batch_size)

    for model_name in SENTENCE_MODELS:
        print(f'Results for model: {model_name}')
        cfg.name = model_name
        model, tokenizer = load_thf_model(cfg, setup, keep_decoder=False)
        for batch_size in batch_sizes:
            print(f"Measuring inference speed for batch size {batch_size}...")
            measure_inference_speed(model, dataset, tokenizer, 'THF', n_batches=50, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(torch.device('cuda'))
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("Results for HuggingFace model: bert-base-uncased")
    print(f"Measuring inference speed for batch size 1...")
    measure_inference_speed(model, dataset, tokenizer, 'HuggingFace', n_batches=50, batch_size=1)
    measure_inference_speed(model, dataset, tokenizer, 'HuggingFace', n_batches=50, batch_size=128)

def flop_experiment(cfg, setup):
    batch_sizes = [1, 128]
    dataset = load_tiny_openwebtext()
    OUR_MODELS = ['ourbert-12-d768', 'thf-sentence-8-4-2-d768']

    for model_name in OUR_MODELS:
        print(f'Results for model: {model_name}')
        cfg.name = model_name
        model, tokenizer = load_thf_model(cfg, setup, keep_decoder=True)
        for batch_size in batch_sizes:
            print(f"Measuring FLOPs for batch size {batch_size}...")
            measure_flops(model, dataset, tokenizer, 'THF', n_batches=50, batch_size=batch_size)

    for model_name in SENTENCE_MODELS:
        print(f'Results for model: {model_name}')
        cfg.name = model_name
        model, tokenizer = load_thf_model(cfg, setup, keep_decoder=False)
        for batch_size in batch_sizes:
            print(f"Measuring FLOPs for batch size {batch_size}...")
            measure_flops(model, dataset, tokenizer, 'THF', n_batches=50, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(torch.device('cuda'))
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("Results for HuggingFace model: bert-base-uncased")
    print(f"Measuring inference speed for batch size 1...")
    measure_flops(model, dataset, tokenizer, 'HuggingFace', n_batches=50, batch_size=1)
    measure_flops(model, dataset, tokenizer, 'HuggingFace', n_batches=50, batch_size=128)

def main_generate_process(cfg, setup):
    """This function controls the central routine."""
    cfg = pathfinder(cfg)
    plot_batch_size_vs_inference_speed(cfg, setup)
    flop_experiment(cfg, setup)

def measure_inference_speed(model, dataset, tokenizer, model_name, n_batches=1, batch_size=1):
    model.eval()
    is_our_model = model_name == 'THF' 
    dataset_iter = iter(dataset)
    n_batches = n_batches + 5  #because first is warmup

    times = []
    kwargs = {}
    for i in range(n_batches):
        examples = []
        for _ in range(batch_size):
            example = next(dataset_iter)['text']
            examples.append(example)

        if is_our_model:
            inputs = tokenize_and_sentenize(tokenizer, prompt=examples)
            args = [inputs['input_ids'][:, :128], inputs['attention_mask'][:, :128].bool()]
            kwargs['sentence_ids'] = inputs['sentence_ids'][:, :128]
        else:
            inputs = tokenizer(examples, return_tensors='pt', max_length=128, truncation=True, padding=True).to(torch.device('cuda'))
            args = [inputs['input_ids'][:, :128]]
            kwargs['attention_mask'] = inputs['attention_mask'][:, :128].bool().to(torch.device('cuda'))

        if i >= 5:
            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                res = model(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        else:
            with torch.no_grad():
                res = model(*args, **kwargs)

    print(f"Average inference time for batch size {batch_size}: {np.mean(times):.2f} ms")
    print(f"Standard deviation: {np.std(times):.2f} ms")
    print(f"Median inference time: {np.median(times):.2f} ms")
    print(f"Throughput: {(batch_size * 128/ np.mean(times) * 1000):.1f} tokens/s")
    return {'time: (ms)':times, 'std: (ms)': times}

def measure_flops(model, dataset, tokenizer, model_name, n_batches=1, batch_size=1):
    prof = FlopsProfiler(model)
    model.eval()
    is_our_model = model_name == 'THF' 
    dataset_iter = iter(dataset)

    flops = []
    kwargs = {}
    for _ in range(n_batches):
        examples = []
        for _ in range(batch_size):
            example = next(dataset_iter)['text']
            examples.append(example)

        if is_our_model:
            inputs = tokenize_and_sentenize(tokenizer, prompt=examples)
            args = [inputs['input_ids'][:, :128], inputs['attention_mask'][:, :128].bool()]
            kwargs['sentence_ids'] = inputs['sentence_ids'][:, :128]
        else:
            inputs = tokenizer(examples, return_tensors='pt', max_length=128, truncation=True, padding=True).to(torch.device('cuda'))
            args = [inputs['input_ids'][:, :128]]
            kwargs['attention_mask'] = inputs['attention_mask'][:, :128].bool().to(torch.device('cuda'))

        with torch.no_grad():
            prof.start_profile()
            res = model(*args, **kwargs)
            prof.stop_profile()
            flops.append(prof.get_total_flops())

    prof.end_profile()
    print(f"Average empirical FLOPs: {np.mean(flops)/(1e9*batch_size):.3f} GFLOPs")
    print(f"Standard deviation: {np.std(flops)/(1e9*batch_size):.3f} GFLOPs")
    print(f"Median FLOPs: {np.median(flops)/(1e9*batch_size):.3f} GFLOPs")
                        
def tokenize_and_sentenize(tokenizer, prompt, max_size=128):
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
            projected_length = len(input_ids) + len(tokenized_sentence['input_ids'])

            if projected_length >= MAX_SIZE:
                remaining_length = MAX_SIZE - len(input_ids)
                truncated_input_ids = tokenized_sentence['input_ids'][:remaining_length]
                truncated_attention_masks = tokenized_sentence['attention_mask'][:remaining_length]

                input_ids.extend(truncated_input_ids)
                sentence_ids.extend([idx] * remaining_length)
                attention_masks.extend(truncated_attention_masks)
                break
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

    return {'input_ids': torch.stack(all_input_ids).to(torch.device('cuda')), 
            'attention_mask': torch.stack(all_attention_masks).to(torch.device('cuda')),
            'sentence_ids': torch.stack(all_sentence_ids).to(torch.device('cuda'))}
        
@hydra.main(config_path="cramming/config", config_name="cfg_speed", version_base="1.1")
def launch(cfg):
    setup, kWh_counter = system_startup(cfg)
    main_generate_process(cfg, setup=setup)

if __name__ == "__main__":
    launch()
