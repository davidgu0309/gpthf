
import json
import pickle
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import cramming
from transformers import AutoTokenizer, BertForMaskedLM
from safetensors.torch import load_file
import logging
import datasets
import os
from nltk import sent_tokenize
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)

def main_perplexity_process(cfg, setup):
    metrics = dict()
    baseline_model = BertForMaskedLM.from_pretrained(cfg.baseline_model).to(setup["device"])
    baseline_tokenizer = AutoTokenizer.from_pretrained(cfg.baseline_model)

    hypothesis_tokenizer = AutoTokenizer.from_pretrained(cfg.hypothesis_model)
    hypothesis_model = load_hypothesis_model(cfg, hypothesis_tokenizer, setup)
    max_context_length = 512
    bucket_size = 16
    bucket_boundaries = list(range(0, max_context_length+1, bucket_size))
    dataset = datasets.load_dataset("Skylion007/openwebtext", "plain_text", streaming=True)['train']
    n_samples_per_bucket = 20
    samples_baseline = [[] for _ in range(len(bucket_boundaries) - 1)]
    samples_hypothesis = [[] for _ in range(len(bucket_boundaries) - 1)]
    baseline_perplexities = []
    hypothesis_perplexities = []
    eval_baseline = True

    dataset_iterator = iter(dataset)
    i = 0
    while not (all_buckets_full(samples_baseline, n_samples_per_bucket) or all_buckets_full(samples_hypothesis, n_samples_per_bucket)):
        sample = next(dataset_iterator)
        log.info(f"Processing sample {i + 1}")
        samples_baseline, samples_hypothesis = create_tokenized_paragraph(sample["text"], baseline_tokenizer, hypothesis_tokenizer, max_context_length, samples_baseline, samples_hypothesis, bucket_boundaries, n_samples_per_bucket, bucket_size)
        # paragraph, baseline_tokenizer, hypothesis_tokenizer, max_length_context, buckets_baseline, buckets_hypothesis, n_samples_per_bucket, bucket_size, max_length_baseline_tokenizer=512, max_length_hypothesis_tokenizer=128
        i += 1
    print(f"length of baseline buckets: {[len(bucket) for bucket in samples_baseline]}")
    print(f"length of hypothesis buckets: {[len(bucket) for bucket in samples_hypothesis]}")
    if eval_baseline:
        for i, bucket in enumerate(samples_baseline):
            print(f"processing bucket {i} of baseline samples")
            for sample in bucket:
                baseline_perplexities.append(score_baseline(baseline_model, baseline_tokenizer, sample, setup["device"]))
    for i, bucket in enumerate(samples_hypothesis):
        print(f"processing bucket {i} of hypothesis samples")
        for sample in bucket:
            hypothesis_perplexities.append(score_thf(hypothesis_model, hypothesis_tokenizer, sample, setup["device"]))
    # log the results
    log.info(f"Baseline model perplexity: {baseline_perplexities}")
    log.info(f"Hypothesis model perplexity: {hypothesis_perplexities}")
    # pickle the results
    with open(os.join(cfg.base_dir, f'baseline_perplexity_{cfg.job_id}.pkl'), 'wb') as f:
        pickle.dump(baseline_perplexities, f)
    with open(os.join(cfg.base_dir, f'hypothesis_perplexity_{cfg.job_id}.pkl'), 'wb') as f:
        pickle.dump(hypothesis_perplexities, f)

    plot_perplexity_results(baseline_perplexities, hypothesis_perplexities, cfg.base_dir, cfg.job_id)
    return metrics


def all_buckets_full(samples_per_bucket, n_samples_per_bucket):
    return all(len(bucket) >= n_samples_per_bucket for bucket in samples_per_bucket)


def plot_perplexity_results(baseline_perplexities, hypothesis_perplexities, save_path, job_id):
    # remove outliers (perplexities > 100)
    baseline_perplexities = sorted([x for x in baseline_perplexities if x[0] < 100], key=lambda x: x[1])
    hypothesis_perplexities = sorted([x for x in hypothesis_perplexities if x[0] < 100], key=lambda x: x[1])

    # store perplexities in a dictionary for each sequence length
    perplexity_per_sequence_length_baseline = {}
    for perplexity, sequence_length in baseline_perplexities:
        if sequence_length not in perplexity_per_sequence_length_baseline:
            perplexity_per_sequence_length_baseline[sequence_length] = []
        perplexity_per_sequence_length_baseline[sequence_length].append(perplexity)

    perplexity_per_sequence_length_hypothesis = {}
    for perplexity, sequence_length in hypothesis_perplexities:
        if sequence_length not in perplexity_per_sequence_length_hypothesis:
            perplexity_per_sequence_length_hypothesis[sequence_length] = []
        perplexity_per_sequence_length_hypothesis[sequence_length].append(perplexity)

    # compute average perplexity for each sequence length
    average_perplexity_per_sequence_length_baseline = {}
    for sequence_length, perplexities in perplexity_per_sequence_length_baseline.items():
        average_perplexity_per_sequence_length_baseline[sequence_length] = np.mean(perplexities)

    average_perplexity_per_sequence_length_hypothesis = {}
    for sequence_length, perplexities in perplexity_per_sequence_length_hypothesis.items():
        average_perplexity_per_sequence_length_hypothesis[sequence_length] = np.mean(perplexities)


    plt.plot(list(average_perplexity_per_sequence_length_baseline.keys()), list(average_perplexity_per_sequence_length_baseline.values()), label='Baseline')
    plt.plot(list(average_perplexity_per_sequence_length_hypothesis.keys()), list(average_perplexity_per_sequence_length_hypothesis.values()), label='Hypothesis')


    plt.xlabel('Sequence Length')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Distribution across Sequence Lengths')
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "plots", f"perplexity_distribution_{job_id}.png"))


def create_tokenized_paragraph(paragraph, baseline_tokenizer, hypothesis_tokenizer, max_length_context, buckets_baseline, buckets_hypothesis, bucket_boundaries, n_samples_per_bucket, bucket_size, max_length_baseline_tokenizer=512, max_length_hypothesis_tokenizer=128):
    sentences = sent_tokenize(paragraph)
    current_sentence_baseline = {"input_ids": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}
    current_sentence_hypothesis = {"input_ids": [], "attention_mask": []}
    for sentence in sentences:
        tokenized_sentence_baseline = baseline_tokenizer(sentence, return_tensors="pt", padding=False, truncation=True, return_token_type_ids=False, max_length=min(max_length_baseline_tokenizer, max_length_context))
        tokenized_sentence_hypothesis = hypothesis_tokenizer(sentence, return_tensors="pt", padding=False, return_token_type_ids=False, max_length=min(max_length_hypothesis_tokenizer, max_length_context), truncation=True)
        current_sentence_baseline["input_ids"] = torch.cat([current_sentence_baseline["input_ids"], tokenized_sentence_baseline["input_ids"]], dim=1)
        current_sentence_baseline["attention_mask"] = torch.cat([current_sentence_baseline["attention_mask"], tokenized_sentence_baseline["attention_mask"]], dim=1)
        current_sentence_hypothesis["input_ids"].extend(tokenized_sentence_hypothesis["input_ids"])
        current_sentence_hypothesis["attention_mask"].extend(tokenized_sentence_hypothesis["attention_mask"])
        baseline_sent_length = current_sentence_baseline["input_ids"].size(-1)
        hypothesis_sent_length = sum([sentence.size(-1) for sentence in current_sentence_hypothesis["input_ids"]])
        bucket_idx_baseline = baseline_sent_length // bucket_size
        bucket_idx_hypothesis = hypothesis_sent_length // bucket_size
        if current_sentence_baseline["input_ids"].size(-1) < max_length_context and len(buckets_baseline[bucket_idx_baseline]) < n_samples_per_bucket:
            buckets_baseline[bucket_idx_baseline].append(current_sentence_baseline)
            buckets_hypothesis[bucket_idx_hypothesis].append(current_sentence_hypothesis)
            current_sentence_baseline = {"input_ids": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}
            current_sentence_hypothesis = {"input_ids": [], "attention_mask": []}
        if current_sentence_baseline["input_ids"].size(-1) > max_length_context:
            # we start a new paragraph if the current sentence is too long
            current_sentence_baseline = {"input_ids": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}
            current_sentence_hypothesis = {"input_ids": [], "attention_mask": []}
    return buckets_baseline, buckets_hypothesis


def load_hypothesis_model(cfg, tokenizer, setup):
    device = str(setup["device"])
    model_file = os.path.join(cfg.hypothesis_model, "model.safetensors")
    model_state = load_file(model_file)
    with open(os.path.join(cfg.hypothesis_model, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))
    model = cramming.construct_model(cfg_arch, cfg.impl, cfg.data, tokenizer.vocab_size)
    try:
        sanitized_state = {}
        for k, v in model_state.items():
            if k.startswith("module."):
                k = k[7:]
            if torch.distributed.is_initialized():
                k = f"module.{k}"
            sanitized_state[k] = v
        model.load_state_dict(sanitized_state, strict=True)
    except RuntimeError as e:
        log.info(f"State dict difference is {str(e).split('Error(s) in loading state_dict for')[1]}... Ok?")
        model.load_state_dict(sanitized_state, strict=False)
    model.to(device)
    model.eval()
    return model


def score_baseline(model, tokenizer, paragraph, device):
    input_ids = paragraph["input_ids"][0]
    attention_mask = paragraph["attention_mask"][0]
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for token_idx, token in enumerate(input_ids):
        # Skip the first and last tokens, padding tokens, and if the sentence is the first token of a sentence
        # Use attention_mask to identify padding tokens
        if token in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or attention_mask[token_idx] == 0:
            continue
        # Create a copy of the input_ids
        input_ids_copy = input_ids.clone()
        # Mask the current token
        input_ids_copy[token_idx] = tokenizer.mask_token_id
        labels_copy = torch.full_like(input_ids, -100)
        # Set the label for the masked token position to its original ID (before masking)
        labels_copy[token_idx] = token
        # Add the labels tensor to the list
        labels_list.append(labels_copy)
        # Add the tensors to the respective lists
        input_ids_list.append(input_ids_copy)
        attention_mask_list.append(attention_mask)
    # Combine the lists of tensors into a single dictionary
    masked_input = {
        "input_ids": torch.stack(input_ids_list).to(device),
        "attention_mask": torch.stack(attention_mask_list).to(device),
    }
    labels = torch.stack(labels_list).to(device)
    loss = 0
    with torch.inference_mode():
        for i in range(masked_input["input_ids"].size(0)):
            output = model(**{k: v[i:i+1] for k, v in masked_input.items()}, labels=labels[i:i+1])
            loss += output.loss.item()
        loss /= masked_input["input_ids"].size(0)
    return np.exp(loss), masked_input["input_ids"].size(0)


def score_thf(model, tokenizer, paragraph, device):
    loss = 0
    max_sentence_length = max(len(sentence) for sentence in paragraph["input_ids"])
    # pad the sentences to the same length
    paragraph = tokenizer.pad(
        paragraph,
        padding='max_length',
        max_length=max_sentence_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    input_ids = paragraph["input_ids"]
    attention_mask = paragraph["attention_mask"]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    # Iterate over each sentence in the input_ids
    for sentence_idx, sentence in enumerate(input_ids):
        # Iterate over each token in the sentence
        for token_idx, token in enumerate(sentence):
            # Skip the first and last tokens, padding tokens, and if the sentence is the first token of a sentence
            # Use attention_mask to identify padding tokens
            if token in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or attention_mask[sentence_idx, token_idx] == 0:
                continue
            # Create a copy of the input_ids
            input_ids_copy = input_ids.clone()
            # Mask the current token
            input_ids_copy[sentence_idx, token_idx] = tokenizer.mask_token_id
            labels_copy = torch.full_like(input_ids, -100)
            # Set the label for the masked token position to its original ID (before masking)
            labels_copy[sentence_idx, token_idx] = token
            # Add the labels tensor to the list
            labels_list.append(labels_copy)
            # Add the tensors to the respective lists
            input_ids_list.append(input_ids_copy)
            attention_mask_list.append(attention_mask)

    # Combine the lists of tensors into a single dictionary
    masked_input = {
        "input_ids": torch.stack(input_ids_list).to(device),
        "attention_mask": torch.stack(attention_mask_list).to(device),
    }
    labels = torch.stack(labels_list).to(device)
    with torch.inference_mode():
        for i in range(masked_input["input_ids"].size(0)):
            output = model(**{k: v[i:i+1] for k, v in masked_input.items()}, labels=labels[i:i+1])
            loss += output['loss'].item()
        loss /= masked_input["input_ids"].size(0)
    return np.exp(loss), masked_input["input_ids"].size(0)



@hydra.main(config_path="cramming/config", config_name="cfg_perplexity", version_base="1.1")
def launch(cfg):
    cfg.name = f"{cfg.name}_{cfg.job_id}"
    cramming.utils.main_launcher(cfg, main_perplexity_process, job_name="perplexity evaluation")


if __name__ == "__main__":
    launch()
