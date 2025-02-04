"""Utilities common to several backends."""
import os


import torch
import transformers
from torch.utils.data import DataLoader

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets.distributed import split_dataset_by_node

def get_num_workers(cfg_impl):
    if cfg_impl.threads > 0:
        return min(torch.get_num_threads() // max(1, torch.cuda.device_count()), cfg_impl.threads)
    else:
        return 0


def group_parameters(model, cfg_train):
    model_parameters = list(model.named_parameters())
    if len(cfg_train.limited_decay_keys) > 0:
        grouped_parameters = optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_parameters if not any(nd in n for nd in cfg_train.limited_decay_keys)],
                "weight_decay": cfg_train.optim.weight_decay,
            },
            {
                "params": [p for n, p in model_parameters if any(nd in n for nd in cfg_train.limited_decay_keys)],
                "weight_decay": 0.0,
            },
        ]
    else:
        grouped_parameters = [p for n, p in model_parameters]
    return grouped_parameters


def update_ema(model_parameters, ema_parameters, model_buffers, ema_buffers, momentum=0.995):
    """Update exponential moving average in parameters and buffers."""
    with torch.no_grad():
        torch._foreach_mul(ema_parameters, momentum)  # want to prevent a second call here, but doesnt seem possible as of now?
        torch._foreach_add_(ema_parameters, model_parameters, alpha=1 - momentum)

        torch._foreach_mul(ema_buffers, momentum)
        torch._foreach_add_(ema_buffers, model_buffers, alpha=1 - momentum)


def updated_latest_weight_average(model_parameters, model_buffers, store, last_k=10):
    if len(store) > last_k:
        store.pop(0)

    store.append(dict(params=model_parameters, buffers=model_buffers))
    param_store = store[0]["params"]
    [torch._foreach_add_(param_store, storage["params"]) for storage in store[1:]]
    torch._foreach_div(param_store, float(last_k))

    buffer_store = store[0]["buffers"]
    [torch._foreach_add_(buffer_store, storage["buffers"]) for storage in store[1:]]
    torch._foreach_div(buffer_store, float(last_k))

    return param_store, buffer_store


def prepare_pretraining_dataloader(dataset, tokenizer, cfg_train, cfg_impl, is_thf=False):
    print("I am making the dataloader now.")
    num_workers = get_num_workers(cfg_impl)
    print("Using workers:", num_workers)

    train_dataset = dataset.select(range(0, int(0.99*len(dataset))))
    val_dataset = dataset.select(range(int(0.99*len(dataset)), len(dataset)))

    if cfg_train.objective.name == "masked-lm":
        if is_thf:
            collate_fn = DataCollatorWithSentenceTokens(tokenizer=tokenizer, max_length=cfg_train.max_seq_length, pad_to_multiple_of=8, mlm_probability=cfg_train.objective.mlm_probability, is_mlm=True)
        else:
            collate_fn = PatchedDataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=not cfg_train.objective.disable_mlm,
                mlm_probability=cfg_train.objective.mlm_probability,
                pad_to_multiple_of=8,
                use_80_20_rule=cfg_train.objective.use_80_20_rule,
                token_drop=cfg_train.objective.token_drop,
            )
    elif cfg_train.objective.name == "causal-lm":
        if is_thf:
            collate_fn = DataCollatorWithSentenceTokens(tokenizer=tokenizer, max_length=cfg_train.max_seq_length, pad_to_multiple_of=8, mlm_probability=cfg_train.objective.mlm_probability, is_mlm=False, is_clm=True)
        else:
            collate_fn = DataCollatorWithSentenceTokens(tokenizer=tokenizer, max_length=cfg_train.max_seq_length, pad_to_multiple_of=8, mlm_probability=cfg_train.objective.mlm_probability, is_mlm=False, is_clm=True)

    else:
        raise ValueError("Objective not implemented.")

    if isinstance(dataset, torch.utils.data.IterableDataset):
        # streaming mode for ready-made datasets, speed not tested
        if torch.distributed.is_initialized():
            train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

        if cfg_impl.shuffle_in_dataloader:
            train_dataset = train_dataset.shuffle(seed=42, buffer_size=256)
        else:
            num_workers = 1  # ordered data is not loaded correctly with multiple workers in this case
        if cfg_train.reverse_dataset_order:
            raise ValueError("Reverse stream not implemented.")
        sampler = None
    else:
        # Normally, we'd just use nice map-style datasets:
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                shuffle=cfg_impl.shuffle_in_dataloader,
                drop_last=True,
            )
        else:
            if cfg_impl.shuffle_in_dataloader:
                sampler = torch.utils.data.RandomSampler(train_dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(train_dataset)
        if cfg_train.reverse_dataset_order:
            train_dataset = train_dataset.select(reversed(range(len(train_dataset))))

    if cfg_train.num_epochs is not None:
        print("Using workers:", num_workers)
        repeated_train_dataloader = FiniteDataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=cfg_impl.microbatch_size,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            drop_last=True,
            prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            max_epochs=cfg_train.num_epochs,
        )
        repeated_val_dataloader = FiniteDataLoader(
            val_dataset,
            batch_size=cfg_impl.microbatch_size,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            drop_last=True,
            prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            max_epochs=cfg_train.num_epochs,
        )
    else:
        repeated_train_dataloader = InfiniteDataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=cfg_impl.microbatch_size,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            drop_last=True,
            prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
        )
        repeated_val_dataloader = InfiniteDataLoader(
            val_dataset,
            batch_size=cfg_impl.microbatch_size,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            drop_last=True,
            prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
        )
    return repeated_train_dataloader, repeated_val_dataloader


def prepare_downstream_dataloader(dataset, tokenizer, mode, cfg_impl, is_thf, seq_length=None):
    if mode == "training":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=cfg_impl.shuffle_in_dataloader,
                drop_last=True,
            )
        else:
            if cfg_impl.shuffle_in_dataloader:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    # Implementation details for dataloaders:
    collate_fn = DataCollatorWithSentenceTokens(tokenizer=tokenizer, max_length=seq_length, pad_to_multiple_of=8, is_mlm=False)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"  # disable here because otherwise the collation generates a ton of errors
    # collate_fn = transformers.DefaultDataCollator()
    num_workers = get_num_workers(cfg_impl)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg_impl.microbatch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=cfg_impl.pin_memory,
        drop_last=True if mode == "training" else False,
        prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    return dataloader


"""This is a minor modification of huggingface's toking masking:"""
"""original source:
https://github.com/huggingface/transformers/blob/130b987880a9b1ade5c76dc1413c12c8924fda50/src/transformers/data/data_collator.py#L748
at commit f00f22a3e290fd377b979124dcf9800b3d73eb11"""

from dataclasses import dataclass

@dataclass
class DataCollatorWithSentenceTokens:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm_probability: float = 0.15
    is_mlm : bool = False
    is_clm: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_length = max([len(batch['input_ids']) for batch in features])
        max_length = min(max_length, self.max_length if self.max_length is not None else max_length)

        if self.pad_to_multiple_of is not None and max_length % self.pad_to_multiple_of > 0:
            max_length += self.pad_to_multiple_of - (max_length % self.pad_to_multiple_of)

        input_ids_out = []
        attention_mask_out = []
        labels_out = []
        sentence_ids_out = []

        has_sentence_ids = 'sentence_ids' in features[0]
        has_labels = 'labels' in features[0]

        for batch_in in features:
            padding_length = max_length - len(batch_in['input_ids']) 

            # if padding_length < 0:
            #     print("Warning: padding length is negative!! Entries will be truncated.")

            batch_padded = self.tokenizer.pad(
                batch_in,
                padding='max_length',
                max_length=max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            input_ids_out.append(batch_padded['input_ids'][:max_length].unsqueeze(0))
            attention_mask_out.append(batch_padded['attention_mask'][:max_length].unsqueeze(0))
            
            if has_sentence_ids:
                sentence_ids_out.append(torch.nn.functional.pad(batch_padded['sentence_ids'], pad=(0, padding_length), mode='constant', value=-1).unsqueeze(0))
            
            if has_labels:
                labels_out.append(batch_padded['labels'].unsqueeze(0))

        if has_sentence_ids:
            sentence_ids = torch.cat(sentence_ids_out, dim=0)

        batched_data = {
            "input_ids": torch.cat(input_ids_out, dim=0),
            "attention_mask": torch.cat(attention_mask_out, dim=0).bool(),
            "sentence_ids": sentence_ids
        }

        if labels_out:
            batched_data["labels"] = torch.cat(labels_out, dim=0)
        
        if self.is_mlm:
            batched_data["input_ids"], batched_data["labels"] = self.torch_mask_tokens(
                batched_data["input_ids"]
            )

        elif self.is_clm:
            batched_data["labels"] = batched_data["input_ids"][:, 1:].clone()
            batched_data["input_ids"] = batched_data["input_ids"][:, :-1]
            batched_data["attention_mask"] = batched_data["attention_mask"][:, :-1]
            if has_sentence_ids:
                batched_data["sentence_ids"] = batched_data["sentence_ids"][:, :-1]
            #pad 0 at the end to preserve original length
            batched_data["input_ids"] = torch.nn.functional.pad(batched_data["input_ids"], pad=(0, 1), mode='constant', value=self.tokenizer.pad_token_id)
            batched_data["attention_mask"] = torch.nn.functional.pad(batched_data["attention_mask"], pad=(0, 1), mode='constant', value=False)
            if has_sentence_ids:
                batched_data["sentence_ids"] = torch.nn.functional.pad(batched_data["sentence_ids"], pad=(0, 1), mode='constant', value=-1)
            batched_data["labels"] = torch.nn.functional.pad(batched_data["labels"], pad=(0, 1), mode='constant', value=-100)

            #replace 50257 with -100 in labels
            #batched_data["labels"] = torch.where(batched_data["labels"] == 50257, -100, batched_data["labels"])

        return batched_data

    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class PatchedDataCollatorForLanguageModeling(transformers.DataCollatorForLanguageModeling):
    def __init__(self, *args, use_80_20_rule=True, token_drop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_80_20_rule = use_80_20_rule
        self.token_drop = token_drop

        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    def torch_mask_tokens(self, inputs=None, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        The ratios in this version are always fixed so that the number of masks is never dynamic!

        Also special_tokens_masks are disregarded in this flavor

        According to timeit this is not slower than the old approach (with was fast enough)
        """
        labels = inputs.clone()

        number_of_masks = round(self.mlm_probability * inputs.shape[1])
        mask_locations = torch.argsort(torch.randint_like(inputs, inputs.shape[1]))[:, :number_of_masks]
        # this was slightly fudged to be faster. A draw of torch.rand would be more random, but take slightly longer to sort

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        masked_indices.scatter_(1, mask_locations, 1)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        if self.use_80_20_rule:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            first_80percent_mask_locations = mask_locations[:, : round(0.8 * number_of_masks)]

            indices_replaced = torch.zeros_like(inputs, dtype=torch.bool)
            indices_replaced.scatter_(1, first_80percent_mask_locations, 1)
            inputs[indices_replaced] = self.mask_token

            # 10% of the time, we replace masked input tokens with random word
            next_10percent_mask_locations = mask_locations[:, round(0.8 * number_of_masks) : round(0.9 * number_of_masks)]

            indices_random = torch.zeros_like(inputs, dtype=torch.bool)
            indices_random.scatter_(1, next_10percent_mask_locations, 1)
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=inputs.dtype)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            pass
        else:
            # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            inputs[masked_indices] = self.mask_token

        if self.token_drop > 0:
            inputs, labels = self._drop_tokens(inputs, labels)
        return inputs, labels

    def _legacy_torch_mask_tokens(self, inputs=None, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        This is the orignal randomized draw.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        if self.use_80_20_rule:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.mask_token

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=inputs.dtype)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        else:
            # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            inputs[masked_indices] = self.mask_token

        if self.token_drop > 0:
            inputs, labels = self._drop_tokens(inputs, labels)
        return inputs, labels

    def torch_call(self, examples):
        """Simplified call assuming all dicts in the list of examples have the same layout and contain tensors.
        Assume further that all these tensors contain vectors of Long Tensors  [AND THEY HAVE TO BE LONG]"""
        # Handle dict or lists with proper padding and conversion to tensor.
        # if isinstance(examples[0], Mapping):
        #     batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        # else:
        #     batch = {"input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}
        # This raises dumb warnings with my latest setup

        # So this is the handmade version
        batch = dict()
        for key in examples[0].keys():
            elem = torch.as_tensor(examples[0][key])
            # block = examples[0][key].new_empty(len(examples), *examples[0][key].shape)
            # for idx, example in enumerate(examples):
            #     block[idx] = example[key]
            out = None
            if torch.utils.data.get_worker_info() is not None:

                # storage = elem._storage()._new_shared(len(examples) * 8 * elem.shape[0], device=elem.device)  # 8 for byte->long
                # storage = elem.untyped_storage()._new_shared(len(examples) * 8 * elem.shape[0], device=elem.device)  # 8 for byte->long
                # out = elem.new(storage).resize_(len(examples), elem.shape[0])
                storage = elem._typed_storage()._new_shared(len(examples) * elem.shape[0], device=elem.device)
                out = elem.new(storage).resize_(len(examples), elem.shape[0])

            batch[key] = torch.stack([torch.as_tensor(example[key]) for example in examples], 0, out=out).contiguous()

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def _drop_tokens(self, input_ids, labels):
        """Drop random tokens. Hou et al., "Token Dropping for Efficient BERT Pretraining" also discuss dropping tokens
        based on more advanced strategies, which might also be helpful.

        This is the simplest strategy, randomly dropping a bunch of tokens for all layers.
        """
        reduced_seq_length = int(input_ids.shape[1] * (1 - self.token_drop))
        # There is probably a faster way to do this, but this works for now?
        token_mask = torch.argsort(torch.rand_like(input_ids, dtype=torch.float), dim=-1)
        fixed_mask = input_ids.scatter(1, token_mask[:, :reduced_seq_length], -1) == -1
        return input_ids[fixed_mask].view(input_ids.shape[0], -1), labels[fixed_mask].view(input_ids.shape[0], -1)


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """Lazy copy-paste from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()
        self.epoch_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            self.epoch_counter += 1
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch_counter)
            batch = next(self.dataset_iterator)
        return batch
    
class FiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, max_epochs, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
        self.epoch_counter = 0
        self.max_epochs = max_epochs  # Maximum number of epochs to run

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_counter >= self.max_epochs:
            # Stop iteration if max_epochs is reached
            raise StopIteration
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            self.epoch_counter += 1
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch_counter)
            # After updating epoch_counter, check again to see if we've reached max_epochs
            if self.epoch_counter >= self.max_epochs:
                raise StopIteration
            batch = next(self.dataset_iterator)
        return batch
