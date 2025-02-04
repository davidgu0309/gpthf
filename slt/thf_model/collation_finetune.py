from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import generate_block_matrix

@dataclass
class DataCollatorWithSentenceTokens:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        sequence_fields = ['input_ids', 'attention_mask']  # Add other sequence fields if necessary
        if 'sentence_ids' in features[0]:
            sequence_fields.append('sentence_ids')

        max_length = max([len(batch['input_ids']) for batch in features])
        max_length = min(max_length, self.max_length if self.max_length is not None else max_length)

        if self.pad_to_multiple_of is not None and max_length % self.pad_to_multiple_of > 0:
            max_length += self.pad_to_multiple_of - (max_length % self.pad_to_multiple_of)

        input_ids_out = []
        attention_mask_out = []
        sentence_ids_out = []
        labels_out = []  

        for batch_in in features:
            sequence_data = {k: v for k, v in batch_in.items() if k in sequence_fields}

            batch_padded = self.tokenizer.pad(
                sequence_data,
                padding='max_length',
                max_length=max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            input_ids_out.append(batch_padded['input_ids'].unsqueeze(0))
            attention_mask_out.append(batch_padded['attention_mask'].unsqueeze(0))

            if 'sentence_ids' in batch_in:
                padding_length = max_length - len(batch_in['input_ids'])
                sentence_ids_out.append(torch.nn.functional.pad(batch_padded['sentence_ids'],
                                                                pad=(0, padding_length), mode='constant', value=-1).unsqueeze(0))

            if 'labels' in batch_in:
                labels_out.append(torch.tensor(batch_in['labels']).unsqueeze(0))

        batched_data = {
            "input_ids": torch.cat(input_ids_out, dim=0),
            "attention_mask": torch.cat(attention_mask_out, dim=0).bool(),
        }

        if sentence_ids_out:
            sentence_ids_out = torch.cat(sentence_ids_out, dim=0)
            word_attention_mask, sentence_indices, sentence_attention_mask = generate_block_matrix(sentence_ids_out)
            batched_data["word_attention_mask"] = word_attention_mask
            batched_data["sentence_indices"] = sentence_indices
            batched_data["sentence_attention_mask"] = sentence_attention_mask
            sentence_ids_out[sentence_ids_out == -1] = 0  # -1 did its job, now we need to dodge cuda indexing error
            batched_data["segment_ids"] = sentence_ids_out

        if labels_out:
            batched_data["labels"] = torch.cat(labels_out, dim=0)

        return batched_data
