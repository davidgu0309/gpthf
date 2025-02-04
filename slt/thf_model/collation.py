from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorWithSentenceTokens:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_length = max([len(batch['input_ids']) for batch in features])
        max_length = min(max_length, self.max_length if self.max_length is not None else max_length)

        if self.pad_to_multiple_of is not None and max_length % self.pad_to_multiple_of > 0:
            max_length += self.pad_to_multiple_of - (max_length % self.pad_to_multiple_of)

        input_ids_out = []
        attention_mask_out = []
        sentence_ids_out = []
        labels_out = []

        for batch_in in features:
            padding_length = max_length - len(batch_in['input_ids'])
            batch_padded = self.tokenizer.pad(
                batch_in,
                padding='max_length',
                max_length=max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            input_ids_out.append(batch_padded['input_ids'].unsqueeze(0))
            attention_mask_out.append(batch_padded['attention_mask'].unsqueeze(0))
            sentence_ids_out.append(torch.nn.functional.pad(batch_padded['sentence_ids'],
                                                            pad=(0, padding_length), mode='constant', value=-1).unsqueeze(0))
            if 'labels' in batch_in:
                labels_out.append(torch.nn.functional.pad(torch.tensor(batch_in['labels']),
                                                         pad=(0, padding_length), mode='constant', value=-100).unsqueeze(0))

        sentence_ids = torch.cat(sentence_ids_out, dim=0)

        batched_data = {
            "input_ids": torch.cat(input_ids_out, dim=0),
            "attention_mask": torch.cat(attention_mask_out, dim=0).bool(),
            "sentence_ids": sentence_ids
        }

        if labels_out:
            batched_data["labels"] = torch.cat(labels_out, dim=0)

        return batched_data