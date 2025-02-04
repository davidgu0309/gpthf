import datasets
import os
from transformers import AutoTokenizer
import nltk
nltk.download('punkt')
import traceback

from . import cli
from . import data
from .sae_model.configuration_sae import SAEConfig
from .sae_model.modeling_sae import SAE
from .thf_model.configuration_thf import THFConfig
from .thf_model.modeling_thf_1d import THF
from .slt_model.configuration_slt import SLTConfig
from .slt_model.modeling_slt import SLT
from . import sae_train
from . import thf_train_old
from . import slt_train

def sentence_piece_transform(tokenizer, row):
	print(f"Length of row['text']: {len(row['text'])}")
	
	assert len(row['text']) == 1

	if 'text' not in row.keys() or len(row['text'][0]) == 0:
		row['text'] = [ "Mary had a little lamb." ] # dummy text is better than returning None or whatever and invoking an undefined behaviour

	sentences = nltk.sent_tokenize(row['text'][0])
	try:
		ret = tokenizer(sentences, truncation=True, max_length=512)
	except IndexError as e:
		print(e)
		traceback.print_exc()
		print("Deliquent sentence list: ", sentences)
		print("Deliquent entry passed from Dataset: ", row['text'])

	ret['input_ids'] = [ ret['input_ids'] ]
	ret['attention_mask'] = [ ret['attention_mask'] ]

	return ret

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset_file_to_load = f"wiki_and_bco-sentence_piece-512.txt"
dataset = datasets.load_dataset('text', data_files=[ os.path.join('../data', dataset_file_to_load) ], split='train', cache_dir='../data')
dataset.set_transform(lambda row: sentence_piece_transform(tokenizer, row))
dataset = dataset.shuffle(seed=1234)
config = THFConfig()
model = THF(config)

