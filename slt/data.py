import os
import sys
import datasets
import nltk
from tqdm import tqdm
import traceback

import numpy as np
from scipy import stats
from pathlib import Path
import numpy as np

from joblib import Parallel, delayed


def form_source_dataset(meta_config):
    # use hugginface.datasets to load the wikipedia en and bookcorpusopen datasets, then concatenate them
	if meta_config.job_name == 'DEBUG':
		combined_dataset = datasets.load_dataset('text', data_files=[ os.path.join('data', 'bert_dataset_shard0_pieced.txt')], cache_dir=meta_config.input_path)
	else:
		combined_dataset = datasets.load_dataset('text', data_files=[ os.path.join('data', f'piece128/bert_dataset_shard{i}_pieced.txt') for i in range(200)], cache_dir=meta_config.input_path)
	#combined_dataset = datasets.concatenate_datasets([ bookcorpusopen, wikipedia ])
	combined_dataset = combined_dataset.with_format("torch")
	combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
	return combined_dataset

def form_new_dataset(meta_config):
	openwebtext = datasets.load_dataset("Skylion007/openwebtext", split='train', cache_dir=meta_config.input_path, num_proc=8)
	wikipedia = datasets.load_dataset("wikipedia", "20220301.en", split='train', cache_dir=meta_config.input_path, num_proc=8)
	arxiv = datasets.load_dataset("scientific_papers", "arxiv", split='train', cache_dir=meta_config.input_path, num_proc=8)
	arxiv = arxiv.rename_column('article', 'text')

	combined_dataset = datasets.concatenate_datasets([ openwebtext, wikipedia, arxiv ])
	combined_dataset = combined_dataset.with_format("torch")
	combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
	return combined_dataset

def form_fineweb_dataset(meta_config):
	ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", cache_dir=meta_config.input_path, num_proc=8)
	ds = ds.with_format("torch")
	ds = ds.shuffle(seed=meta_config.seed)
	return ds

def form_pile_dataset(meta_config):
	dataset = datasets.load_dataset("EleutherAI/the_pile_deduplicated", split='train', cache_dir=meta_config.input_path, num_proc=8)	
	dataset = dataset.with_format("torch")
	#dataset = dataset.shuffle(seed=meta_config.seed)
	return dataset

def form_c4_dataset(meta_config):
	dataset = datasets.load_dataset("allenai/c4", "en", cache_dir=meta_config.input_path, num_proc=80)	
	dataset = dataset.with_format("torch")
	return dataset

def build_dataset_mlm(meta_config, tokenizer):
	if meta_config.model_name == 'thf-sentence':
		# try:
		# 	tokenized_datasets = datasets.load_from_disk(os.path.join('data/piece128', f'tokenized_dataset_for_bert_with_sentence_ids'))
		# except:
		if meta_config.job_name == 'DEBUG':
			combined_dataset = datasets.load_dataset('text', data_files=[ os.path.join('data/piece128', 'bert_dataset_shard0_pieced.txt')], split='train', cache_dir='data/piece128')
		else:
			combined_dataset = datasets.load_dataset('text', data_files=[ os.path.join('data/piece128', f'bert_dataset_shard{i}_pieced.txt') for i in range(200)], split='train', cache_dir='data/piece128')
		combined_dataset.set_transform(lambda row: tokenize_and_sentenize(meta_config, tokenizer, row))
		#tokenized_datasets.save_to_disk(os.path.join('data/piece128', f'tokenized_dataset_for_bert_with_sentence_ids'))
		combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
		return combined_dataset

	else:
		def map_tokenize(examples):
			return tokenizer(examples['text'], add_special_tokens=True, truncation=True, max_length=meta_config.max_length)

		try:
			tokenized_datasets = datasets.load_from_disk(os.path.join('data/piece128', f'tokenized_dataset_for_bert'))
		except:
			if meta_config.job_name == 'DEBUG':
				combined_dataset = datasets.load_dataset('text', data_files=[ os.path.join('data/piece128', 'bert_dataset_shard0_pieced.txt')], split='train', cache_dir='data/piece128')
			else:
				combined_dataset = datasets.load_dataset('text', data_files=[ os.path.join('data/piece128', f'bert_dataset_shard{i}_pieced.txt') for i in range(200)], split='train', cache_dir='data/piece128')
			tokenized_datasets = combined_dataset.map(map_tokenize, batched=True, num_proc=8, remove_columns=["text"])
			tokenized_datasets.save_to_disk(os.path.join('data/piece128', f'tokenized_dataset_for_bert'))
		
		return tokenized_datasets


def form_tiny_source_dataset(meta_config):
    # use hugginface.datasets to load the wikipedia en and bookcorpusopen datasets, then concatenate them
	#bookcorpusopen = datasets.load_dataset("bookcorpusopen", split="train", cache_dir=meta_config.input_path)
	wikipedia = datasets.load_dataset("wikipedia", "20220301.en", split="train[:1000]", cache_dir=meta_config.input_path)

	# concatenate the datasets
	combined_dataset = wikipedia
	#combined_dataset = datasets.concatenate_datasets([ bookcorpusopen, wikipedia ])
	combined_dataset = combined_dataset.with_format("torch")

	combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
	return combined_dataset

from nltk.tokenize import word_tokenize

def clean_text_nltk(text):
    #if text is a list of texts then call word_tokenize on each text
    if isinstance(text, list):
        return [clean_text_nltk(t) for t in text]
    
    tokens = word_tokenize(text)
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def build_piece_dataset(meta_config, source_dataset, tokenizer, output_file_path): 
	print('Using tokenizer: ', tokenizer)
	max_size = meta_config.max_length
	print(source_dataset)
	source_dataset.set_transform(lambda row: tokenizer(clean_text_nltk(row['text'])))
	
	with open(output_file_path, 'w', encoding="utf-8", buffering=16*1024*1024) as f:
		for i in tqdm(range(len(source_dataset))):
			document_len = len(source_dataset[i]['input_ids'])
			for j in range(0, 1 + document_len // max_size):
				start = j * max_size
				end = min((j+1) * max_size, document_len)
				if start == end: # in case the length is exactly a multiple of 512
					break
				f.write(tokenizer.decode(source_dataset[i]['input_ids'][start:end], skip_special_tokens=True) + "\n")

		f.flush()

def build_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=4, mode='pile'):
	max_size = meta_config.max_length

	# Generate unique file paths for each shard
	print("chunking dataset")
	chunked_dataset = [source_dataset.select(range(i*len(source_dataset)//num_threads, (i+1)*len(source_dataset)//num_threads)) for i in range(num_threads)]
	if mode == 'pile':
		shard_paths = [os.path.join(meta_config.input_path, f"pile_dedup_piece_{max_size}/shard-{i}.txt") for i in range(num_threads)]
	elif mode == 'c4':
		shard_paths = [os.path.join(meta_config.input_path, f"c4_piece_{max_size}/shard-{i}.txt") for i in range(num_threads)]
	elif mode == 'newdata':
		shard_paths = [os.path.join(meta_config.input_path, f"newdata_piece_{max_size}/shard-{i}.txt") for i in range(num_threads)]
	else:
		shard_paths = [os.path.join(meta_config.input_path, f"fineweb_{max_size}/shard-{i}.txt") for i in range(num_threads)]

	for i in range(len(chunked_dataset)):
		print(f"shard {i} has {len(chunked_dataset[i])} entries")
		if i == 0:
			print("Example chunk: ",  chunked_dataset[i])

	print("starting jobs")
	Parallel(n_jobs=num_threads)(
		delayed(build_piece_dataset)(meta_config, chunk, tokenizer, shard_paths[i])
		for i, chunk in enumerate(chunked_dataset)
	)

	return shard_paths

def build_sentence_dataset(meta_config, source_dataset, source_dataset_name):
	with open(os.path.join(meta_config.input_path, f"{source_dataset_name}-sentences-{meta_config.job_id}.txt"), 'w', encoding="utf-8", buffering=4*1024*1024) as f:
		for i in tqdm(range(len(source_dataset))):
			sentences = nltk.sent_tokenize(source_dataset[i]['text'])
			for sentence in sentences:
				if sentence.strip() == "" or len(sentence) > 512:
					continue
				sentence_nl_escaped = sentence.replace("\n", "\\n")
				f.write(sentence_nl_escaped + "\n")

		f.flush()

	print("Raw sentence dataset built into the file '" + f"{source_dataset_name}-sentences-{meta_config.job_id}.txt")

def encapsulate_sentence_dataset(meta_config, dataset_file_to_load):
	dataset = load_sentence_dataset(meta_config, dataset_file_to_load)
	print(f"Loaded {dataset_file_to_load}")
	input_filename = Path(dataset_file_to_load).stem
	print(f"Processing dataset {input_filename}")
	dst_path = os.path.join(meta_config.input_path, f"{input_filename}-encapsulated-{meta_config.job_id}")
	dataset.save_to_disk(dst_path)
	print(f"Encapsulated dataset saved to {dst_path}")


def load_sentence_dataset(meta_config, dataset_file_to_load):
	dataset = datasets.load_dataset('text', data_files=[ os.path.join(meta_config.input_path, dataset_file_to_load) ], split='train', cache_dir=meta_config.input_path)
	return dataset

def update_histogram(array: np.array, value: int):
	if value >= len(array):
		array = np.concatenate((array, np.zeros(value - len(array) + 1)))
	array[value] += 1
	return array

def compute_sentence_statistics(meta_config, source_dataset, tokenizer):
	char_lengths = []
	word_lengths = []
	token_lengths = []

	char_histogram = np.array([])
	word_histogram = np.array([])
	token_histogram = np.array([])

	for i in tqdm(range(len(source_dataset))):
		row = source_dataset[i]
		row_text = row['text'].replace("\\n", "\n")
		words = nltk.word_tokenize(row_text)
		tokens = tokenizer([row_text], truncation=False)['input_ids'][0]

		char_lengths.append(len(row_text))
		char_histogram = update_histogram(char_histogram, len(row_text))
		word_lengths.append(len(words))
		word_histogram = update_histogram(word_histogram, len(words))
		token_lengths.append(len(tokens))
		token_histogram = update_histogram(token_histogram, len(tokens))
	
	char_lengths = np.array(char_lengths)
	word_lengths = np.array(word_lengths)
	token_lengths = np.array(token_lengths)

	print("char stats:", stats.describe(char_lengths))
	print("median char length", np.median(char_lengths))
	print("1/4 quantile char length", np.quantile(char_lengths, 0.25))
	print("3/4 quantile char length", np.quantile(char_lengths, 0.75))
	print("iqr char length", np.quantile(char_lengths, 0.75) - np.quantile(char_lengths, 0.25))
	print("95 precentile char length", np.quantile(char_lengths, 0.95))
	print("99 precentile char length", np.quantile(char_lengths, 0.97))           # is that supposed to be 0.99?

	print("word stats:", stats.describe(word_lengths))
	print("median word length", np.median(word_lengths))
	print("1/4 quantile word length", np.quantile(word_lengths, 0.25))
	print("3/4 quantile word length", np.quantile(word_lengths, 0.75))
	print("iqr word length", np.quantile(word_lengths, 0.75) - np.quantile(word_lengths, 0.25))
	print("95 precentile word length", np.quantile(word_lengths, 0.95))
	print("99 precentile word length", np.quantile(word_lengths, 0.97))         

	print("token stats:", stats.describe(token_lengths))
	print("median token length", np.median(token_lengths))
	print("1/4 quantile token length", np.quantile(token_lengths, 0.25))
	print("3/4 quantile token length", np.quantile(token_lengths, 0.75))
	print("iqr token length", np.quantile(token_lengths, 0.75) - np.quantile(token_lengths, 0.25))
	print("95 precentile token length", np.quantile(token_lengths, 0.95))
	print("99 precentile token length", np.quantile(token_lengths, 0.97))

	with np.printoptions(threshold=sys.maxsize):
		print("char histogram:")
		print(char_histogram[0:min(512, char_histogram.size)])
		print("char histogram overflow:", np.sum(word_histogram[512:]))
		print("word histogram:")
		print(word_histogram[0:min(128, word_histogram.size)])
		print("word histogram overflow:", np.sum(word_histogram[128:]))
		print("token histogram:")
		print(token_histogram[0:min(128, token_histogram.size)])
		print("token histogram overflow:", np.sum(token_histogram[128:]))

def build_sentence_piece_dataset(meta_config, source_dataset, tokenizer, output_file_path):
	MAX_SIZE = meta_config.max_length  # Or change to 512 based on your requirement
	print("my output file path is ", output_file_path)
		
	running_input_ids = []  # Maintain running_input_ids across the entire dataset
	print("attempting to write to ", os.path.join(meta_config.input_path, output_file_path))

	with open(output_file_path, 'w', encoding="utf-8", buffering=16*1024*1024) as f:
		for i in tqdm(range(len(source_dataset))):
			sentences = nltk.sent_tokenize(source_dataset[i]['text'])
			running_input_ids = []
			for sentence in sentences:
				sentence_input_ids = tokenizer(sentence)['input_ids']
				sentence_len = len(sentence_input_ids)
				if len(sentence_input_ids) > MAX_SIZE:
					for j in range(0, 1 + sentence_len // MAX_SIZE):
						start = j * MAX_SIZE
						end = min((j+1) * MAX_SIZE, sentence_len)
						if start == end: # in case the length is exactly a multiple of MAX_SIZE
							break
						f.write(tokenizer.decode(sentence_input_ids[start:end], skip_special_tokens=True) + "\n")
				else:
					if len(running_input_ids) + len(sentence_input_ids) < MAX_SIZE:
						running_input_ids += sentence_input_ids
					else:
						f.write(tokenizer.decode(running_input_ids, skip_special_tokens=True) + "\n")
						running_input_ids = sentence_input_ids

			if len(running_input_ids) > 0:
				f.write(tokenizer.decode(running_input_ids, skip_special_tokens=True) + "\n")

def build_old_sentence_piece_dataset(meta_config, source_dataset, tokenizer):
    MAX_SIZE = 1024  # Or change to 512 based on your requirement
    source_dataset = source_dataset['train']
    
    output_file_path = os.path.join(meta_config.input_path, f"wiki_and_bco-sentence_piece-{MAX_SIZE}.txt")
    
    with open(output_file_path, 'w', encoding="utf-8", buffering=16*1024*1024) as f:
        for i in tqdm(range(len(source_dataset))):
            sentences = nltk.sent_tokenize(source_dataset[i]['text'])
            running_input_ids = []
            
            for sentence in sentences:
                sentence_input_ids = tokenizer(sentence)['input_ids']
                
                # Handling the case where a single sentence exceeds MAX_SIZE
                if len(sentence_input_ids) > MAX_SIZE:
                    for start in range(0, len(sentence_input_ids), MAX_SIZE):
                        end = start + MAX_SIZE
                        f.write(tokenizer.decode(sentence_input_ids[start:end], skip_special_tokens=True) + "\n")
                    continue
                
                # If adding the current sentence doesn't exceed MAX_SIZE, add it to the running list
                if len(running_input_ids) + len(sentence_input_ids) <= MAX_SIZE:
                    running_input_ids.extend(sentence_input_ids)
                else:
                    # Write the current accumulated sentences and reset the running list
                    f.write(tokenizer.decode(running_input_ids, skip_special_tokens=True) + "\n")
                    running_input_ids = sentence_input_ids
            
            # Writing any remaining input_ids after processing all sentences
            if running_input_ids:
                f.write(tokenizer.decode(running_input_ids, skip_special_tokens=True) + "\n")

def new_sentence_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=4, mode='pile'):
	max_size = meta_config.max_length

	# Generate unique file paths for each shard
	print("chunking dataset")
	chunked_dataset = [source_dataset.select(range(i*len(source_dataset)//num_threads, (i+1)*len(source_dataset)//num_threads)) for i in range(num_threads)]
	if mode == 'pile':
		shard_paths = [os.path.join(meta_config.input_path, f"pile_dedup_{max_size}/shard-{i}.txt") for i in range(num_threads)]
	elif mode == 'c4':
		shard_paths = [os.path.join(meta_config.input_path, f"c4_{max_size}/shard-{i}.txt") for i in range(num_threads)]
	elif mode == 'newdata':
		shard_paths = [os.path.join(meta_config.input_path, f"newdata_{max_size}/shard-{i}.txt") for i in range(num_threads)]
	else:
		shard_paths = [os.path.join(meta_config.input_path, f"fineweb_{max_size}/shard-{i}.txt") for i in range(num_threads)]

	for i in range(len(chunked_dataset)):
		print(f"shard {i} has {len(chunked_dataset[i])} entries")
		if i == 0:
			print("Example chunk: ",  chunked_dataset[i])

	print("starting jobs")
	Parallel(n_jobs=num_threads)(
		delayed(build_sentence_piece_dataset)(meta_config, chunk, tokenizer, shard_paths[i])
		for i, chunk in enumerate(chunked_dataset)
	)

	return shard_paths

def load_dataset(meta_config, tokenizer):
	# if meta_config.action == 'train-sae' or meta_config.action == 'test-sae':
	# 	dataset = datasets.load_dataset('text', data_files=[ os.path.join('data', 'bert_dataset_shard0_pieced.txt')], cache_dir=meta_config.input_path)

	# 	if meta_config.action == 'train-sae':
	# 		dataset = dataset['train']
	# 	else:
	# 		dataset = dataset['test']
	# 	dataset.set_transform(lambda row: tokenizer(row['text'], truncation=True, max_length=meta_config.max_length))
	# 	dataset = dataset.shuffle(seed=meta_config.seed)
	# else:
	dataset_file_to_load = f"wiki_and_bco-sentence_piece-{meta_config.max_length}.txt"
	dataset = datasets.load_dataset('text', data_files=[ os.path.join(meta_config.input_path, dataset_file_to_load) ], split='train', cache_dir=meta_config.input_path)
	dataset.set_transform(lambda row: sentence_piece_transform_batch(meta_config, tokenizer, row))
	dataset = dataset.shuffle(seed=meta_config.seed)

	print(f"Loaded {dataset_file_to_load}")
	return dataset

def load_dataset_mlm(meta_config, tokenizer):
	#dataset_file_to_load = f"wiki_and_bco-sentence_piece-{meta_config.max_length}.txt"
	#dataset = datasets.load_dataset('text', data_files=[ os.path.join(meta_config.input_path, dataset_file_to_load) ], split='train', cache_dir=meta_config.input_path)
	dataset = form_source_dataset(meta_config)
	dataset.set_transform(lambda row: prepare_mlm(meta_config, tokenizer, row))
	dataset = dataset.shuffle(seed=meta_config.seed)

	print(f"Loaded sharded dataset for MLM")
	return dataset

def load_dataset_distil(meta_config, tokenizer, dataset_name):
	return datasets.load_from_disk(os.path.join(meta_config.input_path, f'{dataset_name}-distil'))

def build_dataset_distil(meta_config, tokenizer, dataset_name):
	dataset = datasets.load_dataset("glue", dataset_name, cache_dir=meta_config.input_path)
	dataset = dataset.map(lambda row: prepare_mrpc(meta_config, tokenizer, row), batched=True, remove_columns=dataset['train'].column_names, batch_size=64)
	dataset = dataset.shuffle(seed=meta_config.seed)
	dataset.save_to_disk(os.path.join(meta_config.input_path, f'{dataset_name}-distil'))

def load_dataset_finetune(meta_config, tokenizer, dataset_name):
	dataset = datasets.load_dataset("glue", dataset_name, cache_dir=meta_config.input_path)
	dataset.set_transform(lambda row: prepare_mrpc(meta_config, tokenizer, row))
	data = dataset.shuffle(seed=meta_config.seed)

	return data

def combine_sentence_datasets(meta_config):
	source_files = meta_config.source_file.split(',')
	print(f"Loading wikipedia from {source_files[0]}")
	wikipedia = datasets.load_from_disk(os.path.join(meta_config.input_path, source_files[0]))
	print(f"Loading bookcorpusopen from {source_files[1]}")
	bookcorpusopen = datasets.load_from_disk(os.path.join(meta_config.input_path, source_files[1]))

	combined_dataset = datasets.concatenate_datasets([ bookcorpusopen, wikipedia ])
	combined_dataset = combined_dataset.with_format("torch")

	combined_dataset = combined_dataset.shuffle(seed=meta_config.seed)
	combined_dataset.save_to_disk(os.path.join(meta_config.input_path, f"wikipedia-and-bco-sentences-{meta_config.job_id}"))

def build_splits(meta_config):
	print(f"Loading dataset from {meta_config.source_file}")
	dataset = datasets.load_from_disk(os.path.join(meta_config.input_path, meta_config.source_file))
	print(f"Splitting dataset")
	split_dataset = dataset.train_test_split(test_size=int(1e6))

	source_file_stem = Path(meta_config.source_file).stem
	print(f"Saving dataset to {source_file_stem}-split")
	split_dataset.save_to_disk(os.path.join(meta_config.input_path, source_file_stem + "-split"))


from .embedder import generate_embedded_dataset
def build_embeddings_race(meta_config, tokenizer, model):
	hfdss = datasets.load_dataset('race', 'all')
	# hfdss['train'], hfdss['validation'], hfdss['test']

	ged = generate_embedded_dataset(meta_config, hfdss['validation'], tokenizer, model)
	hfds_validation_embedded = datasets.Dataset.from_dict(ged)
	hfds_validation_embedded.save_to_disk(os.path.join(meta_config.input_path, "race_validation_embedded"))

	ged = generate_embedded_dataset(meta_config, hfdss['test'], tokenizer, model)
	hfds_test_embedded = datasets.Dataset.from_dict(ged)
	hfds_test_embedded.save_to_disk(os.path.join(meta_config.input_path, "race_test_embedded"))

	ged = generate_embedded_dataset(meta_config, hfdss['train'], tokenizer, model)
	hfds_train_embedded = datasets.Dataset.from_dict(ged)
	hfds_train_embedded.save_to_disk(os.path	.join(meta_config.input_path, "race_train_embedded"))

import torch
def merge_race_mcq(row):
	acc = []
	embedding_dim = len(row['question'][0][0]) * len(row['question'][0][0][0])
	acc.append(torch.zeros((embedding_dim,)))

	acc.append(torch.tensor(row['article']).squeeze())
	acc.append(torch.zeros((embedding_dim,)))
	acc.append(torch.tensor(row['question']).squeeze())
	for option in row['options'][0]:
		acc.append(torch.zeros((embedding_dim,)))
		for sentence in option:
			acc.append(torch.tensor(sentence).squeeze())
	
	acc2 = []
	for tensor in acc:
		tensor2 = torch.reshape(tensor, (-1, embedding_dim))
		acc2.append(tensor2)
		del tensor
	del acc

	embeddings = torch.cat(acc2, dim=0)
	attention_mask = torch.ones((embeddings.shape[0],), dtype=torch.bool)
	amount_to_pad = 32 - embeddings.shape[0]
	embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, amount_to_pad), value=0)
	attention_mask = torch.nn.functional.pad(attention_mask, (0, amount_to_pad), value=False)
	return {
		'embeddings': embeddings.unsqueeze(0),
		'attention_mask': attention_mask.unsqueeze(0),
		'answer': torch.tensor(row['answer'], dtype=torch.long)
	}

def load_race(meta_config):
	race_train = datasets.Dataset.load_from_disk(os.path.join(meta_config.input_path, "race_train_embedded"))
	race_train.set_transform(merge_race_mcq)
	race_validation = datasets.Dataset.load_from_disk(os.path.join(meta_config.input_path, "race_validation_embedded"))
	race_validation.set_transform(merge_race_mcq)
	race_test = datasets.Dataset.load_from_disk(os.path.join(meta_config.input_path, "race_test_embedded"))
	race_test.set_transform(merge_race_mcq)
	
	return race_train, race_validation, race_test

def sentence_piece_transform(meta_config, tokenizer, row):
	#print(f"Length of row['text']: {len(row['text'])}")
	
	#assert len(row['text']) == 1

	if 'text' not in row.keys() or len(row['text'][0]) == 0:
		row['text'] = [ "Mary had a little lamb." ] # dummy text is better than returning None or whatever and invoking an undefined behaviour

	sentences = nltk.sent_tokenize(row['text'][0])
	try:
		ret = tokenizer(sentences, truncation=True, max_length=meta_config.max_length)
	except IndexError as e:
		print(e)
		traceback.print_exc()
		print("Deliquent sentence list: ", sentences)
		print("Deliquent entry passed from Dataset: ", row['text'])

	ret['input_ids'] = [ ret['input_ids'] ]
	ret['attention_mask'] = [ ret['attention_mask'] ]

	return ret

def sentence_piece_transform_batch(meta_config, tokenizer, row):
	# Ensure row['text'] is a list and non-empty
	if 'text' not in row or not row['text']:
		row['text'] = ["Mary had a little lamb."]  # dummy text

	all_input_ids = []
	all_attention_masks = []

	# Iterate through each entry in row['text'] and process them independently
	for text_entry in row['text']:
		sentences = nltk.sent_tokenize(text_entry)

		if not sentences:
			print("Empty sentences for text_entry:", text_entry)
			continue
		try:
			ret = tokenizer(sentences, truncation=True, max_length=meta_config.max_length)
			all_input_ids.append(ret['input_ids'])
			all_attention_masks.append(ret['attention_mask'])
		except IndexError as e:
			print(e)
			traceback.print_exc()
			print("Delinquent entry passed from Dataset: ", text_entry)

	print("all_input_ids", all_input_ids)
	print("all_attention_masks", all_attention_masks)
	return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}

def numel(list_of_lists):
	total_elements = sum(len(sublist) for sublist in list_of_lists)
	return total_elements

def tokenize_and_sentenize(meta_config, tokenizer, row):
	if 'text' not in row or not row['text']:
		row['text'] = ["Mary had a little lamb."]  # dummy text

	all_input_ids = []
	all_attention_masks = []
	all_sentence_ids = []

	MAX_SIZE = meta_config.max_length 

	for text_entry in row['text']:
		sentences = nltk.sent_tokenize(text_entry)

		if not sentences:
			continue
		input_ids = []
		sentence_ids = []
		attention_masks = []

		for idx, sentence in enumerate(sentences):
			tokenized_sentence = tokenizer.encode_plus(sentence, truncation=True, return_attention_mask=True, max_length=MAX_SIZE)
			projected_length = len(input_ids) + len(tokenized_sentence['input_ids'])

			if projected_length > MAX_SIZE:
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

			all_input_ids.append(input_ids)
			all_attention_masks.append(attention_masks)
			all_sentence_ids.append(sentence_ids)

			assert numel(all_input_ids) == numel(all_attention_masks) == numel(all_sentence_ids)
			assert len(all_input_ids) == len(all_attention_masks) == len(all_sentence_ids), f"Length mismatch: {len(all_input_ids)}, {len(all_attention_masks)}, {len(all_sentence_ids)}"
			assert len(all_input_ids[0]) == len(all_attention_masks[0]) == len(all_sentence_ids[0]), f"Length mismatch: {len(all_input_ids[0])}, {len(all_attention_masks[0])}, {len(all_sentence_ids[0])}"

	return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks, 'sentence_ids': all_sentence_ids}

def prepare_mlm(meta_config, tokenizer, row, mask_probability=0.15):
    if 'text' not in row or not row['text']:
        row['text'] = ["Mary had a little lamb."]  # dummy text

    all_input_ids = []
    all_labels = []  # Labels for MLM
    all_attention_masks = []
    all_sentence_ids = []

    MAX_SIZE = meta_config.max_length 

    for text_entry in row['text']:
        sentences = nltk.sent_tokenize(text_entry)

        if not sentences:
            continue

        try:
            input_ids = []
            labels = [] 
            sentence_ids = []
            attention_masks = []

            for idx, sentence in enumerate(sentences):
                if idx >= 64:
                    break

                tokenized_sentence = tokenizer.encode_plus(sentence, truncation=True, return_attention_mask=True, max_length=MAX_SIZE)
                projected_length = len(input_ids) + len(tokenized_sentence['input_ids'])

                if projected_length > MAX_SIZE:
                    remaining_length = MAX_SIZE - len(input_ids)
                    truncated_input_ids = tokenized_sentence['input_ids'][:remaining_length]
                    truncated_attention_masks = tokenized_sentence['attention_mask'][:remaining_length]

                    tokens, label = mask_tokens(truncated_input_ids, tokenizer, mask_probability)

                    input_ids.extend(tokens)
                    labels.extend(label)
                    sentence_ids.extend([idx] * remaining_length)
                    attention_masks.extend(truncated_attention_masks)
                    break
                else:
                    tokens, label = mask_tokens(tokenized_sentence['input_ids'], tokenizer, mask_probability)

                    input_ids.extend(tokens)
                    labels.extend(label)
                    sentence_ids.extend([idx] * len(tokenized_sentence['input_ids']))
                    attention_masks.extend(tokenized_sentence['attention_mask'])

            all_input_ids.append(input_ids)
            all_labels.append(labels) 
            all_attention_masks.append(attention_masks)
            all_sentence_ids.append(sentence_ids)

        except IndexError as e:
            print(e)
            traceback.print_exc()
            print("Delinquent entry passed from Dataset: ", text_entry)

    return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks, 'sentence_ids': all_sentence_ids, 'labels': all_labels}

def mask_tokens(input_ids, tokenizer, mask_probability):
	"""
	Mask tokens for MLM with a probability.
	"""
	input_ids = np.array(input_ids)
	labels = np.full(input_ids.shape, -100)  # Initialize labels with -100

	# Decide which tokens to mask for MLM
	probability_matrix = np.random.rand(input_ids.shape[0])
	masking_indices = (probability_matrix < mask_probability) & (input_ids != tokenizer.pad_token_id)
	labels[masking_indices] = input_ids[masking_indices]

	# 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
	indices_replaced = masking_indices & (np.random.rand(input_ids.shape[0]) < 0.8)
	input_ids[indices_replaced] = tokenizer.mask_token_id

	# 10% of the time, replace masked input tokens with random word
	indices_random = masking_indices & ~indices_replaced & (np.random.rand(input_ids.shape[0]) < 0.5)
	random_words = np.random.randint(1, len(tokenizer.vocab), size=input_ids.shape[0])
	input_ids[indices_random] = random_words[indices_random]

	# The rest of the time (10% of the time) we keep the masked input tokens unchanged

	return input_ids.tolist(), labels.tolist()

from sympy import O, mobius
from transformers import AutoModelForSequenceClassification
import torch

teacher_model = AutoModelForSequenceClassification.from_pretrained("yoshitomo-matsubara/bert-large-uncased-mrpc")
teacher_model.eval()

def prepare_mrpc(meta_config, tokenizer, rows):
	MAX_SIZE = meta_config.max_length

	all_input_ids = []
	all_attention_masks = []
	all_labels = []
	all_sentence_ids = []

	for sentence1, sentence2, label in zip(rows['sentence1'], rows['sentence2'], rows['label']):
		# Process each sentence pair here
		try:
			tokenized_pair = tokenizer(sentence1, sentence2, truncation=True, max_length=MAX_SIZE)
			
			# Add tokenized information to the lists
			all_input_ids.append(tokenized_pair['input_ids'])
			all_attention_masks.append(tokenized_pair['attention_mask'])

			if meta_config.action == 'build_dataset_distil':
				with torch.no_grad():
					teacher_output = teacher_model(**tokenized_pair)
					probabilities = torch.nn.functional.softmax(teacher_output.logits, dim=-1)
				all_labels.append(probabilities)

			else:
				all_labels.append(label)

			l = len(tokenizer(sentence1, truncation=True, max_length=MAX_SIZE)['input_ids'])
			all_sentence_ids.append([0] * l + [1] * (len(tokenized_pair['input_ids']) - l))

		except Exception as e:
			print(e)
			#print("Issue with entry: ", sentence1, sentence2, label)
		# Append results to the lists

	out = {
		'input_ids': all_input_ids,
		'attention_mask': all_attention_masks,
		'labels': all_labels
	}

	batch_size = len(all_input_ids)

	if len(all_sentence_ids) == batch_size:
		out['sentence_ids'] = all_sentence_ids
	else:
		print(len(all_sentence_ids), batch_size)
		raise ValueError("Mismatch in length of sentence_ids")

	return out