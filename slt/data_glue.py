

from sympy import mobius
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

teacher_model = AutoModelForSequenceClassification.from_pretrained("yoshitomo-matsubara/bert-large-uncased-mrpc")
teacher_model.eval()


def prepare_mrpc(meta_config, tokenizer, row):
	if 'sentence1' not in row.keys() or 'sentence2' not in row.keys() or len(row['sentence1'][0]) == 0:
		row['sentence1'] = [ "Mary had a little lamb." ] # dummy text is better than returning None or whatever and invoking an undefined behaviour
		row['sentence2'] = [ "Gary had a little fox." ]

	all_input_ids = []
	all_attention_masks = []
	all_labels = []  
	
	mode = meta_config.mode

	if mode == 'sentence':
		all_sentence_ids = []

	MAX_SIZE = meta_config.max_length 

	# Iterate through each entry in row['sentence'] and process them independently
	#print(len(row['sentence1']), len(row['sentence2']), len(row['label']))
	assert len(row['sentence1']) == len(row['sentence2']) == len(row['label'])

	for sentence1, sentence2, label in zip(row['sentence1'], row['sentence2'], row['label']):
		try:
			# Tokenize the sentence
			tokenized_pair = tokenizer(sentence1, sentence2, padding='max_length', truncation=True, max_length=MAX_SIZE, return_tensors='pt')
			#print(tokenized_pair)
			
			# Add tokenized information to the lists
			all_input_ids.append(tokenized_pair['input_ids'])
			all_attention_masks.append(tokenized_pair['attention_mask'])

			if meta_config.action == 'train-thf-distil':
				with torch.no_grad():
					teacher_output = teacher_model(**tokenized_pair)
					probabilities = torch.nn.functional.softmax(teacher_output.logits, dim=-1)
				all_labels.append(probabilities)

			else:
				all_labels.append(label)

			if mode == 'sentence' and 'token_type_ids' in tokenized_pair.keys():
				all_sentence_ids.append(tokenized_pair['token_type_ids'])


		except Exception as e:
			print(e)
			#print("Issue with entry: ", sentence1, sentence2, label)

	out = {'input_ids': all_input_ids, 'attention_mask': all_attention_masks, 'labels': all_labels}
	if mode == 'sentence':
		out['sentence_ids'] = all_sentence_ids

	#print(out)

	return out
