import nltk
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_sentences(meta_config, string, tokenizer, model):
	MAX_SIZE = 128
	sentences = nltk.sent_tokenize(string)
	sentence_embeddings = []
	for sentence in sentences:
		sentence_input_ids = tokenizer(sentence)['input_ids']
		sentence_len = len(sentence_input_ids)
		if sentence_len > MAX_SIZE:
			sentence_input_ids = sentence_input_ids[:MAX_SIZE]

		sentence_input_ids_pt = torch.tensor(sentence_input_ids).unsqueeze(0).to(device)
		attention_mask = torch.ones(sentence_input_ids_pt.shape, dtype=torch.bool, device=device)
		sentence_embeddings.append(model.forward_encoder(sentence_input_ids_pt, attention_mask).squeeze(0).tolist())

	return sentence_embeddings

def generate_embedded_dataset(meta_config, source_dataset, tokenizer, model):
	ret = {
		'article': [],
		'question': [],
		'answer': [],
		'options': []
	}
	for i in tqdm(range(len(source_dataset))):
		article = source_dataset[i]['article']
		question = source_dataset[i]['question']
		answer = source_dataset[i]['answer']
		options = source_dataset[i]['options']

		article_embeddings = embed_sentences(meta_config, article, tokenizer, model)
		question_embeddings = embed_sentences(meta_config, question, tokenizer, model)
		answer_embedding = ord(answer[0]) - ord('A')
		option_embeddings = [embed_sentences(meta_config, option, tokenizer, model) for option in options]

		ret['article'].append(article_embeddings)
		ret['question'].append(question_embeddings)
		ret['answer'].append(answer_embedding)
		ret['options'].append(option_embeddings)

	return ret
