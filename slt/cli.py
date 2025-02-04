import argparse
import time

def setup_parser():
	# meta config zone
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-j',
		'--job-id',
		type=int,
		default=int(time.time()),
		help='The job id (and the name of the wandb group)'
	)
	parser.add_argument(
		'--job-name',
		type=str,
		default="noname",
		help='job-name'
	)
	parser.add_argument(
		'-o',
		'--output-path',
		type=str,
		default="models",
		help='The directory which will contain saved model checkpoints'
	)

	parser.add_argument(
		'-i',
		'--input-path',
		type=str,
		default="./data",
		help='The path to the directory containing the data to use (default: ./data)'
	)
	parser.add_argument(
		'-s',
		'--source-file',
		type=str,
		default=None,
		help='The path to the source file to use to compute sentence statistics (default: None)'
	)

	parser.add_argument(
		'--model-name',
		type=str,
		default=None,
		help='The name of the model to be used',
		choices=[
			'original-bert', 'our-bert', 'thf-regular', 'thf-sentence', 'crammed'
		]
	)

	parser.add_argument(
		'--action',
		type=str,
		default='build_wikipedia_sentences',
		choices=[
			'build_wikipedia_sentences', 'build_bookcorpusopen_sentences',
			'compute_sentence_statistics',
			'encapsulate_sentence_dataset',
			'combine_sentence_datasets',
			'build_splits', 'build_new_piece',
			'build_piece_dataset', 'build_sentence_piece_dataset', 'build_new_sentence_piece', 'build_pile_sentence_piece', 'build_c4_sentence_piece', 'build_fineweb_sentence_piece',
			'build_dataset_distil', 'train-sae', 'test-sae', 'train-thf',
			'build-embeddings-race', 'train-slt-race',
			'build-tiny-sentence-piece-dataset', 'build_dataset_mlm',
			'train-thf-mlm', 'train-thf-distil', 'train-thf-finetune', 'train-p+f-bert',
			'thf-p+f', 'thf-pretrain-new'
		],
		help='The action to perform (default: build_sentence_dataset)'
	)
	parser.add_argument(
		'--max-length',
		type=int,
		default=128,
		help='The maximum length of the input sequence (default: 128)'
	)
	parser.add_argument(
		'--embedding-dim',
		type=int,
		default=256,
		help='The embedding dimension of each token fed into the transformer (default: 768)'
	)
	parser.add_argument(
		'--transformer-heads',
		type=int,
		default=4,
		help='The number of heads to use in each of the encoder and decoder (default: 12, to go with default --embedding-dim=768)'
	)
	parser.add_argument(
		'--embedding-width',
		type=int,
		default=1,
		help='The number of tokens to try to embed the sentence into (default: 1)'
	)
	parser.add_argument(
		'--every-n',
		type=int,
		default=1,
		help='In regular mode, how many tokens are compressed into a single token'
	)
	parser.add_argument(
		'--embedding-multiplier',
		type=int,
		default=1,
		help='The number of times to use the encoder output sentence embedding in the input to the decoder. If -1, the decoder input is filled with the encoder output (default: 1)'
	)
	parser.add_argument(
		'--transformer-depth',
		type=int,
		default=2,
		help='The number of layers to use in each of the encoder and decoder (default: 1)'
	)
	parser.add_argument(
		'--lr-pretrain',
		type=float,
		default=2e-4,
		help='The learning rate to use in training (default: 1e-3)'
	)
	parser.add_argument(
		'--lr-finetune',
		type=float,
		default=4e-5,
		help='The learning rate to use in training (default: 4e-5)'
	)
	parser.add_argument(
		'--batch-size-pretrain',
		type=int,
		default=128,
		help='The batch size to use for training and evaluation (default: 128)'
	)
	parser.add_argument(
		'--batch-size-finetune',
		type=int,
		default=16,
		help='The batch size to use for training and evaluation (default: 16)'
	)
	parser.add_argument(
		'--dropout-pretrain'
		, type=float,
		default=0,
		help='The dropout rate to use in training (default: 0)'
	)
	parser.add_argument(
		'--dropout-finetune'
		, type=float,
		default=0.1,
		help='The dropout rate to use in training (default: 0.1)'
	)
	parser.add_argument(
		'--epochs-pretrain',
		type=int,
		default=1,
		help='The number of epochs to train for (default: 1)'
	)
	parser.add_argument(
		'--epochs-finetune',
		type=int,
		default=5,
		help='The number of epochs to train for (default: 5)'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42,
		help='The seed for torch, numpy, and python randomness (default: 1234)'
	)
	
	parser.add_argument(
		'--checkpoint-frequency',
		type=int,
		default=100_000,
		help='The frequency at which to save checkpoints (default: 100000)'
	)
	parser.add_argument(
		'--checkpoint',
		type=str,
		default=None,
		help='The name of the checkpoint to load (default: None)'
	)
	parser.add_argument(
		'--log-wandb',
		type=bool,
		default=False,
		help='Whether to log to wandb (default: False)'
	)
	parser.add_argument(
		'--mode',
		type=str,
		default='sentence',
		choices = ['sentence', 'regular'],
		help='Sentence embedding or take every X embeddings'
	)
	parser.add_argument(
		'--task-name',
		type=str,
		default='mrpc',
		choices = ['mrpc', 'rte', 'qqp', 'qnli', 'sst2', 'mnli', 'cola', 'stsb'],
		help='The name of the task to train on'
	)

	return parser