import re
import datasets
import torch
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
os.environ["WANDB__SERVICE_WAIT"] = "300"

from collections import namedtuple
import sys
import wandb
import os
import random
import numpy as np

from transformers import AutoTokenizer
#from datasets import load_dataset

from . import cli
from . import data

from transformers import BertConfig#, BertForMaskedLM
from .bert_model.modeling_original_bert import BertForMaskedLM

from .sae_model.configuration_sae import SAEConfig
from .sae_model.modeling_sae import SAE
from .thf_model.configuration_thf_tiny import THFConfig

from .bert_model.modeling_bert import BERT
from .bert_model.configuration_bert import BERTConfig



#from .thf_model.configuration_thf import THFConfig
from .thf_model.modeling_thf_1d import THF
from .thf_model.modeling_thf_classification import THFClassifier
from .slt_model.configuration_slt import SLTConfig
from .slt_model.modeling_slt import SLT
from . import sae_train
from . import thf_train
from . import slt_train
from . import thf_pretrain_mlm
from . import thf_finetune
from . import pretrain
import cramming

import hydra

#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# META CONFIG
parser = cli.setup_parser()
meta_config = parser.parse_args()
gettrace = getattr(sys, 'gettrace', None)
meta_config.is_debug_instance = False if gettrace is None or not gettrace() else True

# SEEDS
meta_config.seed = int(meta_config.seed)
random.seed(meta_config.seed)
np.random.seed(meta_config.seed)
torch.manual_seed(meta_config.seed)

# EXPERIMENT CONFIG
default_experiment_config = {
	**meta_config.__dict__,
}
ExperimentConfig = namedtuple('ExperimentConfig', default_experiment_config.keys())
experiment_config = ExperimentConfig(**default_experiment_config)

# INITIALIZE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=4096)

# DATA FORMATION
if meta_config.action == 'train-sae' or meta_config.action == 'test-sae' or meta_config.action == 'train-thf':
	dataset = data.load_dataset(meta_config, tokenizer)
elif meta_config.action == 'train-thf-mlm' or meta_config.action == 'thf-p+f' or meta_config.action == 'train-p+f-bert':
	dataset = data.load_dataset_mlm(meta_config, tokenizer)
elif meta_config.action == 'train-thf-distil':
	dataset = data.load_dataset_distil(meta_config, tokenizer, meta_config.task_name)
elif meta_config.action == 'train-thf-finetune':
	dataset = data.load_dataset_finetune(meta_config, tokenizer, meta_config.task_name)
elif meta_config.action == 'thf-pretrain-new':
	dataset = data.build_dataset_mlm(meta_config, tokenizer)
elif meta_config.action == 'build-embeddings-race':
	model = torch.load(os.path.join(meta_config.output_path, meta_config.model_name), map_location=torch.device('cpu'))
	model = model.to(device)
	data.build_embeddings_race(meta_config, tokenizer, model)
	sys.exit(0)
elif meta_config.action == 'train-slt-race':
	race_train, race_validation, race_test = data.load_race(meta_config)
else:
	if meta_config.action == 'build_wikipedia_sentences':
		wikipedia = datasets.load_dataset("wikipedia", "20220301.en", split="train", cache_dir=meta_config.input_path)
		data.build_sentence_dataset(meta_config, wikipedia, "wikipedia")
	elif meta_config.action == 'build_bookcorpusopen_sentences':
		bookcorpusopen = datasets.load_dataset("bookcorpusopen", split="train", cache_dir=meta_config.input_path)
		data.build_sentence_dataset(meta_config, bookcorpusopen, "bookcorpusopen")
	elif meta_config.action == 'compute_sentence_statistics':
		dataset = data.load_sentence_dataset(meta_config, meta_config.source_file)
		data.compute_sentence_statistics(meta_config, dataset, tokenizer)
	elif meta_config.action == 'encapsulate_sentence_dataset':
		data.encapsulate_sentence_dataset(meta_config, meta_config.source_file)
	elif meta_config.action == 'combine_sentence_datasets':
		data.combine_sentence_datasets(meta_config)
	elif meta_config.action == 'build_splits':
		data.build_splits(meta_config)
	elif meta_config.action == 'build_sentence_piece_dataset':
		source_dataset = data.form_source_dataset(meta_config)
		data.build_sentence_piece_dataset(meta_config, source_dataset, tokenizer)
	elif meta_config.action == 'build_tiny_sentence_piece_dataset':
		source_dataset = data.form_tiny_source_dataset(meta_config)
		data.build_sentence_piece_dataset(meta_config, source_dataset, tokenizer)
	elif meta_config.action == 'build_piece_dataset':
		source_dataset = data.form_source_dataset(meta_config)
		data.build_piece_dataset(meta_config, source_dataset, tokenizer)
	elif meta_config.action == 'build_new_piece':
		source_dataset = data.form_new_dataset(meta_config)
		data.build_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=80, mode='new')
	elif meta_config.action == 'build_new_sentence_piece':
		source_dataset = data.form_new_dataset(meta_config)
		data.new_sentence_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=80, mode='new')
	elif meta_config.action == 'build_pile_sentence_piece':
		os.environ['TOKENIZERS_PARALLELISM'] = 'false'
		source_dataset = data.form_pile_dataset(meta_config)
		data.new_sentence_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=80, mode='pile')
	elif meta_config.action == 'build_c4_sentence_piece':
		os.environ['TOKENIZERS_PARALLELISM'] = 'false'
		source_dataset = data.form_c4_dataset(meta_config)
		data.new_sentence_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=80, mode='c4')
	elif meta_config.action == 'build_fineweb_sentence_piece':
		os.environ['TOKENIZERS_PARALLELISM'] = 'false'
		source_dataset = data.form_fineweb_dataset(meta_config)
		data.new_sentence_piece_dataset_parallel(meta_config, source_dataset, tokenizer, num_threads=80, mode='fineweb')
	elif meta_config.action == 'build_dataset_distil':
		data.build_dataset_distil(meta_config, tokenizer, meta_config.task_name)
	else:
		raise Exception("Unknown action: " + meta_config.action)
	sys.exit(0)

wandb_logging_dir_path = os.path.join(meta_config.output_path, "wandb")
if not os.path.exists(wandb_logging_dir_path):
	os.makedirs(wandb_logging_dir_path)

wandb_project_choice = meta_config.action.split('-')[1] 
# INITIALIZE WANDB
experiment_name = f"{meta_config.job_name}"
if meta_config.log_wandb:
	wandb.init(
		project=wandb_project_choice+("-proto" if meta_config.is_debug_instance else ""),
		name=experiment_name,
		tags=[
			"job_id:"+str(meta_config.job_id),
			"name:"+str(meta_config.job_name)
		],
		settings=wandb.Settings(start_method='thread'),
		dir=wandb_logging_dir_path,
		config=dict(experiment_config._asdict()) if type(experiment_config).__name__ == 'ExperimentConfig' else dict(experiment_config._as_dict()),
		save_code=True
	)
else:
	wandb.init(mode="disabled")

# RUN
if meta_config.action == 'train-sae' or meta_config.action == 'test-sae':
	config = SAEConfig(
		dim=meta_config.embedding_dim,
		wlt_encoder_num_attention_heads=meta_config.transformer_heads,
		wlt_decoder_num_attention_heads=meta_config.transformer_heads,
		wlt_embedding_width=meta_config.embedding_width,
		wlt_embedding_multiplier=meta_config.embedding_multiplier,
		wlt_encoder_num_hidden_layers=meta_config.transformer_depth,
		wlt_decoder_num_hidden_layers=meta_config.transformer_depth
	)
	if meta_config.action == 'test-sae':
		model_path = os.path.join(meta_config.output_path, meta_config.checkpoint)
		model = torch.load(model_path, map_location=torch.device('cpu'))
	else:
		model = SAE(config)

	# PRINT THE NUMBER OF PARAMETERS
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)
	
	model = model.to(device)
	sae_train.run(meta_config.action, meta_config, experiment_config, dataset, model, tokenizer)
elif meta_config.action == 'train-thf':
	config = THFConfig()
	model = THF(config)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	thf_train.run(meta_config, experiment_config, dataset, model, tokenizer)
elif meta_config.action == 'train-thf-mlm':
	if meta_config.checkpoint is None:
		config = THFConfig(dropout=meta_config.dropout_pretrain)
		model = THF(config)
	else:
		model = torch.load(os.path.join(meta_config.output_path, meta_config.checkpoint))
		# config = THFConfig(dropout=meta_config.dropout_pretrain)
		# model = THF(config)
		# model.load_state_dict(torch.load(os.path.join(meta_config.output_path, meta_config.checkpoint)), strict=False)
	
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	thf_pretrain_mlm.run(meta_config, experiment_config, dataset, model, tokenizer)
elif meta_config.action == 'train-thf-distil' or meta_config.action == 'train-thf-finetune':
	if meta_config.checkpoint is None:
		config = THFConfig(dropout=meta_config.dropout_finetune)
		model = THFClassifier(config)
		#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

	else:
		base_model = torch.load(os.path.join(meta_config.output_path, meta_config.checkpoint))
		state_dict = base_model.state_dict()
		relevant_state_dict = {k: v for k, v in state_dict.items() if 'wlt_encoder' in k or 'slt_body' in k or 'wlt_decoder' in k or 'word_embeddings' in k}
		model = THFClassifier(THFConfig(dropout=meta_config.dropout_finetune))
		model.load_state_dict(relevant_state_dict, strict=False)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	if meta_config.action == 'train-thf-finetune':
		thf_finetune.run(meta_config, experiment_config, dataset, model, tokenizer, distil=False)
	else:
		thf_finetune.run(meta_config, experiment_config, dataset, model, tokenizer, distil=True)

elif meta_config.action == 'thf-p+f':
	config = THFConfig(dropout=meta_config.dropout_pretrain)
	model = THF(config)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)
	save_path = thf_pretrain_mlm.run(meta_config, experiment_config, dataset, model, tokenizer)

	print(f"Preparing fine-tuning with checkpoint {save_path}")
	base_model = torch.load(os.path.join(meta_config.output_path, save_path))
	state_dict = base_model.state_dict()
	relevant_state_dict = {k: v for k, v in state_dict.items() if 'wlt_encoder' in k or 'slt_body' in k or 'wlt_decoder' in k or 'word_embeddings' in k}
	model = THFClassifier(THFConfig(dropout=meta_config.dropout_finetune))
	model.load_state_dict(relevant_state_dict, strict=False)
	dataset = data.load_dataset_finetune(meta_config, tokenizer, meta_config.task_name)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	if meta_config.action == 'train-thf-finetune':
		thf_finetune.run(meta_config, experiment_config, dataset, model, tokenizer, distil=False)


elif meta_config.action == 'train-slt-race':
	config = SLTConfig(num_hidden_layers=4)
	model = SLT(config)
	slt_train.run("train", meta_config, experiment_config, race_train, race_validation, model)

elif meta_config.action == 'train-p+f-bert':
	config = BERTConfig(dropout=meta_config.dropout_pretrain)
	model = BERT(config)
	
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	thf_pretrain_mlm.run(meta_config, experiment_config, dataset, model, tokenizer)

elif meta_config.action == 'thf-pretrain-new':
	if meta_config.model_name == 'original-bert':
		config = BertConfig(
			vocab_size=tokenizer.vocab_size,
			max_position_embeddings=128,
			num_hidden_layers=6,    #L
			hidden_size=256,        #H
			num_attention_heads=8,  #A
			intermediate_size=1024,  #4H
			hidden_dropout_prob=meta_config.dropout_pretrain,
			attention_probs_dropout_prob=meta_config.dropout_pretrain,
		)
		model = BertForMaskedLM(config=config)
	elif meta_config.model_name == 'our-bert':
		config = BERTConfig(dropout=meta_config.dropout_pretrain)
		model = BERT(config)
	elif meta_config.model_name == 'thf-regular':
		config = THFConfig(dropout=meta_config.dropout_pretrain, mode='regular', every_n=meta_config.every_n)
		model = THF(config)
	elif meta_config.model_name == 'thf-sentence':
		config = THFConfig(dropout=meta_config.dropout_pretrain, mode='sentence')
		model = THF(config)

	# elif meta_config.model_name == 'crammed':
	# 	@hydra.main(config_path="cramming/cramming/config", config_name="cfg_pretrain", version_base="1.1")
	# 	def get_model(cfg):
	# 		model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
	# 		return model
		
	# 	model = get_model()

	else:
		raise Exception("Unknown model name: " + meta_config.model_name)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)
	print("Using model: ", model)

	if meta_config.model_name == 'thf-sentence':
		pretrain.run(meta_config, experiment_config, dataset, model, tokenizer, is_sentence_thf=True)
	else:
		pretrain.run(meta_config, experiment_config, dataset, model, tokenizer, is_sentence_thf=False)
