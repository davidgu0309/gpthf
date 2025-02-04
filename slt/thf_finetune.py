from curses import meta
import re
import torch
import os
import wandb

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from .thf_model.collation_finetune import DataCollatorWithSentenceTokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import collections
from tqdm import tqdm

import numpy as np


eval_iters = 100000 

def precision_recall_f1(preds, labels):
	true_pos = np.sum((preds == 1) & (labels == 1))
	false_pos = np.sum((preds == 1) & (labels == 0))
	false_neg = np.sum((preds == 0) & (labels == 1))

	precision = true_pos / (true_pos + false_pos + 1e-12)
	recall = true_pos / (true_pos + false_neg + 1e-12)
	f1 = 2 * precision * recall / (precision + recall + 1e-12)

	return precision, recall, f1

def run(meta_config, experiment_config, dataset, model, tokenizer, distil=False):
	print('Using model: ', model)
	train_dataset = dataset['train'].select(range(0, int(1 * len(dataset['train'])))) 
	val_dataset = dataset['validation']
	#test_dataset = dataset['test']

	data_collator = DataCollatorWithSentenceTokens(tokenizer, max_length=meta_config.max_length, pad_to_multiple_of=8)
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=experiment_config.batch_size_finetune, collate_fn=data_collator, num_workers=8, pin_memory=True)
	val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=experiment_config.batch_size_finetune, collate_fn=data_collator, num_workers=8, pin_memory=True)

	criterion = torch.nn.KLDivLoss(reduction='batchmean') if distil else torch.nn.CrossEntropyLoss()
	print(criterion)
	optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr_finetune)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*meta_config.epochs_finetune, eta_min=0)
	
	model = model.to(device)

	total_samples = 0
	num_saves = 0

	for epoch in range(experiment_config.epochs_finetune):
		model.train()		

		running_losses = collections.deque()
		running_correct_answers = collections.deque()
		running_answers_count = collections.deque()

		accumulated_correct_answers = 0
		accumulated_positive = 0	
		accumulated_relevant_comparisons = 0

		total_correct_answers = 0
		total_positive = 0
		total_relevant_comparisons = 0

		for batch_id, data in enumerate(tqdm(train_dataloader)):
			for keys in data:
				data[keys] = data[keys].to(device)

			optimizer.zero_grad()

			outputs = model(data["input_ids"], data["attention_mask"], meta_config.mode, data["word_attention_mask"], data["sentence_indices"], data["sentence_attention_mask"], data["segment_ids"], every_n=meta_config.every_n)
			outputs = torch.log_softmax(outputs, dim=-1) if distil else outputs

			labels = data['labels']

			loss = criterion(outputs, labels)
			loss.backward()

			preds = torch.argmax(outputs, dim=-1)
			correct = (preds == torch.argmax(labels, 1)).sum() if distil else (preds == labels).sum()
			positive = (preds == 1).sum()

			accumulated_correct_answers += correct.item()
			accumulated_positive += positive.item()
			accumulated_relevant_comparisons += torch.numel(labels)

			total_correct_answers += correct.item()
			total_positive += positive.item()
			total_relevant_comparisons += torch.numel(labels)

			if batch_id % 100 == 0:
				for name, p in model.named_parameters():
					if p.grad is not None:
						metric = (experiment_config.lr_finetune * p.grad.std() / (p.data.std() + 1e-6)).log10()
						wandb.log({f"param_update_ratio/{name}": metric.item(), "epoch": epoch, "batch": batch_id})

			optimizer.step()
			scheduler.step()

			# Log statistics
			if batch_id % 10 == 0:
				# Reset accumulators
				running_losses.append(loss.item())
				running_correct_answers.append(accumulated_correct_answers)
				running_answers_count.append(accumulated_relevant_comparisons)
				wandb.log({
					"epoch": epoch,
					"batch": batch_id,
					"current_batch_loss": loss.item(),
					"learning_rate": scheduler.get_last_lr()[0],
					"accumulated_accuracy": accumulated_correct_answers / max(accumulated_relevant_comparisons, 1),
					"rolling_mean_loss": sum(running_losses) / len(running_losses),
					"rolling_correctness": sum(running_correct_answers) / max(sum(running_answers_count), 1)
				})
				
				accumulated_correct_answers = 0
				accumulated_relevant_comparisons = 0

				# if device == torch.device("cuda"):
				# 	torch.cuda.empty_cache()

			total_samples += experiment_config.batch_size_pretrain

			if len(running_losses) > 1000:
				running_losses.popleft()
				running_correct_answers.popleft()
				running_answers_count.popleft()
			
			# if total_samples - (num_saves + 1) * meta_config.checkpoint_frequency > 0 or \
			# 	(epoch == experiment_config.epochs_finetune-1 and batch_id == len(train_dataloader)-1):
			# 	num_saves += 1
			# 	torch.save(model, os.path.join(meta_config.output_path, f"thf-{meta_config.job_id}-{total_samples}.pt"))

		print('Training accuracy: ', total_correct_answers / max(total_relevant_comparisons, 1))
		print('Total positive: ', total_positive / max(total_relevant_comparisons, 1))

		print('Evaluating...')
		#model.eval()
		val_losses = []
		val_correct_answers = 0
		val_relevant_comparisons = 0
		val_positive_answers = 0
		all_preds = []
		all_labels = []

		with torch.no_grad():
			for val_data in val_dataloader:
				for keys in val_data:
					val_data[keys] = val_data[keys].to(device)

				outputs = model(val_data["input_ids"], val_data["attention_mask"], meta_config.mode, val_data["word_attention_mask"], val_data["sentence_indices"], val_data["sentence_attention_mask"], val_data["segment_ids"], every_n=meta_config.every_n)
				outputs = torch.log_softmax(outputs, dim=-1) if distil else outputs
				
				labels = val_data['labels']

				loss = criterion(outputs, labels)
				val_losses.append(loss)

				preds = torch.argmax(outputs, dim=-1)
				correct = (preds == torch.argmax(labels, 1)).sum() if distil else (preds == labels).sum()
				positive = (preds == 1).sum()
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(labels.cpu().numpy())

				val_correct_answers += correct.item()
				val_relevant_comparisons += torch.numel(labels)
				val_positive_answers += positive.item()

			avg_val_loss = sum(val_losses) / len(val_losses)
			avg_val_acc = val_correct_answers / val_relevant_comparisons
			avg_val_pos = val_positive_answers / val_relevant_comparisons

			precision, recall, f1 = precision_recall_f1(np.array(all_preds), np.array(all_labels))

			print('Validation loss: ', avg_val_loss)
			print('Validation accuracy: ', avg_val_acc)
			print('Validation positive: ', avg_val_pos)
			print('Validation precision: ', precision, ' recall: ', recall, ' f1: ', f1)

			wandb.log({"validation_loss": avg_val_loss,
						"validation_accuracy": avg_val_acc,
						"validation_f1": f1,
					})

	print('Finished')