
import sys
import torch
import os
import wandb
import numpy as np

from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import collections

def run(action, meta_config, experiment_config, dataset, model, tokenizer):
	data_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=meta_config.max_length, pad_to_multiple_of=8, return_tensors="pt")
	dataloader = DataLoader(dataset, shuffle=(action == 'train-sae'), batch_size=experiment_config.batch_size_pretrain, collate_fn=data_collator)

	criterion = torch.nn.CrossEntropyLoss()
	if action == 'train-sae':
		optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr_pretrain)
	else:
		optimizer = None

	# torch.autograd.set_detect_anomaly(False)
	
	model = model.to(device)
	total_samples = 0
	num_saves = 0
	total_correct_answers = 0
	total_relevant_comparisons = 0
	total_mean_correctnesses = 0
	counts = np.zeros(128)
	correctnesses = np.zeros(128)
	first_example_per_length = [[] for _ in range(128)]
	output_example_per_length = [[] for _ in range(128)]
	for epoch in range(experiment_config.epochs_pretrain):
		for batch_id, data in enumerate(dataloader, 0):
			data['input_ids'] = data['input_ids'].to(device)
			data['attention_mask'] = data['attention_mask'].to(device)

			# forward, backward, optimize
			outputs = model(data['input_ids'], data['attention_mask'])
			output_labels = torch.argmax(outputs, dim=2)
			comparisons = (output_labels == data['input_ids']) * data['attention_mask'].bool()
			relevant_comparisons = torch.masked_select(comparisons, data['attention_mask'].bool())
			batch_correct_answers = torch.sum(relevant_comparisons).item()
			batch_relevant_comparisons = torch.numel(relevant_comparisons)
			batch_correctnessess = comparisons.float().sum(dim=-1) / data['attention_mask'].sum(dim=-1)
			batch_mean_correctness = batch_correctnessess.sum().item()
			loss = criterion(outputs.transpose(1, 2), data['input_ids']) / experiment_config.gradient_accumulation_steps

			if action == 'train-sae':
				loss.backward()

				if (batch_id+1) % experiment_config.gradient_accumulation_steps == 0:
						optimizer.step()
						optimizer.zero_grad()

			total_samples += experiment_config.batch_size
			total_correct_answers += batch_correct_answers
			total_relevant_comparisons += batch_relevant_comparisons
			total_mean_correctnesses += batch_mean_correctness
			wandb.log({
				"epoch": epoch,
				"batch": batch_id,
				"batch_loss": loss.cpu().item(),
				"batch_mean_correctness": batch_mean_correctness / 16,
				"batch_weighted_correctness": batch_correct_answers / max(batch_relevant_comparisons, 1),
				"total_correct_answers": total_correct_answers,
				"total_relevant_comparisons": total_relevant_comparisons,
				"total_mean_correctnesses": total_mean_correctnesses,
			})
			del loss, outputs

			if action == 'test-sae':
				lengths = torch.sum(data['attention_mask'], dim=-1).detach().cpu().numpy()
				for i in range(lengths.shape[0]):
					if lengths[i] > 127:
						continue
					counts[lengths[i]] += 1
					correctnesses[lengths[i]] += batch_correctnessess[i].item()

					if not first_example_per_length[lengths[i]] and batch_correctnessess[i] < 1.0:
						first_example_per_length[lengths[i]] = tokenizer.decode(data['input_ids'][i].tolist(), skip_special_tokens=True)
						output_example_per_length[lengths[i]] = tokenizer.decode(output_labels[i].tolist(), skip_special_tokens=True)
			
			if action == 'train-sae' and (total_samples - (num_saves + 1) * meta_config.checkpoint_frequency > 0 or \
				(epoch == experiment_config.epochs-1 and batch_id == len(dataloader)-1)):
				num_saves += 1
				torch.save(model, os.path.join(meta_config.output_path, f"sae-{meta_config.job_id}-last.pt"))

			if total_samples == 96_000_000:
				break


	print('Finished')
	print(model.config.to_dict())
	print(f"Total samples: {total_samples}")
	print(f"Total correct answers: {total_correct_answers}")
	print(f"Total relevant comparisons: {total_relevant_comparisons}")
	print(f"Total mean correctness: {total_mean_correctnesses}")
	print(f"Overall weighted correctness: {total_correct_answers / total_relevant_comparisons}")
	print(f"Overall mean correctness: {total_mean_correctnesses / total_samples}")
	if action == 'test-sae':
		with np.printoptions(threshold=sys.maxsize):
			print(counts)
			print(correctnesses)
			counts = np.where(counts == 0, 1, counts)
			print(correctnesses / counts)
			print(first_example_per_length[10:30])
			print(output_example_per_length[10:30])
