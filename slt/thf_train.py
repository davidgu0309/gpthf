
from curses import meta
import torch
import os
import wandb

from torch.utils.data.dataloader import DataLoader
from .thf_model.collation import DataCollatorWithSentenceTokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import collections
from tqdm import tqdm

from torch import nn
from functools import partial

eval_iters = 1000000  # Set this to the frequency of evaluation during training
target_batch_size = 2048

def run(meta_config, experiment_config, dataset, model, tokenizer):
	train_dataset = dataset.select(range(0, int(0.99 * len(dataset)))) 
	val_dataset = dataset.select(range(int(0.99 * len(dataset)), len(dataset))) 

	data_collator = DataCollatorWithSentenceTokens(tokenizer, max_length=meta_config.max_length, pad_to_multiple_of=8)
	train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=experiment_config.batch_size, collate_fn=data_collator, num_workers=16, pin_memory=True)
	val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=experiment_config.batch_size, collate_fn=data_collator, num_workers=16, pin_memory=True)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr)
	
	
	model = model.to(device)
	model.init_position_encodings(model.config)

	total_samples = 0
	num_saves = 0

	accumulation_steps = target_batch_size // meta_config.batch_size

	for param in model.wlt_encoder.parameters():
		param.requires_grad = False
	for param in model.slt_body.parameters():
		param.requires_grad = False

	for epoch in range(experiment_config.epochs):
		if epoch == 1:
			for param in model.slt_body.parameters():
				param.requires_grad = True
		if epoch == 2:
			for param in model.encoder.parameters():
				param.requires_grad = True
		
		running_losses = collections.deque()
		running_correct_answers = collections.deque()
		running_answers_count = collections.deque()

		accumulated_correct_answers = 0
		accumulated_relevant_comparisons = 0

		for batch_id, data in enumerate(tqdm(train_dataloader)):
			for keys in data:
				data[keys] = data[keys].to(device)

			#torch.arange(data['input_ids'].shape[1]).unsqueeze(0).expand(data['input_ids'].shape[0], -1)
			outputs = model(**data)
			output_labels = torch.argmax(outputs, dim=-1)
			relevant_comparisons = torch.masked_select(output_labels == data['input_ids'], data['attention_mask'])
			loss = criterion(outputs.permute(0, 2, 1), data['input_ids'])

			loss.backward()

			batch_correct_answers = torch.sum(relevant_comparisons)
			batch_relevant_comparisons = torch.numel(relevant_comparisons)
			accumulated_correct_answers += batch_correct_answers
			accumulated_relevant_comparisons += batch_relevant_comparisons

			if (batch_id+1) % accumulation_steps == 0:
				optimizer.step()
				optimizer.zero_grad()
				#scheduler.step()

			if (batch_id+1) % eval_iters == 0:
				print('Evaluating...')
				model.eval()
				val_losses = []
				with torch.no_grad():
					for val_data in val_dataloader:
						val_data['input_ids'] = val_data['input_ids'].to(device)
						val_data['attention_mask'] = val_data['attention_mask'].to(device)
						val_data['sentence_ids'] = val_data['sentence_ids'].to(device)
						val_outputs = model(val_data['input_ids'], val_data['attention_mask'], val_data['sentence_ids'])
						val_loss = criterion(val_outputs.permute(0, 2, 1), val_data['input_ids'])
						val_losses.append(val_loss.item())
					avg_val_loss = sum(val_losses) / len(val_losses)
					wandb.log({"validation_loss": avg_val_loss})
				model.train()

			# print statistics
			if batch_id % 100 == 0:
				# Reset accumulators
				running_losses.append(loss.item())
				running_correct_answers.append(accumulated_correct_answers)
				running_answers_count.append(accumulated_relevant_comparisons)

				wandb.log({
					"epoch": epoch,
					"batch": batch_id,
					"current_batch_loss": loss.item(),
					"accumulated_accuracy": accumulated_correct_answers / max(accumulated_relevant_comparisons, 1),
					"rolling_mean_loss": sum(running_losses) / len(running_losses),
					"rolling_correctness": sum(running_correct_answers) / max(sum(running_answers_count), 1)
				})

				accumulated_correct_answers = 0
				accumulated_relevant_comparisons = 0

				if device == torch.device("cuda"):
					torch.cuda.empty_cache()

			total_samples += experiment_config.batch_size

			if len(running_losses) > 1000:
				running_losses.popleft()
				running_correct_answers.popleft()
				running_answers_count.popleft()
			
			if total_samples - (num_saves + 1) * meta_config.checkpoint_frequency > 0 or \
				(epoch == experiment_config.epochs-1 and batch_id == len(train_dataloader)-1):
				num_saves += 1
				torch.save(model, os.path.join(meta_config.output_path, f"thf-{meta_config.job_id}-{total_samples}.pt"))
				
	print('Finished')