import torch
import os
import wandb
from torch.utils.data.dataloader import DataLoader
from slt.bert_model.modeling_bert import BERT

from .thf_model.collation import DataCollatorWithSentenceTokens
import torch.optim as optim
import collections
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_iters = 100000  
target_batch_size = 2048	
		
def run(meta_config, experiment_config, dataset, model, tokenizer):	
	dataset = dataset['train']
	train_dataset = dataset.select(range(int(0.5 * len(dataset))))
	#val_dataset = dataset.select(range(int(0.99 * len(dataset)), len(dataset))) 

	data_collator = DataCollatorWithSentenceTokens(tokenizer, max_length=meta_config.max_length, pad_to_multiple_of=8)
	train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=experiment_config.batch_size_pretrain, collate_fn=data_collator, num_workers=16, pin_memory=True)
	#val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=experiment_config.batch_size_pretrain, collate_fn=data_collator, num_workers=16, pin_memory=True)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr_pretrain, weight_decay=0.01)
	total_steps = len(train_dataloader)*experiment_config.epochs_pretrain
	scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=experiment_config.lr_pretrain, total_steps=total_steps, anneal_strategy='linear', pct_start=0.1)
	# def lr_lambda(current_step: int):
	# 	if current_step < 1000:
	# 		return current_step / 1000
	# 	return max(0.0, float(total_steps - current_step) / (total_steps - 1000))

	# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	model = model.to(device)
	model.position_encoding = model.position_encoding.to(device)
	if not isinstance(model, BERT):
		model.sentence_position_encoding = model.sentence_position_encoding.to(device)

	total_samples = 0
	num_saves = 0
	total_tokens_processed = 0	

	accumulation_steps = target_batch_size // meta_config.batch_size_pretrain

	for epoch in range(experiment_config.epochs_pretrain):	
		running_losses = collections.deque()
		running_correct_answers = collections.deque()
		running_answers_count = collections.deque()

		accumulated_correct_answers = 0
		accumulated_relevant_comparisons = 0

		for batch_id, data in enumerate(tqdm(train_dataloader)):
			for keys in data:
				data[keys] = data[keys].to(device)

			#torch.arange(data['input_ids'].shape[1]).unsqueeze(0).expand(data['input_ids'].shape[0], -1)
			#outputs = model(data["input_ids"], data["attention_mask"], meta_config.mode, data["word_attention_mask"], data["sentence_indices"], data["sentence_attention_mask"], every_n=meta_config.every_n)
			if isinstance(model, BERT):
				outputs = model(data["input_ids"], data["attention_mask"])
			else:
				outputs = model(data["input_ids"], data["attention_mask"], meta_config.mode, data["sentence_ids"], every_n=meta_config.every_n)
			#print(data.keys())

			labels = data['labels']
			mask = labels != -100

			loss = criterion(outputs[mask], labels[mask])
			loss.backward()

			preds = torch.argmax(outputs, dim=-1)
			correct = (preds[mask] == labels[mask]).sum()
			total = mask.sum()

			batch_correct_answers = correct.item()
			batch_relevant_comparisons = total.item()
			accumulated_correct_answers += batch_correct_answers
			accumulated_relevant_comparisons += batch_relevant_comparisons

			#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

			if (batch_id+1) % accumulation_steps == 0:
				for param in model.parameters():
					if param.grad is not None:
						param.grad /= accumulation_steps
			
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()

			batch_tokens = data["attention_mask"].sum().item()
			total_tokens_processed += batch_tokens
			# print statistics
			if batch_id % 100 == 0:
				# Reset accumulators
				running_losses.append(loss.item())
				running_correct_answers.append(accumulated_correct_answers)
				running_answers_count.append(accumulated_relevant_comparisons)
				wandb.log({
					"total_tokens": total_tokens_processed,
					"learning_rate": scheduler.get_last_lr()[0],
					"current_batch_loss": loss.item(),
					"accumulated_accuracy": accumulated_correct_answers / max(accumulated_relevant_comparisons, 1),
					"rolling_mean_loss": sum(running_losses) / len(running_losses),
					"rolling_correctness": sum(running_correct_answers) / max(sum(running_answers_count), 1)
				})
				
				accumulated_correct_answers = 0
				accumulated_relevant_comparisons = 0

				if device == torch.device("cuda"):
					torch.cuda.empty_cache()

			if batch_id % 100000 == 0:
				for name, p in model.named_parameters():
					if p.grad is not None:
						metric = (experiment_config.lr_pretrain * p.grad.std() / (p.data.std() + 1e-6)).log10()
						wandb.log({f"param_update_ratio/{name}": metric.item(), "epoch": epoch, "batch": batch_id})

			total_samples += experiment_config.batch_size_pretrain

			if len(running_losses) > 100:
				running_losses.popleft()
				running_correct_answers.popleft()
				running_answers_count.popleft()
			
			if total_samples - (num_saves + 1) * meta_config.checkpoint_frequency > 0 or \
				(epoch == experiment_config.epochs_pretrain-1 and batch_id == len(train_dataloader)-1):
				num_saves += 1
				save_path = os.path.join(meta_config.output_path, f"thf-{meta_config.job_id}-{total_samples}.pt")
				torch.save(model, save_path)
				
	print('Finished pre-training')
	return save_path