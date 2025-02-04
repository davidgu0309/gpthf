
import torch
import os
import wandb

from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import collections

def run(action, meta_config, experiment_config, main_dataset, validation_dataset, model):
	dataloader = DataLoader(main_dataset, shuffle=False, batch_size=experiment_config.batch_size)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr) if action == "train" else None
	
	was_train_before = model.training
	model.train() if action == "train" else model.eval()

	model = model.to(device)
	total_loss = 0
	total_correct_answers = 0
	total_samples = 0
	num_saves = 0
	epochs_to_run_for = experiment_config.epochs if action == "train" else 1
	for epoch in range(epochs_to_run_for):
		running_losses = collections.deque()
		running_correct_answers = collections.deque()
		running_answers_count = collections.deque()
		for batch_id, data in enumerate(dataloader, 0):
			data['embeddings'] = data['embeddings'].to(device)
			data['answer'] = data['answer'].to(device)
			data['attention_mask'] = data['attention_mask'].to(device)

			# zero the parameter gradients
			if action == "train":
				optimizer.zero_grad()

			# forward, backward, optimize
			outputs = model(data['embeddings'], data['attention_mask'])
			output_labels = torch.argmax(outputs, dim=-1)
			relevant_comparisons = output_labels == data['answer']
			batch_correct_answers = torch.sum(relevant_comparisons).item()
			batch_relevant_comparisons = torch.numel(relevant_comparisons)
			loss = criterion(outputs, data['answer'])

			total_loss += loss.item()
			total_correct_answers += batch_correct_answers

			if action == "train":
				loss.backward()
				optimizer.step()

			# let's try to free up all the memory
			if device == torch.device("cuda"):
				torch.cuda.empty_cache()

			# print statistics
			running_losses.append(loss.item())
			running_correct_answers.append(batch_correct_answers)
			running_answers_count.append(batch_relevant_comparisons)

			if len(running_losses) > 1000:
				running_losses.popleft()
				running_correct_answers.popleft()
				running_answers_count.popleft()

			total_samples += experiment_config.batch_size
			
			validation_addendum = {}
			if action == "train" and \
				total_samples - (num_saves + 1) * meta_config.checkpoint_frequency > 0 or \
				(epoch == experiment_config.epochs-1 and batch_id == len(dataloader)-1):

				num_saves += 1
				torch.save(model, os.path.join(meta_config.output_path, f"slt-{meta_config.job_id}-{total_samples}.pt"))
				validation_loss, validation_correctness = run("validation", meta_config, experiment_config, validation_dataset, None, model)
				validation_addendum = {
					"validation_loss": validation_loss,
					"validation_correctness": validation_correctness
				}

			if action == "train":
				wandb.log({
					"epoch": epoch,
					"batch": batch_id,
					"batch_loss": loss.item(),
					"batch_correctness": batch_correct_answers / max(batch_relevant_comparisons, 1),
					"rolling_mean_loss": sum(running_losses) / len(running_losses),
					"rolling_correctness": sum(running_correct_answers) / max(sum(running_answers_count), 1),
					**validation_addendum
				})

	model.train(was_train_before)

	mean_loss = total_loss / len(dataloader)
	mean_correct_answers = total_correct_answers / total_samples
	return mean_loss, mean_correct_answers

