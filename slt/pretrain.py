from curses import meta
from math import exp
from sched import scheduler
from numpy import isin
from transformers import DataCollatorForLanguageModeling
from .thf_model.collation_mlm import DataCollatorWithSentenceTokensForMLM
import torch

from transformers import Trainer, TrainingArguments
import os

from .thf_model.modeling_thf_1d import THF
import torch.optim as optim

from transformers import TrainerCallback

import cramming

from torch.utils.data import DataLoader, Dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_batch_size = 4096

# Create lamda tokenizing function
def run(meta_config, experiment_config, dataset, model, tokenizer, is_sentence_thf=False):	
    if is_sentence_thf:
        data_collator = DataCollatorWithSentenceTokensForMLM(
            tokenizer=tokenizer, max_length=128, pad_to_multiple_of=8
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr_pretrain, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = len(dataset) // target_batch_size * experiment_config.epochs_pretrain
    print(total_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=experiment_config.batch_size_pretrain, collate_fn=data_collator, num_workers=16, pin_memory=True)
    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 5:
            break


    model = model.to(device)

    if isinstance(model, THF):
        model.position_encoding = model.position_encoding.to(device)
        model.sentence_position_encoding = model.sentence_position_encoding.to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join(meta_config.output_path, f'thf-{meta_config.job_id}'),
        overwrite_output_dir=True,
        num_train_epochs=experiment_config.epochs_pretrain,
        per_device_train_batch_size=experiment_config.batch_size_pretrain,
        save_steps=meta_config.checkpoint_frequency,
        prediction_loss_only=True,
        learning_rate=experiment_config.lr_pretrain,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        logging_steps=10,
        #lr_scheduler_type='linear',
        #warmup_steps=10_000,# // gradient_accumulation_steps,
        gradient_accumulation_steps=target_batch_size // experiment_config.batch_size_pretrain,
        report_to='wandb'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),  # Set the custom optimizer and scheduler
        train_dataset=dataset,
    )

    # Train
    if meta_config.checkpoint == None :
        print("Pre-training BERT model")
        trainer.train()
    else:
        print("Pre-training BERT model from checkpoint", meta_config.checkpoint)
        trainer.train(resume_from_checkpoint=meta_config.checkpoint)

    save_path = os.path.join(meta_config.output_path, f'thf-{meta_config.job_id}.pt')
    torch.save(model, save_path)
    print('Finished pre-training')
