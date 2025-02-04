"""Script for a pretraining run."""
from click import File
import pandas as pd
import torch
import hydra

import os
import time
import datetime
import logging
from collections import defaultdict

import cramming
import wandb

log = logging.getLogger(__name__)

#to_validate = ['opt-125m', 'opt-350m', 'llama-small', 'llama-standard']
#to_validate = ['opt-350m']

def main_validation_process(cfg, setup):
    """This function controls the central training loop."""
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    is_thf = False
    val_every_tokens = 1_000_000_000

    for i in range(1, 11):
        path = os.path.join(cfg.base_dir, cfg.name, f"{i}.pth")
        if cfg.impl.resume_run_after_preempt and os.path.isfile(path):
            try:
                metadata = torch.load(path, map_location=torch.device("cpu"))["metadata"]
                initial_step, elapsed_time = metadata["step"], metadata["elapsed"]
            except FileNotFoundError:
                log.info("Checkpoint file not found.")
                continue
            except RuntimeError:
                log.info("Checkpoint file unreadable or corrupted.")
                continue
            
        else:
            initial_step, elapsed_time = 0, 0.0

        model_engine, _, _, _, val_dataloader = cramming.load_backend(model, dataset, tokenizer, cfg.train, cfg.impl, elapsed_time, setup=setup, is_thf=is_thf)
        print("Model engine: ", model_engine)

        log.info(f"Loading intermediate checkpoint from previous run onto device {cfg.impl.local_rank}...")
        model_engine.load_training_checkpoint(path)
        model_engine.eval()
        stats = defaultdict(list)

        val_losses = []
        log.info(f"Starting validation after {i * val_every_tokens} tokens.")
        for val_step, val_batch in enumerate(val_dataloader):
            if val_step > 1000:
                break
            device_val_batch = model_engine.to_device(val_batch,  keys=["input_ids", "labels", "attention_mask", "sentence_ids"] if is_thf else ["input_ids", "labels", "attention_mask"])
            val_loss = model_engine.validation_step(device_val_batch)    # Make sure to use a different method for validation
            val_losses.append(val_loss.item())
    
        # Calculate and log the mean validation loss
        mean_val_loss = torch.tensor(val_losses).mean().item()
        log.info(f"Validation loss after {i * val_every_tokens} tokens: {mean_val_loss}")            
        stats["val_loss"].append(mean_val_loss)
        if cfg.wandb.enabled and cramming.utils.is_main_process():
            wandb.log({"validation_loss": mean_val_loss, "tokens": i * val_every_tokens})


def collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg):
    stats["step"] += [step]
    stats["epoch"] += [dataloader.epoch_counter]

    tokens_per_step = model_engine.record_tokens_per_step()
    stats["tokens"] += [step * tokens_per_step]
    stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss

    current_lr = model_engine.optimizer.param_groups[0].get("lr", float("NaN"))
    log_msg = f"Train loss {loss_vals[-1].item():2.4f} at step {step} with lr {current_lr:.5f}. "
    log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per step ({tokens_per_second:.0f}t/s). "
        log_msg += f"Estimated Total Train: {estimated_train_finish}."

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]

    # Publish
    cramming.utils.wandb_log(stats, cfg)
    log.info(log_msg)

    # Clear:
    loss_vals = []
    train_time = time.time()
    return loss_vals, train_time


def engage_troubleshooting(model_engine, step, training_allowed, no_recovery_necessary, cfg):
    log.info(f"Non-finite loss in step {step} on device {cfg.impl.local_rank}.")

    is_finite_grad = [torch.isfinite(p.grad).all() for p in model_engine.model.parameters() if p.grad is not None]
    has_finite_gradients = torch.stack(is_finite_grad).all() if len(is_finite_grad) > 0 else True
    if not has_finite_gradients:
        if "dump_nan_grads" in cfg.impl.troubleshoot_strategy:
            log.info(f"Non-finite gradients in step {step} on device {cfg.impl.local_rank}, dumping...")
            model_engine.optimizer.zero_grad()
        else:
            if "recover_checkpoint" in cfg.impl.troubleshoot_strategy:
                no_recovery_necessary = False
            else:
                training_allowed = False
                log.info(f"Stopping training due to non-finite grads in step {step} on device {cfg.impl.local_rank}.")

    has_finite_parameters = torch.stack([torch.isfinite(p).all() for p in model_engine.model.parameters()]).all()
    if not has_finite_parameters:
        if "recover_checkpoint" in cfg.impl.troubleshoot_strategy:
            no_recovery_necessary = False
        else:
            training_allowed = False
            log.info(f"Stopping training due to non-finite parameters in step {step} on device {cfg.impl.local_rank}.")
    return training_allowed, no_recovery_necessary


def communicate_flags(training_allowed, no_recovery_necessary):
    """A quick and dirty communication through the comm protocol. Should not be a major burden."""
    if torch.distributed.is_initialized():
        comm_tensor_allowed = torch.as_tensor([training_allowed, no_recovery_necessary])
        comm_tensor_allowed = comm_tensor_allowed.cuda() if torch.cuda.is_available() else comm_tensor_allowed.float()
        torch.distributed.all_reduce(comm_tensor_allowed, torch.distributed.ReduceOp.MIN, async_op=False)
        if comm_tensor_allowed[0] >= 1:  # training indeed allowed on all devices
            return True, comm_tensor_allowed[1] > 0
        else:
            return False, True
    else:
        return training_allowed, no_recovery_necessary


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_validation_process, job_name="pretraining")


if __name__ == "__main__":
    launch()
