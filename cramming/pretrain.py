"""Script for a pretraining run."""
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
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

#hellaswag code stolen from karpathy
from hellaswag import render_example, iterate_examples

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    local_time = time.time()
    checkpoint_rendevous = os.path.join(cfg.base_dir, cfg.name, "intermediate_state.pth")

    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        try:
            metadata = torch.load(checkpoint_rendevous, map_location=torch.device("cpu"))["metadata"]
            initial_step, elapsed_time = metadata["step"], metadata["elapsed"]
        except RuntimeError:
            log.info("Checkpoint file unreadable or corrupted.")
            os.remove(checkpoint_rendevous)
            initial_step, elapsed_time = 0, 0.0
    else:
        initial_step, elapsed_time = 0, 0.0

    is_thf = "ScriptableCrammedTHF" in cfg.arch.architectures or "GPTHF" in cfg.arch.architectures or "GPTHF_Peter" in cfg.arch.architectures or "GPTHF_Llama" in cfg.arch.architectures
    model_engine, _, _, train_dataloader, val_dataloader = cramming.load_backend(model, dataset, tokenizer, cfg.train, cfg.impl, elapsed_time, setup=setup, is_thf=is_thf)
    print("Model engine: ", model_engine)
    if is_thf:
        print("Length of train_dataloader: ", len(train_dataloader))
        print("Length of val_dataloader: ", len(val_dataloader))
    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        log.info(f"Loading intermediate checkpoint from previous run onto device {cfg.impl.local_rank}...")
        model_engine.load_training_checkpoint(checkpoint_rendevous)
    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time() - elapsed_time
    train_time = time.time()
    training_allowed, no_recovery_necessary = True, True
    loss_vals = []

    #pdb.set_trace()

    # Launch training
    last_billion_tokens_checkpoint = 0  
    val_every_tokens = 1_000_000_000
    token_limit = 10_000_000_000
    print("Training allowed: ", cfg.budget)

    for step, batch in enumerate(train_dataloader, initial_step + 1):
        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask", "sentence_ids"] if is_thf else ["input_ids", "labels", "attention_mask"])   
        loss = model_engine.step(device_batch)

        if loss < 0:
            log.info(f"Negative loss in step {step}, batch input: {batch}")
        elif loss > 50:
            log.info(f"Very high loss in step {step}, batch input: {batch}")
    
        loss_vals.append(loss.detach())

        # Check stopping criteria
        if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
            training_allowed = False
            log.info("Reached deadline. Stopping training ...")

        current_token_count = stats["tokens"][-1] if stats["tokens"] else 0

        if current_token_count // val_every_tokens > last_billion_tokens_checkpoint:
            last_billion_tokens_checkpoint = current_token_count // val_every_tokens
            log.info(f"Reached {last_billion_tokens_checkpoint * val_every_tokens} tokens. Running validation...")

        # Resetting and using a subset of the validation data loader
            model_engine.model.eval()
            val_losses = []
            for val_step, val_batch in enumerate(val_dataloader):
                if val_step > 1000:
                    break
                device_val_batch = model_engine.to_device(val_batch,  keys=["input_ids", "labels", "attention_mask", "sentence_ids"] if is_thf else ["input_ids", "labels", "attention_mask"])
                val_loss = model_engine.validation_step(device_val_batch)    # Make sure to use a different method for validation
                val_losses.append(val_loss.item())

            # for i, example in enumerate(iterate_examples("val")):
            #     _, tokens, mask, label = render_example(example)
            #     tokens = tokens.to(model_engine.model.device)
            #     mask = mask.to(model_engine.model.device)
            #     # get the logits
            #     with torch.no_grad():
            #         logits, loss = model(tokens)
            #         pred_norm = get_most_likely_row(tokens, mask, logits)
            #     num_total += 1
            #     num_correct_norm += int(pred_norm == label)
            # # reduce the stats across all processes
            # acc_norm = num_correct_norm / num_total
            # print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            # if cfg.wandb.enabled and cramming.utils.is_main_process():
            #     wandb.log({"hella_swag_accuracy": acc_norm, "tokens": last_billion_tokens_checkpoint * val_every_tokens})

                
        # Calculate and log the mean validation loss
            mean_val_loss = torch.tensor(val_losses).mean().item()
            log.info(f"Validation loss after {last_billion_tokens_checkpoint * val_every_tokens} tokens: {mean_val_loss}")            
            stats["val_loss"].append(mean_val_loss)
            if cfg.wandb.enabled and cramming.utils.is_main_process():
                wandb.log({"validation_loss": mean_val_loss, "tokens": last_billion_tokens_checkpoint * val_every_tokens})

            log.info(f"Saved checkpoint at {last_billion_tokens_checkpoint * val_every_tokens} tokens.")
            model_engine.save_training_checkpoint(identifier=os.path.join(cfg.base_dir, cfg.name, f'{last_billion_tokens_checkpoint}.pth'), metadata=dict(step=step, elapsed=time.time() - wallclock_timer))
            checkpoint_rendevous = os.path.join(cfg.base_dir, cfg.name, f'{last_billion_tokens_checkpoint}.pth')

            model_engine.model.train()

        #if tokens over 10B also stop training
        if stats["tokens"] and stats["tokens"][-1] > token_limit:
            training_allowed = False
            log.info("Reached token limit. Stopping training ...")

        # Collect stats and print to console and upload to wandb
        if step % cfg.impl.print_loss_every_nth_step == 0:
            loss_vals, train_time = collect_stats(step, loss_vals, train_time, stats, model_engine, train_dataloader, cfg)
            if check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        # Checkpointing is triggered from stopping criteria and normal intervals
        if cfg.impl.save_intermediate_checkpoints and step % cfg.impl.save_every_nth_step == 0:
            if loss.detach().isfinite() and cramming.utils.is_main_process() and not cfg.dryrun:
                model_engine.save_training_checkpoint(checkpoint_rendevous, metadata=dict(step=step, elapsed=time.time() - wallclock_timer))

        if not loss.detach().isfinite():
            training_allowed, no_recovery_necessary = engage_troubleshooting(
                model_engine, step, training_allowed, no_recovery_necessary, cfg
            )

        communicate_flags(training_allowed, no_recovery_necessary)

        if (cfg.dryrun and step > 2) or not training_allowed:
            break

        if not no_recovery_necessary:  # synced across devices
            log.info(f"Attempting reload of checkpoint on device {cfg.impl.local_rank}.")
            model_engine.load_training_checkpoint(checkpoint_rendevous)
            no_recovery_necessary = True

    # Save to summary:
    cramming.utils.save_summary("pretrain", cfg, stats, time.time() - local_time, setup)
    if cramming.utils.is_main_process():
        # Save final checkpoint? Might have to recover the latest checkpoint first
        if not loss.detach().isfinite() and cfg.impl.save_intermediate_checkpoints:
            model_engine.load_training_checkpoint(checkpoint_rendevous)
            loss = torch.as_tensor(16.0)  # fake value for model file name
        if loss.detach().isfinite():
            now = datetime.datetime.now()
            long_checkpoint_id = f"{''.join(cfg.arch.architectures)}_{now.strftime('%Y-%m-%d')}_{loss:2.4f}"
            model_engine.save_final_model(os.path.join(cfg.base_dir, cfg.name), long_checkpoint_id, tokenizer, cfg.arch, cfg.dryrun)

            if cfg.impl.push_to_huggingface_hub:
                model_engine.push_to_hub(tokenizer, cfg, dryrun=cfg.dryrun)
    metrics = dict(num_params=sum([p.numel() for p in model.parameters()]))
    return metrics


def check_deadline(launch_time, hour_limit):
    """These measurements are deliberately wall-clock based."""
    current_time = time.time()
    return True if (current_time - launch_time) / 3600 > hour_limit else False


def check_early_termination(launch_time, loss, early_termination):
    """Early termination based on terrible loss."""
    if early_termination.enabled and loss > early_termination.loss_threshold:
        current_time = time.time()
        return True if (current_time - launch_time) / 3600 > early_termination.budget else False
    else:
        return False


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
    cramming.utils.main_launcher(cfg, main_training_process, job_name="pretraining")


if __name__ == "__main__":
    launch()
