"""Script to evaluate a pretrained model."""

import torch
import hydra


import time
import datetime
import logging
from collections import defaultdict

import cramming
import evaluate
import pdb


log = logging.getLogger(__name__)


def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    is_thf = "ScriptableCrammedTHF" in cfg_arch.architectures or "GPTHF" in cfg_arch.architectures or "GPTHF_Peter" in cfg_arch.architectures or "GPTHF_Llama" in cfg_arch.architectures

    if tokenizer.pad_token is None:
        print("Adding pad token")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # if tokenizer.sep_token is None:
    #     print("Adding sep token")
    #     tokenizer.add_special_tokens({'sep_token': '[SEP]'})


    tasks = cramming.prepare_task_dataloaders(tokenizer, cfg.eval, cfg.impl, is_thf=is_thf)

    stats = defaultdict(list)
    metrics_overall = dict()
    metrics_per_task = dict()
    # Start the clocks now:
    for task_name, task in tasks.items():
        metrics = dict()
        for i in range(3):
            cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
            log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
            # Prepare model for finetuning:
            model = cramming.construct_model(cfg_arch, len(tokenizer), downstream_classes=task["num_classes"])
            model_engine, _, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
            model_engine.load_checkpoint(cfg_arch, model_file)

            try:
                assert task_name != "record"
                metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
            except (FileNotFoundError, AssertionError):  # no specific metric downloadable from evaluate, construct directly
                targets = [evaluate.load(metric_name, cache_dir=cfg.impl.path) for metric_name in task["details"]["target_metrics"]]
                metric = evaluate.CombinedEvaluations(targets)
            # Launch training
            model_engine.train()
            loss_vals = []
            for epoch in range(cfg.eval.epochs):
                train_time = time.time()
                for step, batch in enumerate(task["trainloader"]):
                    # Heavy lifting is moved to engines
                    device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask", "sentence_ids"] if is_thf else ["input_ids", "labels", "attention_mask"])   
                    #pdb.set_trace()
                    loss = model_engine.step(device_batch)
                    loss_vals.append(loss.detach())
                    if cfg.dryrun:
                        break

                metrics[task_name] = validate(model_engine, task["validloader"], metric, setup, cfg)

                if task_name not in metrics_per_task:
                    metrics_per_task[task_name] = list(metrics[task_name].keys())
                stats[f"{task_name}_epoch"] += [epoch]
                stats[f"{task_name}_loss"] += [loss.item()]

                stats[f"{task_name}_avg_loss"] += [torch.stack(loss_vals).mean().item()]  # Smoothed loss
                loss_vals = []
                current_lr = model_engine.optimizer.param_groups[0]["lr"]

                log_msg = f"Train loss {loss.item():2.4f} at step {step} with lr {current_lr:.5f}. "
                log_msg += f"[Avg: {stats[f'{task_name}_avg_loss'][-1]:2.4f}] after epoch {epoch}."

                stats[f"{task_name}_train_time"] += [(time.time() - train_time)]
                estimated_train_finish = str(datetime.timedelta(seconds=stats[f"{task_name}_train_time"][-1] * cfg.eval.epochs))
                tokens_per_second = (step + 1) * cfg.eval.max_seq_length * cfg.impl.microbatch_size / stats[f"{task_name}_train_time"][-1]
                log_msg += (
                    f" Perf: {stats[f'{task_name}_train_time'][-1]/60:2.4f}min per epoch ({tokens_per_second:.0f}t/s). "
                    f"Estimated Total Train: {estimated_train_finish}."
                )

                for name, metric_val in metrics[task_name].items():
                    stats[f"{task_name}_{name}"] += [metric_val]
                log.info(log_msg)
                msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in metrics[task_name].items()])
                log.info(f"Validation metric is {msg_metrics} after epoch {epoch}.")
                cramming.utils.wandb_log(stats, cfg)

                if cfg.dryrun:
                    break
            # Launch extra testing if extra validation set exists (as with MNLI-mismatched):
            if task["extra_validloader"] is not None:
                extra_eval_metric = validate(model_engine, task["extra_validloader"], metric, setup, cfg)
                # metrics[task_name + "extra"] = extra_eval_metric
                metrics[task_name].update({f"{k}_extra": v for k, v in extra_eval_metric.items()})
                for name, metric_val in extra_eval_metric.items():
                    stats[f"{task_name}_{name}_extra"] += [metric_val]
                msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in extra_eval_metric.items()])
                log.info(f"Extra validation metric is {msg_metrics} after finetuning.")
                cramming.utils.wandb_log({f"{task_name}_{k}_extra": [v] for k, v in extra_eval_metric.items()}, cfg)
            for name, metric_val in metrics[task_name].items():
                if task_name not in metrics_overall:
                    metrics_overall[task_name] = dict()
                metrics_overall[task_name][name] = metrics_overall[task_name].get(name, []) + [metric_val]
    # Check average metric for each task:
    avg_metrics = dict()
    std_metrics = dict()
    for task_name, task in tasks.items():
        avg_metrics[task_name] = dict()
        std_metrics[task_name] = dict()
        for metric_name in metrics_per_task[task_name]:
            avg_metrics[task_name][metric_name] = torch.as_tensor(metrics_overall[task_name][metric_name]).median().item()
            std_metrics[task_name][metric_name] = torch.as_tensor(metrics_overall[task_name][metric_name]).std().item()
            log.info(f"Average metric on evaluation {cfg.eval.name} for task {task_name} is {avg_metrics[task_name][metric_name]:.2f}.")
            log.info(f"Standard deviation of metric on evaluation {cfg.eval.name} for task {task_name} is {std_metrics[task_name][metric_name]:.2f}.")
            cramming.utils.wandb_log({f"{task_name}_{metric_name}": [avg_metrics[task_name][metric_name]]}, cfg)
    # Check average metric over all tasks:
    target_metrics = []
    for task_name, task in tasks.items():
        target_metric_names = task["details"]["target_metrics"]
        for metric_name in target_metric_names:
            target_metrics.append(metrics_overall[task_name][metric_name])
    metrics_overall[f"{cfg.eval.name}_amean"] = torch.as_tensor(target_metrics).median().item()
    metrics_overall[f"{cfg.eval.name}_hmean"] = torch.as_tensor(target_metrics).pow(-1).median().pow(-1).item()
    log.info(f"Overall average metric on evaluation {cfg.eval.name} is {metrics_overall[f'{cfg.eval.name}_amean']:.2f}.")
    cramming.utils.wandb_log(
        {f"{cfg.eval.name}_amean": [metrics_overall[f"{cfg.eval.name}_amean"]], f"{cfg.eval.name}_hmean": [metrics_overall[f"{cfg.eval.name}_hmean"]]},
        cfg,
    )
    final_metrics = {f"{cfg.eval.name}_amean": metrics_overall[f"{cfg.eval.name}_amean"], f"{cfg.eval.name}_hmean": metrics_overall[f"{cfg.eval.name}_hmean"], "avg_metrics": avg_metrics, "std_metrics": std_metrics}
    print(f"average scores: {avg_metrics}")
    print(f"std: {std_metrics}")
    # Save to summary & db:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return final_metrics  # will be dumped into yaml


@torch.no_grad()
def validate(model_engine, validloader, metric, setup, cfg):
    """Evaluate on validation set."""
    model_engine.eval()
    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask", "sentence_ids"])
        _, predictions = model_engine.forward_inference(**device_batch)

        if getattr(metric, "config_name", "") != "multirc":
            metric.add_batch(predictions=predictions, references=device_batch["labels"])
        else:  # uuuuuughhhhh, whhyyy multirc
            pred_indices = range(step * predictions.shape[0], (step + 1) * predictions.shape[0])
            packages = [dict(idx=validloader.index_lookup[pred_indices[i]], prediction=p) for i, p in enumerate(predictions.cpu())]
            metric.add_batch(predictions=packages, references=batch["labels"])

        if cfg.dryrun and step > 1:
            break

    try:
        eval_metric = metric.compute()
    except ValueError:  # pearson corr computation will raise errors if metric values are NaN
        log.info("Value Error in metrics computation, maybe non-finite values in prediction. Returning backup score.")
        eval_metric = metric.compute(predictions=[0, 1], references=[1, 0])  # spoof terrible result if metric computation fails
    model_engine.train()
    return {k: float(v) for k, v in eval_metric.items()}  # force float returns


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")


if __name__ == "__main__":
    launch()
