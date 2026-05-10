import argparse
import copy
import inspect
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

import config
import distributed
import models
from tak_counting_data import (
    DEFAULT_SHIFTED_START_TASK,
    evaluate_counting_model,
    get_shifted_start_task_spec,
    load_counting_split,
    make_collate_fn,
    default_data_root,
    unpack_model_outputs,
)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base", choices=config.registered_formats())
    parser.add_argument("--ib_task", default=DEFAULT_SHIFTED_START_TASK)
    parser.add_argument("--ib_data_root", default=str(default_data_root()))
    parser.add_argument("--ib_train_split", default="train")
    parser.add_argument("--ib_eval_splits", nargs="+", default=["val", "ood_test"])
    parser.add_argument("--ib_num_workers", type=int, default=0)
    parser.add_argument("--ib_eval_batch_size", type=int, default=None)
    parser.add_argument("--ib_eval_max_batches", type=int, default=None)
    parser.add_argument("--ib_save_every", type=int, default=None)
    parser.add_argument("--ib_log_every", type=int, default=None)
    parser.add_argument("--ib_best_split", default="ood_test")
    parser.add_argument("--ib_best_metric", default="acc")
    parser.add_argument("--ib_best_mode", choices=["max", "min"], default="max")
    parser.add_argument("--ib_big_eval_splits", nargs="+", default=None)
    parser.add_argument("--ib_big_eval_max_batches", type=int, default=None)

    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format,
        base_parser=parser,
        args=rem_args,
        namespace=args,
    )


def print_master(distributed_backend, message):
    if distributed_backend.is_master_process():
        print(message)


def coerce_seed(seed):
    if isinstance(seed, (list, tuple)):
        return int(seed[0])
    return int(seed)


def apply_task_shape(args, spec, distributed_backend):
    if getattr(args, "sequence_length", None) is None:
        args.sequence_length = spec.minimum_sequence_length
    elif int(args.sequence_length) < spec.minimum_sequence_length:
        print_master(
            distributed_backend,
            f"Raising sequence_length from {args.sequence_length} to {spec.minimum_sequence_length}",
        )
        args.sequence_length = spec.minimum_sequence_length

    old_vocab_size = getattr(args, "vocab_size", None)
    if old_vocab_size != spec.vocab_size:
        print_master(
            distributed_backend,
            f"Setting vocab_size to {spec.vocab_size} for {spec.name}",
        )
        args.vocab_size = spec.vocab_size

    if getattr(args, "dataset", None) != spec.name:
        print_master(distributed_backend, f"Setting dataset to {spec.name}")
        args.dataset = spec.name


def maybe_make_sampler(dataset, shuffle, seed):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    return None


def make_loader(dataset, spec, sequence_length, batch_size, shuffle, seed, num_workers, pin_memory):
    sampler = maybe_make_sampler(dataset, shuffle=shuffle, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=make_collate_fn(spec, sequence_length),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def infinite_batches(dataloader):
    epoch = 0
    while True:
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def make_optimizer(args, model, distributed_backend, device_type):
    raw_model = distributed_backend.get_raw_model(model)
    group_specs = raw_model.get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0

    for group in group_specs:
        params = []
        for p_name in group["params"]:
            translated = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[name] for name in translated]
        group["params"] = params
        optimized_params_cnt += sum(param.numel() for param in params)

    print(f"number of optimized parameters: {optimized_params_cnt / 1e6:.2f}M")
    if args.opt == "adamw":
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    if args.opt == "adafactor":
        from optim.adafactor import Adafactor

        return Adafactor(group_specs, lr=args.lr)
    return torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


def make_scheduler(args, optimizer):
    if getattr(args, "scheduler", "none") == "none":
        return None
    if args.scheduler not in ["cos", "linear"]:
        raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.iterations,
        pct_start=args.warmup_percent,
        anneal_strategy=args.scheduler,
        cycle_momentum=False,
        div_factor=1e2,
        final_div_factor=args.final_div_factor,
    )


def latest_checkpoint_name(ckpt_path):
    checkpoints = [file for file in os.listdir(ckpt_path) if file.startswith("ckpt_") and file.endswith(".pt")]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda file: int(file.split("ckpt_")[1].split(".pt")[0]))


def load_checkpoint_if_needed(args, model, optimizer, scheduler, ckpt_path, device):
    use_pretrained = getattr(args, "use_pretrained", None)
    if use_pretrained == "None":
        use_pretrained = None
    if use_pretrained == "auto":
        use_pretrained = latest_checkpoint_name(ckpt_path)
    if use_pretrained is None:
        return 0
    if use_pretrained is False:
        return 0

    print(f"Resuming from {use_pretrained}")
    checkpoint = torch.load(os.path.join(ckpt_path, use_pretrained), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["itr"])


def sanitize_args_for_json(args):
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, torch.device):
            result[key] = str(value)
        elif isinstance(value, torch.dtype):
            result[key] = str(value)
        elif isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def to_loggable_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().float().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_eval_stats_to_wandb_logs(logs, eval_stats, prefix="eval"):
    for split, split_stats in eval_stats.items():
        for metric, value in split_stats.items():
            scalar = to_loggable_float(value)
            if scalar is not None:
                logs[f"{prefix}/{split}/{metric}"] = scalar


def add_fixed_cot_diagnostics_to_wandb_logs(logs, raw_model, diag_outputs):
    if hasattr(raw_model, "backward_metrics"):
        for key, value in raw_model.backward_metrics.items():
            scalar = to_loggable_float(value)
            if scalar is not None:
                logs[f"diag_grad/{key}"] = scalar
        raw_model.backward_metrics.clear()

    if hasattr(raw_model, "forward_metrics"):
        for key, value in raw_model.forward_metrics.items():
            scalar = to_loggable_float(value)
            if scalar is not None:
                logs[f"diag_step/{key}"] = scalar
        raw_model.forward_metrics.clear()

    if not isinstance(diag_outputs, dict):
        return

    for output_key, log_key in [
        ("sim_of_xs", "diag/boundary_sim"),
        ("var_into", "diag/var_into"),
        ("var_outof", "diag/var_outof"),
    ]:
        scalar = to_loggable_float(diag_outputs.get(output_key))
        if scalar is not None:
            logs[log_key] = scalar

    d_metrics = diag_outputs.get("diag_metrics") or {}
    if "macro_rep_entropy" in d_metrics:
        logs["diag_macro/repeat_entropy"] = float(d_metrics["macro_rep_entropy"])

    if "macro_budget" in d_metrics:
        macro_budget = d_metrics["macro_budget"]
        num_repeats = len(macro_budget)
        for repeat_idx in range(num_repeats):
            loop_num = repeat_idx + 1
            logs[f"diag_macro/budget_loop_{loop_num}"] = float(macro_budget[repeat_idx])
            if "macro_in_entropy" in d_metrics:
                logs[f"diag_macro/within_entropy_loop_{loop_num}"] = float(
                    d_metrics["macro_in_entropy"][repeat_idx]
                )
            if "macro_same_pos" in d_metrics:
                logs[f"diag_macro/same_pos_budget_loop_{loop_num}"] = float(
                    d_metrics["macro_same_pos"][repeat_idx]
                )

    if "head_rep_entropy" in d_metrics and "head_budget" in d_metrics:
        head_rep_entropy = np.asarray(d_metrics["head_rep_entropy"])
        head_budget = np.asarray(d_metrics["head_budget"])
        if head_budget.ndim == 2:
            for head_idx in [0, 5, 11]:
                if head_idx >= len(head_rep_entropy):
                    continue
                logs[f"diag_head_{head_idx}/repeat_entropy"] = float(head_rep_entropy[head_idx])
                for repeat_idx in range(head_budget.shape[1]):
                    logs[f"diag_head_{head_idx}/budget_loop_{repeat_idx + 1}"] = float(
                        head_budget[head_idx, repeat_idx]
                    )


def sorted_eval_items(stats):
    return sorted(
        ((int(step), eval_stats) for step, eval_stats in stats.get("eval", {}).items()),
        key=lambda item: item[0],
    )


def make_wandb_line_series(xs, ys, keys, title, xname="step"):
    if not xs or not ys or not keys:
        return None
    try:
        return wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=title, xname=xname)
    except Exception as exc:
        print(f"WARNING: could not build WandB plot '{title}': {exc}")
        return None


def build_eval_metric_plot(stats, metric):
    eval_items = sorted_eval_items(stats)
    if not eval_items:
        return None

    steps = [step for step, _ in eval_items]
    split_names = sorted({
        split
        for _, eval_stats in eval_items
        for split, split_stats in eval_stats.items()
        if metric in split_stats
    })
    series = []
    keys = []
    for split in split_names:
        values = []
        has_value = False
        has_missing = False
        for _, eval_stats in eval_items:
            split_stats = eval_stats.get(split, {})
            value = to_loggable_float(split_stats.get(metric))
            if value is None or not np.isfinite(value) or (metric == "unseen_len_acc" and value < 0):
                has_missing = True
                continue
            values.append(value)
            has_value = True
        if has_value and not has_missing:
            keys.append(split)
            series.append(values)

    return make_wandb_line_series(
        xs=steps,
        ys=series,
        keys=keys,
        title=f"Shifted-start eval {metric}",
    )


def build_train_loss_plot(stats):
    train_rows = stats.get("train_loss", [])
    if not train_rows:
        return None
    return make_wandb_line_series(
        xs=[row["step"] for row in train_rows],
        ys=[[row["loss"] for row in train_rows]],
        keys=["train_loss"],
        title="Shifted-start train loss",
    )


def build_best_eval_bar_plot(best_eval_stats, metric):
    if not best_eval_stats:
        return None
    table = wandb.Table(columns=["split", "value"])
    has_value = False
    for split, split_stats in best_eval_stats.items():
        value = to_loggable_float(split_stats.get(metric))
        if value is None or not np.isfinite(value) or (metric == "unseen_len_acc" and value < 0):
            continue
        table.add_data(split, value)
        has_value = True
    if not has_value:
        return None
    try:
        return wandb.plot.bar(table, "split", "value", title=f"Best checkpoint {metric}")
    except Exception as exc:
        print(f"WARNING: could not build WandB best-eval plot '{metric}': {exc}")
        return None


def log_wandb_summary_plots(stats, best_eval_stats=None, step=None):
    if not wandb.run:
        return

    logs = {}
    train_loss_plot = build_train_loss_plot(stats)
    if train_loss_plot is not None:
        logs["plots/train_loss"] = train_loss_plot

    for metric in ["loss", "acc", "counting_acc", "last_acc", "unseen_len_acc", "average_depth"]:
        plot = build_eval_metric_plot(stats, metric)
        if plot is not None:
            logs[f"plots/eval/{metric}"] = plot

    for metric in ["loss", "acc", "counting_acc", "last_acc", "unseen_len_acc", "average_depth"]:
        plot = build_best_eval_bar_plot(best_eval_stats, metric)
        if plot is not None:
            logs[f"plots/best_eval/{metric}"] = plot

    if logs:
        wandb.log(logs, step=step)


def best_checkpoint_name(split, metric):
    safe_split = split.replace("/", "_")
    safe_metric = metric.replace("/", "_")
    return f"best_{safe_split}_{safe_metric}.pt"


def load_best_info(ckpt_path):
    path = os.path.join(ckpt_path, "best_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_best_info(ckpt_path, best_info):
    with open(os.path.join(ckpt_path, "best_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(best_info, handle, indent=2)


def get_eval_metric(eval_stats, split, metric):
    split_stats = eval_stats.get(split)
    if split_stats is None:
        return None
    value = split_stats.get(metric)
    if value is None:
        return None
    return float(value)


def is_better_metric(value, best_value, mode):
    if best_value is None:
        return True
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"Unsupported best metric mode: {mode}")


def save_checkpoint(model, optimizer, scheduler, itr, ckpt_path, distributed_backend, filename=None):
    if not distributed_backend.is_master_process():
        return
    checkpoint = {
        "model": distributed_backend.get_raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "itr": itr,
        "cpu_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "py_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["gpu_rng_state"] = torch.cuda.get_rng_state()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    checkpoint_name = filename or f"ckpt_{itr}.pt"
    torch.save(checkpoint, os.path.join(ckpt_path, checkpoint_name))


def evaluate_splits(model, eval_loaders, device, max_seen_len, max_batches, ctx):
    return {
        split: evaluate_counting_model(
            model,
            dataloader,
            device=device,
            max_seen_len=max_seen_len,
            max_batches=max_batches,
            ctx=ctx,
        )
        for split, dataloader in eval_loaders.items()
    }


def make_eval_loaders_for_splits(
    split_names,
    data_root,
    spec,
    sequence_length,
    batch_size,
    seed,
    num_workers,
    pin_memory,
):
    datasets = {
        split: load_counting_split(data_root, spec.name, split)
        for split in split_names
    }
    return {
        split: make_loader(
            dataset,
            spec=spec,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)
    type_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type,
        dtype=args.dtype,
    )

    seed = coerce_seed(args.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    spec = get_shifted_start_task_spec(args.ib_task)
    apply_task_shape(args, spec, distributed_backend)

    data_root = Path(args.ib_data_root)
    pin_memory = device_type == "cuda"
    train_dataset = load_counting_split(data_root, spec.name, args.ib_train_split)
    eval_datasets = {
        split: load_counting_split(data_root, spec.name, split)
        for split in args.ib_eval_splits
    }

    print_master(distributed_backend, f"Loading shifted-start task '{spec.name}' from {data_root}")
    print_master(
        distributed_backend,
        f"Train split '{args.ib_train_split}': {len(train_dataset)} examples",
    )
    for split, dataset in eval_datasets.items():
        print_master(distributed_backend, f"Eval split '{split}': {len(dataset)} examples")

    train_loader = make_loader(
        train_dataset,
        spec=spec,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        shuffle=True,
        seed=coerce_seed(getattr(args, "data_seed", seed)),
        num_workers=args.ib_num_workers,
        pin_memory=pin_memory,
    )
    eval_batch_size = args.ib_eval_batch_size or args.batch_size
    eval_loaders = {
        split: make_loader(
            dataset,
            spec=spec,
            sequence_length=args.sequence_length,
            batch_size=eval_batch_size,
            shuffle=False,
            seed=seed,
            num_workers=args.ib_num_workers,
            pin_memory=pin_memory,
        )
        for split, dataset in eval_datasets.items()
    }

    model = models.make_model_from_args(args).to(args.device)
    model = distributed_backend.transform_model(model)
    optimizer = make_optimizer(args, model, distributed_backend, device_type)
    scheduler = make_scheduler(args, optimizer)

    args.world_size = distributed_backend.get_world_size()
    exp_name = args.exp_name
    if distributed_backend.is_master_process() and getattr(args, "wandb", False):
        params_copy = copy.deepcopy(sanitize_args_for_json(args))
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy, entity=args.wandb_entity)

    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    elif os.path.isfile(f"{ckpt_path}/summary.json"):
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)
    distributed_backend.sync()

    raw_model = distributed_backend.get_raw_model(model)
    itr = load_checkpoint_if_needed(args, raw_model, optimizer, scheduler, ckpt_path, args.device)
    print_master(distributed_backend, f"\nTraining shifted-start CoTFormer model={args.model}\n{vars(args)}\n")
    accepts_log_metrics = "log_metrics" in inspect.signature(raw_model.forward).parameters

    batch_iter = infinite_batches(train_loader)
    diag_split = args.ib_eval_splits[0] if args.ib_eval_splits else None
    diag_iter = iter(eval_loaders[diag_split]) if diag_split is not None else None
    stats = {"eval": {}, "train_loss": []}
    save_every = args.ib_save_every if args.ib_save_every is not None else args.eval_freq
    log_every = args.ib_log_every if args.ib_log_every is not None else max(1, min(100, args.eval_freq))
    best_filename = best_checkpoint_name(args.ib_best_split, args.ib_best_metric)
    best_info = load_best_info(ckpt_path)
    if best_info is not None and (
        best_info.get("split") != args.ib_best_split
        or best_info.get("metric") != args.ib_best_metric
        or best_info.get("mode") != args.ib_best_mode
    ):
        best_info = None
    best_value = None if best_info is None else float(best_info["value"])

    def maybe_update_best(eval_stats, step):
        nonlocal best_info, best_value

        value = get_eval_metric(eval_stats, args.ib_best_split, args.ib_best_metric)
        if value is None:
            if step == 0:
                print_master(
                    distributed_backend,
                    "WARNING: cannot track best checkpoint because "
                    f"{args.ib_best_split}.{args.ib_best_metric} is not present in eval stats.",
                )
            return

        if not is_better_metric(value, best_value, args.ib_best_mode):
            return

        best_value = value
        best_info = {
            "step": int(step),
            "split": args.ib_best_split,
            "metric": args.ib_best_metric,
            "mode": args.ib_best_mode,
            "value": float(value),
            "checkpoint": best_filename,
        }
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            step,
            ckpt_path,
            distributed_backend,
            filename=best_filename,
        )
        write_best_info(ckpt_path, best_info)
        print(json.dumps({"step": step, "best_checkpoint": best_info}, indent=2))

        if getattr(args, "wandb", False):
            wandb.log(
                {
                    "iter": step,
                    "best/step": int(step),
                    f"best/{args.ib_best_split}/{args.ib_best_metric}": float(value),
                },
                step=step,
            )

    def next_diag_batch():
        nonlocal diag_iter
        if diag_iter is None:
            return None
        try:
            return next(diag_iter)
        except StopIteration:
            diag_iter = iter(eval_loaders[diag_split])
            return next(diag_iter)

    def collect_diagnostic_outputs():
        diag_batch = next_diag_batch()
        if diag_batch is None:
            return None

        was_training = raw_model.training
        raw_model.eval()
        inputs = diag_batch["input_id"].to(args.device, non_blocking=pin_memory)
        labels = diag_batch["label"].to(args.device, non_blocking=pin_memory)
        with torch.no_grad(), type_ctx:
            outputs = raw_model(inputs, targets=labels, get_logits=False)
        if was_training:
            raw_model.train()
        return outputs

    for step in range(itr, args.iterations):
        if step % args.eval_freq == 0 and distributed_backend.is_master_process():
            eval_stats = evaluate_splits(
                raw_model,
                eval_loaders,
                args.device,
                spec.max_seen_len,
                args.ib_eval_max_batches,
                type_ctx,
            )
            stats["eval"][str(step)] = eval_stats
            print(json.dumps({"step": step, "eval": eval_stats}, indent=2))
            maybe_update_best(eval_stats, step)
            if getattr(args, "wandb", False):
                logs = {"iter": step}
                add_eval_stats_to_wandb_logs(logs, eval_stats)
                diag_outputs = collect_diagnostic_outputs()
                add_fixed_cot_diagnostics_to_wandb_logs(logs, raw_model, diag_outputs)
                wandb.log(logs, step=step)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for microstep_idx in range(args.acc_steps):
            batch = next(batch_iter)
            inputs = batch["input_id"].to(args.device, non_blocking=pin_memory)
            labels = batch["label"].to(args.device, non_blocking=pin_memory)
            is_diagnostic_step = (
                accepts_log_metrics
                and (step + 1) % args.eval_freq == 0
                and microstep_idx == args.acc_steps - 1
            )
            forward_kwargs = {
                "targets": labels,
                "get_logits": True,
            }
            if is_diagnostic_step:
                forward_kwargs["log_metrics"] = True
            with type_ctx:
                outputs = model(inputs, **forward_kwargs)
            loss, _ = unpack_model_outputs(outputs)
            (loss / args.acc_steps).backward()
            total_loss += float(loss.detach().item())

        grad_clip = getattr(args, "grad_clip", None)
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        grad_norm = float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        mean_loss = total_loss / args.acc_steps
        stats["train_loss"].append({"step": step + 1, "loss": mean_loss})
        if distributed_backend.is_master_process() and (step + 1) % log_every == 0:
            print(json.dumps({"step": step + 1, "train_loss": mean_loss}))
            if getattr(args, "wandb", False):
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
                logs = {
                    "iter": step + 1,
                    "train/loss": mean_loss,
                    "train/grad_norm": grad_norm,
                    "lr": current_lr,
                }
                add_fixed_cot_diagnostics_to_wandb_logs(logs, raw_model, None)
                wandb.log(logs, step=step + 1)
        if distributed_backend.is_master_process() and (step + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step + 1, ckpt_path, distributed_backend)

    if distributed_backend.is_master_process():
        eval_stats = evaluate_splits(
            raw_model,
            eval_loaders,
            args.device,
            spec.max_seen_len,
            args.ib_eval_max_batches,
            type_ctx,
        )
        stats["eval"][str(args.iterations)] = eval_stats
        maybe_update_best(eval_stats, args.iterations)
        if getattr(args, "wandb", False):
            logs = {"iter": args.iterations}
            add_eval_stats_to_wandb_logs(logs, eval_stats)
            diag_outputs = collect_diagnostic_outputs()
            add_fixed_cot_diagnostics_to_wandb_logs(logs, raw_model, diag_outputs)
            for split, split_stats in eval_stats.items():
                for metric in ["loss", "acc", "counting_acc", "last_acc", "unseen_len_acc"]:
                    if metric in split_stats:
                        logs[f"final/{split}/{metric}"] = float(split_stats[metric])
            wandb.log(logs, step=args.iterations)
        save_checkpoint(model, optimizer, scheduler, args.iterations, ckpt_path, distributed_backend)
        if best_info is not None:
            stats["best"] = best_info
        final_best_eval_stats = None
        if args.ib_big_eval_splits is not None:
            if best_info is None:
                print(
                    "WARNING: skipping big eval because no best checkpoint was selected. "
                    f"Make sure {args.ib_best_split} is included in --ib_eval_splits."
                )
            else:
                best_checkpoint_path = os.path.join(ckpt_path, best_info["checkpoint"])
                checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
                raw_model.load_state_dict(checkpoint["model"], strict=True)
                big_eval_loaders = make_eval_loaders_for_splits(
                    args.ib_big_eval_splits,
                    data_root=data_root,
                    spec=spec,
                    sequence_length=args.sequence_length,
                    batch_size=eval_batch_size,
                    seed=seed,
                    num_workers=args.ib_num_workers,
                    pin_memory=pin_memory,
                )
                big_eval_stats = evaluate_splits(
                    raw_model,
                    big_eval_loaders,
                    args.device,
                    spec.max_seen_len,
                    args.ib_big_eval_max_batches,
                    type_ctx,
                )
                stats["best_eval"] = big_eval_stats
                final_best_eval_stats = big_eval_stats
                if getattr(args, "wandb", False):
                    logs = {"iter": args.iterations}
                    add_eval_stats_to_wandb_logs(logs, big_eval_stats, prefix="best_eval")
                    logs["best/selected_step"] = int(best_info["step"])
                    logs[f"best_eval/selected/{best_info['split']}/{best_info['metric']}"] = float(
                        best_info["value"]
                    )
                    wandb.log(logs, step=args.iterations)
                with open(f"{ckpt_path}/best_eval.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "best": best_info,
                            "eval": big_eval_stats,
                            "args": sanitize_args_for_json(args),
                        },
                        handle,
                        indent=2,
                    )
                print(json.dumps({"best": best_info, "best_eval": big_eval_stats}, indent=2))
        if getattr(args, "wandb", False):
            log_wandb_summary_plots(
                stats,
                best_eval_stats=final_best_eval_stats,
                step=args.iterations,
            )
        stats["args"] = sanitize_args_for_json(args)
        with open(f"{ckpt_path}/summary.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

    distributed_backend.finalize()


if __name__ == "__main__":
    main(get_args())
