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
from phop_data import (
    DEFAULT_PHOP_TASK,
    evaluate_phop_model,
    get_phop_task_spec,
    load_phop_split,
    make_collate_fn,
    default_data_root,
    unpack_model_outputs,
)
from tak_shifted_start_main import (
    add_eval_stats_to_wandb_logs,
    add_fixed_cot_diagnostics_to_wandb_logs,
    best_checkpoint_name,
    coerce_seed,
    get_eval_metric,
    infinite_batches,
    is_better_metric,
    load_best_info,
    load_checkpoint_if_needed,
    make_optimizer,
    make_scheduler,
    make_wandb_line_series,
    print_master,
    sanitize_args_for_json,
    save_checkpoint,
    sorted_eval_items,
    to_loggable_float,
    write_best_info,
)


PHOP_PLOT_METRICS = ["loss", "acc", "final_acc", "average_depth"]


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base", choices=config.registered_formats())
    parser.add_argument("--phop_task", default=DEFAULT_PHOP_TASK)
    parser.add_argument("--phop_data_root", default=str(default_data_root()))
    parser.add_argument("--phop_train_split", default="train")
    parser.add_argument("--phop_eval_splits", nargs="+", default=["val", "test"])
    parser.add_argument("--phop_num_workers", type=int, default=0)
    parser.add_argument("--phop_eval_batch_size", type=int, default=None)
    parser.add_argument("--phop_eval_max_batches", type=int, default=None)
    parser.add_argument("--phop_save_every", type=int, default=None)
    parser.add_argument("--phop_log_every", type=int, default=None)
    parser.add_argument("--phop_best_split", default="val")
    parser.add_argument("--phop_best_metric", default="acc")
    parser.add_argument("--phop_best_mode", choices=["max", "min"], default="max")
    parser.add_argument("--phop_big_eval_splits", nargs="+", default=None)
    parser.add_argument("--phop_big_eval_max_batches", type=int, default=None)

    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format,
        base_parser=parser,
        args=rem_args,
        namespace=args,
    )


def apply_phop_task_shape(args, spec, distributed_backend):
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


def evaluate_splits(model, eval_loaders, device, max_batches, ctx):
    return {
        split: evaluate_phop_model(
            model,
            dataloader,
            device=device,
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
        split: load_phop_split(data_root, spec.name, split)
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


def build_phop_eval_metric_plot(stats, metric):
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
            if value is None or not np.isfinite(value):
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
        title=f"p-hop eval {metric}",
    )


def build_phop_train_loss_plot(stats):
    train_rows = stats.get("train_loss", [])
    if not train_rows:
        return None
    return make_wandb_line_series(
        xs=[row["step"] for row in train_rows],
        ys=[[row["loss"] for row in train_rows]],
        keys=["train_loss"],
        title="p-hop train loss",
    )


def build_phop_best_eval_bar_plot(best_eval_stats, metric):
    if not best_eval_stats:
        return None
    table = wandb.Table(columns=["split", "value"])
    has_value = False
    for split, split_stats in best_eval_stats.items():
        value = to_loggable_float(split_stats.get(metric))
        if value is None or not np.isfinite(value):
            continue
        table.add_data(split, value)
        has_value = True
    if not has_value:
        return None
    try:
        return wandb.plot.bar(table, "split", "value", title=f"Best p-hop checkpoint {metric}")
    except Exception as exc:
        print(f"WARNING: could not build WandB p-hop best-eval plot '{metric}': {exc}")
        return None


def log_phop_wandb_summary_plots(stats, best_eval_stats=None, step=None):
    if not wandb.run:
        return

    logs = {}
    train_loss_plot = build_phop_train_loss_plot(stats)
    if train_loss_plot is not None:
        logs["plots/train_loss"] = train_loss_plot

    for metric in PHOP_PLOT_METRICS:
        plot = build_phop_eval_metric_plot(stats, metric)
        if plot is not None:
            logs[f"plots/eval/{metric}"] = plot

    for metric in PHOP_PLOT_METRICS:
        plot = build_phop_best_eval_bar_plot(best_eval_stats, metric)
        if plot is not None:
            logs[f"plots/best_eval/{metric}"] = plot

    if logs:
        wandb.log(logs, step=step)


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

    spec = get_phop_task_spec(args.phop_task)
    apply_phop_task_shape(args, spec, distributed_backend)

    data_root = Path(args.phop_data_root)
    pin_memory = device_type == "cuda"
    train_dataset = load_phop_split(data_root, spec.name, args.phop_train_split)
    eval_datasets = {
        split: load_phop_split(data_root, spec.name, split)
        for split in args.phop_eval_splits
    }

    print_master(distributed_backend, f"Loading p-hop task '{spec.name}' from {data_root}")
    print_master(
        distributed_backend,
        f"Train split '{args.phop_train_split}': {len(train_dataset)} examples",
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
        num_workers=args.phop_num_workers,
        pin_memory=pin_memory,
    )
    eval_batch_size = args.phop_eval_batch_size or args.batch_size
    eval_loaders = {
        split: make_loader(
            dataset,
            spec=spec,
            sequence_length=args.sequence_length,
            batch_size=eval_batch_size,
            shuffle=False,
            seed=seed,
            num_workers=args.phop_num_workers,
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
    print_master(distributed_backend, f"\nTraining p-hop model={args.model}\n{vars(args)}\n")
    accepts_log_metrics = "log_metrics" in inspect.signature(raw_model.forward).parameters

    batch_iter = infinite_batches(train_loader)
    diag_split = args.phop_eval_splits[0] if args.phop_eval_splits else None
    diag_iter = iter(eval_loaders[diag_split]) if diag_split is not None else None
    stats = {"eval": {}, "train_loss": []}
    save_every = args.phop_save_every if args.phop_save_every is not None else args.eval_freq
    log_every = args.phop_log_every if args.phop_log_every is not None else max(1, min(100, args.eval_freq))
    best_filename = best_checkpoint_name(args.phop_best_split, args.phop_best_metric)
    best_info = load_best_info(ckpt_path)
    if best_info is not None and (
        best_info.get("split") != args.phop_best_split
        or best_info.get("metric") != args.phop_best_metric
        or best_info.get("mode") != args.phop_best_mode
    ):
        best_info = None
    best_value = None if best_info is None else float(best_info["value"])

    def maybe_update_best(eval_stats, step):
        nonlocal best_info, best_value

        value = get_eval_metric(eval_stats, args.phop_best_split, args.phop_best_metric)
        if value is None:
            if step == 0:
                print_master(
                    distributed_backend,
                    "WARNING: cannot track best checkpoint because "
                    f"{args.phop_best_split}.{args.phop_best_metric} is not present in eval stats.",
                )
            return

        if not is_better_metric(value, best_value, args.phop_best_mode):
            return

        best_value = value
        best_info = {
            "step": int(step),
            "split": args.phop_best_split,
            "metric": args.phop_best_metric,
            "mode": args.phop_best_mode,
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
                    f"best/{args.phop_best_split}/{args.phop_best_metric}": float(value),
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
                args.phop_eval_max_batches,
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
            args.phop_eval_max_batches,
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
                for metric in PHOP_PLOT_METRICS:
                    if metric in split_stats:
                        logs[f"final/{split}/{metric}"] = float(split_stats[metric])
            wandb.log(logs, step=args.iterations)
        save_checkpoint(model, optimizer, scheduler, args.iterations, ckpt_path, distributed_backend)
        if best_info is not None:
            stats["best"] = best_info
        final_best_eval_stats = None
        if args.phop_big_eval_splits is not None:
            if best_info is None:
                print(
                    "WARNING: skipping big eval because no best checkpoint was selected. "
                    f"Make sure {args.phop_best_split} is included in --phop_eval_splits."
                )
            else:
                best_checkpoint_path = os.path.join(ckpt_path, best_info["checkpoint"])
                checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
                raw_model.load_state_dict(checkpoint["model"], strict=True)
                big_eval_loaders = make_eval_loaders_for_splits(
                    args.phop_big_eval_splits,
                    data_root=data_root,
                    spec=spec,
                    sequence_length=args.sequence_length,
                    batch_size=eval_batch_size,
                    seed=seed,
                    num_workers=args.phop_num_workers,
                    pin_memory=pin_memory,
                )
                big_eval_stats = evaluate_splits(
                    raw_model,
                    big_eval_loaders,
                    args.device,
                    args.phop_big_eval_max_batches,
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
            log_phop_wandb_summary_plots(
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
