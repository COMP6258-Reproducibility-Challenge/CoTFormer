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


def save_checkpoint(model, optimizer, scheduler, itr, ckpt_path, distributed_backend):
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
    torch.save(checkpoint, os.path.join(ckpt_path, f"ckpt_{itr}.pt"))


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

    batch_iter = infinite_batches(train_loader)
    stats = {"eval": {}, "train_loss": []}
    save_every = args.ib_save_every if args.ib_save_every is not None else args.eval_freq
    log_every = args.ib_log_every if args.ib_log_every is not None else max(1, min(100, args.eval_freq))

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

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.acc_steps):
            batch = next(batch_iter)
            inputs = batch["input_id"].to(args.device, non_blocking=pin_memory)
            labels = batch["label"].to(args.device, non_blocking=pin_memory)
            with type_ctx:
                outputs = model(inputs, targets=labels, get_logits=True)
            loss, _ = unpack_model_outputs(outputs)
            (loss / args.acc_steps).backward()
            total_loss += float(loss.detach().item())

        grad_clip = getattr(args, "grad_clip", None)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        mean_loss = total_loss / args.acc_steps
        stats["train_loss"].append({"step": step + 1, "loss": mean_loss})
        if distributed_backend.is_master_process() and (step + 1) % log_every == 0:
            print(json.dumps({"step": step + 1, "train_loss": mean_loss}))
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
        save_checkpoint(model, optimizer, scheduler, args.iterations, ckpt_path, distributed_backend)
        stats["args"] = sanitize_args_for_json(args)
        with open(f"{ckpt_path}/summary.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

    distributed_backend.finalize()


if __name__ == "__main__":
    main(get_args())
