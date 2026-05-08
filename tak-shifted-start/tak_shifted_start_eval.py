import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import config
import distributed
import models
from IB_counting_data import (
    DEFAULT_SHIFTED_START_TASK,
    default_data_root,
    evaluate_counting_model,
    get_shifted_start_task_spec,
    load_counting_split,
    make_counting_dataloader,
)
from IB_shifted_start_main import apply_task_shape, coerce_seed


def none_or_str(value):
    if value == "None":
        return None
    return value


def latest_checkpoint_file(checkpoint_dir):
    checkpoints = [
        file for file in os.listdir(checkpoint_dir)
        if file.startswith("ckpt_") and file.endswith(".pt")
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda file: int(file.split("ckpt_")[1].split(".pt")[0]))


def get_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", type=none_or_str, default=None)
    parser.add_argument("--checkpoint_filename", default=None)
    parser.add_argument("--config_format", type=str, required=False)
    parser.add_argument("--ib_task", default=DEFAULT_SHIFTED_START_TASK)
    parser.add_argument("--ib_data_root", default=str(default_data_root()))
    parser.add_argument("--ib_eval_splits", nargs="+", default=["val", "ood_test"])
    parser.add_argument("--ib_num_workers", type=int, default=0)
    parser.add_argument("--ib_eval_batch_size", type=int, default=None)
    parser.add_argument("--ib_eval_max_batches", type=int, default=None)

    args, rem_args = parser.parse_known_args(argv)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            args.checkpoint, inferred_filename = os.path.split(args.checkpoint)
            if args.checkpoint_filename is None:
                args.checkpoint_filename = inferred_filename
        else:
            if args.checkpoint_filename is None:
                args.checkpoint_filename = latest_checkpoint_file(args.checkpoint)

        summary_path = os.path.join(args.checkpoint, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            for key, value in summary.get("args", {}).items():
                if key == "config_format" and args.config_format is not None:
                    continue
                if key not in ["device", "dtype"]:
                    setattr(args, key, value)

    return config.parse_args_with_format(
        format=args.config_format,
        base_parser=argparse.ArgumentParser(allow_abbrev=False),
        args=rem_args,
        namespace=args,
    )


def main(args):
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
    np.random.seed(seed)

    spec = get_shifted_start_task_spec(args.ib_task)
    apply_task_shape(args, spec, distributed_backend)

    model = models.make_model_from_args(args).to(args.device)
    if args.checkpoint is not None:
        if args.checkpoint_filename is None:
            raise FileNotFoundError(f"No ckpt_*.pt file found in {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, args.checkpoint_filename)
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint["model"], strict=True)

    data_root = Path(args.ib_data_root)
    eval_batch_size = args.ib_eval_batch_size or args.batch_size
    eval_loaders = {
        split: make_counting_dataloader(
            data_root=data_root,
            task=spec.name,
            split=split,
            spec=spec,
            sequence_length=args.sequence_length,
            batch_size=eval_batch_size,
            shuffle=False,
            seed=seed,
            num_workers=args.ib_num_workers,
            pin_memory=device_type == "cuda",
        )
        for split in args.ib_eval_splits
    }

    if distributed_backend.is_master_process():
        stats = {}
        for split, dataloader in eval_loaders.items():
            stats[split] = evaluate_counting_model(
                model,
                dataloader,
                device=args.device,
                max_seen_len=spec.max_seen_len,
                max_batches=args.ib_eval_max_batches,
                ctx=type_ctx,
            )
        print(json.dumps(stats, indent=2))

    distributed_backend.finalize()


if __name__ == "__main__":
    main(get_args())
