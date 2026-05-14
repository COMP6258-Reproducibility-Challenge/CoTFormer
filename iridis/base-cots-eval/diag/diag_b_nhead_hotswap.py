#!/usr/bin/env python3
"""Diag B — n_head hot-swap PPL probe on a trained checkpoint.

Background: PyTorch attention Q/K/V projections store weights at shape
(d_model, d_model) regardless of n_head — the head split is a runtime reshape,
not a stored property. So state_dict.load() succeeds under ANY n_head, but
inference PPL is catastrophically wrong if you eval at a different n_head than
the one the weights were ACTUALLY trained for.

This script loads a checkpoint TWICE — once with the n_head claimed by
summary.json, once with the alternate value — and measures val PPL on a small
batch sample for each. A large gap pinpoints the head-grouping the weights
were ACTUALLY trained for; a small gap means summary.json tells the truth.

Run from the repo root (~/CoTFormer/) so model/config imports resolve:
    cd ~/CoTFormer
    python iridis/base-cots-eval/diag/diag_b_nhead_hotswap.py

Requires CUDA (single GPU) + access to /scratch/ab3u21/datasets/openwebtext2/val.bin.
"""

import argparse
import argparse as _ap
import json
import os
import sys


def main() -> None:
    parser = _ap.ArgumentParser(
        description="Hot-swap n_head on a trained checkpoint and compare PPL.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--ckpt-dir",
        default="/scratch/ab3u21/exps/owt2/cotformer_full_depth/BaseCot_12L_5R",
        help="Checkpoint directory containing ckpt.pt + summary.json.",
    )
    parser.add_argument(
        "--data-dir",
        default="/scratch/ab3u21/datasets",
        help="Dataset root (overrides summary['args']['data_dir']).",
    )
    parser.add_argument(
        "--alt-n-head",
        type=int,
        default=24,
        help="Alternate n_head value to probe (claimed n_head comes from summary).",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=20,
        help="Number of val batches to sample for the PPL estimate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per probe (kept small to fit any L4/A100).",
    )
    args = parser.parse_args()

    # Late imports so --help works without GPU.
    import math
    import numpy as np
    import torch
    import models

    # Ensure repo root is on sys.path when invoked from anywhere.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA required for this probe.")

    summary_path = os.path.join(args.ckpt_dir, "summary.json")
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    val_bin = os.path.join(args.data_dir, "openwebtext2", "val.bin")

    for p in (summary_path, ckpt_path, val_bin):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: missing {p}")

    with open(summary_path) as f:
        summary = json.load(f)
    claimed_n_head = summary["args"]["n_head"]

    print(f"Probing {args.ckpt_dir}")
    print(f"  summary claims n_head = {claimed_n_head}")
    print(f"  alternate probe       = {args.alt_n_head}")
    print()

    # Load ckpt once; reuse weights for both probes.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {
        k: v for k, v in ckpt["model"].items()
        if "attn.bias" not in k and "wpe" not in k
    }
    # Strip torch.compile orig_mod prefix if present.
    weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}

    # Memmap val tokens — read-only.
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")
    seq_len = summary["args"]["sequence_length"]

    rng = np.random.default_rng(0)
    max_start = len(val) - (seq_len + 1) * args.batch_size * args.n_batches
    starts = rng.integers(0, max_start, size=args.n_batches * args.batch_size).tolist()
    starts.sort()

    def make_args(n_head: int) -> argparse.Namespace:
        ns = argparse.Namespace(**summary["args"])
        ns.n_head = n_head
        ns.data_dir = args.data_dir
        ns.device = torch.device("cuda")
        ns.dtype = torch.bfloat16
        ns.distributed_backend = None
        return ns

    def ppl_probe(n_head: int) -> float:
        ns = make_args(n_head)
        model = models.make_model_from_args(ns).to("cuda").eval()
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            print(f"  WARN n_head={n_head}: unexpected keys in state_dict (first 5): "
                  f"{list(unexpected)[:5]}")
        if missing:
            print(f"  WARN n_head={n_head}: missing keys in model (first 5): "
                  f"{list(missing)[:5]}")

        losses = []
        with torch.no_grad():
            for batch_idx in range(args.n_batches):
                xs, ys = [], []
                for j in range(args.batch_size):
                    s = starts[batch_idx * args.batch_size + j]
                    chunk = torch.from_numpy(val[s:s + seq_len + 1].astype(np.int64))
                    xs.append(chunk[:-1])
                    ys.append(chunk[1:])
                x = torch.stack(xs).to("cuda", non_blocking=True)
                y = torch.stack(ys).to("cuda", non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, targets=y, get_logits=False)
                key = "cross_entropy_loss" if "cross_entropy_loss" in out else "loss"
                losses.append(out[key].item())

        del model
        torch.cuda.empty_cache()
        return math.e ** (sum(losses) / len(losses))

    print(f"Probing claimed n_head={claimed_n_head}...")
    ppl_claimed = ppl_probe(claimed_n_head)
    print(f"  PPL = {ppl_claimed:.3f}\n")

    print(f"Probing alternate n_head={args.alt_n_head}...")
    ppl_alt = ppl_probe(args.alt_n_head)
    print(f"  PPL = {ppl_alt:.3f}\n")

    ratio = ppl_alt / ppl_claimed if ppl_claimed > 0 else float("inf")
    print("=" * 60)
    print(f"RESULT  n_head={claimed_n_head:<3d} -> PPL={ppl_claimed:.3f}")
    print(f"        n_head={args.alt_n_head:<3d} -> PPL={ppl_alt:.3f}")
    print(f"        ratio (alt / claimed) = {ratio:.3f}x")
    print()

    if ratio > 1.5:
        print(f"Interpretation: weights were genuinely trained at n_head={claimed_n_head}.")
        print(f"                The alternate config is catastrophically worse, as expected")
        print(f"                if summary.json tells the truth. Cause is NOT n_head.")
    elif ratio < 0.7:
        print(f"Interpretation: weights respond BETTER at n_head={args.alt_n_head} than the")
        print(f"                claimed n_head={claimed_n_head}. Strongly suggests summary.json")
        print(f"                lies about n_head and the weights were actually trained at "
              f"{args.alt_n_head}.")
    else:
        print(f"Interpretation: the two probes are within {abs(1-ratio)*100:.0f}% of each other.")
        print(f"                Inconclusive on this sample size. Increase --n-batches or")
        print(f"                inspect state_dict shapes manually (see Diag C).")


if __name__ == "__main__":
    main()
