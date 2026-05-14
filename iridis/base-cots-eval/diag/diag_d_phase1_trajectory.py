#!/usr/bin/env python3
"""Diag D -- Phase-1 trajectory reconstruction via intermediate ckpt eval.

The training-time val_pp[] array in summary.json is reset to empty whenever
training resumes (optim/base.py:46 reinitialises stats). For ablations trained
in two phases (e.g. BaseCot_12L_5R: Phase 1 iter 0->26000, Phase 2 iter
26000->40000), this means the Phase 1 trajectory is LOST from summary.json --
the only data point we have post-Phase-1 is the model state captured in
ckpt_26000.pt.

This script reconstructs the missing trajectory by loading each ckpt_<N>.pt
file in the ablation directory and running a quick val-PPL probe identical
in protocol to Diag B (load via eval.py's filter, sample N batches).
Outputs a table of (iter, val_pp) per ablation, which can be plotted
alongside the surviving summary.json trajectory to see the full picture.

Run from the repo root (~/CoTFormer/) so model/config imports resolve. GPU
required. The default loops over 3 ablations x 6 milestones = 18 evals,
~30 sec each = ~10 min wall on L4.

Usage:
    cd ~/CoTFormer
    sbatch iridis/base-cots-eval/diag/diag_d_job.sh
    # OR interactively on a compute node:
    python iridis/base-cots-eval/diag/diag_d_phase1_trajectory.py
"""

import argparse
import json
import math
import os
import re
import sys


def parse_use_pretrained_offset(s) -> int:
    if not isinstance(s, str) or not s or s == "auto":
        return 0
    m = re.search(r"ckpt_(\d+)", s)
    return int(m.group(1)) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct Phase-1 val_pp trajectory by evaluating "
                    "intermediate checkpoints.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--ckpt-root",
        default="/scratch/ab3u21/exps/owt2/cotformer_full_depth",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["BaseCot_12L_2R", "BaseCot_12L_3R", "BaseCot_12L_5R"],
        help="Ablations to reconstruct (default: 3 sibling 12L variants).",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 26000, 30000, 35000, 40000],
        help="Iter milestones to eval (script picks the nearest existing "
             "ckpt_<N>.pt for each).",
    )
    parser.add_argument(
        "--data-dir",
        default="/scratch/ab3u21/datasets",
        help="Override summary['args']['data_dir'] at eval time.",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=50,
        help="Val batches per probe. Higher = lower noise. 50 ~ 410k tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per probe (fits any L4/A100).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. Default: stdout table only.",
    )
    args = parser.parse_args()

    # Repo root on sys.path BEFORE project imports (login-node-safe).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import numpy as np
    import torch
    import models

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA required. Run via diag_d_job.sh on a GPU node.")

    # Memmap val tokens once -- shared across all probes (read-only).
    val_bin = os.path.join(args.data_dir, "openwebtext2", "val.bin")
    if not os.path.isfile(val_bin):
        sys.exit(f"ERROR: missing {val_bin}")
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")

    # Deterministic sample positions (same across all probes for fair compare).
    seq_len_default = 256
    rng = np.random.default_rng(0)
    max_start = len(val) - (seq_len_default + 1) * args.batch_size * args.n_batches
    sample_starts = sorted(rng.integers(0, max_start,
                                         size=args.n_batches * args.batch_size).tolist())

    def probe_ckpt(ckpt_dir: str, ckpt_filename: str) -> float:
        """Run a quick PPL probe on a specific ckpt file. Returns PPL."""
        summary_path = os.path.join(ckpt_dir, "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        ns = argparse.Namespace(**summary["args"])
        ns.data_dir = args.data_dir
        ns.device = torch.device("cuda")
        ns.dtype = torch.bfloat16
        ns.distributed_backend = None

        ckpt = torch.load(
            os.path.join(ckpt_dir, ckpt_filename),
            map_location="cpu",
            weights_only=False,
        )
        weights = {k.replace("_orig_mod.", ""): v
                   for k, v in ckpt["model"].items()
                   if "attn.bias" not in k and "wpe" not in k}

        model = models.make_model_from_args(ns).to("cuda").eval()
        model.load_state_dict(weights, strict=False)

        seq_len = ns.sequence_length
        losses = []
        with torch.no_grad():
            for batch_idx in range(args.n_batches):
                xs, ys = [], []
                for j in range(args.batch_size):
                    s = sample_starts[batch_idx * args.batch_size + j]
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

    def find_nearest_ckpt(ckpt_dir: str, target_iter: int) -> tuple[str, int]:
        """Find the ckpt_<N>.pt file with N closest to target_iter."""
        pattern = re.compile(r"^ckpt_(\d+)\.pt$")
        candidates = []
        for fn in os.listdir(ckpt_dir):
            m = pattern.match(fn)
            if m:
                candidates.append((int(m.group(1)), fn))
        if not candidates:
            return ("", -1)
        candidates.sort(key=lambda x: abs(x[0] - target_iter))
        return (candidates[0][1], candidates[0][0])

    # ---- Run the probe grid ----
    results: list[dict] = []
    print(f"Probing {len(args.ablations)} ablations x {len(args.milestones)} milestones")
    print(f"Sample: {args.n_batches} batches x {args.batch_size} seq -> "
          f"{args.n_batches * args.batch_size * seq_len_default:,} tokens per probe")
    print()

    for abl in args.ablations:
        ckpt_dir = os.path.join(args.ckpt_root, abl)
        if not os.path.isdir(ckpt_dir):
            print(f"SKIP {abl}: directory not found")
            continue
        print(f"--- {abl} ---")
        for target in args.milestones:
            ckpt_filename, actual_iter = find_nearest_ckpt(ckpt_dir, target)
            if actual_iter < 0 or abs(actual_iter - target) > 2000:
                print(f"  iter ~{target}: no ckpt within +/-2000; skipping")
                continue
            try:
                ppl = probe_ckpt(ckpt_dir, ckpt_filename)
                print(f"  {ckpt_filename:<20s}  (iter {actual_iter:>5d}) -> PPL = {ppl:.3f}")
                results.append({
                    "ablation": abl,
                    "target_iter": target,
                    "actual_iter": actual_iter,
                    "ckpt": ckpt_filename,
                    "ppl": ppl,
                })
            except Exception as e:  # noqa: BLE001
                print(f"  {ckpt_filename}: FAILED ({type(e).__name__}: {e})")
        print()

    # ---- Final table ----
    print()
    print("=" * 80)
    print("PHASE-1 TRAJECTORY RECONSTRUCTION  (val PPL per iter milestone)")
    print("=" * 80)
    all_milestones = sorted({r["target_iter"] for r in results})
    header = f"{'ABLATION':<22s} " + " ".join(f"{m:>8d}" for m in all_milestones)
    print(header)
    print("-" * len(header))
    for abl in args.ablations:
        row = f"{abl:<22s} "
        for m in all_milestones:
            match = next((r for r in results if r["ablation"] == abl and r["target_iter"] == m), None)
            row += f"{match['ppl']:>8.2f} " if match else f"{'--':>8s} "
        print(row)
    print()
    print("Interpretation guide:")
    print("  - If 12L_5R sits 3+ PPL above 12L_2R/3R at ALL milestones:")
    print("      => systematic Phase 1 degradation (config/env)")
    print("  - If 12L_5R matches siblings until iter X, then jumps:")
    print("      => pinpoint event at iter X (NaN spike / hw glitch)")
    print("  - If 12L_5R matches siblings throughout (incl iter 25000-26000):")
    print("      => save-side or resume-side issue at the iter 26000 boundary")

    if args.output_csv:
        with open(args.output_csv, "w") as f:
            f.write("ablation,target_iter,actual_iter,ckpt,ppl\n")
            for r in results:
                f.write(f"{r['ablation']},{r['target_iter']},{r['actual_iter']},"
                        f"{r['ckpt']},{r['ppl']:.4f}\n")
        print(f"\nCSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
