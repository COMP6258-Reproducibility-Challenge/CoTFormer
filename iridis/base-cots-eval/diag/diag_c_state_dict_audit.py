#!/usr/bin/env python3
"""Diag C — state_dict key + shape audit, failing vs working ablation.

Loads both checkpoints, normalises layer indices (h_mid.N.* -> h_mid.LAYER.*),
and reports:
  1. Keys present in one but not the other (after layer-index normalisation).
  2. Shape mismatches on shared keys.
  3. Total parameter count per checkpoint.

A working architecture comparison should report zero unique keys in either set
and zero shape mismatches on shared keys (after layer normalisation). Any
output indicates a structural drift between the two trained models that would
not be detected by summary.json args inspection alone.

Run from anywhere (no GPU, no repo imports required).

Usage:
    python diag_c_state_dict_audit.py
    python diag_c_state_dict_audit.py --failing BaseCot_12L_5R --working BaseCot_24L_5R
"""

import argparse
import os
import re
import sys


_LAYER_INDEX_RE = re.compile(r"\.h_(begin|mid|end)\.\d+\.")


def normalise_key(k: str) -> str:
    """Collapse layer-index components so 12L and 24L checkpoints align."""
    return _LAYER_INDEX_RE.sub(r".h_\1.LAYER.", k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="State_dict structural audit between two ablations.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--ckpt-root",
        default="/scratch/ab3u21/exps/owt2/cotformer_full_depth",
    )
    parser.add_argument("--failing", default="BaseCot_12L_5R")
    parser.add_argument("--working", default="BaseCot_24L_5R")
    args = parser.parse_args()

    import torch  # late import for fast --help

    p_fail = os.path.join(args.ckpt_root, args.failing, "ckpt.pt")
    p_work = os.path.join(args.ckpt_root, args.working, "ckpt.pt")
    for p in (p_fail, p_work):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: missing {p}")

    print(f"Loading {args.failing}/ckpt.pt ...")
    c_fail = torch.load(p_fail, map_location="cpu", weights_only=False)["model"]
    print(f"Loading {args.working}/ckpt.pt ...")
    c_work = torch.load(p_work, map_location="cpu", weights_only=False)["model"]

    # ----- Param counts -----
    n_params_fail = sum(v.numel() for v in c_fail.values())
    n_params_work = sum(v.numel() for v in c_work.values())
    print()
    print(f"{args.failing}: {len(c_fail)} keys, {n_params_fail:,} params")
    print(f"{args.working}: {len(c_work)} keys, {n_params_work:,} params")
    print()

    # ----- Key set diff (after layer normalisation) -----
    norm_fail = {normalise_key(k) for k in c_fail}
    norm_work = {normalise_key(k) for k in c_work}

    only_in_fail = norm_fail - norm_work
    only_in_work = norm_work - norm_fail

    if only_in_fail:
        print(f"Keys present in {args.failing} but NOT in {args.working} "
              f"({len(only_in_fail)}):")
        for k in sorted(only_in_fail):
            print(f"  + {k}")
        print()
    if only_in_work:
        print(f"Keys present in {args.working} but NOT in {args.failing} "
              f"({len(only_in_work)}):")
        for k in sorted(only_in_work):
            print(f"  + {k}")
        print()
    if not only_in_fail and not only_in_work:
        print("Key sets are IDENTICAL after layer-index normalisation.")
        print()

    # ----- Shape audit on shared (normalised) keys -----
    fail_by_norm: dict[str, tuple[str, tuple]] = {}
    for k, v in c_fail.items():
        fail_by_norm.setdefault(normalise_key(k), (k, tuple(v.shape)))
    work_by_norm: dict[str, tuple[str, tuple]] = {}
    for k, v in c_work.items():
        work_by_norm.setdefault(normalise_key(k), (k, tuple(v.shape)))

    shared = norm_fail & norm_work
    mismatches: list[tuple[str, tuple, tuple]] = []
    for nk in sorted(shared):
        _, sf = fail_by_norm[nk]
        _, sw = work_by_norm[nk]
        if sf != sw:
            mismatches.append((nk, sf, sw))

    if mismatches:
        print(f"Shape mismatches on {len(mismatches)} shared key(s):")
        print(f"  {'NORMALISED KEY':<55s} {args.failing:<25s} {args.working:<25s}")
        for nk, sf, sw in mismatches:
            print(f"  {nk:<55s} {str(sf):<25s} {str(sw):<25s}")
        print()
    else:
        print(f"All {len(shared)} shared keys have matching shapes.")
        print()

    # ----- Final verdict -----
    print("=" * 60)
    if not only_in_fail and not only_in_work and not mismatches:
        print("VERDICT: structurally identical (modulo layer count). No drift detected.")
    else:
        print("VERDICT: structural drift detected. Inspect output above for root cause.")


if __name__ == "__main__":
    main()
