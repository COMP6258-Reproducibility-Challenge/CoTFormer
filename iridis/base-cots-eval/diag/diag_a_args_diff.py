#!/usr/bin/env python3
"""Diag A — args diff between failing 12L_5R and working 24L_5R.

Compares summary['args'] field-by-field. Both ablations were trained with the
same notebook setup; one reached paper PPL (24L_5R), the other did not (12L_5R).
Any non-expected field difference is a candidate root cause.

Run from anywhere on iridis after activating the conda env. No GPU required.

Usage:
    python diag_a_args_diff.py
    python diag_a_args_diff.py --ckpt-root /scratch/ab3u21/exps/owt2/cotformer_full_depth
    python diag_a_args_diff.py --failing BaseCot_12L_5R --working BaseCot_24L_5R
"""

import argparse
import json
import os
import sys


# Architectural fields that SHOULD differ between two ablations with different
# (n_layer, n_repeat). Listing them as expected means we suppress them in the
# diff output to surface only suspicious differences.
_EXPECTED_DIFFERENT = {
    "n_layer",
    "n_repeat",
    "min_repeat",
    "exp_name",
}


def load_args(ckpt_root: str, ablation: str) -> dict:
    summary_path = os.path.join(ckpt_root, ablation, "summary.json")
    if not os.path.isfile(summary_path):
        sys.exit(f"ERROR: missing {summary_path}")
    with open(summary_path) as f:
        data = json.load(f)
    if "args" not in data:
        sys.exit(f"ERROR: {summary_path} has no 'args' key")
    return data["args"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff training args between a failing and a working ablation.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--ckpt-root",
        default="/scratch/ab3u21/exps/owt2/cotformer_full_depth",
        help="Root directory containing one subdirectory per ablation.",
    )
    parser.add_argument(
        "--failing",
        default="BaseCot_12L_5R",
        help="Ablation that did NOT reach paper PPL.",
    )
    parser.add_argument(
        "--working",
        default="BaseCot_24L_5R",
        help="Ablation that DID reach paper PPL (sibling control).",
    )
    args = parser.parse_args()

    a = load_args(args.ckpt_root, args.failing)
    b = load_args(args.ckpt_root, args.working)

    keys = sorted(set(a) | set(b))
    diffs = [k for k in keys if a.get(k) != b.get(k) and k not in _EXPECTED_DIFFERENT]

    print(f"Comparing  failing={args.failing}  vs  working={args.working}")
    print(f"Total fields: {len(keys)}    Expected differences (suppressed): "
          f"{sorted(_EXPECTED_DIFFERENT)}")
    print()

    if not diffs:
        print("RESULT: args are IDENTICAL after suppressing expected differences.")
        print("        Cause is NOT in summary['args']. Proceed to Diag B and C.")
        return

    print(f"RESULT: {len(diffs)} suspicious difference(s) found:\n")
    print(f"{'FIELD':<35s} {args.failing:<32s} {args.working:<32s}")
    print("-" * 100)
    for k in diffs:
        va = a.get(k, "<MISSING>")
        vb = b.get(k, "<MISSING>")
        print(f"{k:<35s} {repr(va):<32s} {repr(vb):<32s}")
    print()
    print("Each row above is a candidate root cause. Investigate from most to")
    print("least likely impact on training dynamics (lr, weight_decay, optimizer,")
    print("scheduler, batch size, then misc).")


if __name__ == "__main__":
    main()
