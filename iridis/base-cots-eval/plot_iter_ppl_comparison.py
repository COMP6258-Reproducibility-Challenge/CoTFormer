#!/usr/bin/env python3
"""Plot val_pp vs iteration for BaseCot_12L_{2,3,5}R training trajectories.

Runs locally on your PC after rsync'ing the relevant summary.json files from
iridis. Reconstructs the per-eval iteration number from each summary's
eval_freq and use_pretrained offset, then overlays the three trajectories on a
single iter-axis plot with paper-target horizontal markers.

Usage (from local iridis/base-cots-eval/):
    # 1. Rsync the three summaries from iridis preserving directory structure:
    rsync -av --include="*/" --include="BaseCot_12L_*R/summary.json" --exclude="*" \\
        iridis:/scratch/ab3u21/exps/owt2/cotformer_full_depth/ \\
        ./summaries/

    # 2. Generate the plot:
    python plot_iter_ppl_comparison.py
    # Output: ./iter_ppl_comparison.png + .pdf  (in CWD)

    # Override input root / output basename:
    python plot_iter_ppl_comparison.py --summary-root ./summaries --output iter_ppl

    # Include 12L_15R as a 4th trajectory:
    python plot_iter_ppl_comparison.py --include-15r
"""

import argparse
import json
import os
import sys
from typing import Any


# Paper Table 1 reference values (Mohtashami et al. 2025), used as horizontal
# overlay markers. Kept inline here so this script has zero project imports.
_PAPER_TARGETS = {
    "BaseCot_12L_2R": 27.55,
    "BaseCot_12L_3R": 27.07,
    "BaseCot_12L_5R": 26.64,
    "BaseCot_12L_15R": None,  # not in paper Table 1; from Fig 2 visual ~26.85
}

# Per-ablation colour palette (consistent across plots).
_COLOURS = {
    "BaseCot_12L_2R": "#1f77b4",   # blue
    "BaseCot_12L_3R": "#2ca02c",   # green
    "BaseCot_12L_5R": "#d62728",   # red (the failing one)
    "BaseCot_12L_15R": "#9467bd",  # purple
}


def parse_use_pretrained_offset(use_pretrained: Any) -> int:
    """Extract the resumed-from iteration count from a use_pretrained value.

    Returns 0 for None / 'auto' / unparseable values.
    """
    if use_pretrained is None or use_pretrained == "auto":
        return 0
    if not isinstance(use_pretrained, str):
        return 0
    # Expect 'ckpt_NNNNN.pt'
    base = os.path.splitext(use_pretrained)[0]
    if "_" not in base:
        return 0
    try:
        return int(base.rsplit("_", 1)[1])
    except ValueError:
        return 0


def load_trajectory(summary_path: str) -> dict[str, Any]:
    """Load summary.json and reconstruct iters/val_pp arrays.

    Returns a dict with keys: iters, val_pp, train_loss, val_loss, eval_freq,
    use_pretrained, args (full args dict).
    """
    with open(summary_path) as f:
        s = json.load(f)
    args = s.get("args", {})
    eval_freq = args.get("eval_freq", 100)
    offset = parse_use_pretrained_offset(args.get("use_pretrained"))

    val_pp = s.get("val_pp", [])
    if not val_pp and s.get("val_loss"):
        # Older summaries may store only val_loss; reconstruct val_pp = e^val_loss
        import math
        val_pp = [math.e ** vl for vl in s["val_loss"]]

    # Reconstruct iter axis: each eval happens every eval_freq steps starting
    # from offset + eval_freq, so iter[i] = offset + (i+1) * eval_freq.
    iters = [offset + (i + 1) * eval_freq for i in range(len(val_pp))]

    return {
        "iters": iters,
        "val_pp": val_pp,
        "val_loss": s.get("val_loss", []),
        "train_loss": s.get("train_loss", []),
        "eval_freq": eval_freq,
        "use_pretrained": args.get("use_pretrained"),
        "offset": offset,
        "iterations_target": args.get("iterations", 40000),
        "args": args,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot iter-vs-PPL training trajectories for BaseCot ablations.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--summary-root",
        default=".",
        help=(
            "Root directory containing one subdirectory per ablation, each with "
            "a summary.json. Default: current working directory."
        ),
    )
    parser.add_argument(
        "--output",
        default="iter_ppl_comparison",
        help="Output basename (writes <basename>.png and <basename>.pdf in CWD).",
    )
    parser.add_argument(
        "--include-15r",
        action="store_true",
        help="Also plot BaseCot_12L_15R if a summary is available.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Override y-axis bounds (PPL). Default: auto-fit with paper targets visible.",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Override x-axis bounds (iter). Default: auto-fit.",
    )
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415 — late so --help doesn't import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ablations = ["BaseCot_12L_2R", "BaseCot_12L_3R", "BaseCot_12L_5R"]
    if args.include_15r:
        ablations.append("BaseCot_12L_15R")

    trajectories: dict[str, dict[str, Any]] = {}
    for abl in ablations:
        summary_path = os.path.join(args.summary_root, abl, "summary.json")
        if not os.path.isfile(summary_path):
            print(f"WARN: missing {summary_path} -- skipping {abl}", file=sys.stderr)
            continue
        try:
            trajectories[abl] = load_trajectory(summary_path)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"WARN: failed to load {summary_path}: {e}", file=sys.stderr)
            continue

    if not trajectories:
        sys.exit("ERROR: no trajectories loaded.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for abl, traj in trajectories.items():
        if not traj["val_pp"]:
            print(f"WARN: {abl} has empty val_pp -- skipping curve", file=sys.stderr)
            continue
        colour = _COLOURS.get(abl, "#666666")
        ax.plot(
            traj["iters"],
            traj["val_pp"],
            color=colour,
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            label=f"{abl} (n_repeat={traj['args'].get('n_repeat', '?')})",
            alpha=0.9,
        )
        # Mark the use_pretrained resume point if non-zero (visual flag for the
        # reader: this trajectory does NOT start at iter 0).
        if traj["offset"] > 0:
            ax.axvline(
                traj["offset"],
                color=colour,
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )
            ax.text(
                traj["offset"],
                ax.get_ylim()[1] if ax.get_ylim()[1] else max(traj["val_pp"]),
                f"  resume\n  ({abl})",
                fontsize=7,
                color=colour,
                va="top",
                alpha=0.7,
            )

    # Paper-target horizontal markers.
    for abl, target in _PAPER_TARGETS.items():
        if target is None or abl not in trajectories:
            continue
        colour = _COLOURS.get(abl, "#666666")
        ax.axhline(target, color=colour, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(
            ax.get_xlim()[1] if args.xlim else 40000,
            target,
            f" paper {target}",
            fontsize=7.5,
            color=colour,
            va="center",
            ha="right" if args.xlim else "left",
            alpha=0.85,
        )

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Validation Perplexity", fontsize=11)
    ax.set_title(
        "BaseCot 12L ablations -- training trajectory comparison\n"
        "Dashed horizontals = paper Table 1 targets; dotted verticals = resume points",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if args.xlim:
        ax.set_xlim(*args.xlim)
    if args.ylim:
        ax.set_ylim(*args.ylim)

    fig.tight_layout()

    out_png = os.path.abspath(f"{args.output}.png")
    out_pdf = os.path.abspath(f"{args.output}.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")

    # Summary table to stdout for sanity-check.
    print()
    print(f"{'ABLATION':<22s} {'final_PPL':>10s} {'paper':>8s} {'gap':>7s} "
          f"{'iters':>16s} {'resume_from':>15s}")
    print("-" * 90)
    for abl, traj in trajectories.items():
        if not traj["val_pp"]:
            continue
        final = traj["val_pp"][-1]
        paper = _PAPER_TARGETS.get(abl)
        gap = final - paper if paper else None
        iter_range = (
            f"{traj['iters'][0]}-{traj['iters'][-1]}" if traj["iters"] else "-"
        )
        paper_s = f"{paper:.2f}" if paper else "n/a"
        gap_s = f"{gap:+.2f}" if gap is not None else "n/a"
        resume_s = str(traj["use_pretrained"]) if traj["use_pretrained"] else "fresh"
        print(f"{abl:<22s} {final:>10.3f} {paper_s:>8s} {gap_s:>7s} "
              f"{iter_range:>16s} {resume_s:>15s}")


if __name__ == "__main__":
    main()
