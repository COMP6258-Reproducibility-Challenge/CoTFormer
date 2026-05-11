#!/usr/bin/env python3
"""Plot Figure 3: MACs vs Sequence Length (log-log).

Reproduces Figure 3 of the CoTFormer paper (Section 4.2). Log-log plot
comparing compute cost as a function of sequence length:
  - Block Universal Transformer (12x5): blue #1f77b4
  - CoTFormer (12x3): green #2ca02c

Data is read entirely from macs.json — no PPL values used here.

Produces:
  fig3.png
  fig3.pdf

Usage:
  python scripts/plot_fig3.py --macs <macs.json> --output-dir <run_N/figs/>
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Style constants (from schema_table1_fig23.md)
# ---------------------------------------------------------------------------

COLOR_BUT = "#1f77b4"
COLOR_COT = "#2ca02c"
MARKER = "o"
LINEWIDTH = 1.6
MARKERSIZE = 6
FIG_SIZE = (4.5, 3.4)

# Canonical sequence-length axis from the schema
SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_curve(macs_data: dict, family: str, n_layer: int, n_repeat: int) -> tuple:
    """Return (seq_lens, macs_raw) lists for the given curve config."""
    lookup = {}
    for pt in macs_data["points"]:
        if (
            pt["family"] == family
            and pt["n_layer"] == n_layer
            and pt["n_repeat"] == n_repeat
        ):
            lookup[pt["seq_len"]] = pt["macs"]

    xs, ys = [], []
    for sl in SEQ_LENGTHS:
        if sl in lookup:
            xs.append(sl)
            ys.append(lookup[sl])
    return xs, ys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Figure 3: MACs vs Sequence Length (log-log)"
    )
    parser.add_argument(
        "--macs",
        type=str,
        required=True,
        help="Path to macs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where fig3.png and fig3.pdf are written",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    with open(args.macs, "r") as fh:
        macs_data = json.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    but_x, but_y = extract_curve(macs_data, family="BUT", n_layer=12, n_repeat=5)
    cot_x, cot_y = extract_curve(macs_data, family="CoTFormer", n_layer=12, n_repeat=3)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        but_x, but_y,
        color=COLOR_BUT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="Block Universal Transformer (12x5)",
        zorder=3,
    )
    ax.plot(
        cot_x, cot_y,
        color=COLOR_COT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="CoTFormer (12x3)",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Canonical x-ticks at the paper's sequence-length grid
    all_x = sorted(set(but_x) | set(cot_x))
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(v) for v in all_x], rotation=45, fontsize=7)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_xlabel("Sequence Length", fontsize=9)
    ax.set_ylabel("Multiply-Accumulate Operations", fontsize=9)

    # Legend top-left (matches screenshot)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(False)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"fig3.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Written: {out}")
    plt.close()


if __name__ == "__main__":
    main()
