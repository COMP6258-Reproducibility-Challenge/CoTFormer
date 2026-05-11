#!/usr/bin/env python3
"""Plot Figure 2: Perplexity vs MACs for n_layer=12 and n_layer=24.

Reproduces Figure 2 of the CoTFormer paper (Section 4.2). Two subplots:
  (a) n_layer=12, x-axis log scale
  (b) n_layer=24, x-axis linear scale

Each subplot shows two curves:
  - Block Universal Transformer (blue #1f77b4, circle markers, solid)
  - CoTFormer (green #2ca02c, circle markers, solid)

Per-point text labels "NLxNR" placed slightly above each marker.

Produces:
  fig2a.png, fig2a.pdf
  fig2b.png, fig2b.pdf

Usage:
  python scripts/plot_fig2.py \\
      --results <results_table1_fig2.json> \\
      --macs <macs.json> \\
      --output-dir <run_N/figs/>
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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_macs_lookup(macs_data: dict) -> dict:
    """Return dict: (family, n_layer, n_repeat, seq_len) -> macs (float, ×10⁹)."""
    lookup = {}
    for pt in macs_data["points"]:
        key = (pt["family"], pt["n_layer"], pt["n_repeat"], pt["seq_len"])
        lookup[key] = pt["macs"] / 1e9
    return lookup


def get_but_ppl(results: dict, n_layer: int, n_repeat: int):
    """Return PPL for BUT from paper_reference_table1 or paper_reference_fig2."""
    ref = results["paper_reference_table1"]
    key = f"block_universal_{n_layer}L"
    by_r = ref[key]["by_n_repeat"]
    r_str = str(n_repeat)
    if r_str in by_r:
        return by_r[r_str]["ppl"]
    # Extra points only in paper_reference_fig2 (12L, r=6 and r=15)
    extra = results.get("paper_reference_fig2", {}).get("but_12L_extra", {})
    if r_str in extra:
        return extra[r_str]["ppl_visual"]
    return None


def get_cot_ppl(results: dict, n_layer: int, n_repeat: int):
    """Return PPL for CoTFormer from ablations (ours) if available, else paper ref."""
    ablations = results["ablations"]
    for a in ablations:
        if a["family"] == "CoTFormer" and a["n_layer"] == n_layer and a["n_repeat"] == n_repeat:
            return a["eval"]["val_perplexity"]
    # Fall back to paper reference
    ref = results["paper_reference_table1"]
    key = f"cotformer_{n_layer}L_paper"
    by_r = ref[key]["by_n_repeat"]
    r_str = str(n_repeat)
    if r_str in by_r:
        return by_r[r_str]["ppl"]
    return None


# ---------------------------------------------------------------------------
# Subplot renderer
# ---------------------------------------------------------------------------

def _plot_subplot(
    ax,
    results: dict,
    macs_lookup: dict,
    n_layer: int,
    but_repeats: list,
    cot_repeats: list,
    xlog: bool,
) -> None:
    """Draw one subplot on ax."""
    import matplotlib.pyplot as plt  # noqa: F401 (matplotlib already imported by caller)

    # --- BUT curve ---
    but_x, but_y, but_labels = [], [], []
    for r in but_repeats:
        macs = macs_lookup.get(("BUT", n_layer, r, 256))
        ppl = get_but_ppl(results, n_layer, r)
        if macs is not None and ppl is not None:
            but_x.append(macs)
            but_y.append(ppl)
            but_labels.append(f"{n_layer}x{r}")

    # --- CoTFormer curve ---
    cot_x, cot_y, cot_labels = [], [], []
    for r in cot_repeats:
        macs = macs_lookup.get(("CoTFormer", n_layer, r, 256))
        ppl = get_cot_ppl(results, n_layer, r)
        if macs is not None and ppl is not None:
            cot_x.append(macs)
            cot_y.append(ppl)
            cot_labels.append(f"{n_layer}x{r}")

    # Plot lines
    ax.plot(
        but_x, but_y,
        color=COLOR_BUT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="Block Universal",
        zorder=3,
    )
    ax.plot(
        cot_x, cot_y,
        color=COLOR_COT,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="CoTFormer",
        zorder=3,
    )

    # Per-point text labels (slightly above marker)
    y_range = max(but_y + cot_y) - min(but_y + cot_y) if (but_y + cot_y) else 1.0
    offset = y_range * 0.025
    for x, y, lbl in zip(but_x, but_y, but_labels):
        ax.annotate(lbl, (x, y), xytext=(0, 4), textcoords="offset points",
                    fontsize=7, color=COLOR_BUT, ha="center", va="bottom")
    for x, y, lbl in zip(cot_x, cot_y, cot_labels):
        ax.annotate(lbl, (x, y), xytext=(0, 4), textcoords="offset points",
                    fontsize=7, color=COLOR_COT, ha="center", va="bottom")

    if xlog:
        ax.set_xscale("log")

    ax.set_xlabel(
        r"Multiply-Accumulate Operations ($\times 10^9$)", fontsize=9
    )
    ax.set_ylabel("Perplexity", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(False)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2: Perplexity vs MACs (a) 12L log, (b) 24L linear"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results_table1_fig2.json",
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
        help="Directory where fig2a.png/.pdf and fig2b.png/.pdf are written",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    with open(args.results, "r") as fh:
        results = json.load(fh)
    with open(args.macs, "r") as fh:
        macs_data = json.load(fh)

    macs_lookup = build_macs_lookup(macs_data)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Fig 2(a): n_layer=12, x log scale ---
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    _plot_subplot(
        ax=ax,
        results=results,
        macs_lookup=macs_lookup,
        n_layer=12,
        but_repeats=[2, 3, 5, 6, 15],
        cot_repeats=[2, 3, 5, 15],
        xlog=True,
    )
    fig.text(
        0.5, -0.04,
        r"(a) $n_{\mathrm{layer}} = 12$ (x-axis is in log scale)",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"fig2a.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Written: {out}")
    plt.close()

    # --- Fig 2(b): n_layer=24, x linear scale ---
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    _plot_subplot(
        ax=ax,
        results=results,
        macs_lookup=macs_lookup,
        n_layer=24,
        but_repeats=[2, 3, 5],
        cot_repeats=[2, 3, 5],
        xlog=False,
    )
    fig.text(
        0.5, -0.04,
        r"(b) $n_{\mathrm{layer}} = 24$",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"fig2b.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Written: {out}")
    plt.close()


if __name__ == "__main__":
    main()
