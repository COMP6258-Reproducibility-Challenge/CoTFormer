#!/usr/bin/env python3
"""Plot Table 1: Perplexity comparison across model sizes and n_repeat values.

Produces:
  table1.md   — Markdown table mirroring paper Table 1 layout
  table1.tex  — LaTeX tabular (booktabs) suitable for paste into report
  table1.png  — Rendered PNG of the table (dpi=200)

Usage:
  python scripts/plot_table1.py --results <results_table1_fig2.json> --output-dir <run_N/figs/>
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ci95_half(ci95: list) -> float:
    """Return half the width of a 95 % CI interval [lo, hi]."""
    return (ci95[1] - ci95[0]) / 2.0


def _fmt_ours(ppl: float, ci95_half: float) -> str:
    """Format our row: ppl(±ci95-half), two decimal places."""
    return f"{ppl:.2f}(±{ci95_half:.2f})"


def _fmt_paper(ppl: float, sem: float) -> str:
    """Format paper row: ppl(sem), matching paper convention."""
    return f"{ppl:.2f}({sem:.2f})"


def _fmt_single(ppl: float, sem: float) -> str:
    """Format single-value rows (Standard) shown once across n_repeat cols."""
    return f"{ppl:.2f}({sem:.2f})"


def _bold_md(s: str) -> str:
    return f"**{s}**"


def _bold_tex(s: str) -> str:
    return f"\\textbf{{{s}}}"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_rows(data: dict) -> list:
    """Return a list of row dicts ready for table rendering.

    Each dict has keys:
      model       : display name (str)
      n_layer_col : value for the "Base Layers" column (str)
      r2, r3, r5  : cell strings for n_repeat 2, 3, 5
      is_ours     : bool — True = CoTFormer (ours), bold in output
    """
    ref = data["paper_reference_table1"]
    ablations = data["ablations"]

    # Build lookup: (family, n_layer, n_repeat) -> ablation entry
    abl_lookup: dict = {}
    for a in ablations:
        key = (a["family"], a["n_layer"], a["n_repeat"])
        abl_lookup[key] = a

    def ours_cell(family: str, n_layer: int, n_repeat: int) -> str:
        key = (family, n_layer, n_repeat)
        if key not in abl_lookup:
            return "—"
        a = abl_lookup[key]
        ppl = a["eval"]["val_perplexity"]
        ci95 = a["eval"]["val_perplexity_ci95"]
        return _fmt_ours(ppl, _ci95_half(ci95))

    def paper_cell(by_n_repeat: dict, r: int) -> str:
        s = str(r)
        if s not in by_n_repeat:
            return "—"
        entry = by_n_repeat[s]
        return _fmt_paper(entry["ppl"], entry["sem"])

    rows = []

    # Standard 12L — single ppl spans all n_repeat cols
    s12 = ref["standard_12L"]
    rows.append({
        "model": "Standard",
        "n_layer_col": "12",
        "r2": _fmt_single(s12["ppl"], s12["sem"]),
        "r3": "",
        "r5": "",
        "multispan": True,
        "is_ours": False,
    })

    # BUT 12L
    but12 = ref["block_universal_12L"]["by_n_repeat"]
    rows.append({
        "model": "Block Universal Transformer",
        "n_layer_col": "12",
        "r2": paper_cell(but12, 2),
        "r3": paper_cell(but12, 3),
        "r5": paper_cell(but12, 5),
        "multispan": False,
        "is_ours": False,
    })

    # CoTFormer 12L (ours)
    rows.append({
        "model": "CoTFormer (ours)",
        "n_layer_col": "12",
        "r2": ours_cell("CoTFormer", 12, 2),
        "r3": ours_cell("CoTFormer", 12, 3),
        "r5": ours_cell("CoTFormer", 12, 5),
        "multispan": False,
        "is_ours": True,
    })

    # CoTFormer 12L (paper)
    cot12 = ref["cotformer_12L_paper"]["by_n_repeat"]
    rows.append({
        "model": "CoTFormer (paper)",
        "n_layer_col": "12",
        "r2": paper_cell(cot12, 2),
        "r3": paper_cell(cot12, 3),
        "r5": paper_cell(cot12, 5),
        "multispan": False,
        "is_ours": False,
    })

    # Standard 24L
    s24 = ref["standard_24L"]
    rows.append({
        "model": "Standard",
        "n_layer_col": "24",
        "r2": _fmt_single(s24["ppl"], s24["sem"]),
        "r3": "",
        "r5": "",
        "multispan": True,
        "is_ours": False,
    })

    # BUT 24L
    but24 = ref["block_universal_24L"]["by_n_repeat"]
    rows.append({
        "model": "Block Universal Transformer",
        "n_layer_col": "24",
        "r2": paper_cell(but24, 2),
        "r3": paper_cell(but24, 3),
        "r5": paper_cell(but24, 5),
        "multispan": False,
        "is_ours": False,
    })

    # CoTFormer 24L (ours)
    rows.append({
        "model": "CoTFormer (ours)",
        "n_layer_col": "24",
        "r2": ours_cell("CoTFormer", 24, 2),
        "r3": ours_cell("CoTFormer", 24, 3),
        "r5": ours_cell("CoTFormer", 24, 5),
        "multispan": False,
        "is_ours": True,
    })

    # CoTFormer 24L (paper)
    cot24 = ref["cotformer_24L_paper"]["by_n_repeat"]
    rows.append({
        "model": "CoTFormer (paper)",
        "n_layer_col": "24",
        "r2": paper_cell(cot24, 2),
        "r3": paper_cell(cot24, 3),
        "r5": paper_cell(cot24, 5),
        "multispan": False,
        "is_ours": False,
    })

    # Standard 48L
    s48 = ref["standard_48L"]
    rows.append({
        "model": "Standard",
        "n_layer_col": "48",
        "r2": _fmt_single(s48["ppl"], s48["sem"]),
        "r3": "",
        "r5": "",
        "multispan": True,
        "is_ours": False,
    })

    return rows


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def build_markdown(rows: list) -> str:
    lines = []
    header = (
        "| Model | Base Layers (n_layer) "
        "| n_repeat=2 | n_repeat=3 | n_repeat=5 |"
    )
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for r in rows:
        model = r["model"]
        if r["is_ours"]:
            model = _bold_md(model)
        if r["multispan"]:
            cell2 = r["r2"]
            cell3 = "—"
            cell5 = "—"
        else:
            cell2, cell3, cell5 = r["r2"], r["r3"], r["r5"]
            if r["is_ours"]:
                cell2 = _bold_md(cell2) if cell2 != "—" else cell2
                cell3 = _bold_md(cell3) if cell3 != "—" else cell3
                cell5 = _bold_md(cell5) if cell5 != "—" else cell5
        lines.append(
            f"| {model} | {r['n_layer_col']} "
            f"| {cell2} | {cell3} | {cell5} |"
        )

    lines.append("")
    lines.append(
        "*Our rows (CoTFormer ours): uncertainty shown as ±CI95-half "
        "(per-batch 95% confidence interval from a single seed). "
        "Paper rows: SEM over 3 seeds, as reported in the original paper.*"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def build_latex(rows: list) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{\textbf{Performance of CoTFormer, Block Universal Transformer "
        r"and Standard Transformers on OpenWebText2.} "
        r"Paper rows report SEM over 3 seeds. "
        r"Our rows (CoTFormer ours) report per-batch CI95-half from a single seed "
        r"(see reprod-notes \S A8 for statistical interpretation).}"
    )
    lines.append(r"  \label{tab:table1}")
    lines.append(r"  \begin{tabular}{lc ccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Model & Base Layers ($n_{\mathrm{layer}}$) & "
        r"$n_{\mathrm{repeat}}=2$ & $n_{\mathrm{repeat}}=3$ & $n_{\mathrm{repeat}}=5$ \\"
    )
    lines.append(r"    \midrule")

    prev_n_layer = None
    for r in rows:
        if prev_n_layer is not None and r["n_layer_col"] != prev_n_layer:
            lines.append(r"    \midrule")
        prev_n_layer = r["n_layer_col"]

        model = r["model"]
        if r["is_ours"]:
            model = _bold_tex(model)

        if r["multispan"]:
            cell2 = r["r2"]
            cell3 = "—"
            cell5 = "—"
        else:
            cell2, cell3, cell5 = r["r2"], r["r3"], r["r5"]
            if r["is_ours"]:
                cell2 = _bold_tex(cell2) if cell2 != "—" else cell2
                cell3 = _bold_tex(cell3) if cell3 != "—" else cell3
                cell5 = _bold_tex(cell5) if cell5 != "—" else cell5

        lines.append(
            f"    {model} & {r['n_layer_col']} & {cell2} & {cell3} & {cell5} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG output via matplotlib.table
# ---------------------------------------------------------------------------

def build_png(rows: list, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    col_labels = [
        "Model",
        r"$n_\mathrm{layer}$",
        r"$n_\mathrm{repeat}=2$",
        r"$n_\mathrm{repeat}=3$",
        r"$n_\mathrm{repeat}=5$",
    ]

    table_data = []
    row_colors = []
    ours_color = "#e8f5e9"   # light green tint for our rows
    plain_color = "#ffffff"

    for r in rows:
        if r["multispan"]:
            cell2, cell3, cell5 = r["r2"], "—", "—"
        else:
            cell2, cell3, cell5 = r["r2"], r["r3"], r["r5"]
        table_data.append([r["model"], r["n_layer_col"], cell2, cell3, cell5])
        row_colors.append(
            [ours_color] * 5 if r["is_ours"] else [plain_color] * 5
        )

    n_rows = len(table_data)
    fig_h = max(2.5, 0.32 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.4)

    # Bold header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#d0d0d0")
        tbl[(0, j)].set_text_props(fontweight="bold")

    # Bold "ours" cells
    for i, r in enumerate(rows):
        if r["is_ours"]:
            for j in range(len(col_labels)):
                tbl[(i + 1, j)].set_text_props(fontweight="bold")

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Table 1 as .md, .tex, and .png"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results_table1_fig2.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where table1.md / table1.tex / table1.png are written",
    )
    args = parser.parse_args()

    with open(args.results, "r") as fh:
        data = json.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    rows = extract_rows(data)

    md_path = os.path.join(args.output_dir, "table1.md")
    with open(md_path, "w") as fh:
        fh.write(build_markdown(rows))
    print(f"Written: {md_path}")

    tex_path = os.path.join(args.output_dir, "table1.tex")
    with open(tex_path, "w") as fh:
        fh.write(build_latex(rows))
    print(f"Written: {tex_path}")

    png_path = os.path.join(args.output_dir, "table1.png")
    build_png(rows, png_path)
    print(f"Written: {png_path}")


if __name__ == "__main__":
    main()
