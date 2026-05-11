# reproduce-table1-fig23 — Table 1 + Figure 2 + Figure 3

Reproduces the CoTFormer rows in Table 1 and both Figure 2 and Figure 3 from
the paper (Mohtashami et al., ICLR 2025).

---

## Purpose

| Artefact   | What it shows |
|------------|---------------|
| Table 1    | Perplexity on OWT2 val at matching MACs budget (CoTFormer vs Standard, BUT) |
| Figure 2   | PPL vs MACs Pareto frontier — subplot (a) 12-layer, subplot (b) 24-layer |
| Figure 3   | Analytical MACs vs sequence-length scaling (CoTFormer 12×3 vs BUT 12×5) |

---

## Ablations + paper artefact mapping

| Checkpoint folder  | n_layer | n_repeat | Table 1 | Fig 2(a) | Fig 2(b) |
|--------------------|---------|----------|---------|----------|----------|
| BaseCot_12L_2R     | 12      | 2        | yes     | yes      |          |
| BaseCot_12L_3R     | 12      | 3        | yes     | yes      |          |
| BaseCot_12L_5R     | 12      | 5        | yes     | yes      |          |
| BaseCot_12L_15R    | 12      | 15       |         | yes      |          |
| BaseCot_24L_2R     | 24      | 2        | yes     |          | yes      |
| BaseCot_24L_3R     | 24      | 3        | yes     |          | yes      |
| BaseCot_24L_5R     | 24      | 5        | yes     |          | yes      |

Figure 3 uses only analytical MACs (no checkpoints needed for that panel).

---

## Limitations

- **1 seed per ablation.** The paper reports SEM over 3 seeds. Our uncertainty
  estimates are per-batch CI95 from `eval.py` — a different statistical
  quantity. See `docs/reprod-notes.md §A8`.
- **Standard and BUT rows are paper-reported reference values** (not retrained
  checkpoints in this package). The plot scripts inject these from the
  `paper_reference_table1` block in `results_table1_fig2.json`. See
  `docs/reprod-notes.md §A9–A10`.
- Fig 2 BUT 12×6 and 12×15 PPL y-coordinates are read off the paper figure
  (visual, ±0.05 PPL precision); MACs x-coordinates are recomputed analytically.

---

## Usage

```bash
cd ~/CoTFormer
bash iridis/reproduce-table1-fig23/job.sh
```

The script self-submits via `sbatch` when run on a login node. On first run it
creates `run_0/`; subsequent runs increment the counter.

---

## Output tree (`run_N/`)

```
run_N/
  slurm_%j.out              SLURM stdout
  slurm_%j.err              SLURM stderr
  json/
    macs.json               analytical FLOP counts (all families/configs)
    results_table1_fig2.json eval results joined with MACs
    eval_per_ablation/      per-ablation eval summaries (one subdir each)
      BaseCot_12L_2R/eval_summary_ckpt.json
      ...
  figs/
    table1.md               Table 1 in Markdown
    table1.tex              Table 1 in LaTeX (ready for paper)
    fig2a_12L_pareto.png    Figure 2 subplot (a) — PNG
    fig2a_12L_pareto.pdf    Figure 2 subplot (a) — PDF
    fig2b_24L_pareto.png    Figure 2 subplot (b)
    fig2b_24L_pareto.pdf
    fig3_macs_vs_seqlen.png Figure 3
    fig3_macs_vs_seqlen.pdf
  eval_logs/
    compute_macs.log
    reproduce_table1_fig2.log
    plot_table1.log
    plot_fig2.log
    plot_fig3.log
```

No `.npy`, `.pt`, or checkpoint files land in `run_N`.

---

## Expected wall time

15–30 minutes on L4 GPU (7 ablations × single forward-pass eval, then CPU-bound
plotting). 1.5 h walltime allocation is generous headroom for queue/startup
overhead.

---

## Pre-flight failure modes

| Error message | Cause | Fix |
|---------------|-------|-----|
| `missing summary.json for BaseCot_*` | Checkpoint folder absent or training incomplete | Re-run training package; confirm path under `CKPT_ROOT` |
| `missing ckpt.pt for BaseCot_*` | Training saved with a different filename | Check the folder for `ckpt_*.pt` and rename or symlink to `ckpt.pt` |
| `N file(s) missing` (multiple) | CKPT_ROOT path wrong for this user | Edit `CKPT_ROOT` at the top of `job.sh` |

See also `docs/reprod-notes.md §C4` for the `2.71828 ** loss` truncation quirk
in `eval.py:140` that affects displayed perplexity precision.
