#!/bin/bash
#SBATCH --job-name=base_cots_eval
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --time=02:30:00
################################################################################
# base-cots-eval -- Table 1 + Figure 2 + Figure 3 Reproduction
#
# Evaluates the seven BaseCot_* CoTFormer ablations on OWT2 val and renders
# Table 1, Figure 2 (Pareto), and Figure 3 (MACs vs seq_len) verbatim against
# the paper (Mohtashami et al., ICLR 2025).
#
# Runs end-to-end:
#   1. Compute analytical MACs (closed-form formula, CPU-only, no OOM risk).
#      See docs/reprod-notes.md §A10 for the closed-form derivation; the prior
#      ptflops-backed approach OOM'd at seq_len >= 8192 even on L4 24 GB.
#   2. Eval sweep across 7 CoTFormer ablations -> results_table1_fig2.json
#   3. Plot Table 1 (LaTeX/Markdown table), Figure 2 (PPL vs MACs Pareto),
#      Figure 3 (MACs vs sequence-length scaling)
#
# Ablation checkpoints (all at cotformer_full_depth training config):
#   BaseCot_12L_2R  BaseCot_12L_3R  BaseCot_12L_5R  BaseCot_12L_15R
#   BaseCot_24L_2R  BaseCot_24L_3R  BaseCot_24L_5R
#
# Output structure:
#   run_N/
#     json/
#       macs.json                        analytical FLOP counts (all families)
#       results_table1_fig2.json         eval results joined with MACs
#       eval_per_ablation/               per-ablation eval summaries
#         BaseCot_12L_2R/eval_summary_ckpt.json
#         ...
#     figs/
#       table1.{md,tex}                  Table 1 reproduction
#       fig2a_12L_pareto.{png,pdf}       Figure 2 subplot (a)
#       fig2b_24L_pareto.{png,pdf}       Figure 2 subplot (b)
#       fig3_macs_vs_seqlen.{png,pdf}    Figure 3
#     eval_logs/
#       compute_macs.log
#       reproduce_table1_fig2.log
#       plot_table1.log
#       plot_fig2.log
#       plot_fig3.log
#
# No large intermediates land in run_N. Eval is a single forward pass per
# ablation — no router_weights.npy workspace needed (see eval-adm for that
# pattern). Per-ablation eval JSONs are small (< 10 KB each).
#
# Usage:
#   cd ~/CoTFormer && bash iridis/base-cots-eval/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

CKPT_ROOT="/scratch/ab3u21/exps/owt2/cotformer_full_depth"

# Format: "<folder_name>"  -- the folder name encodes n_layer and n_repeat
ABLATIONS=(
    "BaseCot_12L_2R"
    "BaseCot_12L_3R"
    "BaseCot_12L_5R"
    "BaseCot_12L_15R"
    "BaseCot_24L_2R"
    "BaseCot_24L_3R"
    "BaseCot_24L_5R"
)

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node, no SLURM env yet) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    # Pre-flight: for each ablation verify summary.json AND ckpt.pt exist.
    # Fail fast before queueing — saves hours of queue wait if a ckpt is missing.
    MISSING=0
    for ablation in "${ABLATIONS[@]}"; do
        summary_file="$CKPT_ROOT/$ablation/summary.json"
        ckpt_file="$CKPT_ROOT/$ablation/ckpt.pt"
        if [ ! -f "$summary_file" ]; then
            echo "ERROR: missing summary.json for $ablation"
            echo "       expected: $summary_file"
            MISSING=$((MISSING + 1))
        fi
        if [ ! -f "$ckpt_file" ]; then
            echo "ERROR: missing ckpt.pt for $ablation"
            echo "       expected: $ckpt_file"
            MISSING=$((MISSING + 1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo ""
        echo "$MISSING file(s) missing. Ensure all ablation checkpoints are"
        echo "trained and placed under: $CKPT_ROOT/<ablation>/"
        echo "Required files per ablation: summary.json, ckpt.pt"
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")

    # Pre-create the run_N output tree on login node
    mkdir -p \
        "$RUN_DIR/json/eval_per_ablation" \
        "$RUN_DIR/figs" \
        "$RUN_DIR/eval_logs"

    echo "=== Table 1 + Figure 2 + Figure 3 Reproduction ==="
    echo "  CKPT_ROOT:  $CKPT_ROOT"
    echo "  Ablations:  ${#ABLATIONS[@]} (${ABLATIONS[*]})"
    echo "  RUN_DIR:    $RUN_DIR"
    echo ""

    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$0" "$@"
fi

# --- Actual job (runs on compute node) ---
set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"

echo "========================================="
echo " Table 1 + Figure 2 + Figure 3 Repro"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Started:       $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""


# ========================= PHASE 1: Compute analytical MACs =================
echo ""
echo "###################################################################"
echo "   PHASE 1: Compute analytical MACs"
echo "###################################################################"

python scripts/compute_macs.py \
    --output "$RUN_DIR/json/macs.json" \
    2>&1 | tee "$RUN_DIR/eval_logs/compute_macs.log"

echo "  MACs written to $RUN_DIR/json/macs.json"


# ========================= PHASE 2: Eval sweep + join =======================
echo ""
echo "###################################################################"
echo "   PHASE 2: Eval sweep across ${#ABLATIONS[@]} ablations + join"
echo "###################################################################"

python scripts/reproduce_table1_fig2.py \
    --ckpt-root "$CKPT_ROOT" \
    --ablations "${ABLATIONS[@]}" \
    --output-dir "$RUN_DIR/json/" \
    --macs-json "$RUN_DIR/json/macs.json" \
    --eval-log-dir "$RUN_DIR/eval_logs/" \
    --data-dir "/scratch/ab3u21/datasets" \
    2>&1 | tee "$RUN_DIR/eval_logs/reproduce_table1_fig2.log"

echo "  Results written to $RUN_DIR/json/"


# ========================= PHASE 3: Plots ===================================
echo ""
echo "###################################################################"
echo "   PHASE 3: Plots"
echo "###################################################################"

echo ""
echo "--- plot_table1.py ---"
python scripts/plot_table1.py \
    --results "$RUN_DIR/json/results_table1_fig2.json" \
    --output-dir "$RUN_DIR/figs/" \
    2>&1 | tee "$RUN_DIR/eval_logs/plot_table1.log"

echo ""
echo "--- plot_fig2.py ---"
python scripts/plot_fig2.py \
    --results "$RUN_DIR/json/results_table1_fig2.json" \
    --macs "$RUN_DIR/json/macs.json" \
    --output-dir "$RUN_DIR/figs/" \
    2>&1 | tee "$RUN_DIR/eval_logs/plot_fig2.log"

echo ""
echo "--- plot_fig3.py ---"
python scripts/plot_fig3.py \
    --macs "$RUN_DIR/json/macs.json" \
    --output-dir "$RUN_DIR/figs/" \
    2>&1 | tee "$RUN_DIR/eval_logs/plot_fig3.log"

echo "  Figures written to $RUN_DIR/figs/"


# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " Table 1 + Fig 2 + Fig 3 complete: $(date)"
echo " Results tree: $RUN_DIR/"
echo ""
echo " run_N/ MUST contain:"
echo "   slurm_%j.out / slurm_%j.err"
echo "   json/macs.json"
echo "   json/results_table1_fig2.json"
echo "   json/eval_per_ablation/*/eval_summary_ckpt.json  (one per ablation)"
echo "   figs/*.png  figs/*.pdf  figs/*.md  figs/*.tex"
echo "   eval_logs/*.log"
echo ""
echo " run_N/ MUST NOT contain:"
echo "   *.npy  *.pt  checkpoint files  large intermediates"
echo ""
echo " Output tree (up to depth 3):"
find "$RUN_DIR" -maxdepth 3 -printf '%p\n' | sort
echo "========================================="
