#!/bin/bash
#SBATCH --job-name=eval_cotres
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
################################################################################
# CoTFormer + Reserved Layers Evaluation -- iridis/eval-cot-res
#
# Evaluates the trained CoTFormer + Reserved Layers (cotformer_full_depth)
# at the 40k checkpoint against the full OWT2 validation set.
#
# Paper target (Table 2, row 2):
#   40k steps: PPL ~24.51
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-cot-res/job.sh
#
# Outputs (in this package's run_N/ subdir):
#   eval_40k.txt   -- stdout from eval.py at ckpt.pt (final 40k checkpoint)
#   slurm_*.out    -- combined SLURM log
################################################################################

# ========================= CONFIGURATION ====================================

CKPT_DIR="/scratch/ab3u21/exps/owt2/cotformer_full_depth/cotformer_full_depth_res_only_lr0.001_bs8x16_seqlen256"

# Only 40k checkpoint — Table 2 scope
CKPTS=("40000")

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    # Pre-flight: verify checkpoint exists
    # The final checkpoint might be ckpt.pt or ckpt_40000.pt
    FOUND_CKPT=false
    for step in "${CKPTS[@]}"; do
        if [ -f "$CKPT_DIR/ckpt_${step}.pt" ]; then
            FOUND_CKPT=true
        fi
    done
    if [ -f "$CKPT_DIR/ckpt.pt" ]; then
        FOUND_CKPT=true
    fi
    if [ "$FOUND_CKPT" = false ]; then
        echo "ERROR: No checkpoint found in $CKPT_DIR"
        echo "  Train it with: bash iridis/cot-res-train/job.sh"
        exit 1
    fi

    if [ ! -f "$CKPT_DIR/summary.json" ]; then
        echo "ERROR: summary.json not found in $CKPT_DIR"
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== CoTFormer + Reserved Layers Evaluation ==="
    echo "  Checkpoints: ${CKPTS[*]} steps"
    echo "  Source:       $CKPT_DIR"
    echo "  Logs:         $RUN_DIR/"
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
echo " CoTFormer + Reserved Layers Evaluation"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Model:         cotformer_full_depth"
echo " Checkpoints:   ${CKPTS[*]} steps"
echo " Started:       $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Evaluate each checkpoint ---
for step in "${CKPTS[@]}"; do
    # Try ckpt_N.pt first, fall back to ckpt.pt (final checkpoint)
    CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
    if [ ! -f "$CKPT_FILE" ]; then
        CKPT_FILE="$CKPT_DIR/ckpt.pt"
        echo "Note: ckpt_${step}.pt not found, using ckpt.pt"
    fi
    OUT_FILE="$RUN_DIR/eval_${step/000/k}.txt"

    echo ""
    echo "========================================="
    echo " Evaluating: $(basename $CKPT_FILE)"
    echo " Output:     $OUT_FILE"
    echo "========================================="

    python eval.py \
        --checkpoint "$CKPT_FILE" \
        --config_format base \
        --distributed_backend None \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "$OUT_FILE"

    echo ""
    echo "--- $(basename $CKPT_FILE) DONE ---"
    echo ""
done

echo "========================================="
echo " All evaluations complete: $(date)"
echo " Results in: $RUN_DIR/"
echo ""
echo " Outputs per checkpoint:"
echo "   eval_<step>k.txt               -- stdout log (loss, PPL, accuracy)"
echo "   eval_summary_ckpt_<step>.json  -- full stats incl. val_loss_std,"
echo "                                     n_batches, val_perplexity_ci95"
echo "                                     (SEM = val_loss_std / sqrt(n_batches))"
echo ""
echo " Paper target (Table 2 row 2): PPL ~24.51 (CoTFormer + Reserved Layers)"
echo "========================================="
