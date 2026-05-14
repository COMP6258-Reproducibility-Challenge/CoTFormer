#!/bin/bash
#SBATCH --job-name=diag_b_nhead
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
################################################################################
# diag_b_job.sh -- SLURM wrapper for the n_head hot-swap probe.
#
# Diag B requires CUDA + the repo on PYTHONPATH. Login nodes don't have GPUs,
# so we submit a short (~5 min wall) GPU job. Output lands next to the script
# as diag_b_<job_id>.out / .err for easy rsync-back.
#
# Usage (from login node, after activating conda env):
#   cd ~/CoTFormer
#   sbatch iridis/base-cots-eval/diag/diag_b_job.sh
#   # OR with custom args:
#   sbatch iridis/base-cots-eval/diag/diag_b_job.sh --alt-n-head 6 --n-batches 50
################################################################################

# --- Self-submitting wrapper ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    echo "=== Diag B -- n_head hot-swap probe ==="
    echo "  Repo:    $REPO_DIR"
    echo "  Package: $PACKAGE_DIR"
    echo "  Args:    $*"
    echo ""

    exec sbatch \
        --output="$PACKAGE_DIR/diag_b_%j.out" \
        --error="$PACKAGE_DIR/diag_b_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",DIAG_ARGS="$*" \
        "$0"
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
echo " Diag B -- n_head hot-swap on $(hostname)"
echo " Job ID:        $SLURM_JOB_ID"
echo " Started:       $(date)"
echo " DIAG_ARGS:     ${DIAG_ARGS:-<none>}"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# DIAG_ARGS is passed through unquoted intentionally so multi-word args
# (--alt-n-head 6) split correctly into argv.
# shellcheck disable=SC2086
python iridis/base-cots-eval/diag/diag_b_nhead_hotswap.py $DIAG_ARGS

echo ""
echo "========================================="
echo " Diag B complete: $(date)"
echo "========================================="
