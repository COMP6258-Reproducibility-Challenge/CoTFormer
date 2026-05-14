#!/bin/bash
#SBATCH --job-name=diag_d_phase1
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
################################################################################
# diag_d_job.sh -- SLURM wrapper for Phase-1 trajectory reconstruction.
#
# Evaluates the intermediate ckpt_<N>.pt files of 3 ablations (12L_2R, 12L_3R,
# 12L_5R) at 9 iter milestones each = 27 quick PPL probes. Reconstructs the
# Phase 1 trajectory that was lost when summary.json was overwritten at the
# Phase 2 resume.
#
# Output: diag_d_<job_id>.out next to the script.
# Wall: ~15 min (~30 sec per probe x 27 probes + warmup).
#
# Usage:
#   cd ~/CoTFormer
#   sbatch iridis/base-cots-eval/diag/diag_d_job.sh
#   # OR with extra args:
#   sbatch iridis/base-cots-eval/diag/diag_d_job.sh --n-batches 100 --output-csv phase1.csv
################################################################################

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    echo "=== Diag D -- Phase-1 trajectory reconstruction ==="
    echo "  Repo:    $REPO_DIR"
    echo "  Args:    $*"
    echo ""

    exec sbatch \
        --output="$PACKAGE_DIR/diag_d_%j.out" \
        --error="$PACKAGE_DIR/diag_d_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",DIAG_ARGS="$*" \
        "$0"
fi

set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
fi
source "$REPO_DIR/iridis/env.sh"

echo "========================================="
echo " Diag D on $(hostname), job $SLURM_JOB_ID"
echo " Started: $(date)"
echo " DIAG_ARGS: ${DIAG_ARGS:-<none>}"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# shellcheck disable=SC2086
python iridis/base-cots-eval/diag/diag_d_phase1_trajectory.py $DIAG_ARGS

echo ""
echo "========================================="
echo " Diag D complete: $(date)"
echo "========================================="
