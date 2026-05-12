#!/bin/bash
#SBATCH --job-name=phop_constructive_data
#SBATCH --partition=amd_student
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
################################################################################
# Generate paper-style constructive p-hop JSONL splits under shared scratch.
#
# This writes split names like train_constructive.txt into the same task
# directory as the rejection-sampled data, so a training job can evaluate on
# both distributions:
#
#   TRAIN_SPLIT=train_constructive \
#   EVAL_SPLITS="val test val_constructive test_constructive" \
#   bash iridis/p-hop-cotformer-train/job.sh
#
# Usage:
#   cd ~/CoTFormer && bash iridis/p-hop-constructive-data-prep/job.sh
################################################################################

TASK="${TASK:-phop_p8_seq256_a4_final}"
SPLIT_SIZES="${SPLIT_SIZES:-train_constructive=2000000 val_constructive=50000 test_constructive=50000}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50000}"

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== constructive p-hop data prep ==="
    echo "  Task:        $TASK"
    echo "  Split sizes: $SPLIT_SIZES"
    echo "  Force:       $FORCE"
    echo "  Progress:    every $PROGRESS_EVERY examples"
    echo "  Data:        $DATA_DIR/p-hop/$TASK"
    echo "  Logs:        $RUN_DIR/"
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$0" "$@"
fi

set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
    echo "Tip: use 'bash job.sh' instead of 'sbatch job.sh'"
fi

source "$REPO_DIR/iridis/env.sh"

if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(job_output_dir)
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/output.log") 2> >(tee -a "$RUN_DIR/error.log" >&2)

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

mkdir -p "$DATA_DIR/p-hop" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/p-hop-induction:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "========================================="
echo " Constructive p-hop Data Prep"
echo " User:        $USER"
echo " Node:        $(hostname)"
echo " CPUs:        $SLURM_CPUS_PER_TASK"
echo " Job ID:      $SLURM_JOB_ID"
echo " Task:        $TASK"
echo " Split sizes: $SPLIT_SIZES"
echo " Seed:        $SEED"
echo " Force:       $FORCE"
echo " Progress:    every $PROGRESS_EVERY examples"
echo " Data dir:    $DATA_DIR/p-hop/$TASK"
echo " Run dir:     $RUN_DIR"
echo " Started:     $(date)"
echo "========================================="

FORCE_ARGS=()
if [ "$FORCE" = "1" ]; then
    FORCE_ARGS=(--force)
fi

python p-hop-induction/phop_data.py \
    --task "$TASK" \
    --data_root "$DATA_DIR/p-hop" \
    --split_sizes $SPLIT_SIZES \
    --seed "$SEED" \
    --progress_every "$PROGRESS_EVERY" \
    --sampling_strategy constructive \
    --avoid_adjacent_repeats false \
    "${FORCE_ARGS[@]}"

echo ""
echo "Generated files:"
ls -lh "$DATA_DIR/p-hop/$TASK"
echo ""
for split_spec in $SPLIT_SIZES; do
    split="${split_spec%%=*}"
    if [ -f "$DATA_DIR/p-hop/$TASK/$split.txt" ]; then
        wc -l "$DATA_DIR/p-hop/$TASK/$split.txt"
    fi
done

echo "========================================="
echo " Done: $(date)"
echo "========================================="
