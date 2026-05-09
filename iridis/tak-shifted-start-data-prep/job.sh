#!/bin/bash
#SBATCH --job-name=shifted_start_data_prep
#SBATCH --partition=amd_student
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
################################################################################
# Generate shifted-start JSONL splits under shared scratch.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/tak-shifted-start-data-prep/job.sh
################################################################################

TASK="counting_samesymbol_shiftedstart3__tr25_te200__"
NUM_TRAIN=1000000
SPLITS="${SPLITS:-train_var_len val_var_len ood_test_grid_ltmax ood_test_max_len ood_test_grid ood_test_sampled}"
OOD_SAMPLE_SIZE="${OOD_SAMPLE_SIZE:-2000}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== Shifted-start data prep ==="
    echo "  Task:   $TASK"
    echo "  Splits: $SPLITS"
    echo "  Force:  $FORCE"
    echo "  Data:   $DATA_DIR/rasp_primitives/$TASK"
    echo "  Logs:   $RUN_DIR/"
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

mkdir -p "$DATA_DIR/rasp_primitives" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/tak-shifted-start:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "========================================="
echo " Shifted-start Data Prep"
echo " User:     $USER"
echo " Node:     $(hostname)"
echo " CPUs:     $SLURM_CPUS_PER_TASK"
echo " Job ID:   $SLURM_JOB_ID"
echo " Task:     $TASK"
echo " Splits:   $SPLITS"
echo " OOD samp: $OOD_SAMPLE_SIZE"
echo " Seed:     $SEED"
echo " Force:    $FORCE"
echo " Data dir: $DATA_DIR/rasp_primitives/$TASK"
echo " Run dir:  $RUN_DIR"
echo " Started:  $(date)"
echo "========================================="

FORCE_ARGS=()
if [ "$FORCE" = "1" ]; then
    FORCE_ARGS=(--force)
fi

python tak-shifted-start/tak_gen_shifted_start.py \
    --task "$TASK" \
    --data_root "$DATA_DIR/rasp_primitives" \
    --splits $SPLITS \
    --num_train "$NUM_TRAIN" \
    --ood_sample_size "$OOD_SAMPLE_SIZE" \
    --seed "$SEED" \
    "${FORCE_ARGS[@]}"

echo ""
echo "Generated files:"
ls -lh "$DATA_DIR/rasp_primitives/$TASK"
echo ""
for split in train val ood_test train_var_len val_var_len ood_test_grid_ltmax ood_test_max_len ood_test_grid ood_test_sampled; do
    if [ -f "$DATA_DIR/rasp_primitives/$TASK/$split.txt" ]; then
        wc -l "$DATA_DIR/rasp_primitives/$TASK/$split.txt"
    fi
done

echo "========================================="
echo " Done: $(date)"
echo "========================================="
