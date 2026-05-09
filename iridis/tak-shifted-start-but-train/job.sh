#!/bin/bash
#SBATCH --job-name=shifted_start_but
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
################################################################################
# BUT shifted-start counting baseline.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/tak-shifted-start-but-train/job.sh
#
# Requires:
#   bash iridis/tak-shifted-start-data-prep/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

TASK="counting_samesymbol_shiftedstart3__tr25_te200__"
MODEL_NAME="but_full_depth"
N_GPUS=1
N_LAYER=1
N_REPEAT=4
N_LAYER_BEGIN=0
N_LAYER_END=0
ITERATIONS=100
BATCH_SIZE=8
ACC_STEPS=16
CKPT_FREQ=20
EVAL_FREQ=10
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLITS="${EVAL_SPLITS:-val ood_test}"
BEST_SPLIT="${BEST_SPLIT:-ood_test}"
BEST_METRIC="${BEST_METRIC:-acc}"
BIG_EVAL_SPLITS="${BIG_EVAL_SPLITS:-val val_var_len ood_test ood_test_sampled ood_test_grid_ltmax ood_test_max_len ood_test_grid}"
BIG_EVAL_MAX_BATCHES="${BIG_EVAL_MAX_BATCHES:-}"

# ========================= END CONFIGURATION ================================

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== Shifted-start BUT training ==="
    echo "  Partition: ecsstudents_l4"
    echo "  GPUs:      $N_GPUS"
    echo "  Task:      $TASK"
    echo "  Train:     $TRAIN_SPLIT"
    echo "  Eval:      $EVAL_SPLITS"
    echo "  Best:      $BEST_SPLIT.$BEST_METRIC"
    echo "  Big eval:  $BIG_EVAL_SPLITS"
    echo "  Layers:    $N_LAYER (${N_LAYER_BEGIN}+mid*${N_REPEAT}+${N_LAYER_END})"
    echo "  Steps:     $ITERATIONS"
    echo "  Eff. BS:   $((BATCH_SIZE * ACC_STEPS))"
    echo "  Logs:      $RUN_DIR/"
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

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

EXPS_DIR="/scratch/ab3u21/exps"
SHIFTED_DATA_DIR="$DATA_DIR/rasp_primitives/$TASK"
mkdir -p "$EXPS_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export WANDB_MODE=offline
export PYTHONPATH="$REPO_DIR:$REPO_DIR/tak-shifted-start:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "========================================="
echo " Shifted-start BUT Training"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Task:          $TASK"
echo " Train split:   $TRAIN_SPLIT"
echo " Eval splits:   $EVAL_SPLITS"
echo " Best metric:   $BEST_SPLIT.$BEST_METRIC"
echo " Big eval:      $BIG_EVAL_SPLITS"
echo " Model:         $MODEL_NAME"
echo " Architecture:  ${N_LAYER}L (${N_LAYER_BEGIN}->mid*${N_REPEAT}->${N_LAYER_END})"
echo " Iterations:    $ITERATIONS"
echo " Eff. BS:       $((BATCH_SIZE * ACC_STEPS))"
echo " Checkpoint:    every $CKPT_FREQ steps -> $EXPS_DIR"
echo " Data dir:      $SHIFTED_DATA_DIR"
echo " Run dir:       $RUN_DIR"
echo " Started:       $(date)"
echo "========================================="

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

for split in $TRAIN_SPLIT $EVAL_SPLITS $BIG_EVAL_SPLITS; do
    if [ ! -f "$SHIFTED_DATA_DIR/$split.txt" ]; then
        echo "Missing split: $SHIFTED_DATA_DIR/$split.txt" >&2
        echo "Suggested fix: bash iridis/tak-shifted-start-data-prep/job.sh" >&2
        die "Missing shifted-start split check failed."
    fi
done

echo "Dataset files:"
ls -lh "$SHIFTED_DATA_DIR"
echo ""

DATA_VARIANT_SUFFIX=""
if [ "$TRAIN_SPLIT" != "train" ]; then
    DATA_VARIANT_SUFFIX="_${TRAIN_SPLIT}"
fi

BIG_EVAL_ARGS=()
if [ -n "$BIG_EVAL_SPLITS" ]; then
    BIG_EVAL_ARGS+=(--ib_big_eval_splits $BIG_EVAL_SPLITS)
fi
if [ -n "$BIG_EVAL_MAX_BATCHES" ]; then
    BIG_EVAL_ARGS+=(--ib_big_eval_max_batches "$BIG_EVAL_MAX_BATCHES")
fi

TRAIN_ARGS=(
    --config_format base
    --model "$MODEL_NAME"
    --n_embd 768   #not sure about these whatsoever
    --n_head 12   #not sure about these whatsoever
    --n_layer "$N_LAYER"
    --n_repeat "$N_REPEAT"
    --batch_size "$BATCH_SIZE"
    --sequence_length 256
    --acc_steps "$ACC_STEPS"
    --dropout 0.0
    --iterations "$ITERATIONS"
    --dataset owt2
    --lr 1e-3
    --weight_decay 0.1
    --warmup_percent 0.2
    --eval_freq "$EVAL_FREQ"
    --seed 0
    --depth_random_method uniform_random_range
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --min_repeat "$N_REPEAT"
    --results_base_folder "$EXPS_DIR"
    --exp_name "shifted_start_${TASK}_but_full_depth_${N_LAYER}layer_${N_REPEAT}repeat${DATA_VARIANT_SUFFIX}_bs${BATCH_SIZE}x${ACC_STEPS}_seqlen256"
    --use_pretrained auto
    --ib_task "$TASK"
    --ib_data_root "$DATA_DIR/rasp_primitives"
    --ib_train_split "$TRAIN_SPLIT"
    --ib_eval_splits $EVAL_SPLITS
    --ib_best_split "$BEST_SPLIT"
    --ib_best_metric "$BEST_METRIC"
    "${BIG_EVAL_ARGS[@]}"
    --ib_save_every "$CKPT_FREQ"
    --ib_log_every 10
    --wandb
    --wandb_project rcotformer
    "$@"
)

if [ "$N_GPUS" -gt 1 ]; then
    export OMP_NUM_THREADS=1
    RDZV_HOST=$(hostname)
    RDZV_PORT=$(expr 10000 + $(echo -n "$SLURM_JOB_ID" | tail -c 4))

    echo "Launching DDP shifted-start training: ${N_GPUS} GPUs, RDZV ${RDZV_HOST}:${RDZV_PORT}"
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}" \
        tak-shifted-start/tak_shifted_start_main.py "${TRAIN_ARGS[@]}" \
        --distributed_backend nccl
else
    echo "Launching single-GPU shifted-start training"
    python tak-shifted-start/tak_shifted_start_main.py "${TRAIN_ARGS[@]}"
fi

EXIT_CODE=$?

echo "========================================="
echo " Training finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo ""
echo " Checkpoints: $EXPS_DIR/$TASK/$MODEL_NAME/"
echo ""
echo " If training incomplete, resubmit:"
echo "   bash iridis/tak-shifted-start-but-train/job.sh"
echo ""
echo " After training completes, sync WandB:"
echo "   wandb sync $WANDB_DIR/<offline-run-*>"
echo "========================================="

exit $EXIT_CODE
