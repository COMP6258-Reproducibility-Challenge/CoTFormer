#!/bin/bash
#SBATCH --job-name=shifted_start_cotformer
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
################################################################################
# CoTFormer shifted-start counting experiment.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/shifted-start-train/job.sh
#
# Requires:
#   bash iridis/shifted-start-data-prep/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

TASK="counting_samesymbol_shiftedstart3__tr25_te200__"
N_GPUS=1
N_LAYER=12
N_REPEAT=3
ITERATIONS=100
BATCH_SIZE=8
ACC_STEPS=16
CKPT_FREQ=20
EVAL_FREQ=10

# Reserved layers:
N_LAYER_BEGIN=2
N_LAYER_END=1

# ========================= END CONFIGURATION ================================

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== CoTFormer shifted-start training ==="
    echo "  Partition: ecsstudents_l4"
    echo "  GPUs:      $N_GPUS"
    echo "  Task:      $TASK"
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
export PYTHONPATH="$REPO_DIR:$REPO_DIR/IB_shifted_Start:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "========================================="
echo " CoTFormer Shifted-start Training"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Task:          $TASK"
echo " Model:         fixed_cot_attn"
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

for split in train val ood_test; do
    if [ ! -f "$SHIFTED_DATA_DIR/$split.txt" ]; then
        echo "Missing split: $SHIFTED_DATA_DIR/$split.txt" >&2
        echo "Suggested fix: bash iridis/shifted-start-data-prep/job.sh" >&2
        die "Missing shifted-start split check failed."
    fi
done

echo "Dataset files:"
ls -lh "$SHIFTED_DATA_DIR"
echo ""

TRAIN_ARGS=(
    --config_format base
    --model fixed_cot_attn
    --n_embd 768
    --n_head 12
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
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --results_base_folder "$EXPS_DIR"
    --exp_name "shifted_start_${TASK}_fixed_cot_attn_${N_LAYER}layer_${N_REPEAT}repeat_bs${BATCH_SIZE}x${ACC_STEPS}_seqlen256"
    --use_pretrained auto
    --ib_task "$TASK"
    --ib_data_root "$DATA_DIR/rasp_primitives"
    --ib_eval_splits val ood_test
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
        IB_shifted_Start/IB_shifted_start_main.py "${TRAIN_ARGS[@]}" \
        --distributed_backend nccl
else
    echo "Launching single-GPU shifted-start training"
    python IB_shifted_Start/IB_shifted_start_main.py "${TRAIN_ARGS[@]}"
fi

EXIT_CODE=$?

echo "========================================="
echo " Training finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo ""
echo " Checkpoints: $EXPS_DIR/$TASK/fixed_cot_attn/"
echo ""
echo " If training incomplete, resubmit:"
echo "   bash iridis/shifted-start-train/job.sh"
echo ""
echo " After training completes, sync WandB:"
echo "   wandb sync $WANDB_DIR/<offline-run-*>"
echo "========================================="

exit $EXIT_CODE
