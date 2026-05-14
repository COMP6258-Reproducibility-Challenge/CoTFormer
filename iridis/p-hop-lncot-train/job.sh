#!/bin/bash
#SBATCH --job-name=phop_cotformer
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
################################################################################
# CoTFormer p-hop induction experiment.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/p-hop-cotformer-train/job.sh
#
# Requires:
#   bash iridis/p-hop-data-prep/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

TASK="${TASK:-phop_p32_seq256_a4_final}"
N_GPUS=2
N_LAYER=3
N_REPEAT=4
N_LAYER_BEGIN=1
N_LAYER_END=1
N_EMBD="${N_EMBD:-128}"
N_HEAD="${N_HEAD:-8}"
ITERATIONS="${ITERATIONS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACC_STEPS="${ACC_STEPS:-4}"
CKPT_FREQ="${CKPT_FREQ:-8000}"
EVAL_FREQ="${EVAL_FREQ:-8000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_constructive}"
EVAL_SPLITS="${EVAL_SPLITS:-val_constructive test_constructive}"
BEST_SPLIT="${BEST_SPLIT:-val_constructive}"
BIG_EVAL_SPLITS="${BIG_EVAL_SPLITS:-val_constructive test_constructive}"
BIG_EVAL_MAX_BATCHES="${BIG_EVAL_MAX_BATCHES:-}"
SEED="${SEED:-0}"
BEST_METRIC="${BEST_METRIC:-acc}"
N_MID=$((N_LAYER - N_LAYER_BEGIN - N_LAYER_END))
EFFECTIVE_DEPTH=$((N_LAYER_BEGIN + N_MID * N_REPEAT + N_LAYER_END))
ARCH="${N_LAYER_BEGIN}-${N_MID}x${N_REPEAT}-${N_LAYER_END}"


# ========================= END CONFIGURATION ================================

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== p-hop CoTFormer training ==="
    echo "  Partition: ecsstudents_l4"
    echo "  GPUs:      $N_GPUS"
    echo "  Task:      $TASK"
    echo "  Train:     $TRAIN_SPLIT"
    echo "  Eval:      $EVAL_SPLITS"
    echo "  Seed:      $SEED"
    echo "  Best:      $BEST_SPLIT.$BEST_METRIC"
    echo "  Big eval:  $BIG_EVAL_SPLITS"
    echo "  Layers:    ${N_LAYER} template / ${EFFECTIVE_DEPTH} effective (${ARCH})"
    echo " Architecture:  ${EFFECTIVE_DEPTH} effective layers (${ARCH}; template ${N_LAYER}L)"
    echo "  Width:     d=$N_EMBD h=$N_HEAD"
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

EXPS_DIR="/scratch/ab3u21/exps/p-hop"
PHOP_DATA_DIR="$DATA_DIR/p-hop/$TASK"
mkdir -p "$EXPS_DIR" "$DATA_DIR/p-hop" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export WANDB_MODE=offline
export PYTHONPATH="$REPO_DIR:$REPO_DIR/p-hop-induction:$REPO_DIR/tak-shifted-start:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "========================================="
echo " p-hop CoTFormer Training"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          $SLURM_CPUS_PER_TASK"
echo " GPUs:          $N_GPUS"
echo " Job ID:        $SLURM_JOB_ID"
echo " Task:          $TASK"
echo " Train split:   $TRAIN_SPLIT"
echo " Eval splits:   $EVAL_SPLITS"
echo " Seed:          $SEED"
echo " Best metric:   $BEST_SPLIT.$BEST_METRIC"
echo " Big eval:      $BIG_EVAL_SPLITS"
echo "  Layers:    ${N_LAYER} template / ${EFFECTIVE_DEPTH} effective (${ARCH})"
echo " Architecture:  ${EFFECTIVE_DEPTH} effective layers (${ARCH}; template ${N_LAYER}L)"
echo " Model:         cotformer_full_depth_lnmid_depthemb (LN only, no depth embedding)"
echo " Width:         d=$N_EMBD h=$N_HEAD"
echo " Architecture:  ${N_LAYER}L (${N_LAYER_BEGIN}->mid*${N_REPEAT}->${N_LAYER_END})"
echo " Iterations:    $ITERATIONS"
echo " Eff. BS:       $((BATCH_SIZE * ACC_STEPS))"
echo " Checkpoint:    every $CKPT_FREQ steps -> $EXPS_DIR"
echo " Data dir:      $PHOP_DATA_DIR"
echo " Run dir:       $RUN_DIR"
echo " Started:       $(date)"
echo "========================================="

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

for split in $TRAIN_SPLIT $EVAL_SPLITS $BIG_EVAL_SPLITS; do
    if [ ! -f "$PHOP_DATA_DIR/$split.txt" ]; then
        echo "Missing split: $PHOP_DATA_DIR/$split.txt" >&2
        echo "Suggested fix: bash iridis/p-hop-data-prep/job.sh" >&2
        die "Missing p-hop split check failed."
    fi
done

echo "Dataset files:"
ls -lh "$PHOP_DATA_DIR"
echo ""

BIG_EVAL_ARGS=()
if [ -n "$BIG_EVAL_SPLITS" ]; then
    BIG_EVAL_ARGS+=(--phop_big_eval_splits $BIG_EVAL_SPLITS)
fi
if [ -n "$BIG_EVAL_MAX_BATCHES" ]; then
    BIG_EVAL_ARGS+=(--phop_big_eval_max_batches "$BIG_EVAL_MAX_BATCHES")
fi

TRAIN_ARGS=(
    --config_format base
    --model cotformer_full_depth_lnmid_depthemb
    --n_embd "$N_EMBD"
    --n_head "$N_HEAD"
    --n_layer "$N_LAYER"
    --n_repeat "$N_REPEAT"
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --batch_size "$BATCH_SIZE"
    --sequence_length 256
    --acc_steps "$ACC_STEPS"
    --dropout 0.0
    --iterations "$ITERATIONS"
    --dataset owt2
    --lr 1e-3
    --weight_decay 0.0
    --warmup_percent 0.02
    --eval_freq "$EVAL_FREQ"
    --seed "$SEED"
    --results_base_folder "$EXPS_DIR"
    --exp_name "phop_${TASK}_lnmid_cot_no_depthemb_${ARCH}_${EFFECTIVE_DEPTH}eff_d${N_EMBD}_h${N_HEAD}_bs${BATCH_SIZE}x${ACC_STEPS}_seed${SEED}"
    --use_pretrained auto
    --phop_task "$TASK"
    --phop_data_root "$DATA_DIR/p-hop"
    --phop_train_split "$TRAIN_SPLIT"
    --phop_eval_splits $EVAL_SPLITS
    --phop_best_split "$BEST_SPLIT"
    --phop_best_metric "$BEST_METRIC"
    --depth_embedding None
    "${BIG_EVAL_ARGS[@]}"
    --phop_save_every "$CKPT_FREQ"
    --phop_log_every 100
    --wandb
    --wandb_project rcotformer
    "$@"
)

if [ "$N_GPUS" -gt 1 ]; then
    export OMP_NUM_THREADS=1
    RDZV_HOST=$(hostname)
    RDZV_PORT=$(expr 10000 + $(echo -n "$SLURM_JOB_ID" | tail -c 4))

    echo "Launching DDP p-hop training: ${N_GPUS} GPUs, RDZV ${RDZV_HOST}:${RDZV_PORT}"
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}" \
        p-hop-induction/phop_main.py "${TRAIN_ARGS[@]}" \
        --distributed_backend nccl
else
    echo "Launching single-GPU p-hop training"
    python p-hop-induction/phop_main.py "${TRAIN_ARGS[@]}"
fi

EXIT_CODE=$?

echo "========================================="
echo " Training finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo ""
echo " Checkpoints: $EXPS_DIR/$TASK/cotformer_full_depth_lnmid_depthemb/"
echo ""
echo " If training incomplete, resubmit:"
echo "   bash iridis/p-hop-cotformer-train/job.sh"
echo ""
echo " After training completes, sync WandB:"
echo "   wandb sync $WANDB_DIR/<offline-run-*>"
echo "========================================="

exit $EXIT_CODE
