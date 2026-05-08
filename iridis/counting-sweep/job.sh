#!/bin/bash
#SBATCH --job-name=rq9_counting
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
################################################################################
# RQ9 Counting Sweep -- Single-Cell Training Job
#
# Trains ONE cell of the RQ9 twin-pilot + 12-cell Arm B initial wave per
# docs/extend-notes.md §1.2 RQ9 and DEC-033. The script is dispatched from
# scripts/rq9_launch_pilot1.sh (Arm A, 5 C&B-style seeds) and
# scripts/rq9_launch_arm_b.sh (Arm B, 4 variants x 3 seeds = 12 cells).
#
# The two regimes diverge on seven training-regime hyperparameters enumerated
# in docs/reprod-notes.md §B12/DEC-020. Arm A uses Chang and Bisk 2024 [25]
# codebase-exact values (verified via
# inductive_counting_with_LMs/scripts/causal_transformer/{trainer,config}.py
# reads); Arm B uses the CoTFormer-native defaults.
#
# Usage (from launcher scripts; see scripts/rq9_launch_*.sh):
#   bash iridis/counting-sweep/job.sh --arm A --variant baseline --seed 1234 --n-embd 1024
#   bash iridis/counting-sweep/job.sh --arm B --variant V1 --seed 2357 --n-embd 1024
#   bash iridis/counting-sweep/job.sh --arm B --variant V2 --seed 8191 --n-embd 1024
#   bash iridis/counting-sweep/job.sh --arm B --variant V3 --seed 19937 --n-embd 1024
#   bash iridis/counting-sweep/job.sh --arm B --variant V4 --seed 2357 --n-embd 1024
#
# Arguments:
#   --arm A|B          Training regime. A = Chang and Bisk-exact (Pilot 1);
#                      B = CoTFormer-native (main sweep). Arm C is deferred
#                      to a follow-up DIR per DEC-033.
#   --variant NAME     Architecture variant.
#                        A: baseline  (but_full_depth 4L standard, the ONLY
#                                      strict-reproduction control in RQ9)
#                        B: V1        (cotformer_full_depth, no reserved)
#                        B: V2        (cotformer_full_depth, 2+1 reserved,
#                                      7 effective layers per DEC-033 Option c)
#                        B: V3        (cotformer_full_depth_lnmid_depthemb,
#                                      LN-CoTFormer)
#                        B: V4        (adaptive_cotformer_mod_efficient_sigmoid
#                                      _crw_lnmid_de_random_factor_single_final,
#                                      ADM with routing disabled)
#   --seed N           Integer seed.
#                        A: 1234, 12, ... (Chang and Bisk convention)
#                        B: 2357, 8191, 19937 (project convention)
#   --n-embd N         Embedding dimension. Initial wave is 1024. Secondary
#                      widths {128, 256, 512} are enabled here but not
#                      commissioned in this DIR.
#   --dry-run          Print the intended TRAIN_ARGS without submitting or
#                      executing main.py. Useful for launcher validation.
#
# Self-submitting wrapper: when invoked on the login node ($SLURM_JOB_ID unset)
# the script sbatch-submits itself with --export propagating the parsed
# arguments + REPO_DIR. On the compute node it sources env.sh, activates
# conda, and dispatches to main.py with the variant-specific TRAIN_ARGS.
#
# Path-construction discipline: checkpoint path is constructed inside
# main.py as ckpt_path = $EXPS_DIR / counting / <model> / <exp_name>.
# This script does NOT mirror that construction in bash; the distinctive
# exp_name per cell (rq9_arm_${ARM,,}_${variant}_seed_${SEED}_nembd_${NEMBD})
# guarantees non-collision of run leaves across cells per the
# never-mirror-path-construction rule documented in
# docs/reprod-notes.md §C4.
#
# Reproduction-fidelity reminder: Arm A is the reproduction control;
# its hyperparameters must match the Chang and Bisk codebase (not the
# paper's prose) -- see docs/reprod-notes.md §M6 for the
# reproduction-fidelity vs fair-comparison discipline, §B12 for the
# seven divergences between paper prose and code, and §B13 for the
# max_grad_norm = 0.3 dead-code discovery.
################################################################################

# ========================= ARG PARSING =====================================

ARM=""
VARIANT=""
SEED=""
NEMBD=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --n-embd) NEMBD="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h)
            sed -n '3,70p' "$0" ; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2 ; exit 2 ;;
    esac
done

# Validate required args.
if [ -z "$ARM" ] || [ -z "$VARIANT" ] || [ -z "$SEED" ] || [ -z "$NEMBD" ]; then
    echo "ERROR: --arm, --variant, --seed, --n-embd are all required" >&2
    echo "  Got: --arm='$ARM' --variant='$VARIANT' --seed='$SEED' --n-embd='$NEMBD'" >&2
    exit 2
fi

case "$ARM" in
    A|B) ;;
    *) echo "ERROR: --arm must be A or B, got '$ARM'" >&2 ; exit 2 ;;
esac

case "$ARM:$VARIANT" in
    A:baseline) ;;
    B:V1|B:V2|B:V3|B:V4) ;;
    *)
        echo "ERROR: --variant='$VARIANT' invalid for --arm='$ARM'" >&2
        echo "  Arm A accepts: baseline" >&2
        echo "  Arm B accepts: V1, V2, V3, V4" >&2
        exit 2 ;;
esac

# Normalise integers.
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --seed must be a positive integer, got '$SEED'" >&2 ; exit 2
fi
if ! [[ "$NEMBD" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --n-embd must be a positive integer, got '$NEMBD'" >&2 ; exit 2
fi

ARM_LOWER="$(echo "$ARM" | tr 'A-Z' 'a-z')"
EXP_NAME="rq9_arm_${ARM_LOWER}_${VARIANT}_seed_${SEED}_nembd_${NEMBD}"

# ========================= SELF-SUBMITTING WRAPPER =========================

if [ -z "$SLURM_JOB_ID" ] && [ "$DRY_RUN" -eq 0 ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    # Stamp the cell identity into the run dir so the SLURM .out is greppable.
    echo "$EXP_NAME" > "$RUN_DIR/cell.txt"

    echo "=== RQ9 Counting Sweep ==="
    echo "  Arm:          $ARM"
    echo "  Variant:      $VARIANT"
    echo "  Seed:         $SEED"
    echo "  n_embd:       $NEMBD"
    echo "  Exp name:     $EXP_NAME"
    echo "  Run dir:      $RUN_DIR"
    echo "  Ckpt target:  \$EXPS_DIR/counting/<model>/$EXP_NAME"
    echo ""
    exec sbatch \
        --job-name="rq9_${ARM_LOWER}_${VARIANT}_s${SEED}_n${NEMBD}" \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR",ARM="$ARM",VARIANT="$VARIANT",SEED="$SEED",NEMBD="$NEMBD",EXP_NAME="$EXP_NAME" \
        "$0" "$@"
fi

# Dry-run short-circuit: build TRAIN_ARGS, print, exit.
if [ "$DRY_RUN" -eq 1 ]; then
    # Fall through to TRAIN_ARGS construction below; skip SLURM + main.py.
    :
fi

# ========================= COMPUTE-NODE SETUP ==============================

set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -n "$SLURM_JOB_ID" ]; then
    if [ -z "$REPO_DIR" ]; then
        REPO_DIR="$HOME/CoTFormer"
        echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
    fi
    source "$REPO_DIR/iridis/env.sh"
    # Defensive cache + scratch mkdirs: env.sh exports the path vars but
    # does not create directories on the compute node. The counting task
    # uses the trivial w2i tokeniser (no tiktoken), so TIKTOKEN_CACHE_DIR
    # is included for forward-compat / cross-pkg consistency only.
    mkdir -p "$EXPS_DIR" "$WANDB_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"
fi

if [ "$DRY_RUN" -eq 1 ] && [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="${REPO_DIR:-$(cd "$PACKAGE_DIR/../.." && pwd)}"
    # Silently source env.sh when available to resolve $EXPS_DIR etc. for
    # the printed TRAIN_ARGS; fall back to placeholder strings otherwise.
    if [ -f "$REPO_DIR/iridis/env.sh" ]; then
        source "$REPO_DIR/iridis/env.sh"
    fi
fi

# ========================= REGIME DISPATCH =================================
#
# Arm A (Chang and Bisk-exact) and Arm B (CoTFormer-native) differ on the
# seven training-regime hyperparameters enumerated in docs/reprod-notes.md
# §B12 + DEC-020. The regime block below is the single source of truth for
# these seven values; the variant block below selects the architecture and
# variant-specific flags at fixed regime.

# Shared across arms (DEC-016 + DEC-033):
ITERATIONS=312500      # 312.5K steps matching Chang and Bisk [25]
BATCH_SIZE=32          # DEC-016 step-matched BS=32
ACC_STEPS=1            # Counting task fits on a single GPU; no accumulation
SEQUENCE_LENGTH=256    # supports OOD L up to 200 (> max_out + 1 = 201)
VOCAB_SIZE=203         # te200 trivial vocab per data/counting.py TE200_VOCAB_SIZE
EVAL_FREQ=1000         # Less frequent than base-train's 100 to keep eval cost bounded
CKPT_FREQ=10000        # Checkpoint every 10K steps (~30 checkpoints per cell)
WARMUP_PERCENT="0.0096" # 3000 / 312500 == 0.0096, matches Chang and Bisk warmup_steps=3000

if [ "$ARM" = "A" ]; then
    # Chang and Bisk-exact (DEC-020 seven-divergence reconciliation):
    #   trainer.py:231 max_grad_norm = 1.0 (paper prose says 0.3 but
    #       config.py:28 dead-code per reprod-notes §B13)
    #   trainer.py:72  get_constant_schedule_with_warmup (not cosine)
    #   trainer.py:70  AdamW defaults wd=0.01, beta2=0.999 (no decay grouping)
    #   config.py:13   n_head = 8 at hidden_size = 1024
    #   trainer.py:52  mixed_precision = "fp16"
    #   trainer.py:211 torch.backends.cuda.sdp_kernel(enable_flash = False)
    #   config.py:26   activation_function = "relu"
    #   config.py:20   scale_attn_by_inverse_layer_idx = True
    #   config.py:25   tie_word_embeddings = False
    # (The last three are the three activation / architecture bits; the
    # first seven are the "seven training-regime divergences" from DEC-020.
    # Collectively Arm A differs from Arm B on ten atomic settings.)
    ACTIVATION="relu"
    TIE_WORD_EMB="False"
    SCALE_ATTN="True"
    GRAD_CLIP="1.0"
    SCHEDULER="constant_with_warmup"
    WEIGHT_DECAY="0.01"
    BETA1="0.9"
    BETA2="0.999"
    DTYPE="torch.float16"
    NHEAD=8
    LR="1e-4"
    # Chang and Bisk 2024 trainer.py:70 uses flat AdamW(model.parameters(),
    # lr=...) rather than per-parameter-group decay. Arm A matches this via
    # --disable_decay_grouping True (eighth C&B divergence, logged as
    # DEC-035 in docs/extend-notes.md §1.9).
    DISABLE_DECAY_GROUPING="True"
    # Chang and Bisk 2024 config.py:44-55 defaults every posemb flag to
    # False, so the te200 config_taskspecific.py entry inherits NoPE
    # (no positional encoding). CoTFormer's registered "none" encoder
    # (models/positional_encoders/__init__.py and
    # models/positional_encoders/encoder.py:PositionalEncoder) is an
    # identity NoPE closure that matches this semantics. Ninth C&B
    # divergence, logged as DEC-035.
    POSITIONAL_ENCODER="none"
else
    # CoTFormer-native (main sweep):
    ACTIVATION="gelu"
    TIE_WORD_EMB="True"     # cotformer_full_depth*.py hardcodes wte = lm_head
    SCALE_ATTN="False"
    GRAD_CLIP="1.0"
    SCHEDULER="cos"
    WEIGHT_DECAY="0.1"
    BETA1="0.9"
    BETA2="0.95"
    DTYPE="torch.bfloat16"
    NHEAD=4
    LR="1e-3"
    # Project-standard per-parameter-group decay (biases / norms / embeddings
    # excluded from the decay set) for Arm B.
    DISABLE_DECAY_GROUPING="False"
    # Arm B retains the CoTFormer project default rotary positional
    # encoding; the NoPE divergence is an Arm A strict-fidelity concern
    # only.
    POSITIONAL_ENCODER="rotary"
fi

# ========================= VARIANT DISPATCH ================================

case "$VARIANT" in
    baseline)
        # Arm A only. 4L standard Transformer via but_full_depth with
        # n_repeat = 1 (skip the depth-recurrence loop) and no reserved
        # layers. but_full_depth supports --tie_word_embeddings=False via
        # the BLOCKER 2 conditional at models/but_full_depth.py:289-294.
        MODEL="but_full_depth"
        NLAYER=4
        NREPEAT=1
        MINREPEAT=1
        NLAYER_BEGIN=0
        NLAYER_END=0
        DEPTH_EMB="None"
        ;;
    V1)
        # cotformer_full_depth without reserved layers; 4 effective layers.
        MODEL="cotformer_full_depth"
        NLAYER=1
        NREPEAT=4
        MINREPEAT=4
        NLAYER_BEGIN=0
        NLAYER_END=0
        DEPTH_EMB="None"
        ;;
    V2)
        # cotformer_full_depth with 2+1 reserved layers; 7 effective
        # layers. DEC-033 Option (c): compare against 4L standard and
        # acknowledge the depth asymmetry as a secondary finding about
        # reserved layers plus recurrence jointly.
        # n_layer = n_begin + n_mid + n_end = 2 + 1 + 1 = 4 (model class
        # builds h_mid via range(n_layer_begin, n_layer - n_layer_end);
        # NLAYER=1 with begin=2/end=1 collapses to range(2, 0) = empty).
        MODEL="cotformer_full_depth"
        NLAYER=4
        NREPEAT=4
        MINREPEAT=4
        NLAYER_BEGIN=2
        NLAYER_END=1
        DEPTH_EMB="None"
        ;;
    V3)
        # LN-CoTFormer. n_layer_begin = 0 and n_layer_end = 0 keep depth
        # matched to the 4L standard; ln_mid lives inside the mid block
        # per cotformer_full_depth_lnmid_depthemb.py.
        MODEL="cotformer_full_depth_lnmid_depthemb"
        NLAYER=1
        NREPEAT=4
        MINREPEAT=4
        NLAYER_BEGIN=0
        NLAYER_END=0
        DEPTH_EMB="linear_learned"
        ;;
    V4)
        # ADM with routing DISABLED (min_repeat == n_repeat) for
        # comparability with the non-adaptive V1-V3 variants per DEC-033.
        MODEL="adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final"
        NLAYER=1
        NREPEAT=4
        MINREPEAT=4
        NLAYER_BEGIN=0
        NLAYER_END=0
        DEPTH_EMB="linear_learned"
        ;;
esac

# ========================= TRAIN_ARGS CONSTRUCTION =========================

TRAIN_ARGS=(
    --config_format base
    --model "$MODEL"
    --dataset counting
    --vocab_size "$VOCAB_SIZE"
    --n_embd "$NEMBD"
    --n_head "$NHEAD"
    --n_layer "$NLAYER"
    --n_repeat "$NREPEAT"
    --min_repeat "$MINREPEAT"
    --n_layer_begin "$NLAYER_BEGIN"
    --n_layer_end "$NLAYER_END"
    --depth_embedding "$DEPTH_EMB"
    --sequence_length "$SEQUENCE_LENGTH"
    --batch_size "$BATCH_SIZE"
    --acc_steps "$ACC_STEPS"
    --dropout 0.0
    --iterations "$ITERATIONS"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --beta1 "$BETA1"
    --beta2 "$BETA2"
    --grad_clip "$GRAD_CLIP"
    --scheduler "$SCHEDULER"
    --warmup_percent "$WARMUP_PERCENT"
    --activation "$ACTIVATION"
    --tie_word_embeddings "$TIE_WORD_EMB"
    --scale_attn_by_inverse_layer_idx "$SCALE_ATTN"
    --disable_decay_grouping "$DISABLE_DECAY_GROUPING"
    --positional_encoder "$POSITIONAL_ENCODER"
    --dtype "$DTYPE"
    --seed "$SEED"
    --eval_freq "$EVAL_FREQ"
    --save_checkpoint_freq "$CKPT_FREQ"
    --results_base_folder "${EXPS_DIR:-\$EXPS_DIR}"
    --exp_name "$EXP_NAME"
    --use_pretrained auto
    --wandb
    --wandb_project rq9_counting
)

# ========================= DRY-RUN SHORT-CIRCUIT ===========================

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== RQ9 Counting Sweep DRY RUN ==="
    echo "  Arm:        $ARM"
    echo "  Variant:    $VARIANT"
    echo "  Seed:       $SEED"
    echo "  n_embd:     $NEMBD"
    echo "  Exp name:   $EXP_NAME"
    echo "  Model:      $MODEL"
    echo ""
    echo "  TRAIN_ARGS (would pass to python main.py):"
    for arg in "${TRAIN_ARGS[@]}"; do
        echo "    $arg"
    done
    echo ""
    echo "  Ckpt target: ${EXPS_DIR:-\$EXPS_DIR}/counting/$MODEL/$EXP_NAME/"
    echo "  Dry run complete; no SLURM job submitted and no training started."
    exit 0
fi

# ========================= COMPUTE-NODE EXECUTION ==========================

echo "========================================="
echo " RQ9 Counting Sweep -- single cell"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " CPUs:          ${SLURM_CPUS_PER_TASK:-?}"
echo " Job ID:        ${SLURM_JOB_ID:-none}"
echo " Arm:           $ARM"
echo " Variant:       $VARIANT"
echo " Model:         $MODEL"
echo " Seed:          $SEED"
echo " n_embd:        $NEMBD"
echo " Iterations:    $ITERATIONS"
echo " Eff. BS:       $((BATCH_SIZE * ACC_STEPS))"
echo " Scheduler:     $SCHEDULER (warmup=$WARMUP_PERCENT)"
echo " Grad clip:     $GRAD_CLIP"
echo " Activation:    $ACTIVATION"
echo " Tie wte:       $TIE_WORD_EMB"
echo " Scale attn:    $SCALE_ATTN"
echo " dtype:         $DTYPE"
echo " Exp name:      $EXP_NAME"
echo " Ckpt target:   $EXPS_DIR/counting/$MODEL/$EXP_NAME/"
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

# --- Launch (single-GPU; counting task fits on 1x L4 at n_embd=1024) ---
python -u main.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

echo "========================================="
echo " RQ9 cell finished: $(date)"
echo " Exit code:     $EXIT_CODE"
echo " Checkpoints:   $EXPS_DIR/counting/$MODEL/$EXP_NAME/"
echo ""
echo " If training incomplete, resubmit via the launcher:"
echo "   scripts/rq9_launch_pilot1.sh  (Arm A)"
echo "   scripts/rq9_launch_arm_b.sh   (Arm B)"
echo " --use_pretrained auto will pick up from the latest ckpt_N.pt."
echo "========================================="

# ========================= DV-2 PAIRED-PROBE EVALUATION ====================
#
# Per-(cell, task) linear-probe + MLP-probe sweep over the trained checkpoint
# of this cell. Runs only on a clean training exit so we never probe a
# truncated checkpoint. The probe writes counting_dv2_results.json (+ .png)
# under $CKPT_DIR/dv2/<task>/ -- the bash script does NOT mirror the probe's
# internal artefact paths (per the path-construction discipline at the top
# of this file).
#
# Module-path discriminator: Arm A baseline is the 4L standard transformer
# (but_full_depth, no mid block) so we probe transformer.h. All Arm B
# variants (V1-V4) are CoTFormer variants whose recurrent mid block is the
# representational locus of interest, so we probe transformer.h_mid.
if [ "$EXIT_CODE" -eq 0 ]; then
    if [ "$ARM" = "A" ]; then
        DV2_MODULE_PATH="transformer.h"
    else
        DV2_MODULE_PATH="transformer.h_mid"
    fi
    DV2_CKPT_DIR="$EXPS_DIR/counting/$MODEL/$EXP_NAME"

    echo ""
    echo "========================================="
    echo " DV-2 paired-probe evaluation"
    echo " Checkpoint:    $DV2_CKPT_DIR"
    echo " Module path:   $DV2_MODULE_PATH"
    echo " Started:       $(date)"
    echo "========================================="

    DV2_OVERALL_EC=0
    for DV2_TASK in task1 task2 task3; do
        DV2_OUT_DIR="$DV2_CKPT_DIR/dv2/$DV2_TASK"
        mkdir -p "$DV2_OUT_DIR"

        echo ""
        echo "--- DV-2 task=$DV2_TASK ---"
        python -u -m analysis.counting_dv2_probe \
            --checkpoint "$DV2_CKPT_DIR" \
            --checkpoint-file ckpt.pt \
            --output-dir "$DV2_OUT_DIR" \
            --seed "$SEED" \
            --task "$DV2_TASK" \
            --eval-length 200 \
            --n-samples 500 \
            --batch-size 4 \
            --sequence-length "$SEQUENCE_LENGTH" \
            --device cuda \
            --config-mode raw \
            --module-path "$DV2_MODULE_PATH" \
            --train-fraction 0.8 \
            --mlp-epochs 200 \
            --mlp-lr 1e-3 \
            --ridge-alpha 1.0 \
            --selectivity-threshold 0.05
        DV2_TASK_EC=$?
        if [ "$DV2_TASK_EC" -ne 0 ]; then
            echo "WARNING: DV-2 task=$DV2_TASK exited $DV2_TASK_EC" >&2
            DV2_OVERALL_EC=$DV2_TASK_EC
        fi
    done

    echo ""
    echo "========================================="
    echo " DV-2 finished: $(date)"
    echo " Overall DV-2 EC: $DV2_OVERALL_EC"
    echo " Results under:  $DV2_CKPT_DIR/dv2/{task1,task2,task3}/"
    echo "========================================="
fi

exit $EXIT_CODE
