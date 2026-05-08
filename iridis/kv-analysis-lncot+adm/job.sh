#!/bin/bash
#SBATCH --job-name=kv_analysis
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
################################################################################
# KV Cache Compressibility Analysis — iridis/kv-analysis-lncot+adm
#
# Runs Protocol G (Type A weight-level SVD + Type B activation-level SVD +
# KV-CoRE NER) on the four exploratory LN-CoTFormer / ADM checkpoints that
# pre-date the formal §1.5 Analysis Matrix. The four checkpoints below
# were the original MLA-rank-selection inputs (Phase 4); two of them
# (lncot_40k = c2, adm_v2_40k) are NOT covered by the analyze-lncot+adm
# package's KV_RANK_TAGS_REGEX (which runs Protocol G only on c1, c3, c5),
# so this package is retained -- not redundant -- to keep their KV-rank
# coverage.
#
# Implementation: invokes `python -m analysis.kv_rank` (Protocol G; covers
# Type A + Type B + KV-CoRE NER). The previous entry point
# analyze_kv_compression.py was absorbed into analysis.kv_rank during the
# mvp-b orch wave and no longer exists on disk.
#
# Per checkpoint:
#   Type A (weight-level SVD): fast, no data needed (CPU-only).
#   Type B (activation-level SVD): requires GPU + data; --capture-kv
#                                  triggers the workspace populate when
#                                  the kv_mid_l<L>_r<R>.npy files are absent.
#   KV-CoRE NER: emitted alongside Type B per Chen et al. 2026 [8].
#
# Usage:
#   cd ~/CoTFormer && bash iridis/kv-analysis-lncot+adm/job.sh
#
# ANALYSIS MATRIX:
#   - LN-CoTFormer 40k  (Table 2 baseline, --checkpoint-file ckpt_40000.pt)
#   - LN-CoTFormer 60k  (Section 5, --checkpoint-file ckpt.pt)
#   - ADM v2 40k         (matched step count, --checkpoint-file ckpt_40000.pt)
#   - ADM v2 60k         (best adaptive, --checkpoint-file ckpt.pt)
################################################################################

# ========================= CONFIGURATION ======================================

TARGET_RANK=192
N_BATCHES=10
SEQ_LEN=256
BATCH_SIZE=8
OUTPUT_DIR="/scratch/ab3u21/exps/kv-analysis"

# Checkpoints to analyse. Format: "tag|dir|checkpoint_file"
LNCOT_DIR="/scratch/ab3u21/exps/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256"
ADM_DIR="/scratch/ab3u21/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256"

CHECKPOINTS=(
    "lncot_40k|$LNCOT_DIR|ckpt_40000.pt"
    "lncot_60k|$LNCOT_DIR|ckpt.pt"
    "adm_v2_40k|$ADM_DIR|ckpt_40000.pt"
    "adm_v2_60k|$ADM_DIR|ckpt.pt"
)

# ========================= END CONFIGURATION ==================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== KV Cache Compressibility Analysis (LNCot + ADM) ==="
    echo "  Target rank:  $TARGET_RANK"
    echo "  Checkpoints:  ${#CHECKPOINTS[@]}"
    echo "  Output:       $OUTPUT_DIR/"
    echo "  Logs:         $RUN_DIR/"
    echo ""
    for entry in "${CHECKPOINTS[@]}"; do
        IFS='|' read -r tag path ckpt_file <<< "$entry"
        if [ -f "$path/$ckpt_file" ]; then
            echo "  [$tag] $path/$ckpt_file"
        else
            echo "  [$tag] $path/$ckpt_file  (WARNING: not found)"
        fi
    done
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR" \
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

mkdir -p "$OUTPUT_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR"

echo "========================================="
echo " KV Cache Compressibility Analysis"
echo " User:        $USER"
echo " Node:        $(hostname)"
echo " Job ID:      $SLURM_JOB_ID"
echo " Target rank: $TARGET_RANK"
echo " N batches:   $N_BATCHES"
echo " Output:      $OUTPUT_DIR/"
echo " Started:     $(date)"
echo "========================================="

# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- Verify dataset (needed for activation analysis) ---
if [ ! -f "$DATA_DIR/openwebtext2/train.bin" ]; then
    echo "WARNING: Dataset not found. Activation analysis (Type B) will be skipped."
    SKIP_ACTIVATIONS=1
else
    SKIP_ACTIVATIONS=0
fi

# --- Run analysis on each checkpoint ---
ANALYSED=0
SKIPPED=0

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r tag ckpt_dir ckpt_file <<< "$entry"

    echo ""
    echo "========================================="
    echo " Analysing: $tag"
    echo " Checkpoint: $ckpt_dir/$ckpt_file"
    echo "========================================="

    if [ ! -f "$ckpt_dir/summary.json" ]; then
        echo "  SKIP: no summary.json. Training may not have completed."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ ! -f "$ckpt_dir/$ckpt_file" ]; then
        echo "  SKIP: $ckpt_file not found. Checkpoint may have been cleaned."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # analysis.kv_rank writes outputs to --output-dir directly; we point
    # it at a per-tag subdir so the parallel runs don't collide on shared
    # filenames. Workspace lives on /scratch like the analyze-lncot+adm
    # package's convention, so a later --capture-kv re-run finds the
    # cached kv_mid_l<L>_r<R>.npy and skips re-capture.
    TAG_OUTPUT_DIR="$OUTPUT_DIR/$tag"
    TAG_WORKSPACE="/scratch/$USER/analysis_workspace/kv_legacy_$tag"
    mkdir -p "$TAG_OUTPUT_DIR" "$TAG_WORKSPACE"

    ANALYSIS_ARGS=(
        --checkpoint "$ckpt_dir"
        --checkpoint-file "$ckpt_file"
        --workspace "$TAG_WORKSPACE"
        --output-dir "$TAG_OUTPUT_DIR"
        --target-rank "$TARGET_RANK"
        --seed 2357
    )

    if [ "$SKIP_ACTIVATIONS" -eq 0 ]; then
        ANALYSIS_ARGS+=(
            --capture-kv
            --data-dir "$DATA_DIR"
            --max-tokens $((N_BATCHES * SEQ_LEN * BATCH_SIZE))
            --batch-size "$BATCH_SIZE"
            --seq-length "$SEQ_LEN"
        )
    else
        ANALYSIS_ARGS+=( --skip-activations )
    fi

    python -m analysis.kv_rank "${ANALYSIS_ARGS[@]}"

    echo "  -> $TAG_OUTPUT_DIR/kv_rank_results.json"
    echo "  -> $TAG_OUTPUT_DIR/kv_rank_plots.png"

    ANALYSED=$((ANALYSED + 1))
done

echo ""
echo "========================================="
echo " KV Analysis complete: $(date)"
echo " Analysed: $ANALYSED checkpoints"
echo " Skipped:  $SKIPPED checkpoints"
echo ""
echo " Results: $OUTPUT_DIR/"
echo ""
echo " Next steps:"
echo "   1. Compare LNCot vs ADM effective ranks"
echo "   2. Check if kv_lora_rank=$TARGET_RANK suffices for all layers"
echo "   3. Identify prefix/suffix vs mid-layer compressibility differences"
echo "   4. Use findings to finalise MLA configuration (Phase 4)"
echo "========================================="

exit 0
