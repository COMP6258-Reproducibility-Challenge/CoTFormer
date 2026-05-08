#!/bin/bash
#SBATCH --job-name=analyze_lncot_adm_gpu
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
################################################################################
# analyze-lncot+adm GPU stage -- Protocol A (Logit Lens) + Protocol G Type B
# + Protocol C (Attention Taxonomy).
#
# Wave 1 + 2a + 2c deliverable (MVP observational bundle).
# Runs analysis.logit_lens on every checkpoint in the VERSIONS array; the
# script emits logit_lens_results.json and logit_lens_trajectories.png into
# run_N/<tag>/, and persists the per-(layer, repeat) residual cache into
# /scratch/$USER/analysis_workspace/<tag>/ for Stage-4 consumers
# (analysis.tuned_lens, analysis.residual_diagnostics).
#
# Wave 2a additions (Protocol G Type B, per §1.5 Matrix row "G: KV Compress."
# which applies to C1, C3, C5): analysis.kv_rank runs in --capture-kv mode
# on those three tags to populate the kv_mid_l<L>_r<R>.npy cache and emit
# kv_rank_results.json + kv_rank_plots.png into run_N/<tag>/. Activation-
# level SVD is a ~25 min GPU step per G-enabled checkpoint per §1.3 Protocol
# G row.
#
# Wave 2c additions (Protocol C attention taxonomy + sparsity-across-repeat
# test per §1.3 Protocol C row + §1.4 Prediction 2): analysis.attention_taxonomy
# runs with ATTN_WEIGHTS site + non_flash=True on C1, C2, C3, C4, C5 (all
# five per the §1.3 extended matrix note), emitting attention_taxonomy_results.json
# plus heatmap + sparsity-trajectory PNGs. The non-flash math path + the
# monkey-patched forward's Q/K re-compute incur ~8x the flash-backend per-
# block cost; per-checkpoint wall time is ~9 min L4 (§1.8 compute budget).
#
# Wave 3 additions (RQ5 / RQ7 / RQ8):
#   * Protocol D-calibration (substrate-specific, not per-checkpoint):
#     Tier 1 (GPT-2-large) + Tier 2 (ADM C5) forward passes per
#     §1.3 Protocol D-cal three-tier substrate. Tier 3 is contingent
#     and not wired. Per-tier outputs go to $RUN_DIR/d_cal/tier{1,2}/.
#   * Protocol H Interpolation Validity (RQ7): runs on C4 + C5 only per
#     §1.5 Matrix row "H: Interp. Validity"; clone-set permutation test
#     with N=10000 permutations (DEC-026). ~10 min L4 per checkpoint.
#   * Protocol I Depth-Embedding Freeze (RQ8): runs on C3 only per
#     §1.5 Matrix row "I: Depth-Emb Freeze"; iterates the 8 freeze
#     conditions (Cartesian {freeze, unfreeze} x {zero, preserve, random}
#     plus DEC-026 zero-vector controls). ~21 min L4 across all 8.
#
# Stage-5 synthesis is local-friendly per the synthesis docstring (pure
# plotting + scalar aggregation, no GPU/SVD/forward). It is wired as a
# top-level Makefile target (`make synthesis`) consuming rsync'd
# run_N/<tag>/ artefacts; it does NOT run inside this SLURM job.
#
# Output structure:
#   run_N/
#     slurm_<jobid>.out                 SLURM stdout
#     slurm_<jobid>.err                 SLURM stderr
#     <tag>/
#       logit_lens_results.json         per-(layer, repeat) metrics + CV
#       logit_lens_trajectories.png     per-repeat PNG trajectories
#
# Workspace (intermediate, /scratch):
#   /scratch/$USER/analysis_workspace/<tag>/
#     residual_mid_l{L}_r{R}.npy        pre-ln_mid residuals
#     residual_ln_mid_r{R}.npy          post-ln_mid residuals
#     residual_pre_ln_f.npy             input to ln_f (canonical Belrose 2023 target residual)
#     targets.npy                       next-token targets for Tuned Lens
#     meta.json                         collector metadata
#
# Usage:
#   cd ~/CoTFormer && bash iridis/analyze-lncot+adm/job-gpu.sh
################################################################################

# ========================= CONFIGURATION ====================================

# Version matrix: "tag|ckpt_dir_absolute_path|ckpt_file"
# The VERSIONS list is populated per the §1.5 Analysis Matrix (Protocol A
# runs on C1-C5). Replace each entry's ckpt_dir leaf as your run history
# dictates; the leaf names in /scratch/ab3u21/exps/owt2/ follow the
# pattern exp_name=<model>_lr<lr>_bs<bs>x<acc>_seqlen<seq>__<overrides>_seed=<s>.
#
# IMPORTANT: VERSIONS uses $EXPS_DIR which is exported by env.sh. The array
# values are evaluated at assignment time, so the assignment must run AFTER
# `source env.sh`. We therefore wrap the assignment in _build_versions(),
# called from both the login-node wrapper and the compute-node section.
# Top-level reference matrix (commented for at-a-glance reading):
#   c1_cotres_40k   $EXPS_DIR/owt2/cotformer_full_depth/cotformer_full_depth_res_only_lr0.001_bs8x16_seqlen256/ckpt_40000.pt
#   c2_lncot_40k    $EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256/ckpt_40000.pt
#   c3_lncot_60k    $EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256/ckpt_60000.pt
#   c4_mod_40k      $EXPS_DIR/owt2/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_lr0.001_bs8x16_seqlen256/ckpt_40000.pt
#   c5_adm_v2_60k   $EXPS_DIR/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256/ckpt_60000.pt
_build_versions() {
    VERSIONS=(
        "c1_cotres_40k|$EXPS_DIR/owt2/cotformer_full_depth/cotformer_full_depth_res_only_lr0.001_bs8x16_seqlen256|ckpt_40000.pt"
        "c2_lncot_40k|$EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256|ckpt_40000.pt"
        "c3_lncot_60k|$EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256|ckpt_60000.pt"
        "c4_mod_40k|$EXPS_DIR/owt2/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_lr0.001_bs8x16_seqlen256|ckpt_40000.pt"
        "c5_adm_v2_60k|$EXPS_DIR/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256|ckpt_60000.pt"
    )
}

# Inter-batch robustness batch count and per-batch token budget (§1.2 RQ1).
N_BATCHES=4
MAX_TOKENS=2048
SEQ_LENGTH=256
BATCH_SIZE=8
SEED=2357

# Protocol G tag whitelist per §1.5 Matrix row "G: KV Compress." -- runs on
# C1 (cotres_40k), C3 (lncot_60k), C5 (adm_v2_60k). Other tags fall through
# unchanged.
KV_RANK_TAGS_REGEX='^(c1_cotres_40k|c3_lncot_60k|c5_adm_v2_60k)$'

# Reference MLA kv_lora_rank for the Prediction-1 comparison plot line.
KV_TARGET_RANK=192

# Dataset root on /scratch; used by --capture-kv to bypass a possibly
# absent config.data_dir in the checkpoint's summary.json. Empty means
# "use config.data_dir".
DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/$USER/datasets}"

# Protocol H (interpolation_validity / RQ7) tag whitelist per §1.5 Analysis
# Matrix row "H: Interp. Validity" -- runs only on C4 (MoD baseline) and C5
# (ADM v2). Other tags fall through unchanged.
INTERP_VALIDITY_TAGS_REGEX='^(c4_mod_40k|c5_adm_v2_60k)$'
# Clone-set size and permutation count per docs/extend-notes.md §1.2 RQ7
# (DEC-026: N=10000 permutations).
INTERP_CLONE_SIZE=64
INTERP_N_PERMUTATIONS=10000

# Protocol I (depth_emb_freeze / RQ8) tag whitelist per §1.5 Matrix row
# "I: Depth-Emb Freeze" -- runs only on C3 (LN-CoTFormer at 60k). Other
# tags fall through unchanged.
DEPTH_FREEZE_TAGS_REGEX='^c3_lncot_60k$'
# Bootstrap count per docs/extend-notes.md §1.2 RQ8 (paired-bootstrap p<0.01).
DEPTH_FREEZE_N_BOOTSTRAP=10000
# 8-condition spec per docs/extend-notes.md §1.2 RQ8: Cartesian product of
# {freeze, unfreeze} x {zero, preserve, random} plus two matched controls
# (DEC-026 zero-vector condition addition). Each invocation specifies one
# (freeze_mode, freeze_target) pair; the target index uses the reverse-
# index convention of the forward loop at
# models/cotformer_full_depth_lnmid_depthemb.py:354 (see protocol docstring).
DEPTH_FREEZE_CONDITIONS=(
    "freeze|zero|1"
    "freeze|preserve|1"
    "freeze|random|1"
    "unfreeze|zero|1"
    "unfreeze|preserve|1"
    "unfreeze|random|1"
    "freeze|zero|2"
    "freeze|preserve|2"
)

# Protocol D-calibration (RQ5 entropy calibration) substrate config per
# docs/extend-notes.md §1.3 Protocol D-cal three-tier substrate. Tier 1
# is GPT-2-large (HF identifier; resolved via $HF_HOME by transformers on
# the compute node -- the login-node script cache_huggingface_model.py
# pre-populates the cache). Tier 2 is the ADM C5 checkpoint already in
# VERSIONS. Tier 3 is contingent and NOT wired (per directive).
DCAL_TIER1_CKPT="gpt2-large"
DCAL_TIER2_TAG="c5_adm_v2_60k"
# Primary-ladder n-per-condition default per the protocol docstring;
# AMBIGUOUS-verdict retry uses 4000 and is invoked manually.
DCAL_N_PER_CONDITION=1000

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"
    _build_versions  # $EXPS_DIR now populated; build VERSIONS for login-node use.

    # Pre-flight: verify every checkpoint exists. Fail fast before queueing.
    MISSING=0
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag v_dir v_file <<< "$version_entry"
        if [ ! -f "$v_dir/summary.json" ]; then
            echo "ERROR: $v_tag summary.json not found at $v_dir"
            MISSING=$((MISSING+1))
        fi
        if [ ! -f "$v_dir/$v_file" ]; then
            echo "ERROR: $v_tag $v_file not found at $v_dir"
            MISSING=$((MISSING+1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo ""
        echo "$MISSING checkpoint(s) missing. Update VERSIONS in $0."
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")

    # Create run_N/<tag>/ layout on login node
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag _ _ <<< "$version_entry"
        mkdir -p "$RUN_DIR/$v_tag"
    done

    echo "=== analyze-lncot+adm GPU ==="
    echo "  Versions:    $(IFS=,; echo "${VERSIONS[*]}" | sed 's/|[^,]*//g')"
    echo "  N batches:   $N_BATCHES"
    echo "  Max tokens:  $MAX_TOKENS per batch"
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

# --- Actual job (runs on compute node) ---
set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
fi

source "$REPO_DIR/iridis/env.sh"
_build_versions  # $EXPS_DIR now populated on the compute node; rebuild VERSIONS.

# Defensive cache + scratch mkdirs: env.sh exports the path vars but
# does not create directories. Models load tiktoken("gpt2") at construct
# time (models/base.py); compute nodes have no internet, so the cache
# must already be populated. If TIKTOKEN_CACHE_DIR is empty the run
# will fail at first model construction; in that case re-run
#   python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
# on a login node with conda activated, then resubmit.
mkdir -p "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

echo "========================================="
echo " analyze-lncot+adm GPU stage"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " GPU:           1x L4"
echo " Job ID:        $SLURM_JOB_ID"
echo " Started:       $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# ========================= VERSION LOOP =====================================
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r TAG CKPT_DIR CKPT_FILE <<< "$version_entry"

    TAG_RUN_DIR="$RUN_DIR/$TAG"
    mkdir -p "$TAG_RUN_DIR"

    WORKSPACE="/scratch/$USER/analysis_workspace/$TAG"
    mkdir -p "$WORKSPACE"

    echo ""
    echo "###################################################################"
    echo "   $TAG  ($CKPT_DIR/$CKPT_FILE)"
    echo "###################################################################"

    echo ""
    echo "--- Protocol A: Logit Lens (stage 1 collector + stage 3 metrics) ---"
    python -m analysis.logit_lens \
        --checkpoint "$CKPT_DIR" \
        --checkpoint-file "$CKPT_FILE" \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" \
        --n-batches "$N_BATCHES" \
        --seq-length "$SEQ_LENGTH" \
        --batch-size "$BATCH_SIZE" \
        --device cuda

    echo "  Logit Lens outputs:"
    echo "    $TAG_RUN_DIR/logit_lens_results.json"
    echo "    $TAG_RUN_DIR/logit_lens_trajectories.png"
    echo "  Workspace:"
    echo "    $WORKSPACE/"

    # --- Protocol G Type B: activation-level KV rank + KV-CoRE NER ---
    # Runs only on the §1.5 Matrix G row (C1, C3, C5). --capture-kv
    # triggers the KV_C_ATTN collector pass when the workspace is missing
    # the kv_mid_l<L>_r<R>.npy files (it skips the capture when the files
    # are already present, which is the normal re-run case).
    if [[ "$TAG" =~ $KV_RANK_TAGS_REGEX ]]; then
        echo ""
        echo "--- Protocol G Type B: KV activation rank + NER ---"
        if [ -n "$DATA_DIR_OVERRIDE" ] && [ -d "$DATA_DIR_OVERRIDE" ]; then
            KV_DATA_DIR_ARG=(--data-dir "$DATA_DIR_OVERRIDE")
        else
            KV_DATA_DIR_ARG=()
        fi
        python -m analysis.kv_rank \
            --checkpoint "$CKPT_DIR" \
            --checkpoint-file "$CKPT_FILE" \
            --workspace "$WORKSPACE" \
            --output-dir "$TAG_RUN_DIR" \
            --seed "$SEED" \
            --target-rank "$KV_TARGET_RANK" \
            --capture-kv \
            --device cuda \
            --seq-length "$SEQ_LENGTH" \
            --batch-size "$BATCH_SIZE" \
            --max-tokens "$MAX_TOKENS" \
            "${KV_DATA_DIR_ARG[@]}"

        echo "  Protocol G outputs:"
        echo "    $TAG_RUN_DIR/kv_rank_results.json"
        echo "    $TAG_RUN_DIR/kv_rank_plots.png"
    else
        echo ""
        echo "  Skipping Protocol G for $TAG (not in §1.5 Matrix G row)"
    fi

    # --- Protocol C: attention taxonomy + sparsity-across-repeat test ---
    # §1.3 Protocol C row: runs on C1, C2, C3, C4, C5 (all five). The
    # collector flips attn.flash=False and installs a per-attn monkey-
    # patch that re-runs the Q/K math path to recover att post-softmax;
    # see analysis.common.collector._install_attn_monkey_patch and
    # docs/extend-technical.md §8.6 for the rationale and failure modes.
    # Compute: ~4x non-flash slowdown + ~2x Q/K re-compute cost; per
    # checkpoint ~9 min L4 at max_tokens=2048, seq_length=256.
    echo ""
    echo "--- Protocol C: Attention taxonomy (non-flash ATTN_WEIGHTS) ---"
    if [ -n "$DATA_DIR_OVERRIDE" ] && [ -d "$DATA_DIR_OVERRIDE" ]; then
        PROTC_DATA_DIR_ARG=(--data-dir "$DATA_DIR_OVERRIDE")
    else
        PROTC_DATA_DIR_ARG=()
    fi
    python -m analysis.attention_taxonomy \
        --checkpoint "$CKPT_DIR" \
        --checkpoint-file "$CKPT_FILE" \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" \
        --seq-length "$SEQ_LENGTH" \
        --batch-size "$BATCH_SIZE" \
        --device cuda \
        "${PROTC_DATA_DIR_ARG[@]}"

    echo "  Protocol C outputs:"
    echo "    $TAG_RUN_DIR/attention_taxonomy_results.json"
    echo "    $TAG_RUN_DIR/attention_taxonomy_heatmap.png"
    echo "    $TAG_RUN_DIR/attention_sparsity_trajectory.png"

    # --- Protocol H: Interpolation Validity (RQ7) ---
    # §1.5 Matrix row "H: Interp. Validity" runs on C4 + C5 only (the
    # MoD/ADM checkpoints with adaptive halting). Permutation count per
    # docs/extend-notes.md §1.2 RQ7 / DEC-026 (10000). Per the protocol
    # docstring the analysis emits halting-depth statistics + clone-set
    # entropy + permutation p-value; ~10 min L4 per ckpt at clone-size=64.
    if [[ "$TAG" =~ $INTERP_VALIDITY_TAGS_REGEX ]]; then
        echo ""
        echo "--- Protocol H: Interpolation Validity (RQ7) ---"
        python -m analysis.interpolation_validity \
            --checkpoint "$CKPT_DIR" \
            --checkpoint-file "$CKPT_FILE" \
            --workspace "$WORKSPACE" \
            --output-dir "$TAG_RUN_DIR" \
            --seed "$SEED" \
            --clone-size "$INTERP_CLONE_SIZE" \
            --n-permutations "$INTERP_N_PERMUTATIONS"

        echo "  Protocol H outputs:"
        echo "    $TAG_RUN_DIR/interpolation_validity_results.json"
        echo "    $TAG_RUN_DIR/interpolation_validity_per_layer.png"
    else
        echo ""
        echo "  Skipping Protocol H for $TAG (not in §1.5 Matrix H row)"
    fi

    # --- Protocol I: Depth-Embedding 8-Condition Freeze Ablation (RQ8) ---
    # §1.5 Matrix row "I: Depth-Emb Freeze" runs on C3 only (LN-CoTFormer
    # 60k -- the only checkpoint where the depth_embedding entry has been
    # learned to convergence). The 8 conditions are run sequentially; each
    # invocation specifies one (mode, target) pair and emits its own
    # per-condition perplexity + bootstrap CI. Aggregate ANOVA is computed
    # CPU-side from the per-condition JSON artefacts. ~21 min L4 across
    # all 8 conditions per docs/extend-notes.md §1.8.
    if [[ "$TAG" =~ $DEPTH_FREEZE_TAGS_REGEX ]]; then
        echo ""
        echo "--- Protocol I: Depth-Embedding Freeze (RQ8) ---"
        for condition_entry in "${DEPTH_FREEZE_CONDITIONS[@]}"; do
            IFS='|' read -r FREEZE_MODE_RAW FREEZE_KIND FREEZE_TARGET <<< "$condition_entry"
            COND_DIR="$TAG_RUN_DIR/depth_emb_freeze/${FREEZE_MODE_RAW}_${FREEZE_KIND}_t${FREEZE_TARGET}"
            mkdir -p "$COND_DIR"
            # Map FREEZE_MODE_RAW (freeze | unfreeze) onto the protocol's
            # --freeze-active flag so each row of DEPTH_FREEZE_CONDITIONS
            # maps to a distinct intervention. Without this, the unfreeze
            # rows collapse onto the freeze rows (8 conditions -> 6).
            if [ "$FREEZE_MODE_RAW" = "freeze" ]; then
                FREEZE_ACTIVE=true
            else
                FREEZE_ACTIVE=false
            fi
            echo "    -> condition: ${FREEZE_MODE_RAW} / ${FREEZE_KIND} / target=${FREEZE_TARGET}"
            python -m analysis.depth_emb_freeze \
                --checkpoint "$CKPT_DIR" \
                --checkpoint-file "$CKPT_FILE" \
                --workspace "$WORKSPACE" \
                --output-dir "$COND_DIR" \
                --seed "$SEED" \
                --freeze-mode "$FREEZE_KIND" \
                --freeze-target "$FREEZE_TARGET" \
                --freeze-active "$FREEZE_ACTIVE" \
                --n-bootstrap "$DEPTH_FREEZE_N_BOOTSTRAP"
        done

        echo "  Protocol I outputs (per condition):"
        echo "    $TAG_RUN_DIR/depth_emb_freeze/<mode>_<kind>_t<idx>/depth_emb_freeze_results.json"
    else
        echo ""
        echo "  Skipping Protocol I for $TAG (not in §1.5 Matrix I row)"
    fi

done  # end VERSION LOOP

# ========================= PROTOCOL D-CALIBRATION =============================
# Substrate-specific (not per-checkpoint inside the VERSIONS loop). Runs the
# Tier 1 (GPT-2-large) + Tier 2 (ADM C5) forward passes per the three-tier
# substrate ladder in docs/extend-notes.md §1.3. Each tier writes its own
# metrics.csv + spearman.json + classifier.json + figure.png to a tier-
# specific subdir of $RUN_DIR/d_cal/. The CPU stage (job-cpu.sh) consumes
# these artefacts in its 4-gate verdict aggregation. Tier 3 is contingent
# per the protocol docstring and is NOT wired here.
DCAL_RUN_DIR="$RUN_DIR/d_cal"
mkdir -p "$DCAL_RUN_DIR/tier1" "$DCAL_RUN_DIR/tier2"

# Resolve Tier 2 substrate path from VERSIONS (single source of truth).
DCAL_TIER2_CKPT_DIR=""
DCAL_TIER2_CKPT_FILE=""
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag v_dir v_file <<< "$version_entry"
    if [ "$v_tag" = "$DCAL_TIER2_TAG" ]; then
        DCAL_TIER2_CKPT_DIR="$v_dir"
        DCAL_TIER2_CKPT_FILE="$v_file"
        break
    fi
done
if [ -z "$DCAL_TIER2_CKPT_DIR" ]; then
    echo "ERROR: Tier 2 substrate '$DCAL_TIER2_TAG' not found in VERSIONS; D-cal cannot run."
    exit 1
fi

echo ""
echo "###################################################################"
echo "   Protocol D-calibration (RQ5)"
echo "###################################################################"

echo ""
echo "--- D-calibration Tier 1: $DCAL_TIER1_CKPT (GPT-2-large) ---"
# Tier 1 module-path is model.transformer.h (the GPT-2 layer stack);
# Tiers 2 and 3 use model.transformer.h_mid per the protocol docstring.
python -m analysis.calibration.entropy_calibration \
    --ckpt "$DCAL_TIER1_CKPT" \
    --tier 1 \
    --n-per-condition "$DCAL_N_PER_CONDITION" \
    --seed "$SEED" \
    --out "$DCAL_RUN_DIR/tier1" \
    --module-path model.transformer.h

echo "  Tier 1 outputs:"
echo "    $DCAL_RUN_DIR/tier1/metrics.csv"
echo "    $DCAL_RUN_DIR/tier1/spearman.json"
echo "    $DCAL_RUN_DIR/tier1/classifier.json"
echo "    $DCAL_RUN_DIR/tier1/figure.png"

echo ""
echo "--- D-calibration Tier 2: $DCAL_TIER2_TAG (ADM C5) ---"
python -m analysis.calibration.entropy_calibration \
    --ckpt "$DCAL_TIER2_CKPT_DIR/$DCAL_TIER2_CKPT_FILE" \
    --tier 2 \
    --n-per-condition "$DCAL_N_PER_CONDITION" \
    --seed "$SEED" \
    --out "$DCAL_RUN_DIR/tier2" \
    --module-path model.transformer.h_mid

echo "  Tier 2 outputs:"
echo "    $DCAL_RUN_DIR/tier2/metrics.csv"
echo "    $DCAL_RUN_DIR/tier2/spearman.json"
echo "    $DCAL_RUN_DIR/tier2/classifier.json"
echo "    $DCAL_RUN_DIR/tier2/figure.png"

echo ""
echo "  D-cal aggregation (4-gate verdict) runs CPU-side in job-cpu.sh."
echo "  Tier 3 is contingent per protocol docstring; not wired."

# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " analyze-lncot+adm GPU stage complete: $(date)"
echo " Results tree: $RUN_DIR/"
echo ""
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ _ <<< "$version_entry"
    echo "   $v_tag/"
    echo "     logit_lens_results.json"
    echo "     logit_lens_trajectories.png"
    echo "     attention_taxonomy_results.json"
    echo "     attention_taxonomy_heatmap.png"
    echo "     attention_sparsity_trajectory.png"
done
echo ""
echo " Workspace (intermediate, NOT in run_N):"
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ _ <<< "$version_entry"
    echo "   /scratch/$USER/analysis_workspace/$v_tag/"
    if [[ "$v_tag" =~ $KV_RANK_TAGS_REGEX ]]; then
        echo "     (+ kv_mid_l<L>_r<R>.npy from Protocol G capture)"
    fi
    echo "     (+ attn_weights_mid_l<L>_r<R>.npy from Protocol C capture)"
done
echo ""
echo " D-calibration substrate outputs (not per-checkpoint):"
echo "   $RUN_DIR/d_cal/tier1/  (GPT-2-large)"
echo "   $RUN_DIR/d_cal/tier2/  (ADM C5)"
echo ""
echo " Next step: submit job-cpu.sh for Tuned Lens + Protocol B/E/F/G-weight"
echo " + Protocol D (router) + D-cal aggregation."
echo "========================================="
