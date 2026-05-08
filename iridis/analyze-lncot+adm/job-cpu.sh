#!/bin/bash
#SBATCH --job-name=analyze_lncot_adm_cpu
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
################################################################################
# analyze-lncot+adm CPU stage -- Protocol A-ext (Tuned Lens) + Protocol E
# (Residual Diagnostics) + Protocol B (CKA) + Protocol F (Eff. Dim) +
# Protocol G weight-level (Type A) + Protocol D (Router Analysis) +
# Protocol D-calibration 4-gate aggregation. Synthesis is local-only
# (run `make synthesis` from the repo root after rsync'ing run_N/).
#
# Wave 1 + 2b + 2c deliverable (MVP observational bundle).
# Depends on job-gpu.sh having produced the per-tag workspace on /scratch
# containing residual_mid_l{L}_r{R}.npy, residual_ln_mid_r{R}.npy,
# residual_pre_ln_f.npy, targets.npy, and meta.json. Protocol G-activation
# (Type B) is GPU-side and has already populated kv_mid_l{L}_r{R}.npy
# for C1, C3, C5 when this script runs; Protocol C (GPU-side) has
# populated attn_weights_mid_l{L}_r{R}.npy for every checkpoint.
#
# Tuned Lens training is the dominant cost (~5 h CPU for 105 translators
# per checkpoint per §1.3 Protocol A-ext). Protocol E is cheap (~5 min CPU).
# Protocol B adds ~15 h CPU per checkpoint (5460 pairs x 1000-permutation
# null over debiased HSIC U-statistic); parallelise via --workers if the
# SLURM allocation permits. Protocol F is cheap (~5 min CPU); Protocol G
# weight-level (Type A) is ~1 min CPU per checkpoint. Protocol D runs a
# forward pass on --device cpu for the ROUTER_LOGITS capture (short-circuits
# to not_applicable on non-ADM tags C1/C2/C3) plus a pure-numpy analysis;
# wall time per ADM checkpoint is ~5 min CPU for the capture + ~10 seconds
# for the Jaccard and per-router metrics.
#
# Output structure:
#   run_N/
#     slurm_<jobid>.out                           SLURM stdout
#     slurm_<jobid>.err                           SLURM stderr
#     <tag>/
#       tuned_lens_translators.pt                 105 translator state dicts
#       tuned_lens_diagnostic.json                21x5 ||A-I||_F + bands
#       tuned_lens_triangulation.json             Spearman rho + cross-lens KL
#       tuned_lens_identity_grid.png              diagnostic heatmap
#       residual_diagnostics_results.json         L2 / var / L_inf grids
#       residual_diagnostics_mean_l2.png          mean L2 per (layer, repeat)
#       residual_diagnostics_var_l2.png           variance per (layer, repeat)
#       residual_diagnostics_linf.png             L_inf per (layer, repeat)
#
# Usage:
#   cd ~/CoTFormer && bash iridis/analyze-lncot+adm/job-cpu.sh
################################################################################

# ========================= CONFIGURATION ====================================

# Version matrix: "tag|ckpt_dir_absolute_path|ckpt_file|run_residual_diag"
# run_residual_diag is "yes" or "no" per §1.5 Matrix: Protocol E runs on
# C1/C2/C3 only (the observational CoTFormer + LN-CoTFormer checkpoints);
# C4/C5 are ADM / MoD and not in the Protocol E matrix row.
#
# IMPORTANT: VERSIONS uses $EXPS_DIR which is exported by env.sh. The array
# values are evaluated at assignment time, so the assignment must run AFTER
# `source env.sh`. We therefore wrap the assignment in _build_versions(),
# called from both the login-node wrapper and the compute-node section.
# Top-level reference matrix (commented for at-a-glance reading):
#   c1_cotres_40k   .../cotformer_full_depth/.../ckpt_40000.pt           run_residual_diag=yes
#   c2_lncot_40k    .../cotformer_full_depth_lnmid_depthemb/.../ckpt_40000.pt  run_residual_diag=yes
#   c3_lncot_60k    .../cotformer_full_depth_lnmid_depthemb/.../ckpt_60000.pt  run_residual_diag=yes
#   c4_mod_40k      .../but_mod_efficient_sigmoid_lnmid_depthemb_random_factor/.../ckpt_40000.pt  run_residual_diag=no
#   c5_adm_v2_60k   .../adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/.../ckpt_60000.pt  run_residual_diag=no
_build_versions() {
    VERSIONS=(
        "c1_cotres_40k|$EXPS_DIR/owt2/cotformer_full_depth/cotformer_full_depth_res_only_lr0.001_bs8x16_seqlen256|ckpt_40000.pt|yes"
        "c2_lncot_40k|$EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256|ckpt_40000.pt|yes"
        "c3_lncot_60k|$EXPS_DIR/owt2/cotformer_full_depth_lnmid_depthemb/cotformer_full_depth_lnmid_depthemb_lr0.001_bs8x16_seqlen256|ckpt_60000.pt|yes"
        "c4_mod_40k|$EXPS_DIR/owt2/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_lr0.001_bs8x16_seqlen256|ckpt_40000.pt|no"
        "c5_adm_v2_60k|$EXPS_DIR/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final/adm_v2_lr0.001_bs8x16_seqlen256|ckpt_60000.pt|no"
    )
}

TRANSLATOR_EPOCHS=250
TRANSLATOR_BATCH_SIZE=64
TRANSLATOR_LR=1.0
TRANSLATOR_MOMENTUM=0.9
SEED=2357
CONVERGENCE_GATE_SITES=95

# Protocol B (CKA): 1000-permutation null per pair. Scale N_PERMUTATIONS down
# when iterating in a shorter SLURM allocation (every reduction in permutation
# count widens the per-pair zone thresholds; 1000 is the pre-registered value).
CKA_N_PERMUTATIONS=1000

# Protocol B workers: share the SBATCH --cpus-per-task budget (16) minus a
# small reserve for numpy BLAS oversubscription. Override CKA_WORKERS env
# var when profiling.
CKA_WORKERS="${CKA_WORKERS:-12}"

# Protocol F (effective dim): top-k PCs to remove for the rogue-direction
# cross-check. 2 is the §1.2 RQ6 "Rogue dimensions" default.
EFFDIM_TOP_K_PC_REMOVE=2
EFFDIM_ALPHA_FAMILY=0.01

# Protocol G tag whitelist for weight-only (Type A) pass per §1.5 Matrix row.
KV_RANK_TAGS_REGEX='^(c1_cotres_40k|c3_lncot_60k|c5_adm_v2_60k)$'
KV_TARGET_RANK=192

# Protocol D (router analysis) capture parameters. The forward pass runs
# on --device cpu; max_tokens and seq_length control the per-run data budget
# and per-batch sequence length respectively. Match the defaults used in
# job-gpu.sh so the per-checkpoint budget is consistent across stages.
MAX_TOKENS=2048
SEQ_LENGTH=256
BATCH_SIZE=8

# Protocol D-calibration (RQ5) aggregation: consumes the Tier 1 + Tier 2
# metrics.csv produced GPU-side and runs the Spearman gate + linear-
# classifier fit + 4-gate verdict per docs/extend-notes.md §1.3 D-cal
# verdict table. The per-tier output dirs live under <gpu-run-N>/d_cal/;
# we resolve the most recent sibling run_N when this CPU job's own
# $RUN_DIR is empty (mirrors the convention used for logit_lens_results.json).
DCAL_N_PER_CONDITION=1000

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"
    _build_versions  # $EXPS_DIR now populated; build VERSIONS for login-node use.

    # Pre-flight: verify workspaces exist on /scratch.
    MISSING=0
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag _ _ _ <<< "$version_entry"
        WORKSPACE_CHECK="/scratch/$USER/analysis_workspace/$v_tag"
        if [ ! -f "$WORKSPACE_CHECK/meta.json" ]; then
            echo "ERROR: $v_tag workspace meta.json not found at $WORKSPACE_CHECK/"
            MISSING=$((MISSING+1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo ""
        echo "$MISSING workspace(s) missing. Run job-gpu.sh first to produce the"
        echo "Stage-1 residual cache that this script consumes."
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")

    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag _ _ _ <<< "$version_entry"
        mkdir -p "$RUN_DIR/$v_tag"
    done

    echo "=== analyze-lncot+adm CPU ==="
    echo "  Versions:            $(IFS=,; echo "${VERSIONS[*]}" | sed 's/|[^,]*//g')"
    echo "  Translator epochs:   $TRANSLATOR_EPOCHS"
    echo "  Translator batch:    $TRANSLATOR_BATCH_SIZE tokens per step"
    echo "  Logs:                $RUN_DIR/"
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
# does not create directories. Same rationale as job-gpu.sh; this CPU
# stage may also load tiktoken via Protocol D's CPU forward pass on
# OWT2 ckpts.
mkdir -p "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

echo "========================================="
echo " analyze-lncot+adm CPU stage"
echo " User:          $USER"
echo " Node:          $(hostname)"
echo " Job ID:        $SLURM_JOB_ID"
echo " Started:       $(date)"
echo "========================================="

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

cd "$REPO_DIR"

# ========================= VERSION LOOP =====================================
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r TAG CKPT_DIR CKPT_FILE RUN_RESIDUAL <<< "$version_entry"

    TAG_RUN_DIR="$RUN_DIR/$TAG"
    WORKSPACE="/scratch/$USER/analysis_workspace/$TAG"
    LOGIT_LENS_JSON="$TAG_RUN_DIR/logit_lens_results.json"

    # Allow the logit_lens_results.json to live in the Stage-1 (GPU) run dir.
    # If it lives elsewhere, override this path in your invocation.
    if [ ! -f "$LOGIT_LENS_JSON" ]; then
        GPU_RUN_GUESS="$(dirname "$RUN_DIR")/$(basename "$RUN_DIR" | sed 's/^run_/run_/')"
        # Best-effort: try the most recent sibling run_N with the tag file
        CANDIDATE=$(ls -1d "$PACKAGE_DIR"/run_*/"$TAG"/logit_lens_results.json 2>/dev/null | tail -1 || true)
        if [ -n "$CANDIDATE" ]; then
            LOGIT_LENS_JSON="$CANDIDATE"
        fi
    fi

    echo ""
    echo "###################################################################"
    echo "   $TAG"
    echo "   ckpt:        $CKPT_DIR/$CKPT_FILE"
    echo "   workspace:   $WORKSPACE"
    echo "   residual_diag: $RUN_RESIDUAL"
    echo "###################################################################"

    echo ""
    echo "--- Protocol A-ext: Tuned Lens (Belrose 2023 canonical) ---"

    if [ -f "$LOGIT_LENS_JSON" ]; then
        TUNED_LOGIT_ARG=(--logit-lens-results "$LOGIT_LENS_JSON")
    else
        TUNED_LOGIT_ARG=()
        echo "  WARNING: no logit_lens_results.json at $LOGIT_LENS_JSON;"
        echo "  cross-lens triangulation will be skipped (diagnostic + "
        echo "  translators are still produced)."
    fi

    python -m analysis.tuned_lens \
        --checkpoint "$CKPT_DIR" \
        --checkpoint-file "$CKPT_FILE" \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --translator-epochs "$TRANSLATOR_EPOCHS" \
        --translator-batch-size "$TRANSLATOR_BATCH_SIZE" \
        --translator-lr "$TRANSLATOR_LR" \
        --translator-momentum "$TRANSLATOR_MOMENTUM" \
        --convergence-gate-sites "$CONVERGENCE_GATE_SITES" \
        --device cpu \
        --optimiser sgd-nesterov \
        "${TUNED_LOGIT_ARG[@]}"

    echo "  Tuned Lens outputs:"
    echo "    $TAG_RUN_DIR/tuned_lens_translators.pt"
    echo "    $TAG_RUN_DIR/tuned_lens_diagnostic.json"
    echo "    $TAG_RUN_DIR/tuned_lens_triangulation.json"
    echo "    $TAG_RUN_DIR/tuned_lens_identity_grid.png"

    if [ "$RUN_RESIDUAL" = "yes" ]; then
        echo ""
        echo "--- Protocol E: Residual Diagnostics ---"
        python -m analysis.residual_diagnostics \
            --workspace "$WORKSPACE" \
            --output-dir "$TAG_RUN_DIR" \
            --seed "$SEED"

        echo "  Residual Diagnostics outputs:"
        echo "    $TAG_RUN_DIR/residual_diagnostics_results.json"
        echo "    $TAG_RUN_DIR/residual_diagnostics_{mean_l2,var_l2,linf}.png"
    fi

    # --- Protocol B: debiased HSIC U-statistic CKA (RQ2) ---
    # Runs on every §1.5 Matrix B row entry (C1-C5). Consumes the residual
    # cache only; no model or data dependencies.
    echo ""
    echo "--- Protocol B: debiased CKA matrix (RQ2) ---"
    python -m analysis.cka \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --n-permutations "$CKA_N_PERMUTATIONS" \
        --workers "$CKA_WORKERS"

    echo "  Protocol B outputs:"
    echo "    $TAG_RUN_DIR/cka_results.json"
    echo "    $TAG_RUN_DIR/cka_heatmap.png"

    # --- Protocol F: effective dimensionality (RQ6) ---
    # Runs on every §1.5 Matrix F row entry (C1-C5). Consumes residual cache.
    echo ""
    echo "--- Protocol F: effective dimensionality (RQ6) ---"
    python -m analysis.effective_dim \
        --workspace "$WORKSPACE" \
        --output-dir "$TAG_RUN_DIR" \
        --seed "$SEED" \
        --top-k-pc-remove "$EFFDIM_TOP_K_PC_REMOVE" \
        --alpha-family "$EFFDIM_ALPHA_FAMILY"

    echo "  Protocol F outputs:"
    echo "    $TAG_RUN_DIR/effective_dim_results.json"
    echo "    $TAG_RUN_DIR/effective_dim_trajectory.png"

    # --- Protocol G weight-level Type A (CPU) ---
    # Runs on §1.5 Matrix G row entries (C1, C3, C5). Type A is model-
    # weights-only (no cache dependency); Type B (activation-level) is
    # GPU-side and lives in job-gpu.sh.
    if [[ "$TAG" =~ $KV_RANK_TAGS_REGEX ]]; then
        echo ""
        echo "--- Protocol G weight-level Type A (KV compressibility) ---"
        # The GPU stage has already written kv_rank_results.json with both
        # Type A + Type B; re-running here would overwrite it. Instead we
        # only enter this branch when the GPU-side JSON is absent (e.g.,
        # the GPU stage did not run for this tag, or was skipped for
        # debugging).
        if [ ! -f "$TAG_RUN_DIR/kv_rank_results.json" ]; then
            python -m analysis.kv_rank \
                --checkpoint "$CKPT_DIR" \
                --checkpoint-file "$CKPT_FILE" \
                --workspace "$WORKSPACE" \
                --output-dir "$TAG_RUN_DIR" \
                --seed "$SEED" \
                --target-rank "$KV_TARGET_RANK" \
                --skip-activations

            echo "  Protocol G Type A outputs:"
            echo "    $TAG_RUN_DIR/kv_rank_results.json"
            echo "    $TAG_RUN_DIR/kv_rank_plots.png"
        else
            echo "  Protocol G results already present from GPU stage; skipping"
        fi
    fi

    # --- Protocol D: Router policy analysis (ADM C4/C5 only; else short-circuits) ---
    # Per docs/extend-notes.md §1.3 Protocol D row: runs on every checkpoint;
    # short-circuits with verdict=not_applicable on C1/C2/C3 (no
    # transformer.mod router stack). Capture stage runs a forward pass
    # with the ROUTER_LOGITS site on --device cpu; the analysis stage
    # computes per-router decisiveness / Bernoulli entropy / continue
    # rate plus the full Jaccard overlap matrix. --record-workspace-path
    # embeds the workspace absolute path in the result JSON so a later
    # cross-version run (e.g. C5 v2 against C4 v1) can rebuild the raw
    # continue masks for the stability verdict.
    echo ""
    echo "--- Protocol D: Router policy analysis (ADM only; short-circuits otherwise) ---"
    PROTD_ARGS=(
        --checkpoint "$CKPT_DIR"
        --checkpoint-file "$CKPT_FILE"
        --workspace "$WORKSPACE"
        --output-dir "$TAG_RUN_DIR"
        --seed "$SEED"
        --max-tokens "$MAX_TOKENS"
        --seq-length "$SEQ_LENGTH"
        --batch-size "$BATCH_SIZE"
        --device cpu
        --record-workspace-path
    )
    python -m analysis.router_analysis "${PROTD_ARGS[@]}"

    echo "  Protocol D outputs:"
    echo "    $TAG_RUN_DIR/router_analysis_results.json"
    echo "    $TAG_RUN_DIR/router_policy_heatmap.png    (if ADM)"
    echo "    $TAG_RUN_DIR/router_decisiveness_bars.png (if ADM)"

done  # end VERSION LOOP

# ========================= PROTOCOL D-CALIBRATION AGGREGATION ================
# 4-gate verdict (Spearman + classifier + Tier 1 / Tier 2 triangulation +
# baseline accuracy) consuming the Tier 1 + Tier 2 metrics.csv emitted
# by job-gpu.sh. Per the protocol docstring the verdict logic decides
# whether DV-2 (paired probe) is canonical or replaced by the
# TOP-K-MASS fallback; the AMBIGUOUS path triggers a 4000-per-condition
# retry (manual, not wired here).
#
# The GPU stage writes to <gpu-RUN_DIR>/d_cal/tier{1,2}/. Since this CPU
# job has its own RUN_DIR, we resolve the GPU-side d_cal/ via the most
# recent sibling run_N convention (same approach the per-tag stages use
# for logit_lens_results.json above).
echo ""
echo "###################################################################"
echo "   Protocol D-calibration: 4-gate verdict aggregation"
echo "###################################################################"

# Locate the most recent GPU-side d_cal/ dir. First try our own RUN_DIR
# (in case a future scheduler co-locates GPU and CPU outputs). Then walk
# sibling run_*/ in reverse order picking the first that has tier1/ + tier2/.
DCAL_GPU_RUN=""
if [ -d "$RUN_DIR/d_cal/tier1" ] && [ -d "$RUN_DIR/d_cal/tier2" ]; then
    DCAL_GPU_RUN="$RUN_DIR/d_cal"
else
    # Most recent first: ls -t orders by mtime descending.
    for candidate in $(ls -1dt "$PACKAGE_DIR"/run_*/d_cal 2>/dev/null); do
        if [ -d "$candidate/tier1" ] && [ -d "$candidate/tier2" ]; then
            DCAL_GPU_RUN="$candidate"
            break
        fi
    done
fi

if [ -z "$DCAL_GPU_RUN" ]; then
    echo "  WARNING: no sibling run_N/d_cal/ with tier1/ + tier2/ found; skipping"
    echo "  D-cal aggregation. Submit job-gpu.sh first to populate the per-tier"
    echo "  metrics.csv that this stage consumes."
else
    echo "  Tier inputs: $DCAL_GPU_RUN/{tier1,tier2}/"
    # Aggregation reads sibling tier{1,2}/metrics.csv from the same parent
    # dir as --out, then writes spearman + classifier + verdict + figure
    # into <out>/aggregate/. --analysis-only is the contract surface named
    # by the upstream directive (the module body is Phase-2 deferred and
    # will fail-fast with NotImplementedError until then). Only documented
    # flags from the protocol docstring are used (--ckpt and --tier are
    # forward-pass-only and intentionally omitted in the analysis-only path).
    python -m analysis.calibration.entropy_calibration \
        --analysis-only \
        --n-per-condition "$DCAL_N_PER_CONDITION" \
        --seed "$SEED" \
        --out "$DCAL_GPU_RUN"

    echo "  Protocol D-cal aggregation outputs:"
    echo "    $DCAL_GPU_RUN/aggregate/spearman.json"
    echo "    $DCAL_GPU_RUN/aggregate/classifier.json"
    echo "    $DCAL_GPU_RUN/aggregate/verdict.json"
    echo "    $DCAL_GPU_RUN/aggregate/figure.png"
fi

# ========================= SYNTHESIS (LOCAL; NOT IN HPC) =====================
# analysis.synthesis is pure cross-checkpoint plotting + scalar aggregation
# (no GPU, no model forward, no SVD). It is wired as a top-level Makefile
# target rather than an HPC stage; rsync run_N/<tag>/ artefacts back from
# Iridis and run `make synthesis` locally. See the project root Makefile.
echo ""
echo "--- Stage 5 Synthesis (LOCAL; see top-level Makefile) ---"
echo "  Synthesis is local-friendly per analysis/synthesis.py; rsync the"
echo "  run_N/<tag>/ outputs back from Iridis and run \`make synthesis\`."

# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " analyze-lncot+adm CPU stage complete: $(date)"
echo " Results tree: $RUN_DIR/"
echo ""
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ _ _ <<< "$version_entry"
    echo "   $v_tag/"
done
echo "========================================="
