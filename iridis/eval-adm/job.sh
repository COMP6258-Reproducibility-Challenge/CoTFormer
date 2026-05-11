#!/bin/bash
#SBATCH --job-name=eval_adm
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
################################################################################
# Adaptive LN-CoTFormer (ADM v1 + v2) -- Evaluation + Figures 4 & 5
#
# Single sequential job evaluating BOTH ADM v1 (older training) and ADM v2
# (newer re-training with corrected RNG-restore) across:
#   - PPL at 40k and 60k checkpoints
#   - Figure 4 (PPL vs MACs Pareto) for each version
#   - Figure 5 experiment matrix isolating methodological confounders
#
# v1 serves as a training-dynamics control for Section 5 interpretation.
# v2 is the primary result (matches authors' reported ADM configuration).
#
# Output structure:
#   run_N/
#     adm_v1/
#       eval_40k.txt, eval_60k.txt               PPL text logs
#       eval_summary_ckpt_{40000,60000}.json     Per-ckpt summary w/ CI
#       exp4/                                    Figure 4 (Pareto)
#         figure4_pareto.png
#         eval_per_threshold.npy, eval_per_layer.npy
#       exp5-original/                           CF + raw dict config
#       exp5-hooks/                              Hooks extraction, raw dict
#       exp5-cf-sorted/                          CF + sorted topk
#       exp5-cf-argparse/                        CF + argparse config
#     adm_v2/                                    (same layout as adm_v1)
#
# Output triage: intermediate 130 MB router_weights_*.npy files are
# written to $CKPT_DIR/eval_workspace/ on /scratch (NOT copied to run_N).
# Only meaningful outputs (PNGs, small scalar .npy sweeps, JSON summaries,
# text logs) land in run_N. This keeps the Git-tracked output tree light
# and the large regeneratable workspace on /scratch.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/eval-adm/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

ADM_BASE="/scratch/$USER/exps/owt2/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final"

# Version matrix: "tag|ckpt_dir_absolute_path"
VERSIONS=(
    "v1|$ADM_BASE/adm_lr0.001_bs8x16_seqlen256"
    "v2|$ADM_BASE/adm_v2_lr0.001_bs8x16_seqlen256"
)

CKPTS=("40000" "60000")

# Experiment matrix: "method|config-loading|output-subdir"
# Each entry isolates one confounder in the Figure 5 reproduction:
#   exp5-original    (cf + raw):        authors' default; saves PI, PT, raw
#   exp5-hooks       (hooks + raw):     forward-hook extraction, no cascade
#   exp5-cf-sorted   (cf-sorted + raw): forces topk(sorted=True) to match
#                                        A100/PyTorch-1.x reordering behaviour
#   exp5-cf-argparse (cf + argparse):   loads config via argparse instead of
#                                        summary.json (tests config-loading delta)
EXPERIMENTS=(
    "cf|raw|exp5-original"
    "hooks|raw|exp5-hooks"
    "cf-sorted|raw|exp5-cf-sorted"
    "cf|argparse|exp5-cf-argparse"
)

# ========================= END CONFIGURATION ================================

# --- Self-submitting wrapper (runs on login node) ---
if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    # Pre-flight: verify every (version, step) checkpoint exists. Fail fast
    # before queueing — saves hours of queue wait if a ckpt is missing.
    MISSING=0
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag v_dir <<< "$version_entry"
        if [ ! -f "$v_dir/summary.json" ]; then
            echo "ERROR: $v_tag summary.json not found at $v_dir"
            MISSING=$((MISSING+1))
        fi
        for step in "${CKPTS[@]}"; do
            ckpt_file="$v_dir/ckpt_${step}.pt"
            if [ ! -f "$ckpt_file" ]; then
                echo "ERROR: $v_tag ckpt_${step}.pt not found at $v_dir"
                MISSING=$((MISSING+1))
            fi
        done
    done
    if [ "$MISSING" -gt 0 ]; then
        echo ""
        echo "$MISSING checkpoint(s) missing. Run the training packages first:"
        echo "  adm-train (v2 config at 60k)"
        exit 1
    fi

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")

    # Create nested run_N/adm_{v1,v2}/{exp4,exp5-*}/ layout on login node
    for version_entry in "${VERSIONS[@]}"; do
        IFS='|' read -r v_tag _ <<< "$version_entry"
        mkdir -p "$RUN_DIR/adm_${v_tag}/exp4"
        for exp_spec in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r _m _c _dirname <<< "$exp_spec"
            mkdir -p "$RUN_DIR/adm_${v_tag}/$_dirname"
        done
    done

    echo "=== ADM v1+v2 Evaluation Matrix ==="
    echo "  Versions:    $(IFS=,; echo "${VERSIONS[*]}" | sed 's/|[^,]*//g')"
    echo "  Steps:       ${CKPTS[*]}"
    echo "  Experiments: ${#EXPERIMENTS[@]} exp5 variants + exp4 per version"
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

echo "========================================="
echo " ADM v1+v2 Evaluation Matrix"
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
    IFS='|' read -r VERSION CKPT_DIR <<< "$version_entry"
    V_RUN_DIR="$RUN_DIR/adm_${VERSION}"

    # Workspace: large intermediate .npy files stay here on /scratch
    # (not copied to run_N — regeneratable from the checkpoint).
    WORKSPACE="$CKPT_DIR/eval_workspace"
    mkdir -p "$WORKSPACE"

    echo ""
    echo "###################################################################"
    echo "   ADM $VERSION  ($CKPT_DIR)"
    echo "###################################################################"


    # ========================= PHASE 1: PPL Evaluation ======================
    echo ""
    echo "--- Phase 1: PPL Evaluation ---"

    for step in "${CKPTS[@]}"; do
        CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
        OUT_FILE="$V_RUN_DIR/eval_${step%000}k.txt"
        echo ""
        echo "  Evaluating ckpt_${step}.pt"
        python eval.py \
            --checkpoint "$CKPT_FILE" \
            --config_format base \
            --distributed_backend None \
            --output_dir "$V_RUN_DIR" \
            2>&1 | tee "$OUT_FILE"
    done


    # ========================= PHASE 2: Figure 4 (once per version) =========
    # Figure 4 depends on router_weights.npy in CKPT_DIR. We generate it via
    # the hooks method (raw scores), since Figure 4's MAC sweep only needs
    # weight values, not their cascade ordering. router_weights.npy (130 MB)
    # is an intermediate workspace file left in $CKPT_DIR (not run_N).
    #
    # get_ppl_per_mac.py and plot_fig4.py HARDCODE reads from $CKPT_DIR for
    # router_weights.npy and eval_per_{threshold,layer}.npy. These cannot
    # be redirected without editing the Python scripts. We accept the
    # constraint: workspace .npy files live in $CKPT_DIR, small meaningful
    # sweep outputs + PNG are copied to run_N/adm_$VERSION/exp4/.
    echo ""
    echo "--- Phase 2: Figure 4 (Pareto) ---"

    EXP4_WORKSPACE="$WORKSPACE/exp4_router_extraction"
    mkdir -p "$EXP4_WORKSPACE"

    python get_router_weights.py \
        --method hooks --config-loading raw \
        --checkpoint "$CKPT_DIR" --split val \
        --output-dir "$EXP4_WORKSPACE" \
        --distributed_backend None

    # Stage the ptcascade extraction as the canonical router_weights.npy
    # for get_ppl_per_mac.py (hardcoded read path).
    cp "$EXP4_WORKSPACE/router_weights_ptcascade.npy" "$CKPT_DIR/router_weights.npy"

    python get_ppl_per_mac.py --checkpoint "$CKPT_DIR"
    python plot_fig4.py \
        --checkpoint "$CKPT_DIR" \
        --output "$V_RUN_DIR/exp4/figure4_pareto.png"

    # Copy MEANINGFUL small outputs to run_N (scalar sweeps, ~1 MB each)
    cp "$CKPT_DIR/eval_per_threshold.npy" "$V_RUN_DIR/exp4/"
    cp "$CKPT_DIR/eval_per_layer.npy" "$V_RUN_DIR/exp4/"

    echo "  Figure 4 saved to $V_RUN_DIR/exp4/"


    # ========================= PHASE 3: Figure 5 experiment matrix ==========
    echo ""
    echo "--- Phase 3: Figure 5 experiment matrix ---"

    for exp_spec in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r METHOD CONFIG_LOADING DIRNAME <<< "$exp_spec"
        EXP_DIR="$V_RUN_DIR/$DIRNAME"
        WORKSPACE_EXP="$WORKSPACE/$DIRNAME"
        mkdir -p "$WORKSPACE_EXP"

        echo ""
        echo "  === $DIRNAME (method=$METHOD, config=$CONFIG_LOADING) ==="

        for step in "${CKPTS[@]}"; do
            CKPT_FILE="$CKPT_DIR/ckpt_${step}.pt"
            STEP_WORKSPACE="$WORKSPACE_EXP/${step%000}k"
            mkdir -p "$STEP_WORKSPACE"

            echo "    Extracting ckpt_${step}.pt -> $STEP_WORKSPACE/"
            python get_router_weights.py \
                --method "$METHOD" \
                --config-loading "$CONFIG_LOADING" \
                --checkpoint "$CKPT_FILE" \
                --split train \
                --output-dir "$STEP_WORKSPACE" \
                --distributed_backend None
        done

        # --- Generate Figure 5 plots ---
        # Read from workspace, write PNG to run_N (meaningful output only).
        W40="$WORKSPACE_EXP/${CKPTS[0]%000}k"
        W60="$WORKSPACE_EXP/${CKPTS[1]%000}k"

        if [ -f "$W40/router_weights_picascade.npy" ]; then
            python plot_fig5.py \
                --weights-40k "$W40/router_weights_picascade.npy" \
                --weights-60k "$W60/router_weights_picascade.npy" \
                --output "$EXP_DIR/figure5_picascade.png"
        fi

        if [ -f "$W40/router_weights_ptcascade.npy" ]; then
            python plot_fig5.py \
                --weights-40k "$W40/router_weights_ptcascade.npy" \
                --weights-60k "$W60/router_weights_ptcascade.npy" \
                --output "$EXP_DIR/figure5_ptcascade.png"
        fi

        if [ -f "$W40/router_weights_raw.npy" ]; then
            python plot_fig5.py \
                --weights-40k "$W40/router_weights_raw.npy" \
                --weights-60k "$W60/router_weights_raw.npy" \
                --output "$EXP_DIR/figure5_raw.png"
        fi

        echo "    Figures saved to $EXP_DIR/"
    done

done  # end VERSION LOOP


# ========================= SUMMARY ==========================================
echo ""
echo "========================================="
echo " ADM v1+v2 Evaluation complete: $(date)"
echo " Results tree: $RUN_DIR/"
echo ""
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r v_tag _ <<< "$version_entry"
    echo "   adm_${v_tag}/"
    echo "     eval_{40,60}k.txt                (PPL logs)"
    echo "     eval_summary_ckpt_{40000,60000}.json"
    echo "     exp4/figure4_pareto.png          (Figure 4 Pareto)"
    for exp_spec in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r _m _c _d <<< "$exp_spec"
        echo "     $_d/figure5_{raw,picascade,ptcascade}.png"
    done
done
echo ""
echo " Workspace (intermediate .npy, NOT in run_N):"
for version_entry in "${VERSIONS[@]}"; do
    IFS='|' read -r _ v_dir <<< "$version_entry"
    echo "   $v_dir/eval_workspace/"
done
echo "========================================="
