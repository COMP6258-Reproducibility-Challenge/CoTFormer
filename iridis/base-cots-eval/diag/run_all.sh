#!/bin/bash
################################################################################
# run_all.sh -- run Diag A, B, C in sequence and tee output to a single log
#
# Run from anywhere AFTER activating the conda env. Diag B requires GPU + the
# repo on PYTHONPATH (cd to ~/CoTFormer before invoking, or use a SLURM script).
#
# Usage:
#   cd ~/CoTFormer
#   conda activate /scratch/ab3u21/cotformer-env
#   bash iridis/base-cots-eval/diag/run_all.sh
#
# Output: ./diag_<timestamp>.log + stdout. The single log makes it easy to
# rsync the full diagnostic back for review.
################################################################################

set -uo pipefail

DIAG_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="diag_${TS}.log"

CKPT_ROOT="${CKPT_ROOT:-/scratch/ab3u21/exps/owt2/cotformer_full_depth}"
FAILING="${FAILING:-BaseCot_12L_5R}"
WORKING="${WORKING:-BaseCot_24L_5R}"
DATA_DIR="${DATA_DIR:-/scratch/ab3u21/datasets}"

echo "================================================================" | tee -a "$LOG"
echo " Diagnostic sweep: $FAILING (failing) vs $WORKING (working)"     | tee -a "$LOG"
echo " Timestamp: $TS"                                                  | tee -a "$LOG"
echo " Logging to: $LOG"                                                | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo ""                                                                 | tee -a "$LOG"

run_diag () {
    local label="$1"
    shift
    echo ""                                                             | tee -a "$LOG"
    echo "---------------- $label ----------------"                     | tee -a "$LOG"
    echo "+ $*"                                                         | tee -a "$LOG"
    echo ""                                                             | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    local rc="${PIPESTATUS[0]}"
    if [ "$rc" -ne 0 ]; then
        echo ""                                                         | tee -a "$LOG"
        echo "WARN: $label exited with code $rc (continuing anyway)."   | tee -a "$LOG"
    fi
}

run_diag "Diag A -- args diff" \
    python "$DIAG_DIR/diag_a_args_diff.py" \
        --ckpt-root "$CKPT_ROOT" \
        --failing  "$FAILING" \
        --working  "$WORKING"

run_diag "Diag C -- state_dict audit" \
    python "$DIAG_DIR/diag_c_state_dict_audit.py" \
        --ckpt-root "$CKPT_ROOT" \
        --failing  "$FAILING" \
        --working  "$WORKING"

# --- Diag B requires GPU. Detect node type and dispatch accordingly. ---
echo ""                                                                 | tee -a "$LOG"
echo "---------------- Diag B -- n_head hot-swap ----------------"      | tee -a "$LOG"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "GPU detected on this node ($(hostname)) -- running Diag B directly." | tee -a "$LOG"
    echo ""                                                             | tee -a "$LOG"
    run_diag "Diag B (in-process)" \
        python "$DIAG_DIR/diag_b_nhead_hotswap.py" \
            --ckpt-dir "$CKPT_ROOT/$FAILING" \
            --data-dir "$DATA_DIR"
else
    echo "No GPU on $(hostname) (login node?). Submitting Diag B as SLURM job:" | tee -a "$LOG"
    echo "+ sbatch $DIAG_DIR/diag_b_job.sh --ckpt-dir $CKPT_ROOT/$FAILING --data-dir $DATA_DIR" | tee -a "$LOG"
    echo ""                                                             | tee -a "$LOG"
    SBATCH_OUT="$(sbatch "$DIAG_DIR/diag_b_job.sh" \
        --ckpt-dir "$CKPT_ROOT/$FAILING" \
        --data-dir "$DATA_DIR" 2>&1)"
    echo "$SBATCH_OUT"                                                  | tee -a "$LOG"
    JOB_ID="$(echo "$SBATCH_OUT" | awk '{print $NF}')"
    echo ""                                                             | tee -a "$LOG"
    echo "Diag B submitted. When it finishes (~5 min queue + ~1 min wall) read:" | tee -a "$LOG"
    echo "  $DIAG_DIR/diag_b_${JOB_ID}.out" | tee -a "$LOG"
    echo "  $DIAG_DIR/diag_b_${JOB_ID}.err" | tee -a "$LOG"
    echo ""                                                             | tee -a "$LOG"
    echo "Track with:  squeue -u \$USER -j $JOB_ID" | tee -a "$LOG"
fi

echo ""                                                                 | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo " Done. Full log: $(pwd)/$LOG"                                     | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
