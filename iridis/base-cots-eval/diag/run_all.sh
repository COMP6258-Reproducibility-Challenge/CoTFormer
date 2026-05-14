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

run_diag "Diag B -- n_head hot-swap (requires CUDA + repo root on cwd)" \
    python "$DIAG_DIR/diag_b_nhead_hotswap.py" \
        --ckpt-dir "$CKPT_ROOT/$FAILING" \
        --data-dir "$DATA_DIR"

echo ""                                                                 | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo " Done. Full log: $(pwd)/$LOG"                                     | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
