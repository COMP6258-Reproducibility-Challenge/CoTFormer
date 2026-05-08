#!/bin/bash
# iridis/login-node-prep.sh -- One-shot login-node bootstrap for HPC submission.
#
# Run this ONCE on the Iridis login node before any HPC job submission, to
# warm every cache that compute nodes will need. Compute nodes have no
# outbound internet, so any uncached HuggingFace / tiktoken / pip fetch
# from inside a SLURM job will hang and time out.
#
# Usage (login node only):
#   cd ~/CoTFormer
#   bash iridis/login-node-prep.sh
#
# Idempotent: re-running is a fast no-op once the caches are populated.
#
# What it does (in order):
#   Step A  -- verify the shared conda env exists (no auto-create)
#   Step B  -- regenerate environment.lock.yml.regen if the in-tree lockfile
#              is still a stub (Abe diffs and commits manually; the script
#              never overwrites the in-tree environment.lock.yml)
#   Step C  -- pre-download gpt2-large into $HF_HOME (Protocol D-cal Tier 1)
#   Step D  -- pre-download the tiktoken GPT-2 BPE vocab into $TIKTOKEN_CACHE_DIR
#              (matches data/openwebtext2.py and every model's tokenizer)
#
# Pitfalls handled:
#   - aborts if run on a compute node (no internet)
#   - aborts if any required env var from iridis/env.sh is empty
#   - never overwrites environment.lock.yml; writes a sibling .regen file
#   - prints a copy-paste-ready next-step block on success

set -euo pipefail

# --- Locate repo and source shared env -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_SH="$REPO_DIR/iridis/env.sh"

if [ ! -f "$ENV_SH" ]; then
    echo "ERROR: iridis/env.sh not found at $ENV_SH" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_SH"

# --- ANSI helpers --------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

banner() {
    echo
    echo -e "${BOLD}=== $* ===${NC}"
}

abort() {
    echo -e "${RED}ABORT:${NC} $*" >&2
    exit 1
}

# --- Validate required env vars ------------------------------------------
banner "Pre-flight -- env vars from iridis/env.sh"
for var in SHARED_SCRATCH CONDA_ENV_PREFIX HF_HOME TIKTOKEN_CACHE_DIR DATA_DIR; do
    if [ -z "${!var:-}" ]; then
        abort "Required var \$$var is empty after sourcing iridis/env.sh"
    fi
    printf "  %-22s = %s\n" "$var" "${!var}"
done

# --- Login-node guard ----------------------------------------------------
banner "Pre-flight -- login-node guard"
HOSTNAME_NOW="$(hostname)"
echo "  hostname = $HOSTNAME_NOW"

# Iridis compute nodes follow the pattern red0NNN / green0NNN / amd0NNN /
# navy0NNN. Login nodes are typically iridis5 / iridislogin / similar.
case "$HOSTNAME_NOW" in
    red0*|green0*|amd0*|navy0*|gpu0*|cpu0*)
        abort "Detected compute-node hostname '$HOSTNAME_NOW'. This script must run on a login node (it needs internet)."
        ;;
esac

# Defence-in-depth: a hostname-pattern miss is still caught by an explicit
# reachability probe. Compute nodes have no outbound network at all.
if ! ping -c1 -W2 huggingface.co >/dev/null 2>&1; then
    abort "huggingface.co is unreachable from $HOSTNAME_NOW. Either the login node is offline or this is a compute node."
fi
echo "  internet probe (huggingface.co) -- OK"

# --- Step A: conda env presence ------------------------------------------
banner "Step A -- conda env presence"
if [ -d "$CONDA_ENV_PREFIX" ]; then
    echo -e "  ${GREEN}OK${NC} -- conda env exists at $CONDA_ENV_PREFIX"
else
    echo -e "  ${YELLOW}MISSING${NC} -- conda env not found at $CONDA_ENV_PREFIX"
    cat <<EOF

  Create it manually (long-running, ~10-20 min on Iridis login node):

      module load conda
      conda env create -f "$REPO_DIR/environment.yml" -p "$CONDA_ENV_PREFIX"
      conda activate "$CONDA_ENV_PREFIX"
      pip install --no-deps git+https://github.com/KellerJordan/Muon

  Then re-run this script.
EOF
    abort "Conda env missing -- create it interactively, then re-run."
fi

# Activate the env so subsequent steps invoke the project's python /
# transformers / tiktoken (not whatever happens to be on PATH).
if ! command -v conda >/dev/null 2>&1; then
    if [ -r /etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/conda.sh
    elif command -v module >/dev/null 2>&1; then
        module load conda
    fi
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"
echo "  python = $(command -v python)"

# --- Step B: lockfile regeneration ---------------------------------------
banner "Step B -- environment.lock.yml regeneration (stub-detection)"
LOCKFILE="$REPO_DIR/environment.lock.yml"
LOCKFILE_REGEN="$REPO_DIR/environment.lock.yml.regen"

if [ ! -f "$LOCKFILE" ]; then
    abort "environment.lock.yml not found at $LOCKFILE"
fi

if grep -q "Status: STUB" "$LOCKFILE"; then
    echo "  in-tree lockfile is a STUB -- writing fresh export to sibling .regen file"
    # --no-builds keeps the lockfile cross-host portable while still
    # pinning every version. Writing to a .regen sibling means Abe
    # reviews and commits manually -- the in-tree stub remains the
    # single source of truth until that swap happens.
    conda env export -p "$CONDA_ENV_PREFIX" --no-builds > "$LOCKFILE_REGEN"
    echo -e "  ${GREEN}OK${NC} -- wrote $LOCKFILE_REGEN"
    echo "  Inspect with:"
    echo "      diff -u '$LOCKFILE' '$LOCKFILE_REGEN' | less"
    echo "  When satisfied, replace manually:"
    echo "      mv '$LOCKFILE_REGEN' '$LOCKFILE'"
    echo "      git add '$LOCKFILE' && git commit -m 'chore: regenerate environment.lock.yml on Iridis'"
else
    echo -e "  ${GREEN}OK${NC} -- lockfile is already resolved (not a stub); skipping regen"
fi

# --- Step C: HuggingFace cache (gpt2-large for D-cal Tier 1) -------------
banner "Step C -- HuggingFace cache (gpt2-large)"
mkdir -p "$HF_HOME"
echo "  HF_HOME = $HF_HOME"
# cache_huggingface_model.py is idempotent: HF caches by content hash, so
# re-runs after the first download are fast metadata checks only.
python "$REPO_DIR/cache_huggingface_model.py" --include both --model gpt2-large
echo -e "  ${GREEN}OK${NC} -- gpt2-large cached"

# --- Step D: tiktoken BPE cache (GPT-2 encoding) -------------------------
banner "Step D -- tiktoken BPE cache (gpt2 encoding)"
mkdir -p "$TIKTOKEN_CACHE_DIR"
echo "  TIKTOKEN_CACHE_DIR = $TIKTOKEN_CACHE_DIR"
# tiktoken.get_encoding('gpt2') downloads the vocab + merges on first
# call and writes them to $TIKTOKEN_CACHE_DIR. Re-runs hit the cache.
# Encoding name 'gpt2' matches data/openwebtext2.py and every models/*.py.
python -c "
import tiktoken
enc = tiktoken.get_encoding('gpt2')
# Round-trip a tiny string to force the BPE tables to materialise.
assert enc.decode(enc.encode('hello world')) == 'hello world'
print('  tiktoken gpt2 encoding loaded; n_vocab =', enc.n_vocab)
"
echo -e "  ${GREEN}OK${NC} -- tiktoken gpt2 cached"

# --- Footer --------------------------------------------------------------
echo
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN} OK -- ready for HPC submission${NC}"
echo -e "${GREEN}=========================================${NC}"
cat <<EOF

Next steps (from the repo root):

    cd "$REPO_DIR"
    bash iridis/analyze-lncot+adm/job-gpu.sh    # GPU analysis wave
    # or any other iridis/<package>/job.sh

If you regenerated the lockfile (Step B), diff and commit before submitting:

    diff -u environment.lock.yml environment.lock.yml.regen | less
    mv environment.lock.yml.regen environment.lock.yml
    git add environment.lock.yml && git commit -m 'chore: regenerate environment.lock.yml on Iridis'

EOF
