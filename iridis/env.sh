#!/usr/bin/env bash
# iridis/env.sh -- Iridis X environment config (shared team setup)
# Source this file; do not execute it directly.
#
# Usage (from job.sh scripts):
#   source "$REPO_DIR/iridis/env.sh"
#
# All team members share ab3u21's scratch for conda env, datasets, and caches.
# Job outputs are namespaced by $USER: <USER>-<JOB_ID>-<JOB_NAME>/

# --- Shared scratch (ab3u21, read/write granted to team) ---
SHARED_SCRATCH="/scratch/ab3u21"

# --- Conda environment (shared, single install) ---
export CONDA_ENV_PREFIX="$SHARED_SCRATCH/cotformer-env"

# --- Shared data and output paths ---
export DATA_DIR="$SHARED_SCRATCH/datasets"
export RESULTS_DIR="$SHARED_SCRATCH/job-outputs"
# Single source of truth for the training/eval checkpoint root (main.py
# constructs ckpt_path = EXPS_DIR / dataset / model / exp_name via
# os.makedirs on rank 0). Every iridis/*/job.sh must consume this value
# rather than re-defining it locally.
export EXPS_DIR="$SHARED_SCRATCH/exps"

# --- Caches (shared, keep off home quota) ---
export TIKTOKEN_CACHE_DIR="$SHARED_SCRATCH/.cache/tiktoken"
export HF_HOME="$SHARED_SCRATCH/.cache/huggingface"
export WANDB_DIR="$SHARED_SCRATCH/.cache/wandb"
export PIP_CACHE_DIR="$SHARED_SCRATCH/.cache/pip"
export CONDA_PKGS_DIRS="$SHARED_SCRATCH/.conda/pkgs"

# --- GPU memory allocator (PyTorch) ---
# Reduces VRAM fragmentation on L4 24 GB (1.78 GiB reserved-but-unallocated
# without it). Harmless on CPU-only jobs: PyTorch ignores it without CUDA.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Weights and Biases ---
# Offline on compute nodes (no internet); sync from login nodes.
export WANDB_MODE=offline

# --- NCCL transport protocol ---
# Simple protocol avoids the LL-protocol hang observed after ~52K AllReduces
# on NCCL 2.15-2.17 / CUDA 11.8 (see docs/reprod-notes.md §B9).
export NCCL_PROTO=Simple

# --- SLURM email notifications ---
# Uses Southampton email derived from $USER. Override in ~/.bash_aliases if needed.
export NOTIFY_EMAIL="${USER}@soton.ac.uk"

# --- Run directory helper ---
# Creates the next run_N directory inside a package dir.
# Usage: RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
next_run_dir() {
    local pkg_dir="$1"
    local last_n
    last_n=$(ls -1d "$pkg_dir"/run_* 2>/dev/null | sed 's/.*\/run_//' | sort -n | tail -1)
    local next_n=$(( ${last_n:--1} + 1 ))
    local run_dir="$pkg_dir/run_$next_n"
    mkdir -p "$run_dir"
    echo "$run_dir"
}

# --- Job output directory helper ---
# Creates a structured output dir for non-SLURM artifacts (checkpoints, etc.)
# Format: $RESULTS_DIR/<USER>-<JOB_ID>-<JOB_NAME>/
# Usage (from within a SLURM job): JOB_OUTPUT_DIR=$(job_output_dir)
job_output_dir() {
    local dir="$RESULTS_DIR/${USER}-${SLURM_JOB_ID:-0}-${SLURM_JOB_NAME:-local}"
    mkdir -p "$dir"
    echo "$dir"
}
