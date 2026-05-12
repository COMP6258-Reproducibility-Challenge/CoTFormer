# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
# ---

# %% tags=["notebook-runtime-probe"]
# ---- runtime probe (single source of truth — emitted by /notebook init) ----
# no import side-effect, no ~50 ms compile cost.
import os, sys
from pathlib import Path

if "google.colab" in sys.modules:
    RUNTIME = "colab"
elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    RUNTIME = "kaggle"
elif os.environ.get("CODESPACES") == "true":
    RUNTIME = "codespaces"
elif "BINDER_SERVICE_HOST" in os.environ:
    RUNTIME = "binder"
elif "JUPYTERHUB_USER" in os.environ:
    RUNTIME = "jupyterhub"
else:
    RUNTIME = "local"

IS_COLAB = (RUNTIME == "colab")

# Headless rendering for nbconvert --execute / agent-side validation.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def project_path(rel: str) -> Path:
    """Resolve a project-relative path correctly per runtime."""
    if RUNTIME == "colab":
        from google.colab import drive
        if not Path("/content/drive").is_mount():
            drive.mount("/content/drive")
        return Path("/content/drive/MyDrive") / rel
    if RUNTIME == "kaggle":
        return Path("/kaggle/working") / rel
    return Path.cwd() / rel


print(f"[notebook] runtime={RUNTIME}")


# %% [markdown] tags=["header"]
# # CoTFormer Emergency Retrain — n_head=24 → n_head=12 Fix
#
# **Purpose**: Reproduce 3 paper-spec BaseCoTFormer ablations. Previous iridis runs used `n_head=24` (bug). This notebook retrains with `n_head=12` on Colab Pro+ A100.
#
# ## Before you run
#
# **Step 1 — Set your ablation** in the `params` cell (cell below):
# - Bryan: keep `ABLATION = "12L_5R"` (default)
# - Abe: change to `ABLATION = "24L_3R"`
# - 24L_5R is deferred — go/no-go after first two pass step 8000 cleanly
#
# **Step 2 — Set `REPO_URL`** in the `params` cell (cell below):
# - Replace `"<TODO_repo_url>"` with the actual GitHub URL
# - Example: `REPO_URL = "https://github.com/yourname/CoTFormer"`
#
# **Step 3 — Set Drive paths** in the `params` cell:
# - `DRIVE_ROOT`: your MyDrive root (default `/content/drive/MyDrive` is usually correct)
# - `DATA_TARBALL_PATH`: full path to `owt2.tar.gz` on your Drive (e.g. `/content/drive/MyDrive/cotformer-data/owt2.tar.gz`)
#
# **Step 4 — Set wandb key** (optional): if `WANDB_API_KEY` is set in your environment or Colab Secrets, training will log to wandb project `rcotformer`. If unset, wandb is silently skipped.
#
# ## Calibration verdict (cell 10)
#
# Cell 10 runs a 500-step probe and projects the 40k-step wall time.
# - **GO**: projected wall ≤ 18h → proceed to training cell
# - **NO-GO**: projected wall > 18h → investigate before launching
#
# ## What PASS looks like (cell 12)
#
# | Ablation | Target PPL | PASS criterion |
# |----------|-----------|---------------|
# | 12L_5R   | ~26.64    | PPL ≤ 27.14   |
# | 24L_3R   | ~24.85    | PPL ≤ 25.35   |
# | 24L_5R   | ~24.48    | PPL ≤ 24.98   |
#
# ## Cell execution order
#
# Run all cells top-to-bottom on a fresh kernel. Re-runs after session kill are safe — every cell is idempotent.
#

# %% tags=["params"]
# ============================================================
# PARAMS — edit ABLATION and REPO_URL here, nothing else needs changing
# ============================================================
import os

ABLATION = "24L_3R"  # Bryan: keep this.

# replace with the actual GitHub URL for this repo
REPO_URL = "https://github.com/COMP6258-Reproducibility-Challenge/CoTFormer.git"

# NOTE: cotformer_full_depth ignores --min_repeat / --depth_random_method /
# --depth_embedding (see iridis/base-train/job-template.sh:205-210, and
# models/cotformer_full_depth.py:322 — always runs all n_repeat iterations).
# They were previously passed and only polluted summary.json["args"]. Removed.
ABLATION_CONFIGS = {
    "12L_5R": {
        "n_layer":     12,
        "n_repeat":     5,
        "batch_size":  64,
        "acc_steps":    2,
        "exp_name":    "BaseCot_12L_5R",
        "target_ppl":  26.64,
    },
    "24L_3R": {
        "n_layer":     24,
        "n_repeat":     3,
        "batch_size":  32,
        "acc_steps":    4,
        "exp_name":    "BaseCot_24L_3R",
        "target_ppl":  24.85,
    },
    "24L_5R": {
        "n_layer":     24,
        "n_repeat":     5,
        "batch_size":  32,
        "acc_steps":    4,
        "exp_name":    "BaseCot_24L_5R",
        "target_ppl":  24.48,
    },
}

assert ABLATION in ABLATION_CONFIGS, f"Unknown ABLATION={ABLATION!r}. Must be one of {list(ABLATION_CONFIGS)}"
cfg = ABLATION_CONFIGS[ABLATION]

# Derived globals — do NOT edit below this line
N_HEAD         = 12                    # FIXED: this is the bug-fix, never change
EXP_NAME       = cfg["exp_name"]
DRIVE_ROOT     = "/content/drive/MyDrive"
DATA_TARBALL_PATH = f"{DRIVE_ROOT}/cotformer-data/owt2.tar.gz"
RESULTS_BASE   = f"{DRIVE_ROOT}/cotformer_outputs"
# data/openwebtext2.py:240 appends "openwebtext2/" itself, so pass the parent
DATA_DIR       = "/content/data"
REPO_DIR       = "/content/CoTFormer"

# wandb — silently skipped if no API key
_wandb_key = os.environ.get("WANDB_API_KEY", "")
WANDB_ARGS = "--wandb --wandb_project rcotformer" if _wandb_key else ""

print(f"ABLATION  = {ABLATION}")
print(f"EXP_NAME  = {EXP_NAME}")
print(f"N_HEAD    = {N_HEAD}  (must be 12 — paper fix)")
print(f"n_layer   = {cfg['n_layer']}, n_repeat={cfg['n_repeat']}")
print(f"batch_size= {cfg['batch_size']}, acc_steps={cfg['acc_steps']}  (eff={cfg['batch_size']*cfg['acc_steps']})")
print(f"DRIVE_ROOT        = {DRIVE_ROOT}")
print(f"RESULTS_BASE      = {RESULTS_BASE}")
print(f"DATA_DIR          = {DATA_DIR}")
print(f"DATA_TARBALL_PATH = {DATA_TARBALL_PATH}")
print(f"wandb             = {'enabled' if WANDB_ARGS else 'DISABLED (no WANDB_API_KEY)'}")


# %% tags=["mount-drive"]
# Mount Google Drive (idempotent)
import os

if "google.colab" in __import__("sys").modules:
    if not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
        print("Drive mounted.")
    else:
        print("Drive already mounted — skipping.")
else:
    print("Not running on Colab — skipping drive.mount (local validation mode).")


# %% tags=["gpu-assert"]
# Assert A100 GPU and sufficient system RAM — hard-fail on any deviation
import subprocess
import sys

if "google.colab" in __import__("sys").modules:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    print(f"GPU detected: {result}")
    gpu_name = result.split(",")[0].strip()
    vram_str = result.split(",")[1].strip() if "," in result else "unknown"
    if "A100" not in gpu_name:
        raise RuntimeError(
            f"\n\n[HARD FAIL] Expected A100 but got: {gpu_name}\n"
            f"You are on a Colab Pro+ session but NOT assigned an A100.\n"
            f"Go to Runtime > Change runtime type > GPU > A100 and reconnect.\n"
            f"Do NOT burn a training session on a {gpu_name} — wall-time projections will be wrong."
        )
    print(f"[OK] A100 confirmed. VRAM: {vram_str}")

    # System RAM: --data_in_ram converts the memmap'd train.bin (~17GB uint16)
    # into a numpy array in RAM, plus val.bin. Need >= 40GB system RAM to
    # avoid OOM mid-training. Standard Colab runtimes ship 12-25GB; High-RAM ~83GB.
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 40:
        raise RuntimeError(
            f"\n\n[HARD FAIL] System RAM is {ram_gb:.1f}GB — need >= 40GB for --data_in_ram.\n"
            f"Switch to a high-RAM Colab runtime: Runtime > Change runtime type > High-RAM."
        )
    print(f"[OK] System RAM: {ram_gb:.1f}GB")
else:
    print("Not running on Colab — skipping GPU/RAM assertions (local validation mode).")


# %% tags=["clone-repo"]
# Clone or update CoTFormer repo (idempotent)
# REPO_URL is defined in the params cell (cell 2) — edit it there.
import os
import subprocess

if REPO_URL == "<TODO_repo_url>":
    raise RuntimeError(
        "[SETUP REQUIRED] Replace <TODO_repo_url> in the params cell (cell 2) with the actual GitHub URL.\n"
        "Example: REPO_URL = 'https://github.com/yourname/CoTFormer'"
    )

if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL} → {REPO_DIR}")
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
else:
    print(f"{REPO_DIR} already exists — pulling latest.")
    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])

rev = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"Repo HEAD: {rev}")


# %% tags=["install-deps"]
# Install Python dependencies from requirements.txt (idempotent via pip)
import subprocess, sys

req_path = f"{REPO_DIR}/requirements.txt"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_path])
print(f"[OK] Requirements installed from {req_path}")


# %% tags=["stage-data"]
# Stage OWT2 dataset (idempotent + first-time bootstrap).
# Three dispatch paths:
#   A. Local train.bin AND val.bin already extracted -> skip everything.
#   B. Tarball on Drive   -> copy to local disk, sha256-verify if sidecar
#                            present, extract to /content/data/.
#   C. Tarball MISSING    -> bootstrap: download HF OWT2 (~28GB), tokenise,
#                            write train.bin + val.bin locally, then PACK and
#                            COPY the tarball to Drive (with sha256 sidecar)
#                            for future runs. ONE-TIME op only.
#
# Tarball layout: openwebtext2/{train.bin, val.bin}  (top-level = "openwebtext2")
import os
import sys
import time
import hashlib
import subprocess

TRAIN_BIN = f"{DATA_DIR}/openwebtext2/train.bin"
VAL_BIN   = f"{DATA_DIR}/openwebtext2/val.bin"
LOCAL_TAR = "/content/owt2.tar.gz"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256_if_sidecar(tarball_path: str) -> None:
    sidecar = tarball_path + ".sha256"
    if not os.path.exists(sidecar):
        print(f"[WARN] No .sha256 sidecar at {sidecar} — skipping checksum verification.")
        return
    with open(sidecar) as f:
        expected_sha = f.read().strip().split()[0]
    print(f"Verifying sha256 (expected: {expected_sha[:16]}...)...")
    actual_sha = _sha256_file(tarball_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"[FAIL] sha256 mismatch on {tarball_path}\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual_sha}\n"
            f"Re-upload the tarball, regenerate the sidecar, or remove the sidecar to skip this check."
        )
    print("[OK] sha256 verified.")


def _extract_local_tarball(local_tar: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Extracting {local_tar} to {DATA_DIR}/ ...")
    subprocess.check_call(["tar", "xzf", local_tar, "-C", f"{DATA_DIR}/"])


def _wait_for_drive_tarball(timeout_s: int = 60) -> bool:
    """Wait up to timeout_s for the Drive tarball to become visible.
    Drive FUSE inode cache can lag fresh mounts. True if visible, False if not."""
    deadline = time.time() + timeout_s
    while not os.path.exists(DATA_TARBALL_PATH):
        if time.time() > deadline:
            return False
        print(f"  {DATA_TARBALL_PATH} not yet visible — retrying in 5s...")
        time.sleep(5)
    return True


def _bootstrap_from_huggingface() -> None:
    """Path C: build train.bin + val.bin from scratch via the cloned repo's
    dataset helpers, then pack a tarball and upload it to Drive for future
    runs. ONE-TIME op (~2-3h total: HF download + tokenisation). All
    subsequent sessions skip to Path B."""
    print(f"[BOOTSTRAP] No tarball at {DATA_TARBALL_PATH} — building dataset from scratch.")
    print("            ONE-TIME op (~2-3h). Future runs will reuse the Drive tarball.")
    sys.path.insert(0, REPO_DIR)
    from data.openwebtext2 import (
        cache_tiktoken_bpe,
        download_openwebtext2,
        extract_and_tokenize_openwebtext2,
    )

    bin_dir = f"{DATA_DIR}/openwebtext2/"
    os.makedirs(bin_dir, exist_ok=True)

    print("[BOOTSTRAP 1/4] Warming tiktoken BPE cache (needs internet)...")
    cache_tiktoken_bpe()

    print("[BOOTSTRAP 2/4] Downloading OWT2 tarball from HuggingFace (~28GB)...")
    download_openwebtext2(bin_dir)

    print("[BOOTSTRAP 3/4] Extracting + tokenising -> train.bin + val.bin (CPU-bound; ~1-2h)...")
    extract_and_tokenize_openwebtext2(bin_dir)

    assert os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN), (
        f"[FAIL] Bootstrap produced incomplete output:\n"
        f"  train.bin exists: {os.path.exists(TRAIN_BIN)}\n"
        f"  val.bin   exists: {os.path.exists(VAL_BIN)}"
    )

    print("[BOOTSTRAP 4/4] Packing tarball and uploading to Drive...")
    # Top-level dir = "openwebtext2/" — matches what _extract_local_tarball
    # produces on Path B, so the layouts stay in lockstep.
    subprocess.check_call(["tar", "czf", LOCAL_TAR, "-C", DATA_DIR, "openwebtext2/"])

    drive_dir = os.path.dirname(DATA_TARBALL_PATH)
    os.makedirs(drive_dir, exist_ok=True)
    subprocess.check_call(["cp", LOCAL_TAR, DATA_TARBALL_PATH])

    sha = _sha256_file(LOCAL_TAR)
    sidecar = DATA_TARBALL_PATH + ".sha256"
    with open(sidecar, "w") as f:
        f.write(f"{sha}  {os.path.basename(DATA_TARBALL_PATH)}\n")

    print(f"[OK] Bootstrap complete. Tarball + sidecar live on Drive at {DATA_TARBALL_PATH}")


# ----- Dispatch -----
if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
    print(f"[OK] Dataset already staged at {DATA_DIR}/openwebtext2/ — skipping (Path A).")
else:
    print("Waiting for Drive mount to settle...")
    if _wait_for_drive_tarball(timeout_s=60):
        # Path B: tarball available on Drive
        print(f"Found tarball: {DATA_TARBALL_PATH} (Path B)")
        _verify_sha256_if_sidecar(DATA_TARBALL_PATH)

        if not os.path.exists(LOCAL_TAR):
            print(f"Copying tarball to local disk: {LOCAL_TAR}")
            subprocess.check_call(["cp", DATA_TARBALL_PATH, LOCAL_TAR])
        else:
            print(f"Local tarball already present: {LOCAL_TAR}")

        _extract_local_tarball(LOCAL_TAR)
    else:
        # Path C: tarball missing from Drive — bootstrap once, upload, future runs reuse
        _bootstrap_from_huggingface()

# ----- Final assertions: BOTH train.bin and val.bin required -----
# data/openwebtext2.py:247 raises FileNotFoundError if either is missing,
# so we fail fast here rather than 5 min into the training cell.
assert os.path.exists(TRAIN_BIN), (
    f"[FAIL] {TRAIN_BIN} not found after staging.\n"
    f"Inspect: ls -lh {DATA_DIR}/openwebtext2/"
)
assert os.path.exists(VAL_BIN), (
    f"[FAIL] {VAL_BIN} not found after staging — required by data/openwebtext2.py:247.\n"
    f"Inspect: ls -lh {DATA_DIR}/openwebtext2/"
)

# Sanity listing
result = subprocess.run(["ls", "-lh", f"{DATA_DIR}/openwebtext2"], capture_output=True, text=True)
print(result.stdout)
print("[OK] train.bin + val.bin present.")


# %% tags=["verify-config"]
# Verify resolved config before training — defensive n_head=12 assertion
import os
import subprocess
import sys

_vars = {"ABLATION": ABLATION, "cfg": cfg, "EXP_NAME": EXP_NAME,
         "N_HEAD": N_HEAD, "RESULTS_BASE": RESULTS_BASE,
         "DATA_DIR": DATA_DIR, "REPO_DIR": REPO_DIR}
for k, v in _vars.items():
    print(f"  {k:25s} = {v}")

# The critical assertion — paper fix is n_head=12
assert N_HEAD == 12, f"[BUG] N_HEAD={N_HEAD} — must be 12. Check params cell."
print(f"\n[OK] N_HEAD == 12 assertion passed.")

# Confirm data dir exists (openwebtext2/ subdir, after staging)
owt2_dir = os.path.join(DATA_DIR, "openwebtext2")
assert os.path.exists(owt2_dir), (
    f"[FAIL] {owt2_dir!r} does not exist. Run stage-data cell first."
)
print(f"[OK] OWT2 data dir exists: {owt2_dir}")

# Confirm repo exists
assert os.path.exists(REPO_DIR), f"[FAIL] REPO_DIR={REPO_DIR!r} does not exist. Run clone-repo cell first."
rev = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"[OK] REPO_DIR exists. git HEAD: {rev}")

# Confirm results base is reachable (drive mounted)
os.makedirs(RESULTS_BASE, exist_ok=True)
print(f"[OK] RESULTS_BASE reachable: {RESULTS_BASE}")

# Effective batch size
eff_batch = cfg["batch_size"] * cfg["acc_steps"]
assert eff_batch == 128, f"[WARN] Effective batch={eff_batch}, expected 128. Check params."
print(f"[OK] Effective batch size = {eff_batch} (paper spec: 128)")


# %% tags=["quarantine-check"]
# Phase 5b airtight quarantine: accepts started.json (mid-training resume) OR summary.json (completed).
# Refuses if ckpts exist with NEITHER marker (stale-from-iridis silent-load risk).
import json, os, datetime

CKPT_DIR = f"{RESULTS_BASE}/owt2/cotformer_full_depth/{EXP_NAME}"
date = datetime.datetime.now().strftime("%Y%m%d")

def _fail(msg):
    raise RuntimeError(msg)

if not os.path.exists(CKPT_DIR):
    print(f"[OK] No ckpt dir at {CKPT_DIR} — fresh start.")
else:
    entries = os.listdir(CKPT_DIR)
    ckpt_files = [f for f in entries if f.startswith("ckpt_") and f.endswith(".pt")]
    has_final = "ckpt.pt" in entries
    summary_path = f"{CKPT_DIR}/summary.json"
    marker_path = f"{CKPT_DIR}/started.json"
    has_summary = os.path.exists(summary_path)
    has_marker = os.path.exists(marker_path)
    has_any_ckpt = bool(ckpt_files) or has_final

    if not has_any_ckpt and not has_summary and not has_marker:
        print(f"[OK] Empty dir {CKPT_DIR} — fresh start.")
    elif has_summary:
        with open(summary_path) as f:
            saved = json.load(f).get("args", {})
        saved_n_head = saved.get("n_head")
        if saved_n_head != 12:
            _fail(
                f"STALE summary.json shows n_head={saved_n_head}; this retrain requires n_head=12.\n"
                f"Quarantine:\n  mv {CKPT_DIR} {CKPT_DIR}_STALE_n{saved_n_head}_{date}\nThen re-run."
            )
        print(f"[OK] summary.json confirms n_head=12 (completed run; main.py will extend if --iterations > prev).")
    elif has_marker:
        with open(marker_path) as f:
            marker = json.load(f)
        marker_n_head = marker.get("n_head")
        marker_abl = marker.get("ablation")
        if marker_n_head != 12:
            _fail(
                f"STALE started.json shows n_head={marker_n_head}; quarantine: mv {CKPT_DIR} {CKPT_DIR}_STALE_n{marker_n_head}_{date}"
            )
        if marker_abl != ABLATION:
            _fail(
                f"started.json's ablation={marker_abl} does NOT match current ABLATION={ABLATION}.\n"
                f"Did you switch ABLATION in cell 2 between runs?\n"
                f"Quarantine and start fresh:\n  mv {CKPT_DIR} {CKPT_DIR}_WRONG_ABLATION_{date}"
            )
        print(f"[OK] started.json: n_head=12, ablation={marker_abl}, started {marker.get('first_started_at')} — safe to resume.")
    else:
        _fail(
            f"REFUSE TO RESUME: ckpts exist in {CKPT_DIR} but NEITHER summary.json NOR started.json present.\n"
            f"Cannot verify provenance — could be silently-loaded stale n_head=24 ckpt (state_dict shapes match n_head=12 by accident at main.py:193).\n"
            f"Manually quarantine:\n  mv {CKPT_DIR} {CKPT_DIR}_UNVERIFIED_{date}\nThen re-run."
        )



# %% tags=["ckpt-integrity-check"]
# Phase 5b ckpt integrity sweep: handles partial Drive-writes from Colab kill mid-torch.save.
# Renames un-loadable ckpts to .PARTIAL so main.py's --use_pretrained auto skips them
# (otherwise main.py:155-160 catches corruption but starts FRESH, losing all valid prior ckpts).
import os, re, torch

CKPT_DIR = f"{RESULTS_BASE}/owt2/cotformer_full_depth/{EXP_NAME}"

if not os.path.exists(CKPT_DIR):
    print("[OK] No ckpt dir yet — nothing to integrity-check.")
else:
    pattern = re.compile(r"^ckpt_(\d+)\.pt$")
    candidates = []
    for f in os.listdir(CKPT_DIR):
        m = pattern.match(f)
        if m:
            candidates.append((int(m.group(1)), f))
    candidates.sort(reverse=True)  # highest-iter first

    if not candidates:
        # Also try the final ckpt.pt (post-training)
        if os.path.exists(f"{CKPT_DIR}/ckpt.pt"):
            print(f"[OK] Final ckpt.pt present — completed run.")
        else:
            print(f"[OK] No ckpt_*.pt files in {CKPT_DIR} — fresh start expected.")
    else:
        valid = None
        demoted = []
        for step, fname in candidates:
            path = f"{CKPT_DIR}/{fname}"
            try:
                _ = torch.load(path, map_location="cpu", weights_only=False)
                valid = (step, fname)
                break
            except Exception as e:
                new_name = f"{path}.PARTIAL"
                os.rename(path, new_name)
                demoted.append(fname)
                print(f"[WARN] Demoted partial ckpt {fname} → {fname}.PARTIAL ({type(e).__name__}: {str(e)[:80]})")

        if valid:
            step, fname = valid
            print(f"[OK] Will resume from {fname} (step {step}). {len(demoted)} partial ckpt(s) demoted.")
        else:
            print(f"[WARN] ALL {len(candidates)} ckpts were corrupt — main.py will start fresh from step 0.")
            print(f"      If unexpected: inspect /content/drive/MyDrive/.../{EXP_NAME}/*.PARTIAL files manually.")


# %% tags=["calibration"]
# Calibration probe — 500 steps, project 40k wall time, print GO/NO-GO
# This cell does NOT save checkpoints and does NOT write to Drive.
# Re-running is safe; CALIB ckpt dir is throwaway.
import os
import time
import re
import subprocess
import sys

CALIB_EXP   = f"CALIB_{ABLATION}"
CALIB_DIR   = f"/content/calib_{ABLATION}"
os.makedirs(CALIB_DIR, exist_ok=True)

# wandb intentionally disabled for calibration probe.
# Flags omitted because cotformer_full_depth ignores them:
#   --min_repeat, --depth_random_method  (see params cell note)
calib_cmd = (
    f"cd {REPO_DIR} && python ./main.py "
    f"--config_format base "
    f"--model cotformer_full_depth "
    f"--n_embd 768 "
    f"--n_head {N_HEAD} "
    f"--n_layer {cfg['n_layer']} "
    f"--n_repeat {cfg['n_repeat']} "
    f"--batch_size {cfg['batch_size']} "
    f"--acc_steps {cfg['acc_steps']} "
    f"--sequence_length 256 "
    f"--dropout 0.0 "
    f"--iterations 500 "
    f"--dataset owt2 "
    f"--data_dir {DATA_DIR} "
    f"--data_in_ram "
    f"--lr 1e-3 "
    f"--weight_decay 0.1 "
    f"--warmup_percent 0.2 "
    f"--eval_freq 100000 "    # no eval in 500-step probe -> clean ms/step signal
    f"--seed 0 "
    f"--n_layer_begin 0 "
    f"--n_layer_end 0 "
    f"--save_checkpoint_freq 100000 "
    f"--results_base_folder {CALIB_DIR} "
    f"--exp_name {CALIB_EXP} "
)

print(f"Running 500-step calibration probe for {ABLATION}...")
print(f"Command: {calib_cmd}\n")
t_start = time.time()

proc = subprocess.Popen(
    calib_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)

# Parse step times from "  step N/500  loss=...  Xms/step  ETA ~Yh"
# optim/base.py:124 only prints at `itr % 10 == 0 and itr % eval_freq != 0`.
# With eval_freq=100000 every step-10 multiple emits; collect last 100 of 500.
step_pattern = re.compile(r"step\s+(\d+)/\d+.*?(\d+)ms/step")
collected_ms = []
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    m = step_pattern.search(line)
    if m:
        step_n = int(m.group(1))
        ms_per_step = int(m.group(2))
        if step_n >= 400:  # collect last 100 of 500
            collected_ms.append(ms_per_step)

proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"Calibration probe failed with exit code {proc.returncode}. Check output above.")

if not collected_ms:
    print("[WARN] Could not parse step-time from output. Check log format above.")
else:
    mean_ms = sum(collected_ms) / len(collected_ms)
    projected_h = (40000 * mean_ms) / 3.6e6
    GO_THRESHOLD_H = 18.0

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULT — {ABLATION}")
    print(f"  Mean ms/step (last 100 steps): {mean_ms:.0f} ms")
    print(f"  Projected 40k-step wall time:  {projected_h:.1f} h")
    print(f"  GO threshold:                  {GO_THRESHOLD_H} h")
    if projected_h <= GO_THRESHOLD_H:
        print(f"  VERDICT: *** GO *** — proceed to training cell")
    else:
        print(f"  VERDICT: *** NO-GO *** — projected {projected_h:.1f}h exceeds {GO_THRESHOLD_H}h limit")
        print(f"\n  Diagnostic suggestions:")
        print(f"  1. Confirm /content/data/openwebtext2 is on LOCAL disk (not Drive FUSE mount).")
        print(f"     ls -lh /content/data/ — should show ~10GB+ of .bin files.")
        print(f"  2. Check nvidia-smi for thermal throttling (clock should be ~1410 MHz on A100).")
        print(f"  3. If VRAM is near limit, try batch_size=32 acc_steps=4 in the params cell.")
        print(f"     (Effective batch stays 128, throughput drops ~1.4x.)")
        print(f"  4. Do NOT use --compile — it is known-broken for cotformer_full_depth.")
    print(f"{'='*60}")


# %% tags=["train"]
# Phase 5b: write session marker BEFORE main.py invocation.
# Quarantine-check uses started.json (mid-training resume gate) since
# main.py only writes summary.json on completion (main.py:223-226).
import json, datetime, os
CKPT_DIR = f"{RESULTS_BASE}/owt2/cotformer_full_depth/{EXP_NAME}"
os.makedirs(CKPT_DIR, exist_ok=True)
MARKER = f"{CKPT_DIR}/started.json"
if not os.path.exists(MARKER):
    payload = {
        "n_head": N_HEAD,
        "n_layer": cfg["n_layer"],
        "n_repeat": cfg["n_repeat"],
        "exp_name": EXP_NAME,
        "ablation": ABLATION,
        "first_started_at": datetime.datetime.now().isoformat(),
    }
    tmp = f"{MARKER}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, MARKER)  # atomic on POSIX; Drive FUSE honors rename
    print(f"[INFO] Wrote session marker: {MARKER}")
else:
    with open(MARKER) as f:
        payload = json.load(f)
    print(f"[INFO] Existing marker: started_at={payload.get('first_started_at')}, n_head={payload.get('n_head')}, ablation={payload.get('ablation')}")

# ============================================================
# TRAINING — 40k-step paper-spec run
# Re-running this cell after a session kill will RESUME from
# the latest checkpoint (--use_pretrained auto).
# ============================================================
import os

# Log path mirrors main.py's ckpt layout:
# {results_base_folder}/{dataset}/{model}/{exp_name}/
LOG_FILE = f"{RESULTS_BASE}/owt2/cotformer_full_depth/{EXP_NAME}/train.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

if os.path.exists(LOG_FILE):
    print(f"[INFO] Appending to existing log at {LOG_FILE}")
else:
    print(f"[INFO] Creating new log at {LOG_FILE}")

wandb_flags = WANDB_ARGS  # "" if no key, "--wandb --wandb_project rcotformer" otherwise

# Hyperparameters match iridis/base-train/job-template.sh (paper section 3.2 SOT).
# Flags omitted because cotformer_full_depth ignores them:
#   --min_repeat, --depth_random_method  (see params cell note)
# save_checkpoint_freq 2000: matches iridis SOT — 20 resume points over 40k
# iters, halves Drive FUSE upload churn vs the previous 1000 setting.
train_cmd = (
    f"cd {REPO_DIR} && python ./main.py "
    f"--config_format base "
    f"--model cotformer_full_depth "
    f"--n_embd 768 "
    f"--n_head {N_HEAD} "
    f"--n_layer {cfg['n_layer']} "
    f"--n_repeat {cfg['n_repeat']} "
    f"--batch_size {cfg['batch_size']} "
    f"--acc_steps {cfg['acc_steps']} "
    f"--sequence_length 256 "
    f"--dropout 0.0 "
    f"--iterations 40000 "
    f"--dataset owt2 "
    f"--data_dir {DATA_DIR} "
    f"--data_in_ram "
    f"--lr 1e-3 "
    f"--weight_decay 0.1 "
    f"--warmup_percent 0.2 "
    f"--eval_freq 100 "
    f"--seed 0 "
    f"--n_layer_begin 0 "
    f"--n_layer_end 0 "
    f"--save_checkpoint_freq 2000 "
    f"--remove_intermediary_checkpoints_at_end "
    f"--results_base_folder {RESULTS_BASE} "
    f"--exp_name {EXP_NAME} "
    f"--use_pretrained auto "
    f"{wandb_flags} "
    f"2>&1 | tee -a {LOG_FILE}"  # -a: append on resume, never overwrite
)

print(f"Training {ABLATION} | EXP_NAME={EXP_NAME}")
print(f"Log -> {LOG_FILE}")
print(f"CMD: {train_cmd}\n")
# Shell execution — streams output live via tee -a to Drive log
get_ipython().system(train_cmd)


# %% keep_output=true tags=["eval-ppl"]
# Evaluate perplexity on final checkpoint using eval.py
# eval.py takes --checkpoint <dir or ckpt.pt path>
import os
import re
import subprocess
import sys

CKPT_PATH = os.path.join(RESULTS_BASE, "owt2", "cotformer_full_depth", EXP_NAME, "ckpt.pt")

if not os.path.exists(CKPT_PATH):
    raise RuntimeError(
        f"[FAIL] Final checkpoint not found: {CKPT_PATH}\n"
        f"Training may not have completed (or saved to a different path).\n"
        f"Check ls {os.path.dirname(CKPT_PATH)}"
    )

eval_cmd = (
    f"cd {REPO_DIR} && python ./eval.py "
    f"--checkpoint {CKPT_PATH} "
    f"--distributed_backend None "
)

print(f"Running eval on: {CKPT_PATH}")
proc = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
combined = proc.stdout + proc.stderr
print(combined)

# Parse val PPL from eval.py output (look for "val_pp" or "perplexity")
ppl_match = re.search(r"val_pp[:\s=]+([0-9]+\.?[0-9]*)", combined, re.IGNORECASE)
if not ppl_match:
    ppl_match = re.search(r"perplexity[:\s=]+([0-9]+\.?[0-9]*)", combined, re.IGNORECASE)

target_ppl = cfg["target_ppl"]
PPL_PASS_MARGIN = 0.5

print(f"\n{'='*60}")
print(f"EVAL RESULT — {ABLATION}")
if ppl_match:
    ppl = float(ppl_match.group(1))
    print(f"  Measured PPL:  {ppl:.2f}")
    print(f"  Target PPL:    {target_ppl:.2f}  (paper spec)")
    print(f"  PASS threshold: {target_ppl + PPL_PASS_MARGIN:.2f}")
    if ppl <= target_ppl + PPL_PASS_MARGIN:
        print(f"  VERDICT: *** PASS *** — within {PPL_PASS_MARGIN} of paper target")
    else:
        print(f"  VERDICT: *** INVESTIGATE *** — PPL={ppl:.2f} exceeds target+margin={target_ppl+PPL_PASS_MARGIN:.2f}")
        print(f"  Check: did training converge (train.log)? Was n_head=12 used throughout?")
        print(f"  Verify: grep 'n_head' {os.path.dirname(CKPT_PATH)}/summary.json")
else:
    print(f"  [WARN] Could not parse PPL from eval output — inspect manually above.")
    print(f"  Target PPL: {target_ppl:.2f} ± {PPL_PASS_MARGIN}")
print(f"{'='*60}")


# %% [markdown] tags=["handoff"]
# # Training Complete — Handoff Summary
#
# ## Results to report back
#
# After cell `eval-ppl` completes, report:
#
# 1. **PPL number** printed by the eval cell
# 2. **Train log path** on Drive:
#    - Format: `{DRIVE_ROOT}/cotformer_outputs/owt2/cotformer_full_depth/{EXP_NAME}/train.log`
#    - Example (Bryan): `/content/drive/MyDrive/cotformer_outputs/owt2/cotformer_full_depth/BaseCot_12L_5R/train.log`
#    - Example (Abe):   `/content/drive/MyDrive/cotformer_outputs/owt2/cotformer_full_depth/BaseCot_24L_3R/train.log`
# 3. **GPU type** from the gpu-assert cell output (should say "A100 confirmed")
#
# ## Drive folder layout after training
#
# ```
# MyDrive/cotformer_outputs/
# └── owt2/
#     └── cotformer_full_depth/
#         └── BaseCot_<ABLATION>/
#             ├── ckpt.pt           ← final checkpoint
#             ├── summary.json      ← training args + stats (includes n_head for provenance)
#             ├── started.json      ← session marker (notebook-side quarantine gate)
#             └── train.log         ← full training log (appended across resumes)
# ```
#
# Note: intermediary `ckpt_<step>.pt` files are removed at end of training
# (`--remove_intermediary_checkpoints_at_end`). Only `ckpt.pt` survives.
#
# ## Transferring to iridis for Figure 2 / Table 1 reproduction
#
# Figure 2 and Table 1 require ALL SEVEN BaseCot ablations evaluated together
# (see `iridis/base-cots-eval/`). This notebook produces ONE ablation per run.
#
# The iridis eval job's pre-flight (`iridis/base-cots-eval/job.sh:84-97`)
# requires BOTH files per ablation:
# - `<CKPT_ROOT>/BaseCot_<N>L_<R>R/ckpt.pt`
# - `<CKPT_ROOT>/BaseCot_<N>L_<R>R/summary.json`
#
# **Critical**: `summary.json` is written ONLY when training reaches 40000 iters
# and `main.py` finishes cleanly (`main.py:225`). Mid-training resume cycles
# produce `ckpt_<N>.pt` files but NO `summary.json`.
#
# **Do NOT rsync to iridis** until the eval cell prints `EVAL RESULT` AND
# `summary.json` is present at `.../BaseCot_<ABLATION>/summary.json`.
#
# The Colab `EXP_NAME` (`BaseCot_24L_3R` etc.) is already the exact folder name
# the iridis eval job expects under `CKPT_ROOT` — no rename needed.
#
# ## 24L_5R go/no-go
#
# After both 12L_5R and 24L_3R reach step ≥ 8000 cleanly, decide whether
# to launch 24L_5R. Change `ABLATION = "24L_5R"` in the params cell on a
# separate Colab session and re-run all cells.
#
# ## What PASS looks like
#
# | Ablation | Target PPL | PASS if PPL ≤ |
# |----------|-----------|--------------|
# | 12L_5R   | ~26.64    | 27.14        |
# | 24L_3R   | ~24.85    | 25.35        |
# | 24L_5R   | ~24.48    | 24.98        |
#
# * * *
#
# ## Going AFK / Resume Protocol
#
# **Before stepping away** (laptop sleep, commute, etc.):
# 1. Verify the latest `ckpt_*.pt` step number is increasing on Drive (the FUSE-side files update; the cloud-side upload completes async).
# 2. (Optional, recommended) In a fresh cell, run:
#    ```python
#    from google.colab import drive
#    drive.flush_and_unmount()
#    ```
#    BEFORE closing the tab. This forces the FUSE layer to flush pending writes to Drive cloud. **Do not run this if you intend to keep the session active** -- it unmounts Drive.
# 3. **DO NOT trust Pro+ Background Execution** to keep the session alive while AFK. Per googlecolab/colabtools issues #5793 / #5950, sessions are currently terminating within 40 min to 1 h of browser close even with Pro+. Plan for the runtime to die.
#
# **Resuming at home**:
# 1. Open the notebook fresh in Colab (File > Open notebook > GitHub > your fork).
# 2. Confirm `ABLATION` and `REPO_URL` in cell 2 are still correct (they should persist via the .ipynb on GitHub).
# 3. **Run All** (Ctrl+F9). Order matters and is automatic:
#    - `stage-data` finds the tarball on Drive (Path B) — bootstrap (Path C) only fires on the very first run, never on resume.
#    - `quarantine-check` reads `started.json` from Drive, validates `n_head==12` + `ablation` matches -> PASS.
#    - `ckpt-integrity-check` finds the latest non-partial ckpt; demotes any `*.PARTIAL` siblings caused by interrupted writes.
#    - `calibration` re-projects wall time (will project the REMAINING time, since main.py will resume).
#    - `train` invokes `main.py --use_pretrained auto` -> picks the highest-step VALID `ckpt_*.pt` (integrity-checked) -> resumes from `checkpoint['itr']`.
# 4. Watch `train.log` for the resume confirmation line: `Resuming from ckpt_NNNNN.pt`. If you see `WARN: No checkpoint found ... starting fresh` BUT you expected a resume, **STOP** -- the integrity sweep may have demoted everything; inspect the `*.PARTIAL` files before letting it overwrite.
#
# **Sanity invariants** to spot-check at any point:
# - Drive `.../{EXP_NAME}/started.json` shows `n_head: 12`.
# - Drive `.../{EXP_NAME}/ckpt_*.pt` step numbers are monotonically increasing.
# - `train.log` line count grows (we use `tee -a`, never overwrite).
# - `summary.json` appears ONLY after all 40k iters complete (don't worry if it's absent during training).
# - Drive `cotformer-data/owt2.tar.gz` (and `.sha256` sidecar) — present after first bootstrap; all future Colab sessions reuse it.
