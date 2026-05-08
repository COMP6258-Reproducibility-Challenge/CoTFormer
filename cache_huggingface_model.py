#!/usr/bin/env python3
"""Pre-download a HuggingFace model snapshot to a project-controlled cache.

Run on the Iridis **login node** (which has internet) before submitting
any SLURM job that consumes the model -- compute nodes have no outbound
network, so any uncached HuggingFace fetch on a compute node will hang
and time out. Defaults are tuned for the Protocol D-calibration Tier 1
substrate (GPT-2-large, 36 layers, 774 M parameters).

Cache directory resolution order: ``--cache-dir`` flag, then ``$HF_HOME``,
then ``$SHARED_SCRATCH/.cache/huggingface/`` (the convention from
``iridis/env.sh``), then ``~/.cache/huggingface/``. The resolved path is
printed on success so the caller can verify it matches what compute-node
jobs will see when they re-source ``iridis/env.sh``.

Example:
    python cache_huggingface_model.py --model gpt2-large --include both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL = "gpt2-large"
INCLUDE_CHOICES = ("model", "tokenizer", "both")


def resolve_cache_dir(cli_value: str | None) -> Path:
    """Pick the cache directory in order: CLI > HF_HOME > SHARED_SCRATCH > ~/.cache.

    Mirrors the convention established by ``iridis/env.sh`` (which exports
    ``HF_HOME=$SHARED_SCRATCH/.cache/huggingface``) so this script and the
    SLURM jobs share a single cache root.
    """
    if cli_value:
        return Path(cli_value).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    shared = os.environ.get("SHARED_SCRATCH")
    if shared:
        return Path(shared).expanduser() / ".cache" / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download a HuggingFace model snapshot to a project-controlled "
            "cache directory. Run on the Iridis login node before SLURM "
            "submission; compute nodes have no internet."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "HuggingFace model identifier "
            f"(default: {DEFAULT_MODEL} -- the Protocol D-calibration Tier 1 substrate)."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "HuggingFace revision (commit SHA, branch, or tag). "
            "Default: None -- HuggingFace resolves to the model's default branch."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Override the cache root. Default: $HF_HOME, else "
            "$SHARED_SCRATCH/.cache/huggingface, else ~/.cache/huggingface."
        ),
    )
    parser.add_argument(
        "--include",
        choices=INCLUDE_CHOICES,
        default="both",
        help="Which artefacts to download (default: both).",
    )
    return parser


def download(
    model: str,
    revision: str | None,
    cache_dir: Path,
    include: str,
) -> None:
    """Download the requested artefacts to ``cache_dir``.

    Imports ``transformers`` lazily so ``--help`` works on machines that
    do not have the project conda env activated.
    """
    # Lazy import: keeps `python cache_huggingface_model.py --help`
    # functional on environments without `transformers` installed
    # (e.g. login-node bootstrap before the conda env is active).
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    cache_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = {"revision": revision, "cache_dir": str(cache_dir)}

    if include in ("tokenizer", "both"):
        print(f"[cache_huggingface_model] downloading tokenizer for {model!r}...", flush=True)
        AutoTokenizer.from_pretrained(model, **common_kwargs)

    if include in ("model", "both"):
        print(f"[cache_huggingface_model] downloading weights for {model!r}...", flush=True)
        # torch_dtype="auto" avoids unnecessary upcasting; this script
        # is download-only so the model never moves to GPU.
        AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", **common_kwargs)


def main() -> int:
    args = build_argparser().parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    try:
        download(args.model, args.revision, cache_dir, args.include)
    except Exception as exc:  # noqa: BLE001 -- surface any HF/network error to the shell
        print(
            f"[cache_huggingface_model] ERROR: failed to cache {args.model!r}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"[cache_huggingface_model] OK -- cache root: {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
