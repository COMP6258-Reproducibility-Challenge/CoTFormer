#!/usr/bin/env python3
"""Analytical FLOP (MAC) counter for CoTFormer architecture variants.

Computes MACs at each (family, n_layer, n_repeat, seq_len) combination required
for Figure 2 and Figure 3 of the paper, using a CLOSED-FORM formula derived
from the model source (models/cotformer_full_depth.py, models/but_full_depth.py)
and validated bit-exact against ptflops on the BaseCot training config.

Why not ptflops?
    Earlier versions of this script used ``get_ppl_per_mac.get_macs_for_seqlens``
    which calls ptflops' aten backend with flash attention disabled (so the
    dispatcher can count Q@K^T MACs). At large effective sequence lengths
    (CoTFormer 12L x 3R @ seq_len=8192 -> effective N = 3 x 8192 = 24576), the
    materialised attention score matrix exceeds 22 GiB and OOMs on an L4 GPU --
    SLURM ``--mem`` does NOT cap CUDA VRAM, so bumping host RAM has no effect.

    The MAC count is a deterministic function of architecture constants
    (n_layer, n_repeat, T, d_model, d_ff, vocab) -- there is no need to
    materialise the attention matrix to count multiplications. The formula
    below matches ptflops bit-exact on the 4 reference points that ptflops
    DOES succeed at; this is verified at script startup before any production
    point is computed.

See docs/reprod-notes.md (Section A10) for the methodology rationale and the
historical run_0 OOM that prompted this rewrite.

Usage:
    python scripts/compute_macs.py --output run_1/json/macs.json
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import Any

# Ensure the repo root is on sys.path so config/models resolve correctly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Architecture constants — match config/base.py defaults used by BaseCot_*
# training jobs (see iridis/base-train/job.sh): n_head=12, n_embd=768,
# vocab_size=50304, MLP is 2-linear GELU with d_ff = 4 * d_model = 3072
# (see models/cotformer_full_depth.py:189-190).
# ---------------------------------------------------------------------------

_ARCH_DEFAULTS = {
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50304,
    "d_ff": 3072,  # MLP hidden = 4 * d_model
    "positional_encoder": "rotary",
    "bias": False,
    "n_layer_begin": 0,
    "n_layer_end": 0,
}

# Fig-3 sequence-length sweep values (schema §results_fig3.json)
_FIG3_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]

# Map display family name → model registry name (purely metadata; the
# closed-form formula dispatches on family, not the registry name).
_FAMILY_TO_MODEL = {
    "CoTFormer": "cotformer_full_depth",
    "BUT": "but_full_depth",
    "Standard": "base",
}

# Bit-exact reference values empirically captured from a successful ptflops run
# on the BaseCot training config (Iridis run_0 slurm_920549.out, points 1-9
# before the OOM at point 10). If the formula below ever drifts from these
# values, the validation at script startup will abort.
_PTFLOPS_REFERENCE = [
    # (family, n_layer, n_repeat, seq_len, expected MACs from ptflops)
    ("CoTFormer", 12, 2, 256, 47_149_056_000),
    ("CoTFormer", 12, 3, 256, 72_516_206_592),
    ("CoTFormer", 12, 5, 256, 126_874_386_432),
    ("CoTFormer", 12, 3, 128, 34_465_480_704),
]


# ---------------------------------------------------------------------------
# Closed-form MAC formula
# ---------------------------------------------------------------------------

def _closed_form_macs(
    family: str,
    n_layer: int,
    n_repeat: int,
    seq_len: int,
    d_model: int = _ARCH_DEFAULTS["n_embd"],
    d_ff: int = _ARCH_DEFAULTS["d_ff"],
    vocab_size: int = _ARCH_DEFAULTS["vocab_size"],
) -> int:
    """Closed-form MAC count for {CoTFormer, BUT, Standard} architectures.

    Matches ptflops' aten backend bit-exact for n_layer_begin=n_layer_end=0
    (i.e., the BaseCot_* training config). The ptflops accounting:
      - Counts the embedding-table dispatch once: vocab_size * d_model MACs.
      - Does NOT count the final LM head projection.
      - Per attention layer: Q@K^T + softmax@V both contribute N^2 * d_model
        MACs (where N is the attended-to sequence length).
      - Per layer linear ops (QKV proj + Out proj + MLP up + MLP down):
        4 * d_model^2 + 2 * d_model * d_ff  MACs per token.

    CoTFormer architecture (models/cotformer_full_depth.py:318-325):
        At repeat r in {1..n_repeat}, the model processes T new tokens
        while attending to r*T context. Per repeat r, per layer:
            T * (4 d^2 + 2 d d_ff)  +  2 * r * T^2 * d_model  MACs.
        Summed over r=1..n_repeat:
            linear   = n_repeat * n_layer * T * (4 d^2 + 2 d d_ff)
            quadratic = n_layer * T^2 * d * n_repeat * (n_repeat + 1)

    BUT architecture (models/but_full_depth.py):
        Each of n_repeat passes processes T tokens attending to T context.
            total = n_repeat * n_layer * T * (4 d^2 + 2 d d_ff + 2 T d)

    Standard architecture (single pass, no repeats):
            total = n_layer * T * (4 d^2 + 2 d d_ff + 2 T d)
    """
    T = seq_len
    linear_per_token = 4 * d_model * d_model + 2 * d_model * d_ff
    embedding_const = vocab_size * d_model

    if family == "CoTFormer":
        linear_total = n_repeat * n_layer * T * linear_per_token
        quadratic_total = n_layer * (T * T) * d_model * n_repeat * (n_repeat + 1)
        return embedding_const + linear_total + quadratic_total

    if family == "BUT":
        quadratic_per_layer = 2 * (T * T) * d_model
        return embedding_const + n_repeat * n_layer * (T * linear_per_token + quadratic_per_layer)

    if family == "Standard":
        quadratic_per_layer = 2 * (T * T) * d_model
        return embedding_const + n_layer * (T * linear_per_token + quadratic_per_layer)

    raise ValueError(f"Unknown family: {family!r} (expected one of CoTFormer / BUT / Standard)")


def _validate_against_reference() -> None:
    """Cross-check the closed-form formula against ptflops reference values.

    These reference values were captured from a real ptflops run on the
    BaseCot training config (Iridis slurm_920549.out, before the OOM at
    point 10). If the formula drifts due to a model-arch change, this
    aborts immediately rather than silently producing wrong MACs.
    """
    print("Validating closed-form MAC formula against ptflops reference values...")
    for family, n_layer, n_repeat, seq_len, expected in _PTFLOPS_REFERENCE:
        computed = _closed_form_macs(family, n_layer, n_repeat, seq_len)
        if computed != expected:
            rel_err = abs(computed - expected) / expected
            raise RuntimeError(
                f"FORMULA DRIFT DETECTED for {family} {n_layer}L x {n_repeat}R "
                f"@ seq={seq_len}:\n"
                f"    expected (ptflops): {expected:,}\n"
                f"    closed-form:        {computed:,}\n"
                f"    relative error:     {rel_err:.4%}\n"
                f"This indicates an arch change in models/{_FAMILY_TO_MODEL[family]}.py "
                f"that the formula in compute_macs.py does not account for. "
                f"Re-derive the formula or update _PTFLOPS_REFERENCE."
            )
        print(f"    OK  {family} {n_layer}L x {n_repeat}R @ seq={seq_len}: {computed:,}")
    print(f"All {len(_PTFLOPS_REFERENCE)} reference points match.\n")


# ---------------------------------------------------------------------------
# Required points to compute (schema §Required points)
# ---------------------------------------------------------------------------

def _build_required_points() -> list[dict[str, Any]]:
    """Return the full list of (family, n_layer, n_repeat, seq_len) dicts."""
    points: list[dict[str, Any]] = []

    def _add(family: str, n_layer: int, n_repeat: int, seq_len: int) -> None:
        points.append(
            {
                "family": family,
                "n_layer": n_layer,
                "n_repeat": n_repeat,
                "seq_len": seq_len,
            }
        )

    # CoTFormer 12L — Fig 2(a) @ seq=256, Fig 3 sweep @ n_repeat=3
    for nr in [2, 3, 5, 15]:
        _add("CoTFormer", 12, nr, 256)
    for sl in _FIG3_SEQ_LENS:
        if sl != 256:
            _add("CoTFormer", 12, 3, sl)

    # BUT 12L — Fig 2(a) @ seq=256, Fig 3 sweep @ n_repeat=5
    for nr in [2, 3, 5, 6, 15]:
        _add("BUT", 12, nr, 256)
    for sl in _FIG3_SEQ_LENS:
        if sl != 256:
            _add("BUT", 12, 5, sl)

    # CoTFormer 24L — Fig 2(b) @ seq=256
    for nr in [2, 3, 5]:
        _add("CoTFormer", 24, nr, 256)

    # BUT 24L — Fig 2(b) @ seq=256
    for nr in [2, 3, 5]:
        _add("BUT", 24, nr, 256)

    return points


def _get_git_commit(repo_root: str) -> str:
    """Return HEAD git commit SHA, or '<unknown>' on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "<unknown>"


def compute_all_macs() -> list[dict[str, Any]]:
    """Iterate required points, compute MACs via closed-form formula."""
    required_points = _build_required_points()
    results: list[dict[str, Any]] = []

    for i, pt in enumerate(required_points):
        family = pt["family"]
        n_layer = pt["n_layer"]
        n_repeat = pt["n_repeat"]
        seq_len = pt["seq_len"]
        model_name = _FAMILY_TO_MODEL[family]

        macs = _closed_form_macs(family, n_layer, n_repeat, seq_len)

        print(
            f"[{i + 1}/{len(required_points)}] "
            f"{family} {n_layer}L x {n_repeat}R @ seq={seq_len}  "
            f"model={model_name}  MACs = {macs:,}",
            flush=True,
        )

        results.append(
            {
                "model": model_name,
                "family": family,
                "n_layer": n_layer,
                "n_repeat": n_repeat,
                "seq_len": seq_len,
                "macs": macs,
            }
        )

    return results


def main() -> None:
    """Parse CLI, validate formula, compute MACs, write macs.json."""
    parser = argparse.ArgumentParser(
        description="Compute analytical MAC counts via closed-form formula.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to write macs.json (e.g. run_1/json/macs.json)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="(Vestigial — closed-form formula runs on CPU. Kept for "
             "backward-compatible job.sh invocation; ignored.)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    git_commit = _get_git_commit(_REPO_ROOT)
    compute_date = datetime.datetime.utcnow().isoformat() + "Z"

    print(f"compute_macs.py — closed-form  git={git_commit}", flush=True)
    print(f"Output: {args.output}\n", flush=True)

    _validate_against_reference()

    points = compute_all_macs()

    output = {
        "schema_version": "1.0",
        "metadata": {
            "computation_method": "closed_form",
            "computation_method_note": (
                "Closed-form formula derived from models/cotformer_full_depth.py "
                "and models/but_full_depth.py; validated bit-exact against ptflops "
                "aten-backend on 4 reference points (CoTFormer 12L x {2,3,5}R @ "
                "seq=256 + 12L x 3R @ seq=128) at script startup. See compute_macs.py "
                "module docstring for the OOM rationale that prompted moving off ptflops."
            ),
            "device": "cpu",
            "ptflops_backend": "n/a (closed-form)",
            "arch_defaults": _ARCH_DEFAULTS,
            "git_commit": git_commit,
            "compute_date_utc": compute_date,
        },
        "points": points,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Wrote {len(points)} points to {args.output}", flush=True)


if __name__ == "__main__":
    main()
