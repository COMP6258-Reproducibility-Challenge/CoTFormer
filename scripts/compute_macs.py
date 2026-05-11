#!/usr/bin/env python3
"""Analytical FLOP (MAC) counter for CoTFormer architecture variants.

Builds synthetic model configs (no trained weights) and runs ptflops to
measure MACs at each (family, n_layer, n_repeat, seq_len) combination
required for Fig 2 and Fig 3 of the paper.

Usage:
    python scripts/compute_macs.py --output run_1/json/macs.json [--device cuda:0]
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
# Required points per schema §Required points
# ---------------------------------------------------------------------------

# Fig-3 sequence-length sweep values (schema §results_fig3.json)
_FIG3_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]

# Map display family name → model registry name
_FAMILY_TO_MODEL = {
    "CoTFormer": "cotformer_full_depth",
    "BUT": "but_full_depth",
    "Standard": "base",
}


def _build_required_points() -> list[dict[str, Any]]:
    """Return the full list of (family, n_layer, n_repeat, seq_len) dicts.

    Constructed from schema Table "Required points" section.
    """
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

    # CoTFormer 12L — Fig 2(a) @ seq=256 for all n_repeat, Fig 3 sweep @ n_repeat=3
    for nr in [2, 3, 5, 15]:
        _add("CoTFormer", 12, nr, 256)
    for sl in _FIG3_SEQ_LENS:
        if sl != 256:  # 256 already added above
            _add("CoTFormer", 12, 3, sl)

    # BUT 12L — Fig 2(a) @ seq=256 for all n_repeat, Fig 3 sweep @ n_repeat=5
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


def _build_synthetic_config(
    family: str,
    n_layer: int,
    n_repeat: int,
    seq_len: int,
    device: str,
) -> Any:
    """Build a fully-defaulted config Namespace for the given architecture point.

    Uses config.parse_args_with_format exactly as documented in eval.py:54.
    """
    import config  # noqa: PLC0415 — deferred to keep module importable without GPU

    model_name = _FAMILY_TO_MODEL[family]
    cli_args = [
        "--model", model_name,
        "--n_layer", str(n_layer),
        "--n_repeat", str(n_repeat),
        "--sequence_length", str(seq_len),
        "--vocab_size", "50304",
        "--n_head", "12",
        "--n_embd", "768",
        "--dtype", "torch.bfloat16",
        "--positional_encoder", "rotary",
        "--distributed_backend", "None",
        "--device", device,
    ]
    cfg = config.parse_args_with_format(
        format="base",
        base_parser=argparse.ArgumentParser(allow_abbrev=False),
        args=cli_args,
        namespace=None,
    )
    return cfg


def compute_all_macs(device: str) -> list[dict[str, Any]]:
    """Iterate required points, build model, measure MACs, delete model.

    Returns list of point dicts with 'macs' field added.
    """
    import torch  # noqa: PLC0415 — deferred to keep module importable without GPU
    import models  # noqa: PLC0415
    from get_ppl_per_mac import get_macs_for_seqlens  # noqa: PLC0415

    required_points = _build_required_points()
    results: list[dict[str, Any]] = []

    for i, pt in enumerate(required_points):
        family = pt["family"]
        n_layer = pt["n_layer"]
        n_repeat = pt["n_repeat"]
        seq_len = pt["seq_len"]
        model_name = _FAMILY_TO_MODEL[family]

        print(
            f"[{i + 1}/{len(required_points)}] "
            f"{family} {n_layer}L×{n_repeat}R @ seq={seq_len}  "
            f"model={model_name}",
            flush=True,
        )

        cfg = _build_synthetic_config(family, n_layer, n_repeat, seq_len, device)
        model = models.make_model_from_args(cfg).cuda().eval()

        macs_list = get_macs_for_seqlens(model, [seq_len])
        macs = macs_list[0]

        print(f"    MACs = {macs:,}", flush=True)

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

        del model
        torch.cuda.empty_cache()

    return results


def main() -> None:
    """Parse CLI, compute MACs, write macs.json."""
    parser = argparse.ArgumentParser(
        description="Compute analytical MAC counts for all required architecture points.",
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
        default="cuda:0",
        type=str,
        help="CUDA device string (default: cuda:0)",
    )
    args = parser.parse_args()

    # Deferred imports — keep module-level import-free for testability.
    import torch  # noqa: PLC0415

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    git_commit = _get_git_commit(_REPO_ROOT)
    compute_date = datetime.datetime.utcnow().isoformat() + "Z"

    print(f"compute_macs.py — device={args.device}  git={git_commit}", flush=True)
    print(f"Output: {args.output}", flush=True)

    points = compute_all_macs(args.device)

    output = {
        "schema_version": "1.0",
        "metadata": {
            "torch_version": torch.__version__,
            "device": args.device,
            "ptflops_backend": "aten",
            "arch_defaults": {
                "n_head": 12,
                "n_embd": 768,
                "vocab_size": 50304,
                "positional_encoder": "rotary",
                "bias": False,
                "n_layer_begin": 0,
                "n_layer_end": 0,
            },
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
