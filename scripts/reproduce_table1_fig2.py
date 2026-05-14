#!/usr/bin/env python3
"""Multi-checkpoint eval sweep joined with analytical MAC counts.

Runs eval.main() for each CoTFormer ablation checkpoint, reads the resulting
eval_summary JSON, looks up the corresponding MAC count from a pre-computed
macs.json, and assembles results_table1_fig2.json conforming to the schema
in docs/schema_table1_fig23.md.

Usage:
    python scripts/reproduce_table1_fig2.py \\
        --ckpt-root /scratch/ab3u21/job-outputs/owt2/cotformer_full_depth \\
        --ablations BaseCot_12L_2R BaseCot_12L_3R BaseCot_12L_5R \\
                    BaseCot_12L_15R BaseCot_24L_2R BaseCot_24L_3R BaseCot_24L_5R \\
        --output-dir run_1/json \\
        --macs-json run_1/json/macs.json \\
        [--eval-log-dir run_1/eval_logs] \\
        [--batch-size 32]
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from typing import Any

# Ensure the repo root is on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Constants — paper-reported reference values (schema §paper_reference_table1)
# ---------------------------------------------------------------------------

# Values quoted directly from Table 1 of the paper (Mohtashami et al. 2025,
# ICLR camera-ready, p.5). Format per schema: {ppl, sem} or {by_n_repeat: ...}.
_PAPER_REFERENCE_TABLE1: dict[str, Any] = {
    "standard_12L": {"n_layer": 12, "n_repeat": 1, "ppl": 28.39, "sem": 0.01},
    "block_universal_12L": {
        "n_layer": 12,
        "by_n_repeat": {
            "2": {"ppl": 27.74, "sem": 0.01},
            "3": {"ppl": 27.47, "sem": 0.01},
            "5": {"ppl": 27.15, "sem": 0.02},
        },
    },
    "cotformer_12L_paper": {
        "n_layer": 12,
        "by_n_repeat": {
            "2": {"ppl": 27.55, "sem": 0.02},
            "3": {"ppl": 27.07, "sem": 0.01},
            "5": {"ppl": 26.64, "sem": 0.04},
        },
    },
    "standard_24L": {"n_layer": 24, "n_repeat": 1, "ppl": 25.93, "sem": 0.02},
    "block_universal_24L": {
        "n_layer": 24,
        "by_n_repeat": {
            "2": {"ppl": 25.47, "sem": 0.00},
            "3": {"ppl": 25.19, "sem": 0.03},
            "5": {"ppl": 24.95, "sem": 0.01},
        },
    },
    "cotformer_24L_paper": {
        "n_layer": 24,
        "by_n_repeat": {
            "2": {"ppl": 25.28, "sem": 0.00},
            "3": {"ppl": 24.85, "sem": 0.04},
            "5": {"ppl": 24.48, "sem": 0.03},
        },
    },
    "standard_48L": {"n_layer": 48, "n_repeat": 1, "ppl": 24.17, "sem": 0.00},
}

# Fig 2 extra BUT points not in Table 1 (schema §paper_reference_fig2).
# PPL values read visually from the paper figure (±0.05 PPL precision).
_PAPER_REFERENCE_FIG2: dict[str, Any] = {
    "but_12L_extra": {
        "6": {"ppl_visual": 27.04},
        "15": {"ppl_visual": 26.85},
    }
}

# Regex to parse (n_layer, n_repeat) from ablation folder names like BaseCot_12L_3R.
_ABLATION_NAME_RE = re.compile(r"BaseCot_(\d+)L_(\d+)R$")


def _parse_ablation_name(name: str) -> tuple[int, int]:
    """Parse (n_layer, n_repeat) from an ablation folder name.

    Raises ValueError if the name does not match the expected pattern.
    """
    m = _ABLATION_NAME_RE.match(name)
    if m is None:
        raise ValueError(
            f"Ablation name '{name}' does not match pattern BaseCot_<N>L_<R>R"
        )
    return int(m.group(1)), int(m.group(2))


def _load_macs_json(path: str) -> list[dict[str, Any]]:
    """Load and return the 'points' list from macs.json."""
    with open(path) as f:
        data = json.load(f)
    return data["points"]


def _lookup_mac(
    points: list[dict[str, Any]],
    family: str,
    n_layer: int,
    n_repeat: int,
    seq_len: int = 256,
) -> int:
    """Return the MAC count for the given architecture point.

    Raises KeyError if no matching point is found.
    """
    for pt in points:
        if (
            pt["family"] == family
            and pt["n_layer"] == n_layer
            and pt["n_repeat"] == n_repeat
            and pt["seq_len"] == seq_len
        ):
            return int(pt["macs"])
    raise KeyError(
        f"No MAC point found for family={family} n_layer={n_layer} "
        f"n_repeat={n_repeat} seq_len={seq_len}"
    )


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


def _run_eval_for_ablation(
    ablation: str,
    ckpt_root: str,
    output_dir: str,
    eval_log_dir: str | None,
    batch_size: int | None,
    data_dir_override: str | None,
) -> dict[str, Any]:
    """Run eval.main() for a single ablation checkpoint.

    Returns the eval stats dict produced by eval.main().
    """
    import eval as eval_module  # noqa: PLC0415 — deferred to keep module importable without GPU

    per_ablation_dir = os.path.join(output_dir, "eval_per_ablation", ablation)
    os.makedirs(per_ablation_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_root, ablation, "ckpt.pt")

    eval_args = [
        "--checkpoint", ckpt_path,
        "--config_format", "base",
        "--distributed_backend", "None",
        "--output_dir", per_ablation_dir,
    ]
    if batch_size is not None:
        eval_args += ["--batch_size", str(batch_size)]

    # eval.py:45-52 unconditionally hydrates summary['args'] onto the namespace.
    # Colab-retrained ablations have data_dir='/content/data' baked in — useless
    # and detrimental on iridis. Override AFTER get_args, BEFORE main().
    parsed = eval_module.get_args(eval_args)
    if data_dir_override is not None:
        parsed.data_dir = data_dir_override

    stats = eval_module.main(parsed)
    return stats


def _build_ablation_entry(
    ablation: str,
    ckpt_root: str,
    summary: dict[str, Any],
    stats: dict[str, Any],
    mac_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the ablation entry conforming to schema §results_table1_fig2.json."""
    n_layer, n_repeat = _parse_ablation_name(ablation)

    macs_at_seq256 = _lookup_mac(mac_points, "CoTFormer", n_layer, n_repeat, seq_len=256)

    seed = summary.get("args", {}).get("seed", stats.get("seed", 0))

    entry: dict[str, Any] = {
        "name": ablation,
        "family": "CoTFormer",
        "n_layer": n_layer,
        "n_repeat": n_repeat,
        "model": "cotformer_full_depth",
        "seed": seed,
        "checkpoint_path": os.path.join(ckpt_root, ablation),
        "checkpoint_filename": "ckpt.pt",
        "eval": {
            "val_loss": stats["val_loss"],
            "val_loss_std": stats["val_loss_std"],
            "val_loss_ci95": stats["val_loss_ci95"],
            "val_perplexity": stats["val_perplexity"],
            "val_perplexity_ci95": stats["val_perplexity_ci95"],
            "val_acc": stats["val_acc"],
            "val_acc_std": stats["val_acc_std"],
            "n_batches": stats["n_batches"],
            "sequence_length": summary.get("sequence_length", stats.get("sequence_length", 256)),
            "batch_size": summary.get("batch_size", stats.get("batch_size", 32)),
            "eval_per_batch_time_ms": stats.get("eval_per_batch_time", None),
        },
        "macs_at_seq256": macs_at_seq256,
    }
    return entry


def main() -> None:
    """Entry point: parse CLI args, run eval sweep, write results JSON."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-checkpoint eval sweep joined with MACs. "
            "Writes results_table1_fig2.json conforming to schema_table1_fig23.md."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--ckpt-root",
        required=True,
        type=str,
        help="Root directory containing one subdirectory per ablation (e.g. BaseCot_12L_2R/).",
    )
    parser.add_argument(
        "--ablations",
        required=True,
        nargs="+",
        type=str,
        help="Ablation folder names (e.g. BaseCot_12L_2R BaseCot_12L_3R ...).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write results_table1_fig2.json and eval sub-dirs.",
    )
    parser.add_argument(
        "--macs-json",
        required=True,
        type=str,
        help="Path to pre-computed macs.json (from compute_macs.py).",
    )
    parser.add_argument(
        "--eval-log-dir",
        default=None,
        type=str,
        help="Optional separate directory for eval log files.",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Override batch size for eval (default: use checkpoint summary value).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        type=str,
        help=(
            "Override data_dir at eval time. Required when checkpoints were "
            "trained on a different host (e.g. Colab '/content/data') and the "
            "summary.json bakes in a path that does not exist on the eval host. "
            "If unset, eval.py hydrates data_dir from summary['args'] verbatim."
        ),
    )
    args = parser.parse_args()

    # Validate ablation names up-front before any GPU work.
    for name in args.ablations:
        _parse_ablation_name(name)  # raises ValueError on bad name

    os.makedirs(args.output_dir, exist_ok=True)

    mac_points = _load_macs_json(args.macs_json)

    # Deferred heavy imports — kept here so argparse --help works without GPU.
    import torch  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    git_commit = _get_git_commit(_REPO_ROOT)
    eval_date = datetime.datetime.utcnow().isoformat() + "Z"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ablation_entries: list[dict[str, Any]] = []

    for ablation in tqdm(args.ablations, desc="Ablation eval sweep"):
        print(f"\n--- Running eval: {ablation} ---", flush=True)

        stats = _run_eval_for_ablation(
            ablation=ablation,
            ckpt_root=args.ckpt_root,
            output_dir=args.output_dir,
            eval_log_dir=args.eval_log_dir,
            batch_size=args.batch_size,
            data_dir_override=args.data_dir,
        )

        # Read the checkpoint's summary.json for seed and sequence_length metadata.
        ckpt_dir = os.path.join(args.ckpt_root, ablation)
        summary_json_path = os.path.join(ckpt_dir, "summary.json")
        with open(summary_json_path) as f:
            summary = json.load(f)

        entry = _build_ablation_entry(
            ablation=ablation,
            ckpt_root=args.ckpt_root,
            summary=summary,
            stats=stats,
            mac_points=mac_points,
        )
        ablation_entries.append(entry)
        print(
            f"    val_ppl={entry['eval']['val_perplexity']:.4f}  "
            f"macs_256={entry['macs_at_seq256']:,}",
            flush=True,
        )

    output = {
        "schema_version": "1.0",
        "metadata": {
            "ckpt_root": args.ckpt_root,
            "eval_date_utc": eval_date,
            "torch_version": torch.__version__,
            "git_commit": git_commit,
            "device": device,
            "notes": (
                "Paper Table 1 reports SEM over 3 seeds. We have 1 seed per "
                "ablation; uncertainty is the per-batch CI95 from eval.py "
                "(different statistical interpretation — see reprod-notes §A8)."
            ),
        },
        "ablations": ablation_entries,
        "paper_reference_table1": _PAPER_REFERENCE_TABLE1,
        "paper_reference_fig2": _PAPER_REFERENCE_FIG2,
    }

    out_path = os.path.join(args.output_dir, "results_table1_fig2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Wrote {len(ablation_entries)} ablation entries to {out_path}", flush=True)


if __name__ == "__main__":
    main()
