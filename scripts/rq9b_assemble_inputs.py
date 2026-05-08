"""Assemble RQ9b ANOVA inputs from a counting-sweep RUN_DIR.

Bridges the per-cell training output (``summary.json`` written by
``main.py:249``) plus the post-eval per-cell output (``eval_summary_*.json``
written by ``eval.py:382``) into the per-cell schema consumed by
``analysis.counting_anova_rq9b`` (one ``<variant>_seed_<N>.json`` per cell
under ``--results-dir``; see ``analysis/counting_anova_rq9b.py`` docstring
"Input format" for the field list).

Discovery
---------
``--run-dir`` is walked recursively; each directory containing a
``summary.json`` file is treated as one cell. The cell's variant is parsed
from ``summary.json["args"]["exp_name"]`` (job.sh names cells
``rq9_arm_<arm>_<variant>_seed_<seed>_nembd_<nembd>``); seed and n_embd are
read from ``summary.json["args"]``. The exact-match OOD accuracy is read
from the most recent ``eval_summary_*.json`` in the same directory under
the ``exact_match_acc`` key (eval.py:293) and renamed to
``exact_match_accuracy`` for the canonical RQ9b schema.

Variant filter
--------------
RQ9b's mapping covers V1, V2, V3, V4 only (per DIR-001 / DEC-033). Cells
whose parsed variant falls outside this set (e.g. Arm A "baseline") are
skipped with a warning -- emitting them would make
``counting_anova_rq9b.run_rq9b_anova`` raise on the unknown variant. Pass
``--include-all-variants`` to keep them anyway (escape hatch for ad-hoc
inspection).

Defensive design
----------------
- Missing ``summary.json``: not a cell directory, skipped silently
  (handled implicitly by the os.walk gate).
- Malformed ``summary.json``: warned, skipped, continues.
- No ``eval_summary_*.json``: warned, skipped (no accuracy available).
- Multiple ``eval_summary_*.json``: take the most recent by mtime.
- Missing ``exact_match_acc`` key in eval summary: warned, skipped.
- Unknown variant token (and ``--include-all-variants`` not set): warned,
  skipped.

Single source of truth
----------------------
The output filename ``<variant>_seed_<seed>_n<n_embd>.json`` matches the
naming pattern expected by ``analysis.counting_anova_rq9b``'s loader
(``load_cells_from_results_dir``: any ``*.json`` in the directory). The
bash launcher does NOT mirror this construction; it is computed solely in
this module.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any


# Variant token regex: matches the variant slug embedded in EXP_NAME (see
# iridis/counting-sweep/job.sh L130: rq9_arm_<arm>_<VARIANT>_seed_...).
# The pattern is anchored on "_arm_<arm_lower>_" prefix and "_seed_" suffix
# so the variant capture is unambiguous on both arms.
_EXP_NAME_VARIANT_RE = re.compile(r"_arm_[ab]_(?P<variant>[A-Za-z0-9]+)_seed_")


# Canonical RQ9b variant set per DIR-001 / DEC-033 (see
# analysis/counting_anova_rq9b.py _DIR001_FACTOR_MAPPING / _STRICT_FACTOR_MAPPING).
RQ9B_VARIANTS = ("V1", "V2", "V3", "V4")


# Required keys that the glue script verifies in each source file before
# emitting a cell. Listed as a comma-separated string on the CLI; the
# default mirrors the canonical RQ9b output schema.
DEFAULT_REQUIRED_KEYS = "variant,seed,n_embd,exact_match_accuracy"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Walk a counting-sweep RUN_DIR for per-cell summary.json + "
            "eval_summary_*.json files; emit one <variant>_seed_<N>.json "
            "per cell in the schema consumed by "
            "analysis.counting_anova_rq9b."
        )
    )
    parser.add_argument(
        "--run-dir", required=True, type=str,
        help="Counting-sweep run directory to walk (e.g. "
             "iridis/counting-sweep/run_0). Discovery is recursive: any "
             "directory containing summary.json is treated as one cell.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str,
        help="Output directory; one <variant>_seed_<seed>_n<n_embd>.json "
             "is written per discovered cell.",
    )
    parser.add_argument(
        "--required-keys", type=str, default=DEFAULT_REQUIRED_KEYS,
        help=f"Comma-separated keys that must be derivable for each cell "
             f"(default: {DEFAULT_REQUIRED_KEYS}). Cells missing any "
             "required key are skipped with a warning.",
    )
    parser.add_argument(
        "--include-all-variants", action="store_true",
        help="Emit cells whose parsed variant is outside the RQ9b "
             "{V1, V2, V3, V4} set (default: skip with warning). "
             "Escape hatch; counting_anova_rq9b will reject unknown "
             "variants at load time.",
    )
    return parser


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _load_summary_json(cell_dir: str) -> dict[str, Any] | None:
    path = os.path.join(cell_dir, "summary.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"failed to load {path}: {exc}; skipping cell")
        return None


def _latest_eval_summary(cell_dir: str) -> dict[str, Any] | None:
    """Return the parsed contents of the most recent eval_summary_*.json.

    Returns None when no matching file exists OR when the file fails to
    parse. The caller is expected to handle the None case (skip cell with
    warning).
    """
    candidates = sorted(
        glob.glob(os.path.join(cell_dir, "eval_summary_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        return None
    latest = candidates[0]
    try:
        with open(latest, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"failed to load {latest}: {exc}; skipping cell")
        return None


def _parse_variant_from_exp_name(exp_name: str) -> str | None:
    m = _EXP_NAME_VARIANT_RE.search(exp_name)
    if m is None:
        return None
    return m.group("variant")


def _extract_cell_record(
    cell_dir: str, include_all_variants: bool,
) -> dict[str, Any] | None:
    """Build the canonical cell record from one cell directory.

    Returns ``None`` when the cell cannot be assembled (missing inputs,
    parse failure, unknown variant under default filtering). All skip
    paths log a warning so the caller's tally is informative.
    """
    summary = _load_summary_json(cell_dir)
    if summary is None:
        return None

    args_dict = summary.get("args")
    if not isinstance(args_dict, dict):
        _warn(f"{cell_dir}/summary.json has no 'args' dict; skipping cell")
        return None

    exp_name = args_dict.get("exp_name")
    if not isinstance(exp_name, str):
        _warn(
            f"{cell_dir}/summary.json args.exp_name missing/non-str; "
            "skipping cell"
        )
        return None

    variant = _parse_variant_from_exp_name(exp_name)
    if variant is None:
        _warn(
            f"{cell_dir}: could not parse variant from exp_name={exp_name!r}; "
            "skipping cell"
        )
        return None

    if variant not in RQ9B_VARIANTS and not include_all_variants:
        _warn(
            f"{cell_dir}: variant={variant!r} not in RQ9b set "
            f"{RQ9B_VARIANTS}; skipping (pass --include-all-variants to keep)"
        )
        return None

    seed = args_dict.get("seed")
    n_embd = args_dict.get("n_embd")
    if seed is None or n_embd is None:
        _warn(
            f"{cell_dir}/summary.json args missing seed/n_embd "
            f"(seed={seed!r}, n_embd={n_embd!r}); skipping cell"
        )
        return None

    eval_summary = _latest_eval_summary(cell_dir)
    if eval_summary is None:
        _warn(
            f"{cell_dir}: no eval_summary_*.json found or parseable; "
            "skipping cell"
        )
        return None

    accuracy = eval_summary.get("exact_match_acc")
    if accuracy is None:
        _warn(
            f"{cell_dir}: eval_summary missing 'exact_match_acc' key; "
            "skipping cell"
        )
        return None

    return {
        "variant": str(variant),
        "seed": int(seed),
        "n_embd": int(n_embd),
        "exact_match_accuracy": float(accuracy),
    }


def _discover_cell_dirs(run_dir: str) -> list[str]:
    """Walk run_dir for directories containing summary.json."""
    cells: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        if "summary.json" in filenames:
            cells.append(dirpath)
    return sorted(cells)


def _validate_required_keys(record: dict[str, Any], required: list[str]) -> str | None:
    """Return None if all required keys are present + non-None; else a
    descriptive missing-key message."""
    missing = [k for k in required if record.get(k) is None]
    if missing:
        return f"missing required keys {missing}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        print(f"ERROR: --run-dir {args.run_dir!r} is not a directory",
              file=sys.stderr)
        return 2

    required = [k.strip() for k in args.required_keys.split(",") if k.strip()]

    cell_dirs = _discover_cell_dirs(args.run_dir)
    if not cell_dirs:
        print(
            f"ERROR: no cell directories (containing summary.json) found "
            f"under {args.run_dir!r}",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    n_processed = 0
    n_skipped = 0
    written: list[str] = []
    for cell_dir in cell_dirs:
        record = _extract_cell_record(
            cell_dir, include_all_variants=args.include_all_variants,
        )
        if record is None:
            n_skipped += 1
            continue
        missing_msg = _validate_required_keys(record, required)
        if missing_msg is not None:
            _warn(f"{cell_dir}: {missing_msg}; skipping cell")
            n_skipped += 1
            continue
        out_basename = (
            f"{record['variant']}_seed_{record['seed']}"
            f"_n{record['n_embd']}.json"
        )
        out_path = os.path.join(args.output_dir, out_basename)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        written.append(out_path)
        n_processed += 1

    print(
        f"rq9b_assemble_inputs: processed {n_processed} cell(s), "
        f"skipped {n_skipped} cell(s); wrote {len(written)} file(s) to "
        f"{args.output_dir}"
    )
    if n_processed == 0:
        print(
            "ERROR: no cells produced a valid record; nothing to ANOVA",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
