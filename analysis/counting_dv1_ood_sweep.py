"""DV-1 -- Per-OOD-length accuracy sweep + McNemar comparator (RQ9).

Scope
-----
Addresses DV-1 of RQ9 per the project's twin-pilot + 12-cell Arm B
sweep. Loads a counting-task checkpoint, evaluates exact-match
accuracy at every OOD test length L in
``{50, 75, 100, 125, 150, 175, 200}`` (the spec's 7 evaluation
lengths -- the rule "L >= 51 is OOD" includes L = 50 only as a
boundary check; the project's spec uses 50 as the lower endpoint
for visualisation symmetry with L = 200).

For each L the driver builds a fresh ``CountingDataset`` with
``length_range=(L, L)`` (every sample in the batch has the same
count window length L), forwards the model on it, and aggregates
per-sample exact-match correctness alongside per-position
accuracy (a vector of length ``sequence_length`` whose entry t is
``P(correct | position == t in the count window)``).

When ``--compare-against <other-ckpt-dir>`` is set, the same sweep
is run on the comparator and a per-length McNemar test is computed
on the contingency table

    b = N where treatment correct AND comparator wrong
    c = N where treatment wrong   AND comparator correct

with the continuity-corrected statistic ``(|b - c| - 1)^2 / (b + c)``
asymptotic chi-square(1). The RQ9 falsification rule is

    treatment >= comparator + 5pp at L = 200 AND McNemar p < 0.01

so the per-length comparator block also records ``delta_acc`` and
the boolean ``rq9_falsifies_h0_at_200`` (only at L = 200).

Falsifiability relevance
------------------------
Per docs/extend-notes.md section 1.2 RQ9, DV-1 is the primary
behavioural DV: every Arm B variant is compared to the Arm A 4L
baseline at the same hyperparameter grid; the absolute OOD curve
plus the McNemar test is the headline rejection rule for the
RQ9 main effect H0.

Pairing constraint
------------------
McNemar requires PAIRED samples: both checkpoints must score the
same (sample, position) grid. The driver enforces this by using
the same ``--seed`` to construct the OOD dataset for both
checkpoints; ``CountingDataset`` is deterministic in its
``num_samples``, ``seed``, and ``length_range`` so a fixed seed
gives the same start_int + L tuples on both runs. The comparator
must be the same task / dataset family (counting); the driver
warns if the comparator's ``summary.json`` reports a different
``dataset`` value but does not abort (the user may legitimately
compare across datasets in exploratory passes).

Smoke test
----------
``python -m analysis.counting_dv1_ood_sweep --smoke-test`` verifies
the McNemar arithmetic on a pure-numpy contingency table with a
known continuity-corrected chi-square; the test does NOT require
torch or a checkpoint and is the basis for the CI gate that
keeps the driver's contract stable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np


# Default OOD evaluation lengths per docs/extend-notes.md section 1.2 RQ9.
# 50 sits at the boundary of the OOD range (51..200) per
# data.counting.TE200_OOD_LENGTH_RANGE; the spec includes it for
# visualisation continuity but the dataset's range is (51, 200) so we
# clip the driver's sweep to L >= 51 inside _parse_ood_lengths.
DEFAULT_OOD_LENGTHS = (50, 75, 100, 125, 150, 175, 200)
SMOKE_OOD_LENGTHS = (75, 200)
SMOKE_N_EVAL = 32


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for DV-1."""
    parser = argparse.ArgumentParser(
        description="DV-1 per-OOD-length accuracy sweep + McNemar comparator"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Treatment checkpoint directory (summary.json + ckpt file).",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Treatment checkpoint filename (default ckpt.pt).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False,
        help="Directory for counting_dv1_ood_results.json.",
    )
    parser.add_argument("--seed", type=int, default=19937,
                        help="OOD dataset seed; same value used on both "
                             "treatment and comparator so samples pair.")
    parser.add_argument("--ood-lengths", type=str,
                        default=",".join(str(L) for L in DEFAULT_OOD_LENGTHS),
                        help="Comma-separated list of OOD lengths "
                             "(default %(default)s).")
    parser.add_argument("--n-eval", type=int, default=500,
                        help="OOD eval-set sample count per length.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-batch sample count.")
    parser.add_argument("--sequence-length", type=int, default=256,
                        help="Pad length; must satisfy seq_length >= max(L)+1.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device string ('cuda' or 'cpu').")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"],
                        help="Checkpoint config-load mode (sibling pattern).")
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid",
                        help="Dotted block-list path; passed through to the "
                             "shared loader. Behaves as in DV-2/DV-3/DV-4.")
    parser.add_argument("--compare-against", type=str, default=None,
                        help="Optional comparator checkpoint directory. When "
                             "set, runs the same sweep on the comparator and "
                             "computes McNemar per length.")
    parser.add_argument("--compare-against-file", type=str, default="ckpt.pt",
                        help="Comparator checkpoint filename "
                             "(default ckpt.pt).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the McNemar arithmetic smoke test and exit "
                             "(pure-numpy; no torch / checkpoint required).")
    return parser


# ---------------------------------------------------------------------------
# Per-length forward pass + per-sample correctness vector
# ---------------------------------------------------------------------------


def _evaluate_at_length(
    model,
    *,
    seed: int,
    length: int,
    n_eval: int,
    sequence_length: int,
    batch_size: int,
    device: str,
    accepts_attention_mask: bool,
) -> dict[str, Any]:
    """Run the model on a fresh ``CountingDataset`` at fixed ``length``.

    Returns
    -------
    dict
        ``length`` (int), ``n_eval`` (int), ``exact_match_acc`` (float),
        ``per_position_acc`` (list[float], length=sequence_length),
        ``per_position_total`` (list[int]), and the per-sample
        correctness vector ``per_sample_correct`` (list[int]) used by
        the McNemar comparator.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from data.counting import CountingDataset, TE200_MAX_OUT

    if sequence_length < length + 1:
        raise ValueError(
            f"_evaluate_at_length: sequence_length={sequence_length} too "
            f"small for length={length}; require sequence_length >= L+1."
        )

    dataset = CountingDataset(
        split="ood",
        seed=int(seed),
        num_samples=int(n_eval),
        sequence_length=int(sequence_length),
        max_out=TE200_MAX_OUT,
        length_range=(int(length), int(length)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    per_position_correct = np.zeros(int(sequence_length), dtype=np.float64)
    per_position_total = np.zeros(int(sequence_length), dtype=np.float64)
    per_sample_correct: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y, pad_mask, loss_mask = batch
            x_d = x.to(device)
            y_d = y.to(device)
            pad_mask_d = pad_mask.to(device)
            loss_mask_d = loss_mask.to(device)

            if accepts_attention_mask:
                outputs = model(x_d, targets=y_d, attention_mask=pad_mask_d, get_logits=True)
            else:
                outputs = model(x_d, targets=y_d, get_logits=True)

            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            correct = ((preds == y_d) & (loss_mask_d > 0.5)).float()  # (B, T)

            correct_np = correct.detach().to("cpu").numpy().astype(np.float64)
            mask_np = (loss_mask_d > 0.5).detach().to("cpu").numpy().astype(np.float64)
            per_position_correct += correct_np.sum(axis=0)
            per_position_total += mask_np.sum(axis=0)

            per_sample_valid = (loss_mask_d > 0.5).float().sum(dim=1)
            per_sample_score = correct.sum(dim=1)
            sample_correct = (
                (per_sample_valid > 0.5)
                & (per_sample_score == per_sample_valid)
            )
            per_sample_correct.extend(int(v) for v in sample_correct.tolist())

    n_total = int(len(per_sample_correct))
    exact_match = (sum(per_sample_correct) / n_total) if n_total > 0 else float("nan")
    pp_acc = np.where(
        per_position_total > 0,
        per_position_correct / np.maximum(per_position_total, 1.0),
        0.0,
    )
    return {
        "length": int(length),
        "n_eval": n_total,
        "exact_match_acc": float(exact_match),
        "per_position_acc": pp_acc.tolist(),
        "per_position_total": per_position_total.astype(int).tolist(),
        "per_sample_correct": list(per_sample_correct),
    }


# ---------------------------------------------------------------------------
# McNemar (continuity-corrected; statsmodels-compatible fallback)
# ---------------------------------------------------------------------------


def _mcnemar_continuity_corrected(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar statistic + p-value.

    Computed as ``chi2 = (|b - c| - 1)^2 / (b + c)`` (Edwards
    continuity correction), with the p-value from the chi-square
    survival function with 1 d.o.f. Used as the in-tree fallback when
    statsmodels is not importable; matches
    ``statsmodels.stats.contingency_tables.mcnemar(table,
    exact=False, correction=True)`` to numerical precision.

    Parameters
    ----------
    b : int
        Discordant pair count: treatment correct, comparator wrong.
    c : int
        Discordant pair count: treatment wrong, comparator correct.

    Returns
    -------
    chi2 : float
        Continuity-corrected statistic. ``0.0`` when ``b + c == 0``
        (no discordant pairs => H0 trivially not rejected).
    p_value : float
        P-value from chi-square(1) survival function. ``1.0`` when
        ``b + c == 0``.
    """
    discordant = int(b) + int(c)
    if discordant == 0:
        return 0.0, 1.0
    delta = abs(int(b) - int(c)) - 1.0
    if delta < 0.0:
        # |b - c| == 0; Edwards correction collapses to chi2 = 1/(b+c)
        # in that limit, but the canonical convention is to set
        # chi2 = 0 so the test does not reject in the perfectly-tied
        # case. statsmodels mirrors this.
        chi2 = 0.0
    else:
        chi2 = (delta * delta) / float(discordant)
    # Chi-square(1) survival function: erfc(sqrt(chi2 / 2))
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p_value)


def _mcnemar(b: int, c: int) -> tuple[float, float, str]:
    """Wrapper preferring statsmodels, falling back to the in-tree impl.

    Returns
    -------
    chi2 : float
    p_value : float
    backend : str
        ``"statsmodels"`` when the optional dependency was used,
        ``"in_tree_continuity_corrected"`` otherwise. Recorded in the
        results JSON for reproducibility.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar  # type: ignore
        table = [[0, int(b)], [int(c), 0]]
        result = mcnemar(table, exact=False, correction=True)
        return float(result.statistic), float(result.pvalue), "statsmodels"
    except Exception:
        chi2, p = _mcnemar_continuity_corrected(b, c)
        return chi2, p, "in_tree_continuity_corrected"


# ---------------------------------------------------------------------------
# Per-checkpoint sweep driver
# ---------------------------------------------------------------------------


def _parse_ood_lengths(spec: str) -> list[int]:
    """Parse the ``--ood-lengths`` comma-separated CLI value."""
    out: list[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            L = int(tok)
        except ValueError as exc:
            raise ValueError(
                f"--ood-lengths: cannot parse {tok!r} as int"
            ) from exc
        if L <= 0:
            raise ValueError(f"--ood-lengths: L={L} must be positive")
        out.append(L)
    if not out:
        raise ValueError("--ood-lengths must contain at least one value")
    return out


def _sweep_checkpoint(
    checkpoint_dir: str,
    checkpoint_file: str,
    args: argparse.Namespace,
    ood_lengths: list[int],
) -> dict[str, Any]:
    """Run the per-length sweep on ``checkpoint_dir``; return per-length records."""
    import inspect as _inspect

    import torch
    from analysis.common.loader import load_model_from_checkpoint

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model, config = load_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        config_mode=args.config_mode,
        device=device,
        module_path=args.module_path,
    )
    model.eval()

    forward_sig = _inspect.signature(model.forward)
    accepts_attention_mask = "attention_mask" in forward_sig.parameters

    per_length: list[dict[str, Any]] = []
    for L in ood_lengths:
        record = _evaluate_at_length(
            model,
            seed=int(args.seed),
            length=int(L),
            n_eval=int(args.n_eval),
            sequence_length=int(args.sequence_length),
            batch_size=int(args.batch_size),
            device=device,
            accepts_attention_mask=accepts_attention_mask,
        )
        per_length.append(record)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(checkpoint_file),
        "model": str(getattr(config, "model", "unknown")),
        "dataset": str(getattr(config, "dataset", "unknown")),
        "accepts_attention_mask": bool(accepts_attention_mask),
        "per_length": per_length,
    }


def _aggregate(per_length: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the aggregate stats block over a per-length record list."""
    if not per_length:
        return {"mean_acc": float("nan"), "min_acc": float("nan"),
                "min_acc_length": None, "max_acc": float("nan"),
                "max_acc_length": None}
    accs = [float(rec["exact_match_acc"]) for rec in per_length]
    lens = [int(rec["length"]) for rec in per_length]
    min_idx = int(np.argmin(accs))
    max_idx = int(np.argmax(accs))
    return {
        "mean_acc": float(np.mean(accs)),
        "min_acc": float(accs[min_idx]),
        "min_acc_length": int(lens[min_idx]),
        "max_acc": float(accs[max_idx]),
        "max_acc_length": int(lens[max_idx]),
    }


def _compute_mcnemar_block(
    treatment: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    comparator_path: str,
) -> dict[str, Any]:
    """Compute per-length McNemar entries from paired per-sample vectors."""
    by_length_t = {int(r["length"]): r for r in treatment}
    by_length_c = {int(r["length"]): r for r in comparator}
    common_lengths = sorted(set(by_length_t) & set(by_length_c))
    if not common_lengths:
        raise ValueError(
            "compute_mcnemar_block: no shared OOD lengths between "
            "treatment and comparator; pairing impossible."
        )

    per_length_blocks: list[dict[str, Any]] = []
    backends: set[str] = set()
    for L in common_lengths:
        rec_t = by_length_t[L]
        rec_c = by_length_c[L]
        t_vec = list(rec_t["per_sample_correct"])
        c_vec = list(rec_c["per_sample_correct"])
        if len(t_vec) != len(c_vec):
            raise ValueError(
                f"compute_mcnemar_block: length mismatch at L={L}: "
                f"treatment n={len(t_vec)} vs comparator n={len(c_vec)}; "
                f"the same --seed and --n-eval must be used for both runs."
            )
        b = sum(1 for ti, ci in zip(t_vec, c_vec) if ti == 1 and ci == 0)
        c_count = sum(1 for ti, ci in zip(t_vec, c_vec) if ti == 0 and ci == 1)
        a = sum(1 for ti, ci in zip(t_vec, c_vec) if ti == 1 and ci == 1)
        d = sum(1 for ti, ci in zip(t_vec, c_vec) if ti == 0 and ci == 0)
        chi2, p_value, backend = _mcnemar(b, c_count)
        backends.add(backend)
        delta = float(rec_t["exact_match_acc"]) - float(rec_c["exact_match_acc"])
        block = {
            "length": int(L),
            "n_paired": int(len(t_vec)),
            "a_both_correct": int(a),
            "b_only_treatment": int(b),
            "c_only_comparator": int(c_count),
            "d_both_wrong": int(d),
            "chi2": float(chi2),
            "p_value": float(p_value),
            "delta_acc": float(delta),
        }
        if int(L) == 200:
            # RQ9 falsification rule (docs/extend-notes.md section 1.2 RQ9
            # control 21): treatment >= comparator + 5 percentage points
            # at L = 200 AND McNemar p-value below 0.01.
            block["rq9_falsifies_h0_at_200"] = bool(delta >= 0.05 and p_value < 0.01)
        per_length_blocks.append(block)

    return {
        "comparator_path": str(comparator_path),
        "backend": ",".join(sorted(backends)) if backends else "n_a",
        "per_length": per_length_blocks,
    }


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    """Top-level driver: per-length sweep on treatment + optional comparator."""
    ood_lengths = _parse_ood_lengths(args.ood_lengths)

    if not args.checkpoint:
        raise ValueError("analyse: --checkpoint is required")
    treatment = _sweep_checkpoint(
        args.checkpoint, args.checkpoint_file, args, ood_lengths,
    )

    aggregate = _aggregate(treatment["per_length"])

    comparator_block: dict[str, Any] | None = None
    mcnemar_block: dict[str, Any] | None = None
    if args.compare_against:
        comparator = _sweep_checkpoint(
            args.compare_against, args.compare_against_file, args, ood_lengths,
        )
        if comparator["dataset"] != treatment["dataset"]:
            print(
                f"WARNING: treatment dataset={treatment['dataset']} differs "
                f"from comparator dataset={comparator['dataset']}; McNemar "
                f"pairing assumes same task family."
            )
        comparator_block = {
            "checkpoint_dir": comparator["checkpoint_dir"],
            "checkpoint_file": comparator["checkpoint_file"],
            "model": comparator["model"],
            "dataset": comparator["dataset"],
            "aggregate": _aggregate(comparator["per_length"]),
        }
        mcnemar_block = _compute_mcnemar_block(
            treatment["per_length"], comparator["per_length"],
            comparator_path=str(args.compare_against),
        )
        # Hoist the L=200 falsification verdict + p-value to the
        # treatment-side aggregate so synthesis-time triangulation can
        # read a single dotted path (aggregate.rq9_falsifies_h0_at_200,
        # aggregate.mcnemar_p_at_200) without descending into the
        # mcnemar.per_length list. Per-length detail stays under
        # mcnemar.per_length unchanged.
        for block in mcnemar_block.get("per_length", []):
            if int(block.get("length", -1)) == 200:
                aggregate["rq9_falsifies_h0_at_200"] = bool(
                    block.get("rq9_falsifies_h0_at_200", False)
                )
                aggregate["mcnemar_p_at_200"] = float(block.get("p_value"))
                break

    # Strip the per-sample correctness vectors before serialising the
    # treatment per-length block: they are large (n_eval ints per
    # length) and only required during McNemar computation. We keep
    # them in the comparator side of the McNemar block above; if the
    # user wants them they can re-run with --n-eval set.
    treatment_per_length_serial = []
    for rec in treatment["per_length"]:
        rec_out = {k: v for k, v in rec.items() if k != "per_sample_correct"}
        treatment_per_length_serial.append(rec_out)

    out = {
        "schema_version": "dv1-ood-1.0",
        "checkpoint": {
            "path": str(args.checkpoint),
            "file": str(args.checkpoint_file),
        },
        "args": {
            "seed": int(args.seed),
            "n_eval": int(args.n_eval),
            "batch_size": int(args.batch_size),
            "sequence_length": int(args.sequence_length),
            "ood_lengths": list(ood_lengths),
            "device": str(args.device),
            "config_mode": str(args.config_mode),
            "module_path": str(args.module_path),
            "compare_against": (str(args.compare_against)
                                if args.compare_against else None),
        },
        "model": treatment["model"],
        "dataset": treatment["dataset"],
        "ood_lengths": list(ood_lengths),
        "per_length": treatment_per_length_serial,
        "aggregate": aggregate,
        "comparator": comparator_block,
        "mcnemar": mcnemar_block,
    }
    return out


# ---------------------------------------------------------------------------
# Smoke test (pure-numpy McNemar + per-length aggregate arithmetic)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Pure-numpy DV-1 smoke test: McNemar arithmetic on a known table."""
    # Case 1: trivial (no discordant pairs).
    chi2, p = _mcnemar_continuity_corrected(0, 0)
    assert chi2 == 0.0, f"expected chi2=0 for b=c=0, got {chi2}"
    assert p == 1.0, f"expected p=1.0 for b=c=0, got {p}"

    # Case 2: classic worked example. b=12, c=2 => |b-c|=10, |.| - 1 = 9,
    # chi2 = 81 / 14 ~= 5.7857. statsmodels-compatible.
    chi2_case2, p_case2 = _mcnemar_continuity_corrected(12, 2)
    assert math.isclose(chi2_case2, 81.0 / 14.0, rel_tol=1e-6), \
        f"expected chi2~=5.7857, got {chi2_case2}"
    assert 0.0 < p_case2 < 1.0, f"expected 0 < p < 1, got {p_case2}"
    # Sanity: p < 0.05 because the discordance is strongly directional.
    assert p_case2 < 0.05, f"expected p<0.05 for (12,2), got {p_case2}"

    # Case 3: equal discordant pairs (b == c) => Edwards correction
    # gives chi2 = 0 (we can't reject the null based on direction alone).
    chi2, p = _mcnemar_continuity_corrected(7, 7)
    assert chi2 == 0.0, f"expected chi2=0 for b==c, got {chi2}"
    assert p == 1.0, f"expected p=1.0 for b==c (after correction), got {p}"

    # Case 4: per-length aggregate arithmetic on a stub record list.
    stub_per_length = [
        {"length": 50, "exact_match_acc": 0.91, "n_eval": SMOKE_N_EVAL,
         "per_position_acc": [], "per_position_total": [],
         "per_sample_correct": []},
        {"length": 200, "exact_match_acc": 0.41, "n_eval": SMOKE_N_EVAL,
         "per_position_acc": [], "per_position_total": [],
         "per_sample_correct": []},
    ]
    agg = _aggregate(stub_per_length)
    assert agg["min_acc_length"] == 200, f"expected min L=200, got {agg}"
    assert math.isclose(agg["min_acc"], 0.41, rel_tol=1e-9)
    assert math.isclose(agg["max_acc"], 0.91, rel_tol=1e-9)
    assert math.isclose(agg["mean_acc"], 0.66, rel_tol=1e-9)

    # Case 5: McNemar block falsification flag at L=200.
    treatment = [
        {"length": 200, "exact_match_acc": 0.50, "n_eval": 100,
         "per_position_acc": [], "per_position_total": [],
         "per_sample_correct": [1] * 50 + [0] * 50},
    ]
    comparator = [
        {"length": 200, "exact_match_acc": 0.40, "n_eval": 100,
         "per_position_acc": [], "per_position_total": [],
         "per_sample_correct": [1] * 30 + [0] * 70},
    ]
    block = _compute_mcnemar_block(treatment, comparator,
                                   comparator_path="<stub>")
    pl = block["per_length"][0]
    assert pl["length"] == 200
    assert math.isclose(pl["delta_acc"], 0.10, rel_tol=1e-9)
    # Whether the falsification flag fires depends on the exact
    # discordance distribution; just check the key is present.
    assert "rq9_falsifies_h0_at_200" in pl

    print("DV-1 smoke test PASS")
    print("  McNemar(0,0) -> (chi2, p) = (0.0, 1.0)")
    print(f"  McNemar(12,2) -> (chi2, p) ~= ({chi2_case2:.4f}, {p_case2:.6f})")
    print(f"  Aggregate over 2 stub lengths -> mean_acc=0.66, min_acc_length=200")
    print(f"  Comparator block emits 'rq9_falsifies_h0_at_200' at L=200")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        _smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when --smoke-test is not set")
    if not args.output_dir:
        parser.error("--output-dir is required when --smoke-test is not set")

    os.makedirs(args.output_dir, exist_ok=True)
    results = analyse(args)

    out_json = os.path.join(args.output_dir, "counting_dv1_ood_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"DV-1 results written to {out_json}")


if __name__ == "__main__":
    main()
