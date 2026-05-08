"""Cross-checkpoint synthesis (plotting-only).

Scope
-----
Not a scientific protocol; presentation-only. Reads per-checkpoint
JSON artefacts produced by Protocols A through I and emits
cross-checkpoint comparative figures: RQ1 convergence curves
overlaid for C1 / C2 / C3 / C4 / C5, CKA grand matrices side-by-side,
RQ6 d_eff trajectories, Protocol G effective-rank summary, and an
overall ``synthesis_report.md`` with embedded figure references.

Ontological purpose
-------------------
Communication artefact; no variables, no hypotheses, no confounder
analysis. Exempt from the ontological audit of a scientific protocol
because it computes nothing -- every datum plotted is computed
upstream by the protocol scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable

# Tag pattern for the analyze-lncot+adm package: c1_, c2_, ..., c5_, c5b_, ...
TAG_PATTERN = re.compile(r"^c[1-5][a-z]?_")

# Per-protocol JSON filenames discovered under each <run-dir>/<tag>/.
# Tuned Lens emits TWO artefacts (diagnostic + triangulation); both are
# loaded under distinct keys so synthesis can pick the right verdict
# domain (training-convergence vs cross-lens agreement).
PROTOCOL_FILES: dict[str, str] = {
    "logit_lens": "logit_lens_results.json",
    "cka": "cka_results.json",
    "attention_taxonomy": "attention_taxonomy_results.json",
    "router_analysis": "router_analysis_results.json",
    "effective_dim": "effective_dim_results.json",
    "kv_rank": "kv_rank_results.json",
    "interpolation_validity": "interpolation_validity_results.json",
    "depth_emb_freeze": "depth_emb_freeze_results.json",
    "residual_diagnostics": "residual_diagnostics_results.json",
    "tuned_lens_diagnostic": "tuned_lens_diagnostic.json",
    "tuned_lens_triangulation": "tuned_lens_triangulation.json",
}

# d_cal/aggregate/verdict.json lives under <run-dir>/d_cal/aggregate/, not
# under a tag directory: it is run-level, not checkpoint-level.
D_CAL_VERDICT_RELPATH = os.path.join("d_cal", "aggregate", "verdict.json")


# Counting (RQ9) per-cell artefact relative paths under each cell directory
# discovered in --counting-run-dir. DV-1 OOD is a single-file artefact (sister
# worker output); DV-2 is per-task; DV-3 and DV-4 are single-file per-cell.
# Loading is defensive: any missing file is skipped silently, mirroring the
# analytic-tag loader's "every datum is computed upstream" stance.
COUNTING_DV1_OOD_RELPATH = os.path.join(
    "dv1_ood", "counting_dv1_ood_results.json"
)
COUNTING_DV2_TASK_FILENAME = "counting_dv2_results.json"
COUNTING_DV2_TASKS = ("task1", "task2", "task3")
COUNTING_DV3_RELPATH = os.path.join("dv3", "counting_dv3_attention_results.json")
COUNTING_DV4_RELPATH = os.path.join("dv4", "counting_dv4_causal_results.json")

# Cell-name regex for the counting sweep. job.sh emits cell directories named
# rq9_arm_(a|b)_<variant>_seed_<seed>_nembd_<n_embd>; we parse arm/variant/
# n_embd/seed from the directory name. Cells whose name does not match this
# pattern still load (the metadata fields fall through to None) so off-spec
# names do not crash synthesis.
COUNTING_CELL_NAME_RE = re.compile(
    r"^rq9_arm_(?P<arm>[ab])_(?P<variant>[A-Za-z0-9]+)"
    r"_seed_(?P<seed>\d+)_nembd_(?P<nembd>\d+)$"
)

# Synthetic per-cell tag prefix used to namespace counting cells inside the
# unified ``artefacts`` dict. Distinct from the analytic c-tag pattern so the
# analytic-only flow remains byte-identical when --counting-run-dir is unset.
COUNTING_TAG_PREFIX = "__counting__/"

# Triangulation matrix per docs/extend-notes.md section 1.6. Each row is a
# claim with primary / secondary / tertiary measurements and the protocol +
# JSON-key path used to locate the per-checkpoint verdict (or "preliminary"
# downgrade).  A path of None means "not encoded as a single-key verdict in
# the artefact" -- those rows fall back to artefact-presence as evidence.
# Optional keys ``primary_label``, ``secondary_label``, ``tertiary_label``
# carry per-cell prose used in the markdown report (default: capitalised
# tier name).  Optional keys ``status`` and ``deferred_note`` allow rows
# to declare deferred / preliminary status without breaking the schema.
TRIANGULATION_ROWS: list[dict[str, Any]] = [
    {
        # Cross-lens agreement (tuned vs unweighted logit lens) lives in
        # tuned_lens_triangulation.json under "agreement_band". The
        # tuned_lens_diagnostic.json "verdict" is a translator-training
        # convergence label (>=95/105 translators beat baseline KL),
        # NOT the scientific monotonicity claim, so the secondary tier
        # uses logit_lens's paired-t verdict and the tertiary tier
        # demotes CKA to a supplementary check.
        "claim": "Representations refine across repeats",
        "primary":   ("tuned_lens_triangulation", "agreement_band"),
        "secondary": ("logit_lens", "h0_test.rejects_flat"),
        "tertiary":  ("cka", "h0_verdict.rejects_h0_redundant_same_weights"),
        "tertiary_label": "supplementary (CKA)",
    },
    {
        # Top-level verdict key in kv_rank_results.json is
        # "prediction_1_verdict", which is itself a sub-object whose
        # "verdict" field carries the three-way label
        # (evidence-of-compounding / consistent-with-literature /
        # inconsistent-with-literature).
        "claim": "KV cache is compressible",
        "primary":   ("kv_rank", "prediction_1_verdict.verdict"),
        "secondary": ("kv_rank", "prediction_1_verdict.verdict"),
        "tertiary":  ("effective_dim", "rq6_verdict.claim_supported"),
    },
    {
        # Protocol G Type A weight-matrix rank lives inside kv_rank.py
        # (NOT a separate weight_matrix_rank.py script). Type B is the
        # activation-rank line of the same artefact; the Chun PR cross-
        # check is via effective_dim's rq6 claim. Wiring unchanged from
        # row 2; the prose label here is what surfaces in the report.
        "claim": "Weight tying / low rank (Protocol G Type A weight-matrix rank)",
        "primary":   ("kv_rank", "prediction_1_verdict.verdict"),
        "secondary": ("kv_rank", "prediction_1_verdict.verdict"),
        "tertiary":  ("effective_dim", "rq6_verdict.claim_supported"),
        "primary_label": "Protocol G Type A weight-matrix rank",
    },
    {
        # attention_taxonomy implements the Zucchet et al. 2025 triple
        # (paired-t / per-layer Spearman+Holm / mixed-effects). The
        # top-level verdict + per-gate booleans live under
        # "prediction2.{verdict, verdict_meta}".
        "claim": "Attention specialises by repeat",
        "primary":   ("attention_taxonomy", "prediction2.verdict"),
        "secondary": ("attention_taxonomy",
                      "prediction2.verdict_meta.gate2_spearman_majority_reject"),
        "tertiary":  ("attention_taxonomy",
                      "prediction2.verdict_meta.gate3_mixed_effects_reject_flat"),
    },
    {
        # RQ5 DV-1/DV-3 are deferred (router_analysis.py docstring); only
        # the run-level Protocol D-calibration aggregate verdict exists,
        # accessible via the synthetic protocol key set up in
        # collect_artefacts (entropy_calibration_aggregate -> __run__/d_cal).
        "claim": "Contextual isolation",
        "primary":   ("entropy_calibration_aggregate", "verdict"),
        "secondary": None,
        "tertiary":  None,
        "status": "single-measurement-preliminary",
        "deferred_note": (
            "DV-1/DV-3 deferred to Protocol D extension "
            "(see router_analysis.py docstring)"
        ),
    },
    {
        # RQ4 partial-correlation halting-vs-loss is deferred; only the
        # interpolation_validity update-magnitude DV-1 exists. State-delta
        # correlation = DV-2 of the same protocol, NOT operationally
        # independent (drop).
        "claim": "Router learns difficulty proxy",
        "primary":   ("interpolation_validity", "aggregate.n_cells_rejected_holm"),
        "secondary": None,
        "tertiary":  None,
        "status": "single-protocol-preliminary",
        "deferred_note": (
            "RQ4 halting-vs-loss bootstrap CI deferred to Protocol D extension"
        ),
    },
    {
        # Counting cells live in a different RUN_DIR (iridis/counting-
        # sweep/run_N/), not the analyze-lncot+adm dir this synthesis
        # consumes. Marked deferred to a separate synthesis pass; when
        # --counting-run-dir is supplied at the CLI, _resolve_triangulation_rows
        # substitutes the active wiring below at table-build time.
        "claim": "Recurrence helps counting",
        "primary":   None,
        "secondary": None,
        "tertiary":  None,
        "status": "deferred-to-counting-synthesis",
        "deferred_note": (
            "Computed by separate counting synthesis pass: "
            "`python -m analysis.synthesis --run-dir <analytic_run> "
            "--counting-run-dir <counting_run>`"
        ),
    },
]


# Active wiring for Row 7 once --counting-run-dir is supplied. Kept as a
# module-level constant so the wiring is greppable / verifiable from outside
# the resolver. Per docs/extend-notes.md §1.6 row 7: DV-1 OOD McNemar at
# L=200 is the falsifier, DV-2 selectivity gap is the corroborator, DV-4
# causal-ablation max repeat-importance is the tertiary cross-check.
TRIANGULATION_ROW_7_ACTIVE: dict[str, Any] = {
    "claim": "Recurrence helps counting",
    "primary":   ("counting_dv1_ood", "aggregate.rq9_falsifies_h0_at_200"),
    # DV-2's ``monotone_by_repeat`` boolean is the cleanest single-key
    # corroborator for the claim: True iff per-repeat mean Ridge R^2 is
    # non-decreasing across repeats, i.e. each successive repeat extracts
    # at least as much count signal as the previous one. The numeric
    # ``mean_gap`` aggregate is also useful but does not collapse to a
    # binary verdict (per analysis/counting_dv2_probe.py docstring lines
    # 24-30: gap >= 0.05 -> non-linear-canonical, gap < 0.05 -> linear-
    # canonical; either is consistent with the recurrence-helps claim
    # and so cannot be a falsifier on its own).
    "secondary": ("counting_dv2", "aggregate.monotone_by_repeat"),
    "tertiary":  ("counting_dv4_causal", "aggregate.max_repeat_importance"),
    "status": "active-counting",
    "primary_label": "DV-1 OOD McNemar @ L=200",
    "secondary_label": "DV-2 monotone Ridge R^2 by repeat",
    "tertiary_label": "DV-4 max repeat-importance",
}


def _resolve_triangulation_rows(counting_run_dir: str | None) -> list[dict[str, Any]]:
    """Return TRIANGULATION_ROWS with Row 7 substituted when counting is on.

    When ``counting_run_dir`` is None the analytic-only rows are returned
    unchanged (byte-identical to the prior single-run-dir behaviour). When
    ``counting_run_dir`` is set, the deferred Row 7 sentinel is replaced
    with TRIANGULATION_ROW_7_ACTIVE in-place at the same index.
    """
    if counting_run_dir is None:
        return TRIANGULATION_ROWS
    rows = list(TRIANGULATION_ROWS)
    for i, row in enumerate(rows):
        if row.get("claim") == "Recurrence helps counting":
            rows[i] = TRIANGULATION_ROW_7_ACTIVE
            break
    return rows


# RQ-primary p-value paths for DEC-025 family-wise Holm-Bonferroni.
# Each entry maps an RQ id to (protocol_key, dotted_path) where the
# raw scalar p-value lives in the protocol's per-tag JSON artefact.
# A value of None means "deferred / no aggregate p-value emitted at this
# wave" -- those RQs are reported in the Holm table with status=deferred
# and DO NOT contribute to the family size.
#
# Path notes (all verified by reading the protocol scripts):
#   RQ1 logit_lens.h0_test.p_value          paired-t over R1 vs R_last
#   RQ2 cka                                 NO scalar p-value (q90 cut-off
#                                           is the test statistic; deferred)
#   RQ3 attention_taxonomy.prediction2      paired-t per metric, no aggregate
#                                           scalar; minimum across the 3 raw
#                                           metrics is used as the family
#                                           p-value for this RQ
#   RQ4 deferred (DIR-002 reserve)
#   RQ5 D-cal verdict is categorical, no p
#   RQ6 effective_dim                       per-layer Holm; min across
#                                           per_layer_slope.<L>.loglinear.
#                                           p_one_tailed_slope_nonpositive
#                                           is the family p-value for this RQ
#   RQ7 interpolation_validity              per-cell Holm; aggregate count is
#                                           the only emitted scalar; we use
#                                           a synthetic p of 0.0 if any cell
#                                           rejects (>0), else 1.0
#   RQ8 depth_emb_freeze.h0_per_condition.p_values_holm[0]   CE-delta Holm p
#   RQ9 counting_dv1_ood.aggregate.mcnemar_p_at_200   (only when
#       --counting-run-dir is supplied; otherwise None and the family size
#       drops back to the analytic-only count, preserving Holm correctness
#       per DEC-025)
RQ_PRIMARY_PVALUE_PATHS: dict[str, tuple[str, str] | None] = {
    "RQ1": ("logit_lens", "h0_test.p_value"),
    "RQ2": None,
    "RQ3": ("attention_taxonomy", "__rq3_min_paired_p__"),
    "RQ4": None,
    "RQ5": None,
    "RQ6": ("effective_dim", "__rq6_min_layer_p__"),
    "RQ7": ("interpolation_validity", "__rq7_aggregate_synthetic_p__"),
    "RQ8": ("depth_emb_freeze", "h0_per_condition.p_values_holm.0"),
    "RQ9": None,  # Substituted at runtime when counting_run_dir is set.
}


# RQ9 active path used when --counting-run-dir is supplied. Held as a module
# constant so the path is greppable from the outside (verification expects
# at least one occurrence of "counting_dv1_ood" in the file).
RQ9_PRIMARY_PVALUE_PATH_ACTIVE: tuple[str, str] = (
    "counting_dv1_ood",
    "aggregate.mcnemar_p_at_200",
)


def _resolve_rq_pvalue_paths(
    counting_run_dir: str | None,
) -> dict[str, tuple[str, str] | None]:
    """Return RQ_PRIMARY_PVALUE_PATHS with RQ9 substituted when counting is on.

    The family size used by Holm-Bonferroni is computed downstream from the
    count of non-None entries that resolve to a finite p-value; switching
    RQ9 from None to an active path therefore extends the family from 5 to
    6 active tests dynamically per DEC-025 (no new statistical tests are
    introduced -- the McNemar p is computed upstream by the DV-1 OOD probe).
    """
    if counting_run_dir is None:
        return RQ_PRIMARY_PVALUE_PATHS
    out = dict(RQ_PRIMARY_PVALUE_PATHS)
    out["RQ9"] = RQ9_PRIMARY_PVALUE_PATH_ACTIVE
    return out


# Family-wise alpha for DEC-025 Holm-Bonferroni across the RQ-primary
# tests. Synthesis is the only site that sees all RQ artefacts at once,
# so the correction is applied here.
DEC025_FAMILY_ALPHA: float = 0.01


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


def discover_tags(run_dir: str) -> list[str]:
    """Return sorted tag subdirectories matching the c[1-5][a-z]?_ pattern."""
    if not os.path.isdir(run_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if os.path.isdir(path) and TAG_PATTERN.match(name):
            out.append(name)
    return out


def load_json(path: str) -> dict[str, Any] | None:
    """Defensively load a JSON file, returning None on any failure.

    Logs a warning on missing/malformed input rather than raising; synthesis
    must not abort when one protocol's artefact is absent (per scope rule:
    every datum is computed upstream and synthesis does not recompute).
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Failed to load %s: %s", path, exc)
        return None


def collect_artefacts(run_dir: str, tags: list[str]) -> dict[str, dict[str, Any]]:
    """Return ``{tag: {protocol_name: parsed_json_or_None}}``.

    Also stuffs the run-level d_cal aggregate under the synthetic key
    ``__run__`` so downstream code can treat it uniformly. The same
    payload is mirrored under each tag as ``entropy_calibration_aggregate``
    so the triangulation matrix can resolve RQ5's run-level cell with the
    same per-tag resolver as the other rows.
    """
    out: dict[str, dict[str, Any]] = {}
    d_cal_payload = load_json(os.path.join(run_dir, D_CAL_VERDICT_RELPATH))
    for tag in tags:
        per_tag: dict[str, Any] = {}
        for proto, fname in PROTOCOL_FILES.items():
            per_tag[proto] = load_json(os.path.join(run_dir, tag, fname))
        # Mirror the run-level D-calibration aggregate so the triangulation
        # matrix's entropy_calibration_aggregate cell resolves uniformly.
        per_tag["entropy_calibration_aggregate"] = d_cal_payload
        out[tag] = per_tag
    out["__run__"] = {
        "d_cal": d_cal_payload,
    }
    return out


# ---------------------------------------------------------------------------
# Counting (RQ9) cross-cell discovery and aggregation
# ---------------------------------------------------------------------------


def discover_counting_cells(counting_run_dir: str) -> list[dict[str, Any]]:
    """Walk ``counting_run_dir`` and return one record per cell directory.

    A "cell directory" is any directory that contains at least one of the
    canonical RQ9 artefact subdirectories (``dv1_ood`` / ``dv2`` / ``dv3``
    / ``dv4``). The walk is intentionally permissive about the parent
    layout: per-arm checkpoints under ``$EXPS_DIR/counting/<model>/<exp>/``
    and per-run-dir layouts under ``iridis/counting-sweep/run_N/<cell>/``
    both resolve via the same artefact-presence rule.

    Each returned record exposes ``cell_name`` (the leaf directory name),
    ``cell_dir`` (absolute path), and parsed metadata fields ``arm`` /
    ``variant`` / ``seed`` / ``n_embd`` when the cell name matches
    ``COUNTING_CELL_NAME_RE``. Off-spec cell names still load (parsed
    metadata fields fall through to ``None``).
    """
    if not os.path.isdir(counting_run_dir):
        return []
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for current, sub_dirs, _files in os.walk(counting_run_dir):
        sub_set = set(sub_dirs)
        # Cell-anchor rule: directory contains at least one DV subdir.
        if not any(
            anchor in sub_set for anchor in ("dv1_ood", "dv2", "dv3", "dv4")
        ):
            continue
        cell_dir = os.path.abspath(current)
        if cell_dir in seen:
            continue
        seen.add(cell_dir)
        cell_name = os.path.basename(cell_dir.rstrip(os.sep))
        match = COUNTING_CELL_NAME_RE.match(cell_name)
        record: dict[str, Any] = {
            "cell_name": cell_name,
            "cell_dir": cell_dir,
            "arm": None,
            "variant": None,
            "seed": None,
            "n_embd": None,
        }
        if match is not None:
            record["arm"] = match.group("arm").upper()
            record["variant"] = match.group("variant")
            try:
                record["seed"] = int(match.group("seed"))
                record["n_embd"] = int(match.group("nembd"))
            except (TypeError, ValueError):
                pass
        cells.append(record)
        # Do NOT descend into a cell directory: the DV subdirs are
        # artefact leaves, never themselves cell anchors.
        sub_dirs[:] = []
    cells.sort(key=lambda r: r["cell_name"])
    return cells


def _aggregate_counting_dv2(cell_dir: str) -> dict[str, Any] | None:
    """Load every ``dv2/<task>/counting_dv2_results.json`` and aggregate.

    Returns a synthetic per-cell payload with an ``aggregate`` block that
    averages the per-task ``aggregate.mean_*`` keys plus a ``per_task``
    sub-dict preserving each task's raw aggregate. Returns ``None`` if no
    DV-2 task artefact loaded for this cell.
    """
    per_task: dict[str, dict[str, Any]] = {}
    for task in COUNTING_DV2_TASKS:
        path = os.path.join(
            cell_dir, "dv2", task, COUNTING_DV2_TASK_FILENAME
        )
        payload = load_json(path)
        if payload is not None:
            per_task[task] = payload
    if not per_task:
        return None
    # Cross-task means over the per-task aggregates. Mirrors the analytic
    # pattern of "synthesis aggregates upstream-computed scalars; no new
    # statistical tests are run here."
    keys_numeric = ("mean_ridge_r2", "mean_mlp_r2", "mean_gap")
    sums: dict[str, float] = {k: 0.0 for k in keys_numeric}
    counts: dict[str, int] = {k: 0 for k in keys_numeric}
    n_flag_total = 0
    for _task, payload in per_task.items():
        agg = payload.get("aggregate") if isinstance(payload, dict) else None
        if not isinstance(agg, dict):
            continue
        for k in keys_numeric:
            v = agg.get(k)
            if isinstance(v, (int, float)) and not _is_nan(v):
                sums[k] += float(v)
                counts[k] += 1
        nf = agg.get("n_sites_non_linear_flag")
        if isinstance(nf, (int, float)) and not _is_nan(nf):
            n_flag_total += int(nf)
    cross_task_aggregate: dict[str, Any] = {
        k: (sums[k] / counts[k]) if counts[k] > 0 else float("nan")
        for k in keys_numeric
    }
    cross_task_aggregate["n_sites_non_linear_flag"] = int(n_flag_total)
    cross_task_aggregate["n_tasks_loaded"] = int(len(per_task))
    return {
        "schema_version": "synthesis-counting-dv2-1.0",
        "per_task": per_task,
        "aggregate": cross_task_aggregate,
    }


def _aggregate_counting_dv4(cell_dir: str) -> dict[str, Any] | None:
    """Load ``dv4/counting_dv4_causal_results.json`` and surface aggregates.

    Adds an ``aggregate`` block synthesising the max per-repeat importance
    plus the baseline exact-match accuracy. The DV-4 script does not emit
    its own ``aggregate`` block so this helper computes it from the
    upstream ``per_repeat_importance`` map.
    """
    payload = load_json(os.path.join(cell_dir, COUNTING_DV4_RELPATH))
    if payload is None:
        return None
    importance = payload.get("per_repeat_importance")
    max_importance = float("nan")
    if isinstance(importance, dict) and importance:
        finite = [
            float(v) for v in importance.values()
            if isinstance(v, (int, float)) and not _is_nan(v)
        ]
        if finite:
            max_importance = max(finite)
    baseline = payload.get("baseline")
    baseline_em = float("nan")
    if isinstance(baseline, dict):
        v = baseline.get("exact_match_accuracy")
        if isinstance(v, (int, float)) and not _is_nan(v):
            baseline_em = float(v)
    payload_with_agg = dict(payload)
    payload_with_agg["aggregate"] = {
        "max_repeat_importance": max_importance,
        "baseline_exact_match": baseline_em,
    }
    return payload_with_agg


def _aggregate_counting_dv3(cell_dir: str) -> dict[str, Any] | None:
    """Load ``dv3/counting_dv3_attention_results.json`` and surface aggregates.

    DV-3 emits ``per_layer_head`` keyed by ``"<site>_h<head>"`` with stats
    sub-dicts. This helper averages ``target_mass_mean`` and
    ``entropy_mean`` across the per-(layer, head) buckets and exposes them
    under ``aggregate.{mean_target_mass, mean_entropy}`` so the
    triangulation resolver can read a single scalar.
    """
    payload = load_json(os.path.join(cell_dir, COUNTING_DV3_RELPATH))
    if payload is None:
        return None
    plh = payload.get("per_layer_head")
    target_masses: list[float] = []
    entropies: list[float] = []
    if isinstance(plh, dict):
        for _bucket, stats in plh.items():
            if not isinstance(stats, dict):
                continue
            v = stats.get("target_mass_mean")
            if isinstance(v, (int, float)) and not _is_nan(v):
                target_masses.append(float(v))
            v = stats.get("entropy_mean")
            if isinstance(v, (int, float)) and not _is_nan(v):
                entropies.append(float(v))
    payload_with_agg = dict(payload)
    payload_with_agg["aggregate"] = {
        "mean_target_mass": (
            sum(target_masses) / len(target_masses)
            if target_masses else float("nan")
        ),
        "mean_entropy": (
            sum(entropies) / len(entropies)
            if entropies else float("nan")
        ),
        "n_buckets": int(len(target_masses)),
    }
    return payload_with_agg


def collect_counting_artefacts(
    cells: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build per-counting-cell artefact dict keyed by synthetic tag.

    Each cell's tag is ``COUNTING_TAG_PREFIX + cell_name`` to namespace
    counting cells away from the analytic c-tag pattern. The returned
    dict shape mirrors ``collect_artefacts`` so the unified ``artefacts``
    dict accepts both flavours of tag without further plumbing.
    """
    out: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cell_dir = cell["cell_dir"]
        tag = COUNTING_TAG_PREFIX + cell["cell_name"]
        out[tag] = {
            "counting_dv1_ood": load_json(
                os.path.join(cell_dir, COUNTING_DV1_OOD_RELPATH)
            ),
            "counting_dv2": _aggregate_counting_dv2(cell_dir),
            "counting_dv3": _aggregate_counting_dv3(cell_dir),
            "counting_dv4_causal": _aggregate_counting_dv4(cell_dir),
            "__cell_meta__": {
                "cell_name": cell["cell_name"],
                "cell_dir": cell_dir,
                "arm": cell.get("arm"),
                "variant": cell.get("variant"),
                "seed": cell.get("seed"),
                "n_embd": cell.get("n_embd"),
            },
        }
    return out


def _counting_tags(artefacts: dict[str, dict[str, Any]]) -> list[str]:
    """Return the subset of artefact tags that are counting cells."""
    return sorted(t for t in artefacts.keys() if t.startswith(COUNTING_TAG_PREFIX))


# ---------------------------------------------------------------------------
# Triangulation matrix
# ---------------------------------------------------------------------------


def _resolve_dotted(payload: Any, dotted: str | None) -> Any:
    """Look up a dotted key path inside a nested dict; return None on miss."""
    if payload is None or dotted is None:
        return None
    cur: Any = payload
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _verdict_label(payload: Any, dotted: str | None) -> str:
    """Map an artefact lookup into a triangulation cell label.

    Labels are intentionally compact for tabular display. The set is:
    - "supported"    : the underlying H0 was rejected (claim corroborated)
    - "preliminary"  : artefact present but no single boolean verdict; the
                       row downgrades by definition (Reliability Discipline
                       and Triangulation, section 1.6)
    - "contradicted" : H0 not rejected when the artefact does support a
                       boolean test (negative finding)
    - "missing-data" : no artefact for this measurement at this checkpoint
    """
    if payload is None:
        return "missing-data"
    if dotted is None:
        # Artefact exists but no scalar verdict key -- preliminary downgrade.
        return "preliminary"
    val = _resolve_dotted(payload, dotted)
    if val is None:
        return "preliminary"
    if isinstance(val, bool):
        return "supported" if val else "contradicted"
    if isinstance(val, str):
        # Each protocol emits its own string-verdict vocabulary. The maps
        # below collapse them onto the {supported, preliminary, contradicted}
        # triangulation labels. Anything not in the maps falls through to
        # "preliminary" so synthesis never silently lies.
        positive = {
            "pass", "reject_h0", "reject_h0_log_linear", "supported",
            # tuned_lens_triangulation.json -> agreement_band
            "agree",
            # kv_rank.py -> prediction_1_verdict.verdict
            "evidence-of-compounding",
        }
        negative = {
            "fail", "accept_h0", "no_signal", "contradicted",
            # tuned_lens_triangulation.json -> agreement_band
            "disagree",
            # attention_taxonomy.py -> prediction2.verdict
            "refuted",
            # kv_rank.py -> prediction_1_verdict.verdict
            "inconsistent-with-literature",
        }
        ambiguous = {
            # tuned_lens_triangulation.json -> agreement_band
            "ambiguous",
            # attention_taxonomy.py -> prediction2.verdict (no rejection)
            "inconclusive",
            # kv_rank.py -> prediction_1_verdict.verdict (Kobayashi-bracket
            # respected but compounding not yet evident -- conservative
            # downgrade per audit)
            "consistent-with-literature",
        }
        low = val.strip().lower()
        if low in positive:
            return "supported"
        if low in negative:
            return "contradicted"
        if low in ambiguous:
            return "preliminary"
        return "preliminary"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        # Numeric cells: positive count -> supported, zero -> contradicted.
        # Used by RQ7 row's aggregate.n_cells_rejected_holm cell and Row 7's
        # DV-4 max_repeat_importance cell. NaN values downgrade to
        # "preliminary" rather than "contradicted" so a missing-aggregate
        # cell does not flip the row's consensus to a negative finding.
        if _is_nan(val):
            return "preliminary"
        return "supported" if float(val) > 0 else "contradicted"
    return "preliminary"


def _is_counting_row(claim_def: dict[str, Any]) -> bool:
    """Return True iff any of the row's tier protocol keys is a counting key.

    Counting rows iterate counting cell tags rather than analytic c-tags.
    The check tolerates ``None`` tier specs (deferred) by skipping them.
    """
    counting_protocols = {
        "counting_dv1_ood", "counting_dv2", "counting_dv3", "counting_dv4_causal",
    }
    for tier in ("primary", "secondary", "tertiary"):
        spec = claim_def.get(tier)
        if not spec:
            continue
        proto = spec[0] if isinstance(spec, (tuple, list)) and spec else None
        if proto in counting_protocols:
            return True
    return False


def build_triangulation_table(
    artefacts: dict[str, dict[str, Any]],
    tags: list[str],
    counting_run_dir: str | None = None,
) -> list[dict[str, Any]]:
    """For each claim, summarise the verdict across all checkpoints.

    Returns one row per claim with three cells (primary/secondary/tertiary).
    Each cell is a dict ``{"label": <consensus>, "per_tag": {tag: label}}``;
    the consensus rule is conservative: if any tag is "contradicted" the
    consensus is "contradicted"; else if any tag is "supported" the
    consensus is "supported"; else "preliminary"; missing-data only when
    every tag is missing.  This mirrors the section 1.6 convergence rule
    without inventing a new statistical test.

    When ``counting_run_dir`` is supplied, Row 7 ("Recurrence helps
    counting") is substituted from its deferred sentinel to the active
    wiring in TRIANGULATION_ROW_7_ACTIVE, and counting rows iterate the
    counting cell tag set instead of the analytic ``tags`` argument.
    """
    rows: list[dict[str, Any]] = []
    counting_tags = _counting_tags(artefacts) if counting_run_dir else []
    row_defs = _resolve_triangulation_rows(counting_run_dir)
    for claim_def in row_defs:
        cells: dict[str, Any] = {"claim": claim_def["claim"]}
        # Counting rows resolve over counting cell tags (one per cell);
        # analytic rows resolve over the analytic c-tag list. The two tag
        # families are disjoint by construction (COUNTING_TAG_PREFIX).
        row_tags = counting_tags if _is_counting_row(claim_def) else tags
        for tier in ("primary", "secondary", "tertiary"):
            spec = claim_def.get(tier)
            if spec is None:
                # Tier explicitly deferred (RQ4/5/9 wave). Render as a
                # "deferred" cell rather than missing-data; keeps the
                # downstream consensus aggregation accurate.
                cells[tier] = {"label": "deferred", "per_tag": {}}
                continue
            proto, dotted = spec
            per_tag = {
                tag: _verdict_label(
                    artefacts.get(tag, {}).get(proto), dotted
                )
                for tag in row_tags
            }
            cells[tier] = {
                "label": _consensus(per_tag.values()),
                "per_tag": per_tag,
            }
        # Pass through optional schema fields so the markdown renderer can
        # surface deferred-status annotations without breaking older rows.
        for opt in ("status", "deferred_note",
                    "primary_label", "secondary_label", "tertiary_label"):
            if opt in claim_def:
                cells[opt] = claim_def[opt]
        rows.append(cells)
    return rows


def _consensus(labels: Iterable[str]) -> str:
    """Aggregate per-checkpoint labels into a single per-tier label.

    Cross-checkpoint robustness is exempt from the multiple-comparison
    family per section 1.6 ("the claim is the agreement across checkpoints
    rather than the per-checkpoint significance"), so the per-tier label
    reflects whether any checkpoint corroborates the claim while flagging
    discordance:

    - ``missing-data``: every checkpoint missing the artefact
    - ``contradicted``: at least one checkpoint produced a "contradicted"
      verdict AND no checkpoint produced a "supported" verdict
    - ``supported``: at least one checkpoint produced a "supported" verdict
      AND no checkpoint produced a "contradicted" verdict
    - ``mixed``: at least one supported and at least one contradicted
      (the disagreement is the finding per section 1.6)
    - ``preliminary``: artefacts exist but no checkpoint produced a
      boolean verdict (single-measurement downgrade)
    """
    labels_list = list(labels)
    if not labels_list or all(lbl == "missing-data" for lbl in labels_list):
        return "missing-data"
    has_supported = any(lbl == "supported" for lbl in labels_list)
    has_contradicted = any(lbl == "contradicted" for lbl in labels_list)
    if has_supported and has_contradicted:
        return "mixed"
    if has_supported:
        return "supported"
    if has_contradicted:
        return "contradicted"
    return "preliminary"


# ---------------------------------------------------------------------------
# Plotting -- one figure per RQ where data exists
# ---------------------------------------------------------------------------


def _safe_setup_figure(rows: int, cols: int, size: tuple[float, float]):
    """Use the project's setup_figure if importable; else plain matplotlib."""
    try:
        from analysis.common.plotting import setup_figure  # type: ignore

        return setup_figure(rows, cols, size=size)
    except Exception:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt.subplots(rows, cols, figsize=size)


def _safe_savefig(fig: Any, path: str) -> None:
    try:
        from analysis.common.plotting import savefig  # type: ignore

        savefig(fig, path)
        return
    except Exception:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _palette(n: int) -> list[str]:
    try:
        from analysis.common.plotting import palette_for_repeats  # type: ignore

        return palette_for_repeats(n)
    except Exception:
        base = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
        if n <= len(base):
            return base[:n]
        return base + [f"C{i}" for i in range(n - len(base))]


def _emit_logit_lens_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    """Plot logit-lens top-1 (lnf) at the final repeat across checkpoints."""
    points: list[tuple[str, float]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("logit_lens")
        if not payload:
            continue
        agg = payload.get("aggregate") or {}
        # Prefer a "mean_top1_lnf" array shaped (n_layer_mid, n_repeat); if
        # absent, try the aggregate's last-repeat scalar.
        mean = agg.get("mean_top1_lnf")
        if isinstance(mean, list) and mean and isinstance(mean[0], list):
            try:
                last_repeat = [row[-1] for row in mean if row]
                value = float(sum(last_repeat) / len(last_repeat))
            except (TypeError, ZeroDivisionError):
                continue
            points.append((tag, value))
            continue
        # Fall back to the H0 test's mean_rlast scalar.
        mean_rlast = (payload.get("h0_test") or {}).get("mean_rlast")
        if mean_rlast is not None:
            try:
                points.append((tag, float(mean_rlast)))
            except (TypeError, ValueError):
                pass
    if not points:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    xs = list(range(len(points)))
    ax.bar(xs, [v for _, v in points], color=_palette(len(points)))
    ax.set_xticks(xs)
    ax.set_xticklabels([t for t, _ in points], rotation=30, ha="right")
    ax.set_ylabel("mean top-1 (ln_f) at last repeat")
    ax.set_title("Logit Lens (RQ1) -- final-repeat top-1 across checkpoints")
    _safe_savefig(fig, out_path)
    return True


def _emit_cka_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    """Bar-chart of debiased-CKA falsifying-pair counts across checkpoints."""
    rows: list[tuple[str, int, bool]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("cka")
        if not payload:
            continue
        verdict = payload.get("h0_verdict") or {}
        falsifying = verdict.get("falsifying_adjacent_repeat_pairs")
        rejects = bool(verdict.get("rejects_h0_redundant_same_weights", False))
        try:
            n = int(len(falsifying)) if falsifying is not None else 0
        except TypeError:
            n = 0
        rows.append((tag, n, rejects))
    if not rows:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    xs = list(range(len(rows)))
    colors = ["#2ca02c" if r[2] else "#d62728" for r in rows]
    ax.bar(xs, [r[1] for r in rows], color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("falsifying adjacent-repeat pairs (count)")
    ax.set_title(
        "Debiased CKA (RQ2) -- adjacent-repeat redundancy refutation count"
    )
    _safe_savefig(fig, out_path)
    return True


def _emit_attention_taxonomy_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    """Per-checkpoint count of layers where attention specialises by repeat."""
    rows: list[tuple[str, int, int]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("attention_taxonomy")
        if not payload:
            continue
        per_layer = payload.get("per_layer") or []
        n_total = 0
        n_reject = 0
        if isinstance(per_layer, list):
            for entry in per_layer:
                if not isinstance(entry, dict):
                    continue
                n_total += 1
                if entry.get("holm_reject"):
                    n_reject += 1
        rows.append((tag, n_reject, n_total))
    if not rows:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    xs = list(range(len(rows)))
    ax.bar(xs, [r[1] for r in rows], color=_palette(len(rows)))
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("layers with Holm-rejected specialisation")
    ax.set_title(
        "Attention Taxonomy (RQ3) -- specialisation count across checkpoints"
    )
    for i, (_, n, total) in enumerate(rows):
        ax.text(i, n, f"{n}/{total}", ha="center", va="bottom", fontsize=8)
    _safe_savefig(fig, out_path)
    return True


def _emit_router_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    """Lightweight scatter of router presence vs Spearman / mean entropy."""
    rows: list[tuple[str, float | None]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("router_analysis")
        if not payload:
            continue
        # Try a few candidate scalar keys; if none, skip.
        scalar = (
            payload.get("aggregate", {}).get("mean_entropy")
            if isinstance(payload.get("aggregate"), dict)
            else None
        )
        if scalar is None:
            scalar = payload.get("mean_entropy")
        try:
            value = float(scalar) if scalar is not None else None
        except (TypeError, ValueError):
            value = None
        rows.append((tag, value))
    if not rows or all(v is None for _, v in rows):
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    plot_rows = [(t, v) for t, v in rows if v is not None]
    xs = list(range(len(plot_rows)))
    ax.bar(xs, [v for _, v in plot_rows], color=_palette(len(plot_rows)))
    ax.set_xticks(xs)
    ax.set_xticklabels([t for t, _ in plot_rows], rotation=30, ha="right")
    ax.set_ylabel("router mean entropy")
    ax.set_title("Router Analysis (RQ4) -- mean entropy across checkpoints")
    _safe_savefig(fig, out_path)
    return True


def _emit_effective_dim_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    """Per-checkpoint Mann-Kendall agreement fraction (RQ6)."""
    rows: list[tuple[str, float, int, int]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("effective_dim")
        if not payload:
            continue
        mk = payload.get("mann_kendall_agreement") or {}
        n_agree = int(mk.get("n_agreeing_layers", 0))
        n_total = int(mk.get("n_layers_compared", 0))
        frac = float(mk.get("agreement_fraction", 0.0))
        rows.append((tag, frac, n_agree, n_total))
    if not rows:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    xs = list(range(len(rows)))
    ax.bar(xs, [r[1] for r in rows], color=_palette(len(rows)))
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("MK agreement fraction")
    ax.set_title("Effective Dim (RQ6) -- log-linear vs Mann-Kendall agreement")
    for i, (_, frac, n_agree, n_total) in enumerate(rows):
        ax.text(
            i,
            frac,
            f"{n_agree}/{n_total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _safe_savefig(fig, out_path)
    return True


def _emit_kv_rank_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    """Plot kv-rank verdict label per checkpoint (categorical)."""
    rows: list[tuple[str, str]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("kv_rank")
        if not payload:
            continue
        verdict = payload.get("verdict")
        rows.append((tag, str(verdict) if verdict is not None else "no_verdict"))
    if not rows:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    xs = list(range(len(rows)))
    # Encode categorical verdicts as integers for plotting.
    ordering = sorted({v for _, v in rows})
    code_map = {v: i for i, v in enumerate(ordering)}
    ax.bar(xs, [code_map[v] for _, v in rows], color=_palette(len(rows)))
    ax.set_xticks(xs)
    ax.set_xticklabels([t for t, _ in rows], rotation=30, ha="right")
    ax.set_yticks(list(code_map.values()))
    ax.set_yticklabels(ordering)
    ax.set_title("KV Rank (Protocol G) -- verdict per checkpoint")
    _safe_savefig(fig, out_path)
    return True


def _emit_d_cal_figure(
    artefacts: dict[str, dict[str, Any]], out_path: str
) -> bool:
    """Plot the run-level Protocol D-calibration four-gate verdict."""
    payload = artefacts.get("__run__", {}).get("d_cal")
    if not payload:
        return False
    gates = payload.get("gates") if isinstance(payload, dict) else None
    if not isinstance(gates, dict) or not gates:
        # No structured gates -- record as a 1-cell pass/fail figure.
        verdict = (
            payload.get("verdict") if isinstance(payload, dict) else None
        )
        fig, ax = _safe_setup_figure(1, 1, (6.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"D-calibration verdict: {verdict!r}",
            ha="center",
            va="center",
            fontsize=12,
        )
        _safe_savefig(fig, out_path)
        return True
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    names = list(gates.keys())
    passes = [1 if bool(gates[n]) else 0 for n in names]
    ax.bar(
        range(len(names)),
        passes,
        color=["#2ca02c" if p else "#d62728" for p in passes],
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fail", "pass"])
    ax.set_title("Protocol D-calibration four-gate ladder")
    _safe_savefig(fig, out_path)
    return True


def _emit_interpolation_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    rows: list[tuple[str, str]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("interpolation_validity")
        if not payload:
            continue
        verdict = payload.get("verdict") if isinstance(payload, dict) else None
        rows.append((tag, str(verdict) if verdict is not None else "no_verdict"))
    if not rows:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    ordering = sorted({v for _, v in rows})
    code_map = {v: i for i, v in enumerate(ordering)}
    ax.bar(
        range(len(rows)),
        [code_map[v] for _, v in rows],
        color=_palette(len(rows)),
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([t for t, _ in rows], rotation=30, ha="right")
    ax.set_yticks(list(code_map.values()))
    ax.set_yticklabels(ordering)
    ax.set_title("Interpolation Validity (RQ7) -- verdict per checkpoint")
    _safe_savefig(fig, out_path)
    return True


def _emit_depth_emb_freeze_figure(
    artefacts: dict[str, dict[str, Any]], tags: list[str], out_path: str
) -> bool:
    rows: list[tuple[str, str]] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get("depth_emb_freeze")
        if not payload:
            continue
        verdict = payload.get("verdict") if isinstance(payload, dict) else None
        rows.append((tag, str(verdict) if verdict is not None else "no_verdict"))
    if not rows:
        return False
    fig, ax = _safe_setup_figure(1, 1, (8.0, 5.0))
    ordering = sorted({v for _, v in rows})
    code_map = {v: i for i, v in enumerate(ordering)}
    ax.bar(
        range(len(rows)),
        [code_map[v] for _, v in rows],
        color=_palette(len(rows)),
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([t for t, _ in rows], rotation=30, ha="right")
    ax.set_yticks(list(code_map.values()))
    ax.set_yticklabels(ordering)
    ax.set_title("Depth-Emb Freeze (RQ8) -- verdict per checkpoint")
    _safe_savefig(fig, out_path)
    return True


def _emit_triangulation_matrix_figure(
    table: list[dict[str, Any]], out_path: str
) -> bool:
    """Render the triangulation matrix as a coloured heatmap-table image."""
    if not table:
        return False
    # Map labels to integer codes for imshow rendering.
    label_codes = {
        "supported": 2,
        "preliminary": 1,
        "missing-data": 0,
        "contradicted": -1,
        "mixed": -2,
        "deferred": -3,
    }
    label_colors = {
        "supported": "#2ca02c",
        "preliminary": "#ff7f0e",
        "missing-data": "#bdbdbd",
        "contradicted": "#d62728",
        "mixed": "#9467bd",
        "deferred": "#7f7f7f",
    }
    rows = [r["claim"] for r in table]
    cols = ["primary", "secondary", "tertiary"]
    import numpy as np  # local import keeps top-level deps lean

    matrix = np.zeros((len(rows), len(cols)), dtype=float)
    for i, row in enumerate(table):
        for j, tier in enumerate(cols):
            matrix[i, j] = label_codes.get(row[tier]["label"], 0)

    fig, ax = _safe_setup_figure(1, 1, (10.0, max(3.0, 0.5 * len(rows) + 2)))
    # Render each cell as a coloured rectangle plus its label.
    for i, row in enumerate(table):
        for j, tier in enumerate(cols):
            label = row[tier]["label"]
            ax.add_patch(
                __import__("matplotlib").patches.Rectangle(
                    (j, len(rows) - 1 - i),
                    1,
                    1,
                    facecolor=label_colors.get(label, "#cccccc"),
                    edgecolor="white",
                )
            )
            ax.text(
                j + 0.5,
                len(rows) - 1 - i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels(cols)
    ax.set_yticks([len(rows) - 1 - i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels(rows)
    ax.set_aspect("equal")
    ax.set_title("Triangulation Matrix (extend-notes section 1.6)")
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _safe_savefig(fig, out_path)
    return True


def _emit_counting_summary_figure(
    artefacts: dict[str, dict[str, Any]], out_path: str
) -> bool:
    """Per-cell counting summary: DV-1 min-acc, DV-1 McNemar p, DV-2 mean gap.

    Renders one row of grouped bars per counting cell. Cells without any
    counting artefact loaded are dropped so the figure never includes
    placeholder rows. Returns False (no figure emitted) when no counting
    cell has loaded any of the three scalar metrics.
    """
    cell_tags = _counting_tags(artefacts)
    if not cell_tags:
        return False
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for tag in cell_tags:
        per_cell = artefacts.get(tag, {})
        meta = per_cell.get("__cell_meta__") or {}
        # Compose a short, distinctive label per cell. Falls back to the
        # raw tag stripped of the counting prefix when metadata is absent.
        if meta.get("variant") and meta.get("n_embd") is not None:
            label = (
                f"{meta['variant']}@n{meta['n_embd']}"
                + (f"/s{meta['seed']}" if meta.get("seed") is not None else "")
            )
        else:
            label = tag.removeprefix(COUNTING_TAG_PREFIX)

        dv1 = per_cell.get("counting_dv1_ood")
        dv2 = per_cell.get("counting_dv2")
        min_acc = _resolve_dotted(dv1, "aggregate.min_acc") if dv1 else None
        mcnemar_p = _resolve_dotted(dv1, "aggregate.mcnemar_p_at_200") if dv1 else None
        mean_gap = _resolve_dotted(dv2, "aggregate.mean_gap") if dv2 else None

        # Coerce to plottable floats; non-numeric / NaN -> drop to None so
        # the bar stays empty for that cell rather than crashing the cast.
        def _coerce(v: Any) -> float | None:
            if isinstance(v, (int, float)) and not _is_nan(v):
                return float(v)
            return None

        rows.append(
            (label, _coerce(min_acc), _coerce(mcnemar_p), _coerce(mean_gap))
        )
    # Drop rows with NO loaded scalars so the figure does not display
    # purely-empty bars; if every row drops, return False.
    rows = [r for r in rows if any(v is not None for v in r[1:])]
    if not rows:
        return False

    import numpy as np  # local import to keep top-level deps lean
    fig, ax = _safe_setup_figure(1, 1, (max(8.0, 0.6 * len(rows) + 4), 5.5))
    x = np.arange(len(rows))
    width = 0.27

    def _series(idx: int) -> list[float]:
        # NaN is the matplotlib convention for "leave the bar blank".
        return [
            float("nan") if r[idx] is None else float(r[idx])
            for r in rows
        ]

    ax.bar(x - width, _series(1), width, label="DV-1 min acc",
           color="#1f77b4")
    ax.bar(x, _series(2), width, label="DV-1 McNemar p @ L=200",
           color="#d62728")
    ax.bar(x + width, _series(3), width, label="DV-2 mean gap",
           color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel("scalar (units differ per series)")
    ax.set_title("Counting (RQ9) per-cell summary")
    ax.axhline(0.0, color="#333333", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _safe_savefig(fig, out_path)
    return True


# ---------------------------------------------------------------------------
# Family-wise Holm-Bonferroni (DEC-025)
# ---------------------------------------------------------------------------


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni-adjusted p-values for a list of raw p-values.

    Prefers ``statsmodels.stats.multitest.multipletests`` when importable;
    falls back to a numpy implementation that mirrors Holm 1979 to keep
    synthesis runnable in stripped-down environments (matching the
    portability pattern of ``_fit_logistic_classifier`` elsewhere in this
    package).  NaN entries propagate as NaN; the family size used by the
    correction is the count of finite p-values, so deferred RQs (mapped
    to None upstream) never contribute.
    """
    try:
        from statsmodels.stats.multitest import multipletests  # type: ignore

        finite_mask = [
            isinstance(p, (int, float)) and not _is_nan(p) for p in p_values
        ]
        finite_ps = [float(p) for p, m in zip(p_values, finite_mask) if m]
        if not finite_ps:
            return [float("nan")] * len(p_values)
        _reject, p_adj, _aS, _aB = multipletests(
            finite_ps, alpha=DEC025_FAMILY_ALPHA, method="holm"
        )
        out: list[float] = []
        ai = 0
        for m in finite_mask:
            if m:
                out.append(float(p_adj[ai]))
                ai += 1
            else:
                out.append(float("nan"))
        return out
    except Exception:
        # Numpy fallback: rank-sort finite p-values, multiply by descending
        # family size, then enforce monotonicity (Holm step-down).
        import numpy as np

        n = len(p_values)
        finite_idx = [
            i for i, p in enumerate(p_values)
            if isinstance(p, (int, float)) and not _is_nan(p)
        ]
        if not finite_idx:
            return [float("nan")] * n
        finite_ps = np.asarray(
            [float(p_values[i]) for i in finite_idx], dtype=np.float64
        )
        order = np.argsort(finite_ps)
        m = len(finite_ps)
        adj = np.empty(m, dtype=np.float64)
        running_max = 0.0
        for rank, sorted_pos in enumerate(order):
            scaled = (m - rank) * finite_ps[sorted_pos]
            running_max = max(running_max, min(1.0, scaled))
            adj[sorted_pos] = running_max
        out = [float("nan")] * n
        for ai, src_i in enumerate(finite_idx):
            out[src_i] = float(adj[ai])
        return out


def _is_nan(x: Any) -> bool:
    """Tiny portability shim: math.isnan that accepts ints / non-floats."""
    try:
        import math

        return bool(math.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def _resolve_rq_pvalue(
    rq_id: str,
    spec: tuple[str, str] | None,
    artefacts: dict[str, dict[str, Any]],
    tags: list[str],
) -> tuple[float | None, str]:
    """Resolve one RQ's family-wise p-value across all tags.

    Returns ``(p_value_or_None, status)`` where status is one of
    ``"ok"``, ``"deferred-DIR-002"`` (spec is None), or
    ``"deferred-missing-data"`` (artefact or path missing in every tag).

    For RQ ids whose protocol does NOT expose a single scalar p-value at
    the top level (RQ3, RQ6, RQ7), ``spec[1]`` is a synthetic sentinel
    handled here: the worst (smallest) p-value across the relevant
    sub-records is taken as the family-level p-value, mirroring the
    "min-across-tests" convention common in family-wise correction
    cookbooks.

    Across tags, the maximum (most conservative) p-value is reported so
    a single noisy checkpoint can downgrade the family-level claim
    without inflating the global Type-I rate.
    """
    if spec is None:
        return None, "deferred-DIR-002"

    proto, dotted = spec
    candidate_ps: list[float] = []
    for tag in tags:
        payload = artefacts.get(tag, {}).get(proto)
        if payload is None:
            continue
        p = _extract_synthetic_or_dotted_p(rq_id, payload, dotted)
        if p is not None and not _is_nan(p):
            candidate_ps.append(float(p))

    if not candidate_ps:
        return None, "deferred-missing-data"
    # Most-conservative-tag rule: max across checkpoints. Single bad tag
    # blocks the family rejection; this matches the spirit of the
    # cross-checkpoint consensus rule in section 1.6 ("the disagreement
    # itself is the finding").
    return max(candidate_ps), "ok"


def _extract_synthetic_or_dotted_p(
    rq_id: str, payload: Any, dotted: str
) -> float | None:
    """Extract a raw p-value from a per-tag payload.

    Handles the synthetic sentinels documented in
    ``RQ_PRIMARY_PVALUE_PATHS`` and falls back to dotted resolution for
    plain scalar paths.
    """
    if dotted == "__rq3_min_paired_p__":
        # attention_taxonomy.prediction2.paired_t_tests is a dict keyed
        # by metric name with sub-fields {p_value, ...}. Minimum across
        # the three raw metrics (entropy / gini / top_k_mass) is the
        # tightest paired-t evidence; per-layer Spearman + mixed-effects
        # are cross-checks rather than primary tests for DEC-025 purposes.
        paired = _resolve_dotted(payload, "prediction2.paired_t_tests")
        if not isinstance(paired, dict):
            return None
        ps = []
        for metric in ("entropy", "gini", "top_k_mass"):
            sub = paired.get(metric)
            if isinstance(sub, dict):
                p = sub.get("p_value")
                if isinstance(p, (int, float)) and not _is_nan(p):
                    ps.append(float(p))
        return min(ps) if ps else None
    if dotted == "__rq6_min_layer_p__":
        # Minimum one-tailed slope p across per_layer_slope.<L>.loglinear.
        per_layer = _resolve_dotted(payload, "per_layer_slope")
        if not isinstance(per_layer, dict):
            return None
        ps = []
        for _, layer_blob in per_layer.items():
            if not isinstance(layer_blob, dict):
                continue
            ll = layer_blob.get("loglinear")
            if not isinstance(ll, dict):
                continue
            p = ll.get("p_one_tailed_slope_nonpositive")
            if isinstance(p, (int, float)) and not _is_nan(p):
                ps.append(float(p))
        return min(ps) if ps else None
    if dotted == "__rq7_aggregate_synthetic_p__":
        # interpolation_validity emits an aggregate cell-rejection count
        # but no single scalar p. Synthesise: 0.0 if any cell rejected
        # at family alpha, else 1.0. Conservative.
        n_rej = _resolve_dotted(payload, "aggregate.n_cells_rejected_holm")
        if isinstance(n_rej, (int, float)) and not _is_nan(n_rej):
            return 0.0 if float(n_rej) > 0 else 1.0
        return None
    if dotted.endswith(".0"):
        # Tail "<key>.0" indexes into the first element of a list under
        # the parent dotted path. depth_emb_freeze stores Holm-adjusted
        # p-values as a 2-element list (CE delta, accuracy delta); the
        # CE-delta entry is the RQ8 primary.
        parent_dotted = dotted[:-2]
        list_payload = _resolve_dotted(payload, parent_dotted)
        if isinstance(list_payload, list) and list_payload:
            head = list_payload[0]
            if isinstance(head, (int, float)) and not _is_nan(head):
                return float(head)
        return None
    val = _resolve_dotted(payload, dotted)
    if isinstance(val, (int, float)) and not _is_nan(val):
        return float(val)
    return None


def family_wise_holm(
    artefacts: dict[str, dict[str, Any]],
    tags: list[str],
    output_dir: str,
    counting_run_dir: str | None = None,
) -> dict[str, Any]:
    """Apply Holm-Bonferroni at family-wise alpha=0.01 across RQ-primary tests.

    Reads the raw p-value for each RQ via ``RQ_PRIMARY_PVALUE_PATHS``,
    runs ``_holm_adjust`` (statsmodels with numpy fallback), and writes
    ``synthesis_holm_table.json`` next to the synthesis report. Returns
    the parsed table so the markdown renderer can surface it inline.

    Synthesis applies the correction; it does NOT run any new statistical
    tests (DEC-025 first clause: corrections, not new tests).

    When ``counting_run_dir`` is supplied, RQ9's deferred path is replaced
    by ``RQ9_PRIMARY_PVALUE_PATH_ACTIVE`` and resolution iterates the
    counting cell tag set. The family size used by Holm grows from 5 to 6
    in lock-step (DEC-025 dynamic-family rule: deferred tests do not count
    against the correction).
    """
    rq_paths = _resolve_rq_pvalue_paths(counting_run_dir)
    counting_tags = _counting_tags(artefacts) if counting_run_dir else []
    raw_entries: list[dict[str, Any]] = []
    raw_ps_for_correction: list[float] = []
    raw_idx_to_correct: list[int] = []
    for rq_id, spec in rq_paths.items():
        # RQ9's resolution scope is the counting cell tag set, not the
        # analytic c-tags (the counting_dv1_ood payload only lives under
        # __counting__/<cell> namespaces). Other RQs continue to resolve
        # over ``tags`` per the analytic synthesis contract.
        if rq_id == "RQ9" and counting_run_dir:
            tags_for_rq = counting_tags
        else:
            tags_for_rq = tags
        p, status = _resolve_rq_pvalue(rq_id, spec, artefacts, tags_for_rq)
        entry: dict[str, Any] = {
            "rq": rq_id,
            "raw_p": p,
            "holm_adjusted_p": None,
            "rejects_at_alpha_001": None,
            "status": status,
        }
        if p is not None:
            raw_idx_to_correct.append(len(raw_entries))
            raw_ps_for_correction.append(p)
        raw_entries.append(entry)

    if raw_ps_for_correction:
        adjusted = _holm_adjust(raw_ps_for_correction)
        for slot, p_adj in zip(raw_idx_to_correct, adjusted):
            raw_entries[slot]["holm_adjusted_p"] = (
                None if _is_nan(p_adj) else float(p_adj)
            )
            if p_adj is not None and not _is_nan(p_adj):
                raw_entries[slot]["rejects_at_alpha_001"] = bool(
                    float(p_adj) < DEC025_FAMILY_ALPHA
                )

    n_applied = sum(1 for e in raw_entries if e["status"] == "ok")
    n_deferred = sum(1 for e in raw_entries if e["status"] != "ok")

    table_payload = {
        "alpha_family_wise": DEC025_FAMILY_ALPHA,
        "n_tests_applied": int(n_applied),
        "n_tests_deferred": int(n_deferred),
        "table": raw_entries,
    }

    json_path = os.path.join(output_dir, "synthesis_holm_table.json")
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(table_payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        logging.warning("Failed to write %s: %s", json_path, exc)
    return table_payload


def _format_holm_table_md(table_payload: dict[str, Any]) -> str:
    """Render the family-wise Holm-Bonferroni table as a markdown block."""
    if not table_payload or "table" not in table_payload:
        return "Family-wise Holm-Bonferroni table unavailable."
    alpha = table_payload.get("alpha_family_wise", 0.01)
    lines = [
        (
            f"Family-wise alpha = {alpha} across "
            f"{table_payload.get('n_tests_applied', 0)} applied tests "
            f"({table_payload.get('n_tests_deferred', 0)} deferred). "
            "Synthesis applies the correction; no new statistical "
            "tests are run here."
        ),
        "",
        "| RQ | Raw p | Holm-adjusted p | Rejects at alpha=0.01 | Status |",
        "|----|-------|-----------------|-----------------------|--------|",
    ]
    for row in table_payload["table"]:
        raw_p = row.get("raw_p")
        adj_p = row.get("holm_adjusted_p")
        reject = row.get("rejects_at_alpha_001")
        lines.append(
            "| {rq} | {raw} | {adj} | {rej} | {st} |".format(
                rq=row.get("rq", "?"),
                raw=("--" if raw_p is None else f"{float(raw_p):.4g}"),
                adj=("--" if adj_p is None else f"{float(adj_p):.4g}"),
                rej=("--" if reject is None else ("yes" if reject else "no")),
                st=row.get("status", "?"),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_triangulation_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Claim | Primary | Secondary | Tertiary | Status |",
        "|-------|---------|-----------|----------|--------|",
    ]
    for row in rows:
        status_bits: list[str] = []
        if "status" in row:
            status_bits.append(str(row["status"]))
        if "deferred_note" in row:
            status_bits.append(str(row["deferred_note"]))
        lines.append(
            "| {claim} | {p} | {s} | {t} | {st} |".format(
                claim=row["claim"],
                p=row["primary"]["label"],
                s=row["secondary"]["label"],
                t=row["tertiary"]["label"],
                st="; ".join(status_bits) if status_bits else "",
            )
        )
    return "\n".join(lines)


def _format_protocol_inventory_md(
    artefacts: dict[str, dict[str, Any]], tags: list[str]
) -> str:
    header = ["| Tag | " + " | ".join(PROTOCOL_FILES.keys()) + " |"]
    sep = ["|" + "---|" * (len(PROTOCOL_FILES) + 1)]
    body: list[str] = []
    for tag in tags:
        per_tag = artefacts.get(tag, {})
        cells = [
            "yes" if per_tag.get(proto) is not None else "no"
            for proto in PROTOCOL_FILES
        ]
        body.append("| " + tag + " | " + " | ".join(cells) + " |")
    return "\n".join(header + sep + body)


def _format_limitations_md(
    artefacts: dict[str, dict[str, Any]], tags: list[str]
) -> str:
    missing: list[str] = []
    for tag in tags:
        per_tag = artefacts.get(tag, {})
        for proto in PROTOCOL_FILES:
            if per_tag.get(proto) is None:
                missing.append(f"  - {tag}: {proto} artefact absent")
    run_level = artefacts.get("__run__", {})
    if run_level.get("d_cal") is None:
        missing.append("  - run-level: Protocol D-calibration verdict absent")
    if not missing:
        return "No missing artefacts detected at synthesis time."
    return "Missing artefacts (synthesis report is partial):\n" + "\n".join(
        missing
    )


def _emitted_figures_md(emitted: dict[str, bool], output_dir: str) -> str:
    lines: list[str] = []
    for fname, was_emitted in emitted.items():
        if not was_emitted:
            lines.append(f"- {fname}: skipped (no upstream artefacts)")
            continue
        lines.append(f"- ![{fname}]({fname})")
    return "\n".join(lines)


def _format_counting_cells_md(artefacts: dict[str, dict[str, Any]]) -> str:
    """Render a per-cell summary table for counting RQ9 cells.

    Columns: variant | n_embd | seed | dv1_min_acc | dv1_McNemar_p_at_200
    | dv2_mean_gap | dv3_mean_target_mass | dv4_max_repeat_importance.
    Cells with a missing scalar render that column as ``--``. A cell with
    no DV artefacts at all is included as a row with all dashes so its
    presence under the run-dir is auditable.
    """
    cell_tags = _counting_tags(artefacts)
    if not cell_tags:
        return ""
    lines = [
        (
            "| variant | n_embd | seed | dv1_min_acc | "
            "dv1_McNemar_p_at_200 | dv2_mean_gap | "
            "dv3_mean_target_mass | dv4_max_repeat_importance |"
        ),
        "|---------|--------|------|-------------|----------------------|"
        "--------------|----------------------|---------------------------|",
    ]

    def _fmt_scalar(v: Any) -> str:
        if v is None:
            return "--"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, float)) and not _is_nan(v):
            return f"{float(v):.4g}"
        return "--"

    for tag in cell_tags:
        per_cell = artefacts.get(tag, {})
        meta = per_cell.get("__cell_meta__") or {}
        dv1 = per_cell.get("counting_dv1_ood")
        dv2 = per_cell.get("counting_dv2")
        dv3 = per_cell.get("counting_dv3")
        dv4 = per_cell.get("counting_dv4_causal")

        row = [
            str(meta.get("variant") or "?"),
            ("?" if meta.get("n_embd") is None else str(meta["n_embd"])),
            ("?" if meta.get("seed") is None else str(meta["seed"])),
            _fmt_scalar(_resolve_dotted(dv1, "aggregate.min_acc")),
            _fmt_scalar(
                _resolve_dotted(dv1, "aggregate.mcnemar_p_at_200")
            ),
            _fmt_scalar(_resolve_dotted(dv2, "aggregate.mean_gap")),
            _fmt_scalar(
                _resolve_dotted(dv3, "aggregate.mean_target_mass")
            ),
            _fmt_scalar(
                _resolve_dotted(dv4, "aggregate.max_repeat_importance")
            ),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: str,
    run_dir: str,
    template: str,
    tags: list[str],
    artefacts: dict[str, dict[str, Any]],
    table: list[dict[str, Any]],
    emitted: dict[str, bool],
    holm_table: dict[str, Any] | None = None,
    counting_run_dir: str | None = None,
) -> str:
    """Write synthesis_report.md and return its absolute path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    counting_tags = _counting_tags(artefacts)
    counting_cells_md = _format_counting_cells_md(artefacts)
    body: list[str] = [
        "# Cross-Checkpoint Synthesis Report",
        "",
        f"- Generated: {timestamp}",
        f"- Run directory: `{run_dir}`",
        (
            f"- Counting run directory: `{counting_run_dir}` "
            f"({len(counting_tags)} counting cell(s) loaded)"
            if counting_run_dir
            else "- Counting run directory: not supplied (Row 7 deferred)"
        ),
        f"- Template: `{template}`",
        f"- Tags found ({len(tags)}): " + ", ".join(tags) if tags else
        "- Tags found: none",
        "",
        "## Protocol artefact inventory",
        "",
        _format_protocol_inventory_md(artefacts, tags) if tags else
        "No tag subdirectories matched the c[1-5][a-z]?_ pattern.",
        "",
        "## Triangulation matrix (extend-notes section 1.6)",
        "",
        ("Per-tier consensus across checkpoints: 'supported' if at least "
         "one checkpoint rejects H0 and none contradict; 'contradicted' "
         "if at least one rejects in the opposite direction and none "
         "support; 'mixed' if checkpoints disagree (the disagreement "
         "itself is the finding per section 1.6); 'preliminary' when "
         "artefacts are present but lack a single boolean verdict "
         "(single-measurement downgrade); 'deferred' when the cell is "
         "scheduled for a later wave (e.g. RQ4/5/9 reserve); "
         "'missing-data' when no artefact exists. Synthesis does not "
         "run any new statistical tests -- every datum is computed "
         "upstream."),
        "",
        _format_triangulation_md(table) if table else
        "Triangulation matrix unavailable (no claims defined).",
        "",
        "## Family-wise Holm-Bonferroni results (DEC-025)",
        "",
        _format_holm_table_md(holm_table) if holm_table else
        "Family-wise Holm-Bonferroni table unavailable (no RQ p-values resolved).",
        "",
        *(
            [
                "## Counting (RQ9) per-cell summary",
                "",
                (
                    "Per-cell scalar surface from the counting sweep RUN_DIR. "
                    "Each row corresponds to one trained checkpoint discovered "
                    "under `--counting-run-dir`; the four DV scalars are "
                    "loaded defensively (a missing artefact renders as `--` "
                    "rather than aborting synthesis). Row 7 of the "
                    "triangulation matrix above resolves over these cells."
                ),
                "",
                counting_cells_md,
                "",
            ]
            if counting_cells_md
            else []
        ),
        "## Cross-checkpoint figures",
        "",
        _emitted_figures_md(emitted, output_dir),
        "",
        "## Limitations",
        "",
        _format_limitations_md(artefacts, tags),
        "",
        "_This report is a communication artefact: it aggregates and "
        "plots upstream-computed values only. Cells marked 'preliminary' "
        "indicate the row downgraded under the operational-independence "
        "audit; cells marked 'missing-data' indicate the artefact was "
        "not present in the run directory at synthesis time._",
        "",
    ]
    report_path = os.path.join(output_dir, "synthesis_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body))
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for the synthesis driver.

    Expected inputs: ``--run-dir`` (the ``run_N/`` output root containing
    per-checkpoint subdirectories), ``--output-dir`` (defaults to
    ``--run-dir``), ``--template`` (template name; reserved for future
    multi-template extension; default ``mech-analysis-v1``).
    """
    parser = argparse.ArgumentParser(
        prog="analysis.synthesis",
        description=(
            "Cross-checkpoint plotting + scalar aggregation. Reads "
            "per-checkpoint protocol JSON artefacts and emits "
            "synthesis_report.md plus per-RQ figures. No GPU, no model "
            "loading, no recomputation of any statistic."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing <tag>/protocol_*_results.json.",
    )
    parser.add_argument(
        "--counting-run-dir",
        default=None,
        help=(
            "Optional separate run directory containing per-cell counting "
            "(RQ9) artefacts (dv1_ood/, dv2/<task>/, dv3/, dv4/). When "
            "supplied, Row 7 of the triangulation matrix resolves "
            "actively, RQ9 is added to the family-wise Holm-Bonferroni "
            "correction (family size grows from 5 to 6 active tests), "
            "and a synthesis_counting_summary.png + per-cell markdown "
            "table are emitted. Loading is defensive: missing per-cell "
            "artefacts are skipped silently. Default unset preserves the "
            "single-run-dir analytic-only behaviour."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write synthesis_*.png + synthesis_report.md "
            "(defaults to --run-dir)."
        ),
    )
    parser.add_argument(
        "--template",
        default="mech-analysis-v1",
        help="Report template name (default: mech-analysis-v1).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log verbosity (default: INFO).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir or args.run_dir)
    os.makedirs(output_dir, exist_ok=True)

    counting_run_dir = (
        os.path.abspath(args.counting_run_dir)
        if args.counting_run_dir
        else None
    )

    if not os.path.isdir(run_dir):
        logging.error("--run-dir does not exist or is not a directory: %s", run_dir)
        # Still emit a stub report so downstream rsync paths are always populated.
        emitted: dict[str, bool] = {}
        # Run the Holm pass with empty inputs so the report still has the
        # section header and the JSON sidecar lands at a predictable path.
        holm_table = family_wise_holm({}, [], output_dir, counting_run_dir)
        write_report(
            output_dir, run_dir, args.template, [], {}, [], emitted, holm_table,
            counting_run_dir,
        )
        return

    tags = discover_tags(run_dir)
    logging.info("Discovered %d tag(s) under %s: %s", len(tags), run_dir, tags)
    artefacts = collect_artefacts(run_dir, tags)

    # Counting cell discovery is gated on --counting-run-dir; absent the
    # flag, the analytic-only flow continues unchanged. Defensive layering:
    # discover_counting_cells returns [] for a non-existent path so a
    # mistyped flag value degrades to the "no counting cells found" code
    # path rather than crashing.
    if counting_run_dir is not None:
        counting_cells = discover_counting_cells(counting_run_dir)
        logging.info(
            "Discovered %d counting cell(s) under %s",
            len(counting_cells), counting_run_dir,
        )
        counting_artefacts = collect_counting_artefacts(counting_cells)
        # Merge into the unified artefacts dict using the counting tag
        # prefix to namespace away from analytic c-tags. The prefix is
        # disjoint from the c-tag pattern by construction.
        for c_tag, c_payload in counting_artefacts.items():
            artefacts[c_tag] = c_payload
    else:
        counting_cells = []

    figures: dict[str, Any] = {
        "synthesis_logit_lens.png": _emit_logit_lens_figure,
        "synthesis_cka.png": _emit_cka_figure,
        "synthesis_attention_taxonomy.png": _emit_attention_taxonomy_figure,
        "synthesis_router.png": _emit_router_figure,
        "synthesis_effective_dim.png": _emit_effective_dim_figure,
        "synthesis_kv_rank.png": _emit_kv_rank_figure,
        "synthesis_interpolation.png": _emit_interpolation_figure,
        "synthesis_depth_emb_freeze.png": _emit_depth_emb_freeze_figure,
    }
    emitted: dict[str, bool] = {}
    for fname, fn in figures.items():
        try:
            ok = bool(fn(artefacts, tags, os.path.join(output_dir, fname)))
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.warning("Figure %s emission failed: %s", fname, exc)
            ok = False
        emitted[fname] = ok

    # Run-level (single-call signature differs).
    try:
        emitted["synthesis_d_cal_verdict.png"] = bool(
            _emit_d_cal_figure(
                artefacts, os.path.join(output_dir, "synthesis_d_cal_verdict.png")
            )
        )
    except Exception as exc:  # pragma: no cover
        logging.warning("D-cal figure emission failed: %s", exc)
        emitted["synthesis_d_cal_verdict.png"] = False

    table = build_triangulation_table(artefacts, tags, counting_run_dir)
    try:
        emitted["synthesis_triangulation_matrix.png"] = bool(
            _emit_triangulation_matrix_figure(
                table,
                os.path.join(output_dir, "synthesis_triangulation_matrix.png"),
            )
        )
    except Exception as exc:  # pragma: no cover
        logging.warning("Triangulation matrix figure emission failed: %s", exc)
        emitted["synthesis_triangulation_matrix.png"] = False

    # Counting (RQ9) per-cell summary figure. Emitted only when
    # --counting-run-dir is supplied and at least one cell loaded a scalar.
    if counting_run_dir is not None:
        try:
            emitted["synthesis_counting_summary.png"] = bool(
                _emit_counting_summary_figure(
                    artefacts,
                    os.path.join(output_dir, "synthesis_counting_summary.png"),
                )
            )
        except Exception as exc:  # pragma: no cover
            logging.warning(
                "Counting summary figure emission failed: %s", exc
            )
            emitted["synthesis_counting_summary.png"] = False

    # DEC-025 family-wise Holm-Bonferroni across the RQ-primary tests.
    try:
        holm_table = family_wise_holm(
            artefacts, tags, output_dir, counting_run_dir
        )
        logging.info(
            "Holm-Bonferroni: %d tests applied, %d deferred",
            holm_table.get("n_tests_applied", 0),
            holm_table.get("n_tests_deferred", 0),
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.warning("family_wise_holm() failed: %s", exc)
        holm_table = None

    report_path = write_report(
        output_dir,
        run_dir,
        args.template,
        tags,
        artefacts,
        table,
        emitted,
        holm_table,
        counting_run_dir,
    )
    logging.info("Wrote synthesis report: %s", report_path)


if __name__ == "__main__":
    main()
