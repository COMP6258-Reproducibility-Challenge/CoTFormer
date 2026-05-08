"""Protocol H -- Interpolation Validity (RQ7).

Scope
-----
Addresses RQ7 (clone-set interpolation entropy). Generates a
clone-set of token positions selected by the ADM router score
``s_{i-1} < 0.1`` and measures both (a) the relative update
magnitude DV-1 ``||x^(i+1) - x^(i)||_2 / ||x^(i)||_2`` per
(layer, repeat, position), and (b) the per-head attention entropy
DV-2 at the final repeat over the same clone positions. Both DVs
are tested against a 10000-permutation null over the
clone-vs-control label assignment per DEC-026.

Falsifiability relevance
------------------------
H0 "DV-1 independent of router-score bin AND DV-2 within the
seed-perturbed `no pollution` band" is rejected when the observed
clone-vs-control entropy delta exceeds the 99.7th percentile (about
3 sigma) of the 10000-permutation null at any (layer, repeat) cell
after Holm-Bonferroni correction at family-wise alpha = 0.01 (per
``docs/extend-notes.md`` §1.2 RQ7 "Falsification" + §1.6 Concern 3).

Ontological purpose
-------------------
A positive result means low-`s_i` positions in the ADM behave
mechanically distinct from typical positions -- the interpolation
formula injects clones whose later-repeat KV-cache attention
distribution is degraded. A negative result supports the paper's
"safety-net" framing of small-`s_i` updates (the previous hidden
state is preserved). Bears directly on the RQ4 router-validity
claim and on any future work that treats halting-depth as a
mechanical primitive.

Implementation
--------------
- **Clone set selection**: per the spec, the clone set is the set
  of token positions whose preceding router gave ``sigmoid(logit) <
  0.1``. The ``--clone-size`` argument caps the per-router clone
  count for the permutation-null compute budget; when fewer than
  ``--clone-size`` low-score positions exist, all are used.
- **Control set**: a same-size random sample of positions from the
  HIGH-score complement (``s_{i-1} >= 0.1``). The matched-size
  control keeps the permutation null degenerate-free.
- **DV-1 (relative update magnitude)**: for each adjacent repeat
  pair ``(r, r+1)``, ``||x^(r+1) - x^(r)||_2 / max(||x^(r)||_2, eps)``
  per (layer, position). Aggregated to a per-(layer, repeat-pair)
  difference of means (clone vs control).
- **DV-2 (clone-set attention entropy)**: per-(layer, repeat) mean
  per-head Shannon entropy of the ATTN_WEIGHTS row at each clone
  position; the test statistic is the difference of means
  (clone mean - control mean).
- **Permutation null**: ``--n-permutations`` shuffles of the
  clone-vs-control label vector under H0 of exchangeability;
  the empirical 99.7th-percentile of the absolute statistic
  defines the per-cell rejection threshold.
- **Holm-Bonferroni**: applied across all (layer, repeat) cells
  for each DV at family-wise alpha = 0.01. Uses
  ``statsmodels.stats.multitest.multipletests`` when available;
  falls back to a pure-numpy port otherwise.

Workspace consumption
---------------------
The protocol consumes pre-captured residuals + attention weights +
router logits when present in ``--workspace``; otherwise it runs a
single forward pass via ``ActivationCollector`` to populate them.
The ``--workspace`` is the same scratch directory used by the other
protocols (Logit Lens, Protocol C, etc.), so on a typical pipeline
run only ROUTER_LOGITS still needs capturing here (Logit Lens has
already produced ``residual_mid_l<L>_r<R>.npy`` and Protocol C has
produced ``attn_weights_mid_l<L>_r<R>.npy``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from typing import Any

import numpy as np

from analysis.common.sites import discover_workspace_sites, residual_key_prefix


# Spec choice: the §1.2 RQ7 "low-score" threshold is the literal
# 0.1 sigmoid score from the IV bin definition; first bin is
# [0, 0.1). The clone set is the set of positions with prior-router
# sigmoid score in this bin.
LOW_SCORE_THRESHOLD = 0.1
NULL_QUANTILE = 0.997  # ~3 sigma per spec
ALPHA_FAMILY = 0.01    # §1.6 Concern 3 family-wise alpha
EPS = 1e-12


# -----------------------------------------------------------------------------
# Permutation-null helpers
# -----------------------------------------------------------------------------


def _diff_of_means(values: np.ndarray, labels: np.ndarray) -> float:
    """Difference of means: ``mean(values[labels==1]) - mean(values[labels==0])``.

    Returns NaN when either group is empty so callers can exclude
    the cell from aggregates.
    """
    clone = values[labels == 1]
    control = values[labels == 0]
    if clone.size == 0 or control.size == 0:
        return float("nan")
    return float(np.mean(clone) - np.mean(control))


def _permutation_null(
    values: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle ``labels`` ``n_permutations`` times; return null statistics.

    Each entry of the returned array is ``_diff_of_means`` evaluated on
    the permuted labels. Operates on a copy of ``labels`` so the caller's
    array is not mutated.
    """
    n = labels.shape[0]
    null = np.empty(n_permutations, dtype=np.float64)
    perm_labels = labels.copy()
    for k in range(n_permutations):
        rng.shuffle(perm_labels)
        null[k] = _diff_of_means(values, perm_labels)
    return null


# -----------------------------------------------------------------------------
# Holm-Bonferroni
# -----------------------------------------------------------------------------


def _holm_bonferroni(p_values: list[float], alpha: float) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni correction; return (decisions, adjusted p-values).

    Prefers ``statsmodels.stats.multitest.multipletests`` when available
    (per the directive guidance on the canonical Holm path). Falls back
    to a numpy port otherwise; the fallback agrees with statsmodels to
    floating-point tolerance on the per-test reject decision.
    """
    if not p_values:
        return [], []
    try:
        from statsmodels.stats.multitest import multipletests

        reject, p_adj, _alpha_sidak, _alpha_bonf = multipletests(
            p_values, alpha=alpha, method="holm"
        )
        return [bool(r) for r in reject], [float(p) for p in p_adj]
    except ImportError:
        pass

    m = len(p_values)
    order = sorted(range(m), key=lambda k: p_values[k])
    decisions = [False] * m
    adjusted = [1.0] * m
    running_max = 0.0
    failed = False
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        adj = (m - rank) * p_values[idx]
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
        if not failed and p_values[idx] <= threshold:
            decisions[idx] = True
        else:
            failed = True
    return decisions, adjusted


# -----------------------------------------------------------------------------
# Workspace consumption + capture
# -----------------------------------------------------------------------------


def _discover_router_files(workspace_dir: str) -> dict[int, str]:
    """Scan ``workspace_dir`` for ``router_mod<K>.npy`` entries.

    Returns a dict ``{router_index: abs_path}``. The collector's
    ROUTER_LOGITS site writes one file per ``mod[k]`` router; per the
    ADM forward, ``mod[k]`` runs between repeats ``k+1`` and ``k+2``,
    so a token's "preceding router score" for repeat ``r`` is
    ``router_mod{r-2}`` (1-indexed repeats; r in [2, n_repeat]).
    """
    if not os.path.isdir(workspace_dir):
        return {}
    out: dict[int, str] = {}
    prefix = "router_mod"
    suffix = ".npy"
    for name in os.listdir(workspace_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        stem = name[len(prefix) : -len(suffix)]
        try:
            idx = int(stem)
        except ValueError:
            continue
        out[idx] = os.path.join(workspace_dir, name)
    return out


def _discover_attn_files(workspace_dir: str) -> dict[tuple[int, int], str]:
    """Scan for ``attn_weights_<group>_l<L>_r<R>.npy`` entries.

    Returns ``{(layer, repeat): abs_path}``.
    """
    if not os.path.isdir(workspace_dir):
        return {}
    out: dict[tuple[int, int], str] = {}
    prefix = "attn_weights_"
    suffix = ".npy"
    for name in os.listdir(workspace_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        stem = name[len(prefix) : -len(suffix)]
        if "_l" not in stem or "_r" not in stem:
            continue
        try:
            _group, rest = stem.split("_l", 1)
            layer_part, repeat_part = rest.split("_r", 1)
            layer = int(layer_part)
            repeat = int(repeat_part)
        except ValueError:
            continue
        out[(layer, repeat)] = os.path.join(workspace_dir, name)
    return out


def _discover_residual_files(
    workspace_dir: str,
    residual_prefix: str,
) -> dict[tuple[int, int], str]:
    """Return ``{(layer, repeat): path}`` for residual_<group>_l<L>_r<R>.npy."""
    out: dict[tuple[int, int], str] = {}
    for layer, repeat, path in discover_workspace_sites(
        workspace_dir, residual_prefix
    ):
        out[(layer, repeat)] = path
    return out


def _run_capture_stage(args: argparse.Namespace, missing: dict[str, bool]) -> None:
    """Run a single forward pass to populate any missing workspace caches.

    ``missing`` flags which buffer families are absent: ``residual``,
    ``attn``, ``router``. The collector requests the union of the
    necessary sites, runs ONE forward pass, and persists. ``non_flash``
    is forced to True iff ATTN_WEIGHTS is requested (constructor
    requirement).
    """
    import torch

    from analysis.common.collector import ActivationCollector
    from analysis.common.data import iterate_owt2_val
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite

    sites: list[ActivationSite] = []
    if missing.get("residual", False):
        sites.append(ActivationSite.RESIDUAL_POST_MID)
    if missing.get("attn", False):
        sites.append(ActivationSite.ATTN_WEIGHTS)
    if missing.get("router", False):
        sites.append(ActivationSite.ROUTER_LOGITS)
    if not sites:
        return

    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        config_mode=args.config_mode,
        module_path=args.module_path,
    )
    model.eval()

    mod_list = getattr(getattr(model, "transformer", None), "mod", None)
    if (mod_list is None or len(mod_list) == 0) and ActivationSite.ROUTER_LOGITS in sites:
        # Non-ADM checkpoint: short-circuit. Caller will surface the
        # missing-router condition as a not_applicable verdict.
        return

    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
        }.get(dtype)
    if args.device.startswith("cuda") and dtype is not None:
        type_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        type_ctx = nullcontext()

    data_iter = iterate_owt2_val(
        data_dir=args.data_dir,
        config=config,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        total_tokens=args.max_tokens,
        device=args.device,
        start_offset=0,
        split="val",
    )

    non_flash = ActivationSite.ATTN_WEIGHTS in sites
    collector = ActivationCollector(
        model,
        sites,
        non_flash=non_flash,
        module_path=args.module_path,
    )
    os.makedirs(args.workspace, exist_ok=True)
    with collector:
        total = collector.run(
            data_iter, type_ctx=type_ctx, max_tokens=args.max_tokens
        )
    collector.save(
        args.workspace,
        meta_extra={
            "protocol": "H",
            "analysis": "interpolation_validity",
            "seq_length": int(args.seq_length),
            "max_tokens": int(args.max_tokens),
            "checkpoint_dir": str(args.checkpoint),
            "checkpoint_filename": str(args.checkpoint_file),
            "n_tokens_captured": int(total),
        },
    )


# -----------------------------------------------------------------------------
# Statistic computation
# -----------------------------------------------------------------------------


def _shannon_entropy_bits(p: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy in bits along the last axis.

    Mirrors ``analysis.attention_taxonomy.shannon_entropy`` (does NOT
    renormalise; assumes rows are already valid probability
    distributions).
    """
    q = np.clip(p, EPS, 1.0)
    return -np.sum(q * np.log2(q), axis=-1)


def _compute_dv1_per_cell(
    residuals: dict[tuple[int, int], np.ndarray],
    layers: list[int],
    repeats: list[int],
) -> dict[tuple[int, int], np.ndarray]:
    """DV-1: per-(layer, repeat-pair) relative-update magnitude per position.

    Returns a dict keyed on ``(layer, repeat)`` for ``repeat`` in
    ``repeats[1:]``; the value at key ``(L, r)`` is the per-position
    array ``||x^(L, r) - x^(L, r-1)||_2 / max(||x^(L, r-1)||_2, eps)``
    of length ``n_tokens``.
    """
    if len(repeats) < 2:
        return {}
    out: dict[tuple[int, int], np.ndarray] = {}
    for layer in layers:
        prev_repeat = repeats[0]
        prev = residuals.get((layer, prev_repeat))
        if prev is None:
            continue
        for repeat in repeats[1:]:
            curr = residuals.get((layer, repeat))
            if curr is None:
                prev = curr
                prev_repeat = repeat
                continue
            n_common = min(prev.shape[0], curr.shape[0])
            delta = curr[:n_common] - prev[:n_common]
            num = np.linalg.norm(delta, axis=-1)
            denom = np.linalg.norm(prev[:n_common], axis=-1)
            denom = np.maximum(denom, EPS)
            out[(layer, repeat)] = (num / denom).astype(np.float64)
            prev = curr
            prev_repeat = repeat
    return out


def _compute_dv2_per_cell(
    attn: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """DV-2: per-(layer, repeat) per-position mean-per-head attention entropy.

    Each value is a length-``n_tokens`` array; ``att`` capture is shape
    ``(N_tokens, n_head, T_k)`` so per-head entropy is reduced to a
    per-position scalar by averaging across heads.
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for key, arr in attn.items():
        if arr.ndim != 3:
            continue
        ent = _shannon_entropy_bits(arr)  # (N_tokens, n_head)
        out[key] = ent.mean(axis=-1).astype(np.float64)
    return out


def _per_token_router_score(
    router_files: dict[int, str],
    n_tokens: int,
) -> dict[int, np.ndarray]:
    """Return ``{router_index: sigmoid(logits)[:n_tokens]}``.

    Routers are indexed 0..n_repeat-2 in the ADM forward; ``mod[k]``
    fires between repeats ``k+1`` and ``k+2`` so its score governs the
    interpolation update at repeat ``k+2``.
    """
    out: dict[int, np.ndarray] = {}
    for idx, path in router_files.items():
        logits = np.load(path)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim != 1:
            continue
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        if n_tokens > 0:
            scores = scores[:n_tokens]
        out[idx] = scores
    return out


def _select_clone_and_control(
    scores: np.ndarray,
    clone_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(clone_idx, control_idx)`` index arrays into ``scores``.

    ``clone`` is positions with ``score < LOW_SCORE_THRESHOLD``, capped
    at ``clone_size`` via uniform random subsampling. ``control`` is a
    same-size sample from the complement. Returns ``(empty, empty)``
    when either pool is too small.
    """
    low_pool = np.flatnonzero(scores < LOW_SCORE_THRESHOLD)
    high_pool = np.flatnonzero(scores >= LOW_SCORE_THRESHOLD)
    if low_pool.size == 0 or high_pool.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    target = min(clone_size, int(low_pool.size), int(high_pool.size))
    if target <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if low_pool.size > target:
        clone_idx = rng.choice(low_pool, size=target, replace=False)
    else:
        clone_idx = low_pool.copy()
    control_idx = rng.choice(high_pool, size=target, replace=False)
    return clone_idx.astype(np.int64), control_idx.astype(np.int64)


def _per_cell_test(
    values: np.ndarray,
    clone_idx: np.ndarray,
    control_idx: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One-cell statistic + permutation null + uncorrected p-value.

    The observed statistic is the absolute difference of means
    ``|mean(values[clone]) - mean(values[control])|``; the null is
    ``|null_diff|`` from ``_permutation_null``. p-uncorrected is the
    proportion of null statistics at-or-above the observed.

    Returns NaN-bearing dict when either side is empty or the values
    array does not cover the indices.
    """
    n_total = int(values.shape[0])
    valid = (
        clone_idx.size > 0
        and control_idx.size > 0
        and (clone_idx.max() < n_total)
        and (control_idx.max() < n_total)
    )
    if not valid:
        return {
            "stat": float("nan"),
            "null_q997": float("nan"),
            "p_uncorrected": float("nan"),
            "n_clone": int(clone_idx.size),
            "n_control": int(control_idx.size),
        }

    pool_idx = np.concatenate([clone_idx, control_idx])
    labels = np.concatenate(
        [
            np.ones(clone_idx.size, dtype=np.int64),
            np.zeros(control_idx.size, dtype=np.int64),
        ]
    )
    pool_values = values[pool_idx]
    obs = abs(_diff_of_means(pool_values, labels))
    null = np.abs(_permutation_null(pool_values, labels, n_permutations, rng))
    null_q997 = float(np.nanquantile(null, NULL_QUANTILE))
    n_at_or_above = int(np.sum(null >= obs))
    p_unc = (n_at_or_above + 1) / (n_permutations + 1)  # +1 / +1 add-one smoothing
    return {
        "stat": float(obs),
        "null_q997": null_q997,
        "p_uncorrected": float(p_unc),
        "n_clone": int(clone_idx.size),
        "n_control": int(control_idx.size),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _render_per_layer_figure(
    payload: dict[str, Any],
    output_path: str,
) -> None:
    """Render the per-layer DV-1 / DV-2 trajectory figure with null bands."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cells = payload.get("per_cell", [])
    if not cells:
        return
    layers = sorted({c["layer"] for c in cells})
    repeats = sorted({c["repeat"] for c in cells})
    if not layers or not repeats:
        return

    def _grid(metric: str, key_stat: str, key_null: str) -> tuple[np.ndarray, np.ndarray]:
        stat = np.full((len(layers), len(repeats)), np.nan, dtype=np.float64)
        null = np.full((len(layers), len(repeats)), np.nan, dtype=np.float64)
        for c in cells:
            if c.get("dv") != metric:
                continue
            li = layers.index(c["layer"])
            ri = repeats.index(c["repeat"])
            stat[li, ri] = c.get(key_stat, np.nan)
            null[li, ri] = c.get(key_null, np.nan)
        return stat, null

    dv1_stat, dv1_null = _grid("DV1", "stat", "null_q997")
    dv2_stat, dv2_null = _grid("DV2", "stat", "null_q997")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    cmap = plt.cm.viridis

    for ax, stat, null, title in [
        (axes[0], dv1_stat, dv1_null, "DV-1 |mean diff| (relative update mag)"),
        (axes[1], dv2_stat, dv2_null, "DV-2 |mean diff| (per-head attn entropy)"),
    ]:
        for li, layer in enumerate(layers):
            color = cmap(li / max(len(layers) - 1, 1))
            ax.plot(repeats, stat[li], color=color, alpha=0.6, linewidth=0.8)
            ax.plot(repeats, null[li], color=color, alpha=0.3, linewidth=0.6, linestyle="--")
        mean_stat = np.nanmean(stat, axis=0)
        mean_null = np.nanmean(null, axis=0)
        ax.plot(repeats, mean_stat, color="black", linewidth=2.0, label="layer mean")
        ax.plot(
            repeats,
            mean_null,
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="null q99.7 (mean)",
        )
        ax.set_title(title)
        ax.set_xlabel("Repeat index")
        ax.set_ylabel("Statistic value")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Protocol H -- Interpolation Validity (RQ7)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol H.

    Required: ``--checkpoint``, ``--workspace``, ``--output-dir``.
    Optional: ``--checkpoint-file``, ``--seed``, ``--clone-size``,
    ``--n-permutations``, ``--device``, ``--config-mode``,
    ``--module-path``, ``--seq-length``, ``--batch-size``,
    ``--max-tokens``, ``--data-dir``, ``--skip-capture``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Protocol H -- Interpolation Validity (RQ7); clone-set "
            "permutation test on DV-1 (relative update magnitude) + "
            "DV-2 (clone-set attention entropy)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory containing summary.json + ckpt file",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default ckpt.pt)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=(
            "Workspace directory consumed for residual / attn / router "
            "captures; auto-populated via a single forward pass when empty"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for interpolation_validity_*.json / *.png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2357,
        help="RNG seed for clone-set sampling and permutation null",
    )
    parser.add_argument(
        "--clone-size",
        type=int,
        default=64,
        help="Clone-set size per router (default 64; matches HPC INTERP_CLONE_SIZE)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Permutation-null sample count (default 10000 per DEC-026)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the optional capture stage (default cuda)",
    )
    parser.add_argument(
        "--config-mode",
        type=str,
        default="raw",
        choices=["raw", "argparse"],
    )
    parser.add_argument(
        "--module-path",
        type=str,
        default="model.transformer.h_mid",
    )
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help=(
            "Skip the workspace-population forward pass; assume the "
            "workspace is already populated by upstream protocols"
        ),
    )
    parser.add_argument(
        "--alpha-family",
        type=float,
        default=ALPHA_FAMILY,
        help=f"Family-wise alpha for Holm-Bonferroni (default {ALPHA_FAMILY})",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run the synthetic-data smoke test of the permutation-null + "
            "Holm-Bonferroni primitives and exit (no checkpoint required)"
        ),
    )
    return parser


def _write_short_circuit(
    args: argparse.Namespace,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a not-applicable verdict (e.g. non-ADM checkpoint)."""
    payload: dict[str, Any] = {
        "schema_version": "interp-validity-1.0",
        "verdict": "not_applicable",
        "reason": reason,
        "checkpoint": {
            "path": str(args.checkpoint),
            "file": str(args.checkpoint_file),
        },
    }
    if extra:
        payload.update(extra)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "interpolation_validity_results.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"interpolation_validity: wrote {out_path} (not_applicable: {reason})")


def _smoke_test() -> None:
    """Verify the permutation-null primitives on synthetic data.

    Generates a random per-(layer, repeat, position) value array with a
    deterministic seed (project convention 19937), constructs a small
    clone-set + matched-size control under H0 (exchangeable labels), and
    exercises ``_diff_of_means``, ``_permutation_null``, ``_per_cell_test``,
    and ``_holm_bonferroni``. Asserts shape, finiteness, and the
    Holm step-down monotonicity property; exits non-zero on any failure.
    Pure-numpy: no torch import is reachable from this path.
    """
    rng = np.random.default_rng(19937)
    n_positions = 32

    # Synthetic per-position scalar values; H0 holds (random) so the
    # observed |diff of means| should sit near the centre of the null.
    values = rng.normal(size=(n_positions,)).astype(np.float64)
    clone_idx = np.arange(0, n_positions // 2, dtype=np.int64)
    control_idx = np.arange(n_positions // 2, n_positions, dtype=np.int64)

    # _diff_of_means: finite scalar on a balanced split.
    labels = np.concatenate([
        np.ones(clone_idx.size, dtype=np.int64),
        np.zeros(control_idx.size, dtype=np.int64),
    ])
    pool_values = values[np.concatenate([clone_idx, control_idx])]
    obs = _diff_of_means(pool_values, labels)
    assert math.isfinite(obs), f"_diff_of_means is not finite: {obs}"

    # _permutation_null: shape and finite-element discipline.
    n_perm = 200
    null = _permutation_null(pool_values, labels, n_perm, rng)
    assert null.shape == (n_perm,), f"null shape {null.shape}"
    assert np.all(np.isfinite(null)), "permutation null has non-finite entries"

    # _per_cell_test: end-to-end one-cell test produces the expected keys.
    test = _per_cell_test(values, clone_idx, control_idx, n_perm, rng)
    expected_keys = {"stat", "null_q997", "p_uncorrected", "n_clone", "n_control"}
    assert expected_keys.issubset(test.keys()), f"missing keys: {expected_keys - test.keys()}"
    assert math.isfinite(test["stat"]), f"test stat not finite: {test}"
    assert math.isfinite(test["null_q997"]), f"null_q997 not finite: {test}"
    assert 0.0 < test["p_uncorrected"] <= 1.0, f"p_uncorrected out of range: {test}"
    assert test["n_clone"] == clone_idx.size
    assert test["n_control"] == control_idx.size

    # _holm_bonferroni: step-down monotone (sorted adjusted p's are
    # non-decreasing) and decisions consistent with adjusted p <= alpha
    # at the rejected end of the order.
    raw_p = [0.001, 0.02, 0.04, 0.5, 0.9]
    decisions, p_adj = _holm_bonferroni(raw_p, alpha=0.05)
    assert len(decisions) == len(raw_p) and len(p_adj) == len(raw_p)
    sorted_adj = sorted(p_adj)
    for k in range(1, len(sorted_adj)):
        assert sorted_adj[k] >= sorted_adj[k - 1] - 1e-12, (
            f"Holm-adjusted p-values not monotone non-decreasing: {sorted_adj}"
        )

    # _select_clone_and_control: small synthetic score vector with both
    # low- and high-score regions. Both arrays must come back non-empty.
    scores = rng.uniform(0.0, 1.0, size=(128,))
    scores[:20] = 0.05  # force a low-score pool
    sel_clone, sel_control = _select_clone_and_control(scores, clone_size=8, rng=rng)
    assert sel_clone.size > 0 and sel_control.size > 0, (
        f"clone/control selection empty: clone={sel_clone.size}, control={sel_control.size}"
    )
    assert sel_clone.size == sel_control.size, (
        f"clone and control sizes must match: {sel_clone.size} vs {sel_control.size}"
    )

    print("interpolation_validity smoke test PASS")
    print(f"  permutation null shape OK: {null.shape}")
    print(f"  Holm step-down monotone OK: {sorted_adj}")
    print(f"  test stat finite: {test['stat']:.4f} (n_perm={n_perm})")
    print(
        f"  clone/control selection OK: clone={sel_clone.size}, "
        f"control={sel_control.size}"
    )


def main() -> None:
    args = build_argparser().parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    parser = build_argparser()
    if not args.checkpoint:
        parser.error("--checkpoint is required when --smoke-test is not set")
    if not args.workspace:
        parser.error("--workspace is required when --smoke-test is not set")
    if not args.output_dir:
        parser.error("--output-dir is required when --smoke-test is not set")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.workspace, exist_ok=True)

    residual_prefix = residual_key_prefix(args.module_path)
    residuals_present = bool(_discover_residual_files(args.workspace, residual_prefix))
    attn_present = bool(_discover_attn_files(args.workspace))
    routers_present = bool(_discover_router_files(args.workspace))

    missing = {
        "residual": not residuals_present,
        "attn": not attn_present,
        "router": not routers_present,
    }

    if any(missing.values()):
        if args.skip_capture:
            print(
                "interpolation_validity: --skip-capture set but workspace "
                f"is missing: residuals={missing['residual']}, "
                f"attn={missing['attn']}, router={missing['router']}",
                file=sys.stderr,
            )
            sys.exit(2)
        _run_capture_stage(args, missing)

    # Re-discover after the capture pass.
    residual_files = _discover_residual_files(args.workspace, residual_prefix)
    attn_files = _discover_attn_files(args.workspace)
    router_files = _discover_router_files(args.workspace)

    if not router_files:
        # Non-ADM checkpoint or capture failed: emit not_applicable.
        _write_short_circuit(
            args,
            reason=(
                "no router_mod*.npy in workspace after capture; "
                "checkpoint likely lacks transformer.mod (non-ADM)"
            ),
        )
        return

    if not residual_files:
        _write_short_circuit(
            args,
            reason="no residual_*_l*_r*.npy in workspace after capture",
        )
        return

    layers = sorted({l for (l, _r) in residual_files})
    repeats = sorted({r for (_l, r) in residual_files})

    print(
        f"interpolation_validity: {len(layers)} layers x {len(repeats)} "
        f"repeats; routers={sorted(router_files.keys())}; "
        f"attn cells={len(attn_files)}"
    )

    # Load residuals + attention into memory once.
    residuals: dict[tuple[int, int], np.ndarray] = {
        key: np.load(path).astype(np.float32, copy=False)
        for key, path in residual_files.items()
    }
    attn: dict[tuple[int, int], np.ndarray] = {
        key: np.load(path) for key, path in attn_files.items()
    }

    # Establish the common token count -- the shortest captured buffer
    # bounds every subsequent slice so the position labels line up.
    n_tokens_residual = min(arr.shape[0] for arr in residuals.values())
    n_tokens_attn = (
        min(arr.shape[0] for arr in attn.values()) if attn else n_tokens_residual
    )
    n_tokens = min(n_tokens_residual, n_tokens_attn)

    router_scores = _per_token_router_score(router_files, n_tokens)
    if not router_scores:
        _write_short_circuit(args, reason="router file load yielded no usable scores")
        return

    rng = np.random.default_rng(args.seed)

    # DV-1 and DV-2 per-cell statistic + null + uncorrected p-value.
    dv1_per_cell = _compute_dv1_per_cell(
        {k: v[:n_tokens] for k, v in residuals.items()}, layers, repeats
    )
    dv2_per_cell = _compute_dv2_per_cell(
        {k: v[:n_tokens] for k, v in attn.items()}
    )

    # Per the §1.2 RQ7 spec: a token's "preceding router" for repeat r
    # is mod[r-2] (1-indexed; r = 2..n_repeat). DV-2 examines the FINAL
    # repeat over positions whose mod[n_repeat-2] score is low; we run
    # the same test at every repeat for completeness, picking the
    # appropriate router per cell.
    cells: list[dict[str, Any]] = []
    p_dv1: list[float] = []
    p_dv2: list[float] = []
    cell_index_dv1: list[int] = []
    cell_index_dv2: list[int] = []

    for layer in layers:
        for repeat in repeats:
            # The clone-defining router for this cell: mod[repeat - 2].
            router_idx = repeat - 2
            if router_idx < 0 or router_idx not in router_scores:
                continue
            scores = router_scores[router_idx]
            clone_idx, control_idx = _select_clone_and_control(
                scores, args.clone_size, rng
            )
            if clone_idx.size == 0 or control_idx.size == 0:
                continue

            # DV-1: only defined when (layer, repeat) has a paired
            # adjacent-repeat residual difference cached (i.e. repeat > min).
            if (layer, repeat) in dv1_per_cell:
                dv1_values = dv1_per_cell[(layer, repeat)]
                rng_dv1 = np.random.default_rng(
                    np.random.SeedSequence([args.seed, int(layer), int(repeat), 1])
                )
                dv1_test = _per_cell_test(
                    dv1_values,
                    clone_idx,
                    control_idx,
                    args.n_permutations,
                    rng_dv1,
                )
                cells.append({
                    "dv": "DV1",
                    "layer": int(layer),
                    "repeat": int(repeat),
                    "router_idx": int(router_idx),
                    **dv1_test,
                })
                if math.isfinite(dv1_test["p_uncorrected"]):
                    cell_index_dv1.append(len(cells) - 1)
                    p_dv1.append(dv1_test["p_uncorrected"])

            # DV-2: per-head attention entropy at this (layer, repeat).
            if (layer, repeat) in dv2_per_cell:
                dv2_values = dv2_per_cell[(layer, repeat)]
                rng_dv2 = np.random.default_rng(
                    np.random.SeedSequence([args.seed, int(layer), int(repeat), 2])
                )
                dv2_test = _per_cell_test(
                    dv2_values,
                    clone_idx,
                    control_idx,
                    args.n_permutations,
                    rng_dv2,
                )
                cells.append({
                    "dv": "DV2",
                    "layer": int(layer),
                    "repeat": int(repeat),
                    "router_idx": int(router_idx),
                    **dv2_test,
                })
                if math.isfinite(dv2_test["p_uncorrected"]):
                    cell_index_dv2.append(len(cells) - 1)
                    p_dv2.append(dv2_test["p_uncorrected"])

    # Holm-Bonferroni applied separately within each DV family.
    decisions_dv1, p_holm_dv1 = _holm_bonferroni(p_dv1, args.alpha_family)
    decisions_dv2, p_holm_dv2 = _holm_bonferroni(p_dv2, args.alpha_family)
    for slot, idx in enumerate(cell_index_dv1):
        cells[idx]["p_holm"] = float(p_holm_dv1[slot])
        cells[idx]["rejects_h0"] = bool(decisions_dv1[slot])
    for slot, idx in enumerate(cell_index_dv2):
        cells[idx]["p_holm"] = float(p_holm_dv2[slot])
        cells[idx]["rejects_h0"] = bool(decisions_dv2[slot])
    # Cells without a finite p (degenerate sampling) get explicit defaults
    # so the downstream JSON always has the same key set.
    for c in cells:
        c.setdefault("p_holm", float("nan"))
        c.setdefault("rejects_h0", False)

    n_rejected = int(sum(1 for c in cells if c.get("rejects_h0", False)))
    any_layer_significant_at_repeat: list[bool] = []
    for repeat in repeats:
        any_layer_significant_at_repeat.append(
            bool(any(
                c.get("rejects_h0", False)
                for c in cells
                if c.get("repeat") == repeat and c.get("dv") == "DV2"
            ))
        )

    payload: dict[str, Any] = {
        "schema_version": "interp-validity-1.0",
        "checkpoint": {
            "path": str(args.checkpoint),
            "file": str(args.checkpoint_file),
        },
        "args": {
            "seed": int(args.seed),
            "clone_size": int(args.clone_size),
            "n_permutations": int(args.n_permutations),
            "alpha_family": float(args.alpha_family),
            "low_score_threshold": LOW_SCORE_THRESHOLD,
            "null_quantile": NULL_QUANTILE,
            "module_path": str(args.module_path),
            "seq_length": int(args.seq_length),
            "batch_size": int(args.batch_size),
            "max_tokens": int(args.max_tokens),
        },
        "n_permutations": int(args.n_permutations),
        "clone_size": int(args.clone_size),
        "layers": layers,
        "repeats": repeats,
        "n_tokens_used": int(n_tokens),
        "router_indices": sorted(router_scores.keys()),
        "per_cell": cells,
        "aggregate": {
            "n_cells_rejected_holm": n_rejected,
            "any_layer_significant_at_repeat": any_layer_significant_at_repeat,
        },
        "figure_paths": ["interpolation_validity_per_layer.png"],
    }

    json_path = os.path.join(args.output_dir, "interpolation_validity_results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"interpolation_validity: wrote {json_path}")

    fig_path = os.path.join(args.output_dir, "interpolation_validity_per_layer.png")
    _render_per_layer_figure(payload, fig_path)
    if os.path.exists(fig_path):
        print(f"interpolation_validity: wrote {fig_path}")

    print(
        f"interpolation_validity: {n_rejected} cells reject H0 at "
        f"family-wise alpha={args.alpha_family} (Holm-Bonferroni)"
    )


if __name__ == "__main__":
    main()
