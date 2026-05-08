"""Protocol C -- Attention-Head Taxonomy + Sparsity-across-Repeat test.

Scope
-----
Captures per-(layer, repeat, head) attention weights at every mid-layer
across the checkpoint matrix (C1, C2, C3, C4, C5 per
``docs/extend-notes.md`` §1.5 Analysis Matrix) using the ``ATTN_WEIGHTS``
site in ``analysis.common.collector.ActivationCollector`` (unblocked by
this wave via the ``_install_attn_monkey_patch`` hook on
``CausalSelfAttention.forward``). The forward pass is forced to the
non-flash math backend because Flash Attention does not return per-head
weights; the collector flips ``attn.flash=False`` and the monkey-patch
re-runs the Q/K math path to recover ``att`` post-softmax.

For every (layer, repeat, head) triplet Protocol C computes three
operationally independent sparsity metrics per query position, then
aggregates over the query axis to produce per-head means:

- Shannon entropy of the causal attention distribution (bits).
- Gini coefficient per the Hurley and Rickard 2009 [42] normal form
  (the only axis-6-compliant sparsity measure alongside the pq-mean;
  distinct from entropy under cloning of duplicate mass, so the two
  are operationally independent in the triangulation sense).
- Top-5 attention mass: fraction of probability on the top-5 key
  positions.

Two position-pattern detectors are computed per head:

- Attention-sink fraction: fraction of queries whose top-1 attention
  lands at key position 0 (BOS token); threshold > 0.8 flags "sink"
  following Xiao et al. 2023 [26] / [36].
- Previous-token fraction: fraction of queries at position t whose
  top-1 attention lands at position t - 1; threshold > 0.5 flags
  "previous-token" following Voita et al. 2019 [37].

Sink correction (mask position 0 and renormalise) is applied to every
metric in a parallel "no-sink" pass; both the raw and sink-corrected
variants are reported per ``docs/extend-notes.md`` §1.3 Protocol C
implementation note.

A five-way classification cascade assigns each (layer, repeat, head)
triplet a label in {sink, previous-token, uniform, content, other}:

1. sink if ``sink_fraction > 0.8``.
2. previous-token if not-sink and ``prev_token_fraction > 0.5``.
3. uniform if not-sink, not-previous-token, and ``gini < 0.2``.
4. content if not-sink, not-previous-token, not-uniform, and
   ``top5_mass > 0.7``.
5. other otherwise.

Falsifiability relevance
------------------------
Prediction 2 (sparse attention from repetition; Zucchet et al. 2025
[12] / [35]) predicts that attention entropy at repeat 5 is
significantly lower than at repeat 1 on per-head paired samples.
Primary test per DIR-001 T2: paired t-test on per-head repeat-1 vs
repeat-5 mean Gini across all 252 heads (21 mid-layers x 12 heads),
two-sided at ``alpha = 0.01``; the direction must match the
prediction (repeat-5 Gini > repeat-1 Gini) to count as a positive
falsification of the null.

Per ``docs/extend-notes.md`` §1.4 Prediction 2, the full Prediction-2
test protocol includes three concordant checks:

1. The primary paired t-test above, extended to entropy and top-5 mass
   (concordant directionality required across all three sparsity
   metrics; metric-dependent outcomes are reported as "metric-dependent"
   rather than supporting P2).
2. Per-layer Spearman rho between repeat index and per-head mean
   metric, with Holm-Bonferroni correction across the 21 mid-layers;
   rho < -0.5 is Cohen 1988 [49] "large effect" threshold.
3. Mixed-effects model ``metric ~ repeat + (1 | layer) + (1 | head:layer)``
   via ``statsmodels.regression.mixed_linear_model.MixedLM`` if
   ``statsmodels`` is importable; the fixed-effect slope on ``repeat``
   with bootstrap 95 % CI excluding zero is the formal per-§1.4 test.
   When statsmodels is unavailable the mixed-effects row records a
   "statsmodels_unavailable" sentinel and the verdict falls back to
   the paired t-test + per-layer Spearman combination.

Validation gate per §1.4 Prediction 2: the mean repeat-1 entropy must
exceed 2.0 bits across all 252 heads; if already near zero the head
cannot become sparser and Prediction 2 is vacuously unfalsifiable.

Ontological purpose
-------------------
Zucchet et al. 2025 [35] document sparse attention emergence under
data repetition; Protocol C tests whether the weight-tied repeat
regime in CoTFormer produces an analogous repetition-induced sparsity
in the depth (not data) axis. A positive result extends the Zucchet
observation to a novel operating regime; a negative result refutes
the analogy and triggers downgrade of the §1.6 triangulation row on
attention-pattern evidence.

Entry point
-----------
``python -m analysis.attention_taxonomy \\
    --checkpoint <ckpt_dir> \\
    --checkpoint-file <ckpt_file> \\
    --workspace <path> \\
    --output-dir <path>``

The capture stage runs if no ``attn_weights_*_r*.npy`` files are
present in ``--workspace``; otherwise the analysis stage reads the
pre-captured files. Capture is GPU-bound (~9 min per checkpoint on L4
per §1.8) due to the non-flash + per-layer re-computation cost; the
analysis stage is CPU-bound.

Outputs
-------
- ``attention_taxonomy_results.json``: full per-(layer, repeat, head)
  panel of metrics, classification labels, and Prediction-2 verdict.
- ``attention_taxonomy_heatmap.png``: (n_layer_mid, n_repeat) grid
  coloured by dominant taxonomy cell per (layer, repeat).
- ``attention_sparsity_trajectory.png``: per-layer mean Gini / entropy
  / top-5 mass curves across repeats 1..n_repeat.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from typing import Any

import numpy as np


# --------------------------------------------------------------
# Frozen configuration constants (DIR-001 commit; see §1.3 Protocol C)
# --------------------------------------------------------------

SINK_THRESHOLD = 0.8           # top-1 fraction on key position 0
PREV_TOKEN_THRESHOLD = 0.5     # top-1 fraction on position t-1
UNIFORM_GINI_THRESHOLD = 0.2   # per-head mean Gini for "uniform" label
CONTENT_TOP5_THRESHOLD = 0.7   # top-5 mass cutoff for "content" label
TOP_K = 5                      # top-k mass k
PREDICTION2_ALPHA = 0.01       # paired t-test alpha (primary gate)
VALIDATION_ENTROPY_GATE = 2.0  # repeat-1 mean entropy must exceed this (bits)
SPEARMAN_LARGE_EFFECT = -0.5   # Cohen 1988 "large effect"
MIXED_EFFECTS_ALPHA = 0.05     # §1.4 fixed-effect slope threshold
EPS = 1e-12                    # numerical floor for log and divisions

# RQ3 silhouette permutation-null gates per docs/extend-notes.md §1.2 RQ3.
# The chi-squared p-value gate is the formal H0 falsifier; the silhouette
# permutation null gates the "meaningful clustering detected" claim and
# replaces the prior fixed Kaufman-Rousseeuw 0.25 cutoff (categorical
# convention, not data-derived).
RQ3_CHI2_ALPHA = 0.01           # cluster x layer chi-squared independence
RQ3_SILHOUETTE_ALPHA = 0.05     # one-tailed empirical p-value vs perm null
RQ3_DEFAULT_PERMUTATIONS = 1000  # DEC-026 family permutation count
RQ3_FEATURE_REPEAT = 5           # canonical analysis anchor (final repeat)
RQ3_MAX_CLUSTERS = 8             # hard cap on K = min(8, ceil(log2(H)))


# --------------------------------------------------------------
# Sparsity primitives (pure numpy; unit-test-friendly)
# --------------------------------------------------------------


def shannon_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits, summed over the last axis.

    The 0 log 0 = 0 limit is handled by clipping to ``EPS`` before the
    log. Input is NOT renormalised; callers must pass a probability
    distribution (each row summing to one within floating-point
    tolerance). Entries above the causal horizon are expected to be
    exactly zero after the ``masked_fill(-inf)`` + softmax pipeline;
    clipping and summing is mathematically equivalent to the
    correct-support entropy.
    """
    q = np.clip(p, EPS, 1.0)
    return -np.sum(q * np.log2(q), axis=-1)


def gini_coefficient(p: np.ndarray) -> np.ndarray:
    """Gini coefficient per Hurley and Rickard 2009 [42].

    Uses the O(N log N) sorted form to avoid the O(N^2) pairwise
    formula. Each row of ``p`` is a probability distribution over the
    last axis; the output has that axis removed.

    Closed form:
        G = 1 - 2 * sum_{i=0}^{N-1} ((N - i - 0.5) / N) * p_{(i)}
    where ``p_{(i)}`` is the i-th smallest entry.

    Gini is 0 for a uniform distribution and 1 for a delta (one-hot).
    Scales between these bounds per the Scaling axiom; insensitive to
    cloning of duplicate mass per the Cloning axiom. Hurley-Rickard
    prove Gini satisfies all six sparsity axioms (Robin-Hood, Scaling,
    Rising-Tide, Cloning, Bill-Gates, Babies) whereas Shannon entropy
    fails Cloning. The two metrics are therefore operationally
    independent per ``docs/extend-notes.md`` §1.3 triangulation
    footnote [W-18].
    """
    N = p.shape[-1]
    p_sorted = np.sort(p, axis=-1)
    total = np.sum(p, axis=-1, keepdims=True) + EPS
    p_norm = p_sorted / total
    weights = (N - np.arange(N) - 0.5) / N  # shape (N,)
    return 1.0 - 2.0 * np.sum(weights * p_norm, axis=-1)


def top_k_mass(p: np.ndarray, k: int) -> np.ndarray:
    """Fraction of probability mass on the top-k entries along the last axis.

    Useful as an upper-tail sparsity measure that is also operationally
    independent of Shannon entropy per Pagliardini et al. [45]. When
    ``k >= p.shape[-1]``, returns ones (all mass is on the top-k by
    definition).
    """
    k_eff = min(k, p.shape[-1])
    partitioned = np.partition(p, -k_eff, axis=-1)
    top = partitioned[..., -k_eff:]
    total = np.sum(p, axis=-1) + EPS
    return np.sum(top, axis=-1) / total


def _query_positions(n_tokens: int, seq_length: int) -> np.ndarray:
    """Derive query positions from the collector flatten convention.

    The collector flattens ``(B, T_q, n_head, T_k)`` to
    ``(B * T_q, n_head, T_k)`` with the B axis outer and T_q axis
    inner. So row ``i`` of the flattened capture corresponds to query
    position ``i % T_q`` within its originating batch.
    """
    return np.arange(n_tokens) % seq_length


def sink_fraction_per_head(att: np.ndarray) -> np.ndarray:
    """Fraction of queries whose top-1 attention lands at key position 0.

    Parameters
    ----------
    att : np.ndarray
        Shape ``(N_tokens, n_head, T_k)``. Each row is a per-head
        attention distribution over key positions for one query.

    Returns
    -------
    frac : np.ndarray
        Shape ``(n_head,)`` fraction per head across all queries.
    """
    top1 = np.argmax(att, axis=-1)  # (N_tokens, n_head)
    return np.mean(top1 == 0, axis=0)


def prev_token_fraction_per_head(att: np.ndarray, seq_length: int) -> np.ndarray:
    """Fraction of queries at position t > 0 whose top-1 lands at t - 1.

    The query at position 0 has no predecessor and is excluded from the
    denominator. The count is taken over all (batch, query-position) pairs
    in ``att`` using the ``(i % seq_length)`` position convention.
    """
    N, H, _Tk = att.shape
    q_pos = _query_positions(N, seq_length)  # (N,)
    expected = (q_pos - 1)
    valid = expected >= 0  # shape (N,)
    top1 = np.argmax(att, axis=-1)  # (N, H)
    match = (top1 == expected[:, None]) & valid[:, None]  # (N, H)
    denom = max(int(np.sum(valid)), 1)
    return np.sum(match, axis=0) / denom


def apply_sink_correction(att: np.ndarray) -> np.ndarray:
    """Zero out attention on key position 0 and renormalise per-row.

    Per §1.3 sink-correction note: "mask position 0, renormalise". The
    per-(query, head) row is renormalised so the remaining mass sums
    to one; if the mass outside position 0 is near zero (i.e. the head
    attends almost exclusively to the BOS sink), the row is left as
    the original distribution minus position 0 and is effectively a
    near-zero vector after renormalisation. Callers should interpret
    the corrected metrics alongside the raw metrics to avoid aliasing
    total-mass effects with sink-mass effects.
    """
    corrected = att.copy()
    corrected[..., 0] = 0.0
    row_sums = np.sum(corrected, axis=-1, keepdims=True)
    # Guard against zero rows (pure-sink queries): keep them as zero
    # after division; they contribute zero to both entropy (clipped)
    # and Gini (0 distribution -> G = 0 by the formula in the limit).
    safe = np.where(row_sums > EPS, row_sums, 1.0)
    return corrected / safe


def summarise_attention(
    att: np.ndarray,
    seq_length: int,
    top_k: int = TOP_K,
) -> dict[str, np.ndarray]:
    """Per-head summary statistics on one (layer, repeat) capture.

    Parameters
    ----------
    att : np.ndarray
        Shape ``(N_tokens, n_head, T_k)`` attention weights after
        softmax, as written by the collector for the ATTN_WEIGHTS site.
    seq_length : int
        The per-batch sequence length; required for the
        previous-token detector's position bookkeeping.
    top_k : int
        k for the top-k mass metric.

    Returns
    -------
    stats : dict[str, np.ndarray]
        Per-head means across all queries. Keys:
        ``entropy``, ``entropy_no_sink``,
        ``gini``, ``gini_no_sink``,
        ``top_k_mass``, ``top_k_mass_no_sink``,
        ``sink_fraction``, ``prev_token_fraction``.
        Each value has shape ``(n_head,)``.
    """
    if att.ndim != 3:
        raise ValueError(
            f"summarise_attention: expected (N_tokens, n_head, T_k); "
            f"got shape {att.shape}"
        )

    # Raw metrics per query, then mean over queries.
    entropy_per_q = shannon_entropy(att)            # (N, H)
    gini_per_q = gini_coefficient(att)              # (N, H)
    top_k_per_q = top_k_mass(att, top_k)            # (N, H)

    # Sink-corrected metrics.
    att_ns = apply_sink_correction(att)
    entropy_per_q_ns = shannon_entropy(att_ns)
    gini_per_q_ns = gini_coefficient(att_ns)
    top_k_per_q_ns = top_k_mass(att_ns, top_k)

    sink_frac = sink_fraction_per_head(att)
    prev_frac = prev_token_fraction_per_head(att, seq_length)

    return {
        "entropy": np.mean(entropy_per_q, axis=0).astype(np.float32),
        "entropy_no_sink": np.mean(entropy_per_q_ns, axis=0).astype(np.float32),
        "gini": np.mean(gini_per_q, axis=0).astype(np.float32),
        "gini_no_sink": np.mean(gini_per_q_ns, axis=0).astype(np.float32),
        "top_k_mass": np.mean(top_k_per_q, axis=0).astype(np.float32),
        "top_k_mass_no_sink": np.mean(top_k_per_q_ns, axis=0).astype(np.float32),
        "sink_fraction": sink_frac.astype(np.float32),
        "prev_token_fraction": prev_frac.astype(np.float32),
    }


# --------------------------------------------------------------
# Classification cascade
# --------------------------------------------------------------


def classify_head(
    sink_fraction: float,
    prev_token_fraction: float,
    gini: float,
    top_k_mass_val: float,
) -> str:
    """Assign a taxonomy label per the DIR-001 T2 classification cascade.

    Returns one of {"sink", "previous-token", "uniform", "content",
    "other"}. Thresholds are the frozen constants at the top of this
    module; any downstream adjustment requires a DEC-NNN amendment per
    ``docs/extend-notes.md`` §1.9 Decision Log.
    """
    if sink_fraction > SINK_THRESHOLD:
        return "sink"
    if prev_token_fraction > PREV_TOKEN_THRESHOLD:
        return "previous-token"
    if gini < UNIFORM_GINI_THRESHOLD:
        return "uniform"
    if top_k_mass_val > CONTENT_TOP5_THRESHOLD:
        return "content"
    return "other"


# --------------------------------------------------------------
# Prediction 2 falsification
# --------------------------------------------------------------


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation + two-sided p-value.

    Falls back to a pure-numpy implementation via rank + Pearson on
    ranks + the Abramowitz-Stegun normal-approximation p-value when
    ``scipy`` is not importable in the session environment.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("_spearman_rho: shape mismatch")
    if a.shape[0] < 3:
        return float("nan"), float("nan")
    try:
        from scipy import stats as _scipy_stats
        result = _scipy_stats.spearmanr(a, b)
        return float(result.correlation), float(result.pvalue)
    except ImportError:
        ra = _rankdata(a)
        rb = _rankdata(b)
        ra_c = ra - np.mean(ra)
        rb_c = rb - np.mean(rb)
        num = float(np.sum(ra_c * rb_c))
        den = float(np.sqrt(np.sum(ra_c ** 2) * np.sum(rb_c ** 2)))
        if den == 0.0:
            return 0.0, 1.0
        rho = num / den
        # Normal approximation to the Spearman p-value at n >= 10;
        # good to ~0.01 absolute for n=5 per Zar 1999.
        n = a.shape[0]
        t = rho * math.sqrt(max(n - 2, 1) / max(1.0 - rho * rho, EPS))
        # Abramowitz-Stegun 7.1.26 erf approximation.
        p_half = 0.5 * (1.0 + _erf_approx(abs(t) / math.sqrt(2.0)))
        p_value = 2.0 * (1.0 - p_half)
        return float(rho), float(p_value)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank (ties resolved by midrank) numpy equivalent."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    # Handle ties: average the ranks across identical values.
    sorted_x = x[order]
    i = 0
    n = x.shape[0]
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


def _erf_approx(x: float) -> float:
    """Abramowitz-Stegun 7.1.26 erf approximation; max abs error ~1.5e-7."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return per-p reject/accept under Holm-Bonferroni at family-wise alpha.

    Preserves NaN entries as non-rejections. See Holm 1979 [47] for
    the monotone procedure; the step-down threshold is
    ``alpha / (m - k + 1)`` at the k-th smallest p across ``m`` tests.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = int(np.sum(np.isfinite(p_arr)))
    if m == 0:
        return [False] * len(p_values)
    order = np.argsort(np.where(np.isfinite(p_arr), p_arr, np.inf))
    rejected = [False] * len(p_values)
    for k, idx in enumerate(order):
        if not np.isfinite(p_arr[idx]):
            continue
        threshold = alpha / (m - k)
        if p_arr[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def prediction2_paired_tests(
    per_head: dict[str, np.ndarray],
    n_repeat: int,
) -> dict[str, Any]:
    """Run the DIR-001 T2 primary paired t-test on (r=1, r=n_repeat) Gini.

    Extended to entropy and top-5 mass per §1.4 Prediction 2 "concordant
    directionality across all three sparsity metrics"; the verdict is
    reported per-metric plus a joint concordance flag.

    Note: per DEC-026, this test is one of three gates and pools per-(layer,
    head) cells as i.i.d.; the formal §1.4 monotonicity test is
    ``prediction2_mixed_effects``.

    Parameters
    ----------
    per_head : dict[str, np.ndarray]
        Each metric array has shape ``(n_layer, n_repeat, n_head)``.
    n_repeat : int
        The per-checkpoint repeat count.

    Returns
    -------
    out : dict
        Keys: ``entropy``, ``gini``, ``top_k_mass`` each with per-metric
        sub-dict ``{t_statistic, p_value, effect_size, reject_flat,
        direction_agrees}``. Plus a joint ``concordant`` flag that is
        True only if all three rejects AND directions agree with the
        Prediction-2 claim (entropy decreases; Gini increases; top-k
        mass increases across repeats).
    """
    from analysis.common.stats import paired_t_test

    if n_repeat < 2:
        return {
            "n_repeat": int(n_repeat),
            "note": "n_repeat < 2; paired test undefined",
        }

    directions = {
        "entropy": -1.0,    # entropy decreases under P2
        "entropy_no_sink": -1.0,
        "gini": +1.0,       # Gini increases under P2
        "gini_no_sink": +1.0,
        "top_k_mass": +1.0,
        "top_k_mass_no_sink": +1.0,
    }

    out: dict[str, Any] = {"alpha": PREDICTION2_ALPHA}
    concordant_rejected = True
    for metric_key, expected_sign in directions.items():
        if metric_key not in per_head:
            out[metric_key] = {"present": False}
            continue
        panel = per_head[metric_key]  # (n_layer, n_repeat, n_head)
        r1 = panel[:, 0, :].reshape(-1)
        r_last = panel[:, n_repeat - 1, :].reshape(-1)
        t_stat, p_value = paired_t_test(r_last, r1)
        effect_size = float(np.nanmean(r_last - r1))
        direction_agrees = (effect_size * expected_sign) > 0
        rejects_flat = bool(np.isfinite(p_value) and p_value < PREDICTION2_ALPHA
                            and direction_agrees)
        if not rejects_flat:
            concordant_rejected = False
        out[metric_key] = {
            "present": True,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "effect_size": effect_size,
            "expected_sign": float(expected_sign),
            "direction_agrees": bool(direction_agrees),
            "reject_flat": rejects_flat,
        }

    out["concordant_rejection"] = bool(concordant_rejected)
    # Split concordance flags so callers can gate on the raw 3-metric
    # set without conflating with sink-corrected variants. A run that
    # rejects on raw entropy/gini/top_k_mass but fails one of the
    # sink-stripped variants should still pass the §1.4 Prediction-2
    # primary claim; the sink-corrected concordance is informational.
    raw_metrics = ["entropy", "gini", "top_k_mass"]
    sink_metrics = ["entropy_no_sink", "gini_no_sink", "top_k_mass_no_sink"]
    out["concordant_rejection_raw"] = all(
        out.get(m, {}).get("reject_flat", False) for m in raw_metrics
    )
    out["concordant_rejection_sink"] = all(
        out.get(m, {}).get("reject_flat", False)
        for m in sink_metrics
        if out.get(m, {}).get("present", False)
    )
    return out


def prediction2_per_layer_spearman(
    panel: np.ndarray,
    direction: float,
    alpha_family: float = MIXED_EFFECTS_ALPHA,
) -> dict[str, Any]:
    """Per-layer Spearman rho between repeat index and per-head mean metric.

    Holm-Bonferroni across the ``n_layer`` comparisons at ``alpha_family``;
    a layer rejects H0 if ``rho * direction <= -0.5`` (Cohen 1988 [49]
    "large effect" along the predicted direction) AND the Holm-adjusted
    p-value passes.

    Parameters
    ----------
    panel : np.ndarray
        Shape ``(n_layer, n_repeat, n_head)`` per-metric mean values.
    direction : float
        +1 for metrics expected to INCREASE with repeat index (gini,
        top_k_mass), -1 for metrics expected to DECREASE (entropy).
        Used to direction-orient the large-effect gate so that
        Spearman rho is judged against ``-0.5 * direction``.
    """
    n_layer, n_repeat, n_head = panel.shape
    per_layer: list[dict[str, Any]] = []
    p_values: list[float] = []
    for layer_idx in range(n_layer):
        # Average across heads at each repeat, then Spearman rho of
        # (repeat_index, mean_metric); expected negative for entropy
        # and positive for Gini / top-k. Returned rho in full range.
        mean_across_heads = panel[layer_idx].mean(axis=-1)  # (n_repeat,)
        repeats = np.arange(1, n_repeat + 1, dtype=np.float64)
        rho, p_val = _spearman_rho(repeats, mean_across_heads)
        per_layer.append({
            "layer": layer_idx,
            "rho": float(rho),
            "p_value": float(p_val),
            "large_effect": bool(
                np.isfinite(rho) and (rho * direction) <= SPEARMAN_LARGE_EFFECT
            ),
        })
        p_values.append(p_val)
    rejected = holm_bonferroni(p_values, alpha=alpha_family)
    for i, row in enumerate(per_layer):
        row["holm_reject"] = bool(rejected[i]) and row["large_effect"]
    n_reject = int(sum(1 for row in per_layer if row["holm_reject"]))
    return {
        "alpha_family": alpha_family,
        "per_layer": per_layer,
        "n_reject": n_reject,
        "n_layer": n_layer,
        "majority_reject": bool(n_reject > n_layer // 2),
    }


def prediction2_mixed_effects(
    panel: np.ndarray,
) -> dict[str, Any]:
    """Optional mixed-effects slope test per §1.4 Prediction 2 #1.

    Fits ``metric ~ repeat + (1 | layer) + (1 | head:layer)`` via
    ``statsmodels.regression.mixed_linear_model.MixedLM`` on the long-
    format panel. Returns ``statsmodels_unavailable`` when the package
    cannot be imported, which lets the overall verdict fall back to
    the paired t-test + per-layer Spearman combination.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return {"statsmodels_available": False}

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return {"statsmodels_available": False, "pandas_available": False}

    import pandas as pd

    n_layer, n_repeat, n_head = panel.shape
    records = []
    for layer in range(n_layer):
        for repeat in range(1, n_repeat + 1):
            for head in range(n_head):
                records.append({
                    "metric": float(panel[layer, repeat - 1, head]),
                    "repeat": float(repeat),
                    "layer": int(layer),
                    "head_in_layer": f"{layer}_{head}",
                })
    df = pd.DataFrame(records)
    try:
        model = smf.mixedlm(
            "metric ~ repeat",
            df,
            groups=df["layer"],
            re_formula="~1",
            vc_formula={"head_in_layer": "0 + C(head_in_layer)"},
        )
        fit = model.fit(reml=False, method="lbfgs", disp=False)
        slope = float(fit.fe_params.get("repeat", float("nan")))
        se = float(fit.bse.get("repeat", float("nan")))
        p_value = float(fit.pvalues.get("repeat", float("nan")))
        ci_low = slope - 1.96 * se if math.isfinite(se) else float("nan")
        ci_high = slope + 1.96 * se if math.isfinite(se) else float("nan")
        return {
            "statsmodels_available": True,
            "slope": slope,
            "se": se,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reject_flat": bool(
                math.isfinite(p_value) and p_value < MIXED_EFFECTS_ALPHA
                and math.isfinite(ci_low) and math.isfinite(ci_high)
                and (ci_low * ci_high > 0.0)
            ),
        }
    except Exception as exc:
        return {
            "statsmodels_available": True,
            "fit_error": str(exc),
            "reject_flat": False,
        }


# --------------------------------------------------------------
# RQ3: silhouette permutation-null + chi-squared independence test
# --------------------------------------------------------------


# Feature columns used to embed each (layer, head) cell at the canonical
# analysis repeat for k-means clustering. These are the per-head sparsity +
# position-pattern features that the existing classify_head cascade already
# consumes; reusing them keeps RQ3's "head specialisation" axis aligned with
# the taxonomy axis defined by Protocol C. Sink-corrected variants are
# included so a head that is purely sink-driven does not collapse onto a
# content-attending head once the BOS column is masked.
_RQ3_FEATURE_KEYS: tuple[str, ...] = (
    "entropy",
    "entropy_no_sink",
    "gini",
    "gini_no_sink",
    "top_k_mass",
    "top_k_mass_no_sink",
    "sink_fraction",
    "prev_token_fraction",
)


def _rq3_default_n_clusters(n_heads_total: int) -> int:
    """Project convention K = ``min(RQ3_MAX_CLUSTERS, ceil(log2(n_heads)))``.

    The log2 rule scales the cluster budget with cell count without
    overfitting small panels: 4 heads -> K=2, 12 heads -> K=4, 252 heads
    (21 layers x 12 heads) -> K=8. The hard cap prevents the silhouette
    geometry from collapsing on high-dimensional, low-sample-count
    splits when the panel grows beyond ~256 cells.
    """
    n = max(int(n_heads_total), 2)
    if n <= 2:
        return 2
    return min(RQ3_MAX_CLUSTERS, int(math.ceil(math.log2(n))))


def _rq3_build_feature_matrix(
    panel: dict[str, np.ndarray],
    repeats: list[int],
    feature_repeat: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stack per-(layer, head) feature vectors at the requested repeat.

    Parameters
    ----------
    panel : dict[str, np.ndarray]
        Each metric array has shape ``(n_layer, n_repeat, n_head)``.
    repeats : list[int]
        The 1-indexed repeat values present in the workspace, in the same
        order as the panel's repeat axis.
    feature_repeat : int
        The 1-indexed repeat to anchor on (canonical: 5). When the
        requested repeat is absent (e.g. a checkpoint trained with
        n_repeat < 5), the largest available repeat is used as a
        graceful fallback.

    Returns
    -------
    features : np.ndarray
        Shape ``(n_layer * n_head, n_features)`` standardised feature
        matrix. Rows are layer-major: row index ``l * n_head + h``.
    layer_index : np.ndarray
        Shape ``(n_layer * n_head,)`` integer layer index per row;
        used to build the L x K contingency table.
    repeat_used : int
        The actual repeat anchor index resolved (1-indexed).
    """
    if not repeats:
        raise ValueError("_rq3_build_feature_matrix: empty repeats list")
    if feature_repeat in repeats:
        repeat_slot = repeats.index(feature_repeat)
        repeat_used = feature_repeat
    else:
        # Fallback: largest available repeat (closest to canonical r=5).
        repeat_used = max(repeats)
        repeat_slot = repeats.index(repeat_used)

    columns: list[np.ndarray] = []
    for key in _RQ3_FEATURE_KEYS:
        if key not in panel:
            continue
        slab = panel[key][:, repeat_slot, :]  # (n_layer, n_head)
        columns.append(slab.reshape(-1).astype(np.float64))
    if not columns:
        raise ValueError(
            "_rq3_build_feature_matrix: panel missing all RQ3 feature keys"
        )
    n_layer, _n_repeat, n_head = panel[_RQ3_FEATURE_KEYS[0]].shape
    feats = np.column_stack(columns)  # (n_layer * n_head, n_features)

    # Z-score standardisation: features span heterogeneous scales (entropy
    # in bits, gini in [0, 1], fractions in [0, 1]); without rescaling the
    # entropy axis dominates Euclidean distances and the silhouette
    # collapses onto a single feature. Constant columns (zero variance)
    # are left at the centred-zero state to avoid divide-by-zero.
    mean = np.nanmean(feats, axis=0, keepdims=True)
    std = np.nanstd(feats, axis=0, keepdims=True)
    std_safe = np.where(std > EPS, std, 1.0)
    feats_std = (feats - mean) / std_safe
    # Replace any residual NaN (from a checkpoint where a feature column
    # is uniformly NaN, e.g. n_repeat=1 paired-t fallback) with zero so
    # k-means and silhouette do not raise.
    feats_std = np.nan_to_num(feats_std, nan=0.0, posinf=0.0, neginf=0.0)

    layer_index = np.repeat(np.arange(n_layer, dtype=np.int64), n_head)
    return feats_std, layer_index, int(repeat_used)


def _rq3_silhouette_permutation_null(
    features: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the silhouette empirical null by shuffling cluster labels.

    Standard permutation-test convention: the cluster-label vector is
    permuted (not the feature rows), so each draw assesses the
    silhouette under the H0 "labels are independent of the feature
    geometry". The features themselves are held fixed across draws,
    which preserves all geometric structure that is not relational with
    the cluster assignment.

    Returns
    -------
    null : np.ndarray
        Shape ``(n_permutations,)`` array of silhouette scores under
        the shuffled-label null. NaN entries (e.g. when a permutation
        happens to produce only one effective cluster) are propagated
        and treated as non-rejecting in the empirical p-value.
    """
    from sklearn.metrics import silhouette_score

    null = np.full(int(n_permutations), np.nan, dtype=np.float64)
    labels_copy = labels.copy()
    for i in range(int(n_permutations)):
        rng.shuffle(labels_copy)
        # Skip degenerate permutations that collapse to a single cluster.
        if np.unique(labels_copy).shape[0] < 2:
            continue
        try:
            null[i] = float(silhouette_score(features, labels_copy, metric="euclidean"))
        except ValueError:
            # silhouette undefined for n_clusters <= 1; leave NaN.
            continue
    return null


def rq3_silhouette_chi2_test(
    panel: dict[str, np.ndarray],
    repeats: list[int],
    n_clusters: int | None = None,
    n_permutations: int = RQ3_DEFAULT_PERMUTATIONS,
    seed: int = 2357,
    feature_repeat: int = RQ3_FEATURE_REPEAT,
) -> dict[str, Any]:
    """RQ3 silhouette permutation-null + chi-squared independence test.

    Tests H0 "head-cluster assignments are uniformly distributed across
    layers" (``docs/extend-notes.md`` §1.2 RQ3). Falsification requires
    BOTH gates: chi-squared independence p < 0.01 AND silhouette beats
    the 95-percentile of a 1000-permutation label-shuffle null (one-tailed
    p < 0.05). The silhouette permutation null replaces the prior fixed
    Kaufman-Rousseeuw 0.25 cutoff per §1.2 RQ3.

    Operates as a SECONDARY output alongside Prediction-2 paired-t
    triangulation; both are emitted in the per-checkpoint JSON.

    Parameters
    ----------
    panel : dict[str, np.ndarray]
        Per-metric ``(n_layer, n_repeat, n_head)`` arrays produced by
        ``_analyse_workspace``.
    repeats : list[int]
        1-indexed repeat values present in the panel (same order as the
        panel's repeat axis).
    n_clusters : int, optional
        Override for K. Defaults to the project convention
        ``min(RQ3_MAX_CLUSTERS, ceil(log2(n_layer * n_head)))``.
    n_permutations : int
        Number of label-shuffle draws for the silhouette null. Default
        ``RQ3_DEFAULT_PERMUTATIONS`` (1000) per the DEC-026 permutation-count
        family used by Prediction 2.
    seed : int
        Seeds both ``KMeans`` (deterministic init) and the permutation
        ``np.random.Generator``.
    feature_repeat : int
        1-indexed repeat to anchor the feature vectors on. Default 5
        (final repeat). Falls back to the largest available repeat when
        ``feature_repeat`` is absent.

    Returns
    -------
    out : dict
        Result dict with the schema documented in the module docstring
        / DIR-001 §1.6 row 4 spec; verdict is one of "supported",
        "preliminary", or "contradicted".
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return {
            "sklearn_available": False,
            "verdict": "preliminary",
            "rejects_h0": False,
            "note": "sklearn unavailable; RQ3 silhouette test could not run",
        }
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return {
            "scipy_available": False,
            "verdict": "preliminary",
            "rejects_h0": False,
            "note": "scipy unavailable; RQ3 chi-squared test could not run",
        }

    features, layer_index, repeat_used = _rq3_build_feature_matrix(
        panel, repeats, feature_repeat
    )
    n_cells = features.shape[0]
    if n_cells < 4:
        return {
            "sklearn_available": True,
            "scipy_available": True,
            "verdict": "preliminary",
            "rejects_h0": False,
            "note": f"insufficient cells for clustering (n={n_cells})",
        }

    K = int(n_clusters) if n_clusters is not None else _rq3_default_n_clusters(n_cells)
    K = max(2, min(K, n_cells - 1))

    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(features)
    n_effective = int(np.unique(labels).shape[0])

    # L x K contingency table for the chi-squared independence test.
    n_layer = int(layer_index.max()) + 1
    contingency = np.zeros((n_layer, K), dtype=np.int64)
    for row, lab in zip(layer_index, labels):
        contingency[int(row), int(lab)] += 1

    chi2_stat = float("nan")
    chi2_p = float("nan")
    chi2_dof = -1
    if n_effective >= 2:
        # Drop empty cluster columns; chi2_contingency rejects all-zero rows or
        # columns with a "internally computed table of expected frequencies has
        # a zero element" error.
        col_mask = contingency.sum(axis=0) > 0
        row_mask = contingency.sum(axis=1) > 0
        reduced = contingency[np.ix_(row_mask, col_mask)]
        if reduced.shape[0] >= 2 and reduced.shape[1] >= 2:
            try:
                chi2_stat_v, chi2_p_v, chi2_dof_v, _expected = chi2_contingency(reduced)
                chi2_stat = float(chi2_stat_v)
                chi2_p = float(chi2_p_v)
                chi2_dof = int(chi2_dof_v)
            except ValueError:
                # Degenerate table; leave as NaN -> chi2 gate fails.
                pass

    # Observed silhouette + permutation null.
    if n_effective >= 2:
        observed_silhouette = float(silhouette_score(features, labels, metric="euclidean"))
        rng = np.random.default_rng(seed)
        null = _rq3_silhouette_permutation_null(
            features, labels, n_permutations=n_permutations, rng=rng,
        )
        finite_null = null[np.isfinite(null)]
        if finite_null.size > 0:
            # Empirical one-tailed p-value: (1 + #(null >= observed)) / (M + 1).
            silhouette_p = float(
                (1.0 + np.sum(finite_null >= observed_silhouette))
                / (finite_null.size + 1.0)
            )
            null_q95 = float(np.quantile(finite_null, 0.95))
        else:
            silhouette_p = float("nan")
            null_q95 = float("nan")
    else:
        observed_silhouette = float("nan")
        silhouette_p = float("nan")
        null_q95 = float("nan")

    chi2_pass = bool(np.isfinite(chi2_p) and chi2_p < RQ3_CHI2_ALPHA)
    silhouette_pass = bool(
        np.isfinite(silhouette_p) and silhouette_p < RQ3_SILHOUETTE_ALPHA
    )
    rejects_h0 = chi2_pass and silhouette_pass
    if rejects_h0:
        verdict = "supported"
    elif chi2_pass or silhouette_pass:
        verdict = "preliminary"
    else:
        # Neither gate passes (or chi2_p is NaN / above 0.05): contradicted.
        verdict = "contradicted"

    return {
        "sklearn_available": True,
        "scipy_available": True,
        "n_clusters": int(K),
        "n_clusters_effective": n_effective,
        "n_permutations": int(n_permutations),
        "feature_repeat": int(repeat_used),
        "feature_keys": list(_RQ3_FEATURE_KEYS),
        "n_features": int(features.shape[1]),
        "n_cells": int(n_cells),
        "chi2": {
            "statistic": chi2_stat,
            "p_value": chi2_p,
            "dof": chi2_dof,
            "alpha": RQ3_CHI2_ALPHA,
            "rejects_uniform": chi2_pass,
        },
        "silhouette": {
            "observed": observed_silhouette,
            "null_p_value": silhouette_p,
            "null_q95": null_q95,
            "alpha": RQ3_SILHOUETTE_ALPHA,
            "beats_null_q95": silhouette_pass,
        },
        "verdict": verdict,
        "rejects_h0": bool(rejects_h0),
    }


# --------------------------------------------------------------
# Workspace I/O
# --------------------------------------------------------------


def _discover_layer_repeat_grid(workspace: str) -> dict[str, Any]:
    """Scan workspace for ``attn_weights_*_l*_r*.npy`` entries.

    Returns ``{"group": str, "n_layer": int, "n_repeat": int,
    "files": dict[(int, int), str]}`` so the analysis loop can
    validate completeness before loading.
    """
    if not os.path.isdir(workspace):
        raise FileNotFoundError(
            f"attention_taxonomy: workspace {workspace!r} does not exist"
        )
    files = os.listdir(workspace)
    prefix = "attn_weights_"
    suffix = ".npy"
    group: str | None = None
    grid: dict[tuple[int, int], str] = {}
    for name in files:
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        stem = name[len(prefix):-len(suffix)]
        # Expected: ``<group>_l<L>_r<R>`` where group is mid/flat.
        if "_l" not in stem or "_r" not in stem:
            continue
        try:
            group_part, rest = stem.split("_l", 1)
            layer_part, repeat_part = rest.split("_r", 1)
            layer = int(layer_part)
            repeat = int(repeat_part)
        except ValueError:
            continue
        if group is None:
            group = group_part
        if group_part != group:
            # Guard: multiple groups in one workspace (should not happen).
            continue
        grid[(layer, repeat)] = os.path.join(workspace, name)
    if not grid:
        return {"group": None, "n_layer": 0, "n_repeat": 0, "files": {}}
    layers = sorted({l for (l, _r) in grid})
    repeats = sorted({r for (_l, r) in grid})
    return {
        "group": group,
        "n_layer": len(layers),
        "n_repeat": len(repeats),
        "layers": layers,
        "repeats": repeats,
        "files": grid,
    }


def _load_meta_seq_length(workspace: str, fallback: int) -> int:
    """Read the collector's ``meta.json`` to recover seq_length, else fallback."""
    meta_path = os.path.join(workspace, "meta.json")
    if not os.path.isfile(meta_path):
        return fallback
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return fallback
    # Prefer per_key_shape on one attn_weights entry; its last axis is T_k.
    shapes = meta.get("per_key_shape", {})
    for key, shape in shapes.items():
        if key.startswith("attn_weights_") and isinstance(shape, list) and len(shape) == 3:
            return int(shape[2])
    # Fallback: the meta may record seq_length directly in a later extension.
    if "seq_length" in meta:
        return int(meta["seq_length"])
    return fallback


# --------------------------------------------------------------
# Capture stage (calls collector)
# --------------------------------------------------------------


def _run_capture_stage(args: argparse.Namespace) -> None:
    """Run a non-flash forward pass and persist ATTN_WEIGHTS to ``workspace``.

    The capture stage imports torch lazily so the analysis-only code path
    (running on pre-captured workspace files) does not require torch in
    the session environment.
    """
    import torch  # local import; not needed for analysis-only path

    from analysis.common.collector import ActivationCollector
    from analysis.common.data import iterate_owt2_val
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite

    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        config_mode=args.config_mode,
        module_path=args.module_path,
    )
    model.eval()

    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}.get(dtype)
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

    sites = [ActivationSite.ATTN_WEIGHTS]
    collector = ActivationCollector(
        model,
        sites,
        non_flash=True,   # ATTN_WEIGHTS requires the non-flash math path
        module_path=args.module_path,
    )
    os.makedirs(args.workspace, exist_ok=True)
    with collector:
        total = collector.run(data_iter, type_ctx=type_ctx, max_tokens=args.max_tokens)
    collector.save(
        args.workspace,
        meta_extra={
            "protocol": "C",
            "analysis": "attention_taxonomy",
            "seq_length": int(args.seq_length),
            "max_tokens": int(args.max_tokens),
            "checkpoint_dir": str(args.checkpoint),
            "checkpoint_filename": str(args.checkpoint_file),
            "n_tokens_captured": int(total),
        },
    )


# --------------------------------------------------------------
# Main analysis stage
# --------------------------------------------------------------


def _analyse_workspace(
    workspace: str,
    seq_length_fallback: int,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Load every ``attn_weights_*_l*_r*.npy`` and compute summary panel.

    Returns a dict with per-metric ``(n_layer, n_repeat, n_head)`` arrays
    plus the per-head classification cascade. The caller hands this to
    the Prediction-2 test functions and the JSON writer.
    """
    grid_info = _discover_layer_repeat_grid(workspace)
    if not grid_info["files"]:
        raise FileNotFoundError(
            f"attention_taxonomy: no attn_weights_* files in {workspace!r}; "
            f"run the capture stage first or point --workspace at the "
            f"collector output directory"
        )
    seq_length = _load_meta_seq_length(workspace, fallback=seq_length_fallback)
    layers: list[int] = grid_info["layers"]
    repeats: list[int] = grid_info["repeats"]
    n_layer = len(layers)
    n_repeat = len(repeats)

    per_head_accum: dict[str, np.ndarray] = {}
    classification_panel: list[list[list[str]]] = []

    n_head: int | None = None
    for layer_slot, layer in enumerate(layers):
        layer_row: list[list[str]] = []
        for repeat_slot, repeat in enumerate(repeats):
            file_path = grid_info["files"].get((layer, repeat))
            if file_path is None:
                layer_row.append([])
                continue
            att = np.load(file_path)
            if att.ndim != 3:
                raise ValueError(
                    f"attention_taxonomy: expected 3D attn capture at "
                    f"{file_path!r}; got shape {att.shape}"
                )
            summary = summarise_attention(att, seq_length=seq_length, top_k=top_k)
            if n_head is None:
                n_head = int(summary["entropy"].shape[0])
                for key in summary:
                    per_head_accum[key] = np.full(
                        (n_layer, n_repeat, n_head),
                        np.nan,
                        dtype=np.float32,
                    )
            for key, value in summary.items():
                per_head_accum[key][layer_slot, repeat_slot, :] = value

            # Classification per head at this (layer, repeat).
            labels: list[str] = []
            for head in range(n_head):
                labels.append(classify_head(
                    sink_fraction=float(summary["sink_fraction"][head]),
                    prev_token_fraction=float(summary["prev_token_fraction"][head]),
                    gini=float(summary["gini"][head]),
                    top_k_mass_val=float(summary["top_k_mass"][head]),
                ))
            layer_row.append(labels)
        classification_panel.append(layer_row)

    return {
        "group": grid_info["group"],
        "layers": layers,
        "repeats": repeats,
        "n_layer": n_layer,
        "n_repeat": n_repeat,
        "n_head": int(n_head or 0),
        "seq_length": seq_length,
        "panel": per_head_accum,
        "classification": classification_panel,
    }


def _build_results_payload(
    analysed: dict[str, Any],
    rq3_n_permutations: int = RQ3_DEFAULT_PERMUTATIONS,
    rq3_seed: int = 2357,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable results from the analysed panel."""
    panel = analysed["panel"]
    n_layer = analysed["n_layer"]
    n_repeat = analysed["n_repeat"]
    n_head = analysed["n_head"]

    # Validation gate: mean repeat-1 entropy across all heads must exceed 2 bits.
    r1_entropy_mean = float(np.nanmean(panel["entropy"][:, 0, :])) if n_repeat >= 1 else float("nan")
    validation_gate = {
        "threshold_bits": VALIDATION_ENTROPY_GATE,
        "observed_repeat1_mean_entropy_bits": r1_entropy_mean,
        "vacuous": bool(np.isfinite(r1_entropy_mean) and r1_entropy_mean < VALIDATION_ENTROPY_GATE),
    }

    paired = prediction2_paired_tests(panel, n_repeat)
    per_layer_spearman: dict[str, Any] = {}
    mixed_effects: dict[str, Any] = {}
    # Direction map: entropy is expected to DECREASE with repeat (slope < 0),
    # gini and top_k_mass are expected to INCREASE (slope > 0). Threading
    # direction through the per-layer Spearman gates the large-effect flag
    # against the predicted sign rather than always against rho < -0.5.
    direction_map = {"entropy": -1.0, "gini": +1.0, "top_k_mass": +1.0}
    for metric_key in ("entropy", "gini", "top_k_mass"):
        per_layer_spearman[metric_key] = prediction2_per_layer_spearman(
            panel[metric_key], direction=direction_map[metric_key]
        )
        mixed_effects[metric_key] = prediction2_mixed_effects(panel[metric_key])

    # Triple-gate AND-concordance verdict: a Prediction-2 claim is only
    # ratified when an aggregate test, a per-layer multiple-comparison
    # test, and a formal mixed-effects slope test all reject the
    # flat-null in concordant directions. DEC-026 placed the
    # mixed-effects model as the formal §1.4 test of monotonicity.
    #
    # Gate 1 -- paired-t concordant rejection across the six metrics
    #           (entropy / gini / top_k_mass + sink-corrected variants).
    #           Pools per-(layer, head) cells so it has high power but
    #           treats them as i.i.d.; DEC-026 demoted it from the SOLE
    #           primary gate to "one of three" because the
    #           exchangeability assumption is violated by per-layer
    #           clustering.
    # Gate 2 -- per-layer Spearman majority-reject on entropy OR Gini
    #           with Holm-Bonferroni across n_layer comparisons. Trades
    #           power for per-layer multiple-comparison rigour.
    # Gate 3 -- mixed-effects slope on `repeat` rejects flat-null on
    #           entropy OR Gini (random intercepts on layer + nested
    #           head-in-layer variance component). The formal §1.4 test
    #           of Prediction 2 monotonicity. Falls open (cannot reject)
    #           when statsmodels is unavailable; in that environment
    #           the verdict caps at "inconclusive" until the dependency
    #           is installed and the run is repeated.
    gate1_paired_concordant = bool(paired.get("concordant_rejection_raw", False))
    gate2_spearman_majority = bool(
        per_layer_spearman.get("entropy", {}).get("majority_reject", False)
        or per_layer_spearman.get("gini", {}).get("majority_reject", False)
        or per_layer_spearman.get("top_k_mass", {}).get("majority_reject", False)
    )
    statsmodels_available = bool(
        mixed_effects.get("entropy", {}).get("statsmodels_available", False)
        or mixed_effects.get("gini", {}).get("statsmodels_available", False)
    )
    gate3_mixed_effects = bool(
        mixed_effects.get("entropy", {}).get("reject_flat", False)
        or mixed_effects.get("gini", {}).get("reject_flat", False)
    )

    # A "refuted" verdict requires that the formal mixed-effects fits
    # actually ran successfully -- a fit_error in either metric means
    # gate3 is uninformatively False (fit never converged), which must
    # not be conflated with a genuine null finding. When any fit_error
    # is present, the verdict caps at "inconclusive".
    mixed_effects_no_fit_error = not (
        bool(mixed_effects.get("entropy", {}).get("fit_error"))
        or bool(mixed_effects.get("gini", {}).get("fit_error"))
    )

    if (
        gate1_paired_concordant
        and gate2_spearman_majority
        and gate3_mixed_effects
    ):
        verdict = "supported"
    elif (
        statsmodels_available
        and mixed_effects_no_fit_error
        and not gate2_spearman_majority
        and not gate3_mixed_effects
    ):
        # All formal gates fail, statsmodels was available, AND no
        # mixed-effects fit error fired -> genuine refutation.
        # Single-metric Gini-paired failure no longer overrides
        # multi-metric concordance (the previous
        # ``not paired["gini"]["reject_flat"]`` shortcut is
        # deprecated; it produced "refuted" verdicts whenever Gini's
        # paired-t alone failed even when entropy + top_k_mass agreed).
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    verdict_meta = {
        "gate1_paired_concordant_rejection": gate1_paired_concordant,
        # Informational: the legacy all-6-metric concordance flag
        # (raw + sink-corrected). Retained for backward comparison;
        # gate1 now uses raw-only concordance.
        "gate1_paired_concordant_rejection_all6": bool(
            paired.get("concordant_rejection", False)
        ),
        "gate1_paired_concordant_rejection_sink": bool(
            paired.get("concordant_rejection_sink", False)
        ),
        "gate2_spearman_majority_reject": gate2_spearman_majority,
        "gate3_mixed_effects_reject_flat": gate3_mixed_effects,
        "statsmodels_available": statsmodels_available,
        "mixed_effects_fit_error": not mixed_effects_no_fit_error,
        "validation_gate_vacuous": bool(validation_gate.get("vacuous", False)),
    }

    # Per-(layer, repeat, head) results laid out for easy JSON pick-up.
    per_head_records = []
    panel_gini = panel["gini"]
    panel_entropy = panel["entropy"]
    panel_top_k = panel["top_k_mass"]
    panel_sink = panel["sink_fraction"]
    panel_prev = panel["prev_token_fraction"]
    for layer_slot, layer in enumerate(analysed["layers"]):
        for repeat_slot, repeat in enumerate(analysed["repeats"]):
            if repeat_slot >= len(analysed["classification"][layer_slot]):
                continue
            labels = analysed["classification"][layer_slot][repeat_slot]
            if not labels:
                continue
            for head in range(len(labels)):
                per_head_records.append({
                    "layer": int(layer),
                    "repeat": int(repeat),
                    "head": int(head),
                    "entropy": float(panel_entropy[layer_slot, repeat_slot, head]),
                    "gini": float(panel_gini[layer_slot, repeat_slot, head]),
                    "top_k_mass": float(panel_top_k[layer_slot, repeat_slot, head]),
                    "sink_fraction": float(panel_sink[layer_slot, repeat_slot, head]),
                    "prev_token_fraction": float(panel_prev[layer_slot, repeat_slot, head]),
                    "taxonomy": labels[head],
                })

    # Per-(layer, repeat) taxonomy histogram for the heatmap.
    histogram = []
    for layer_slot, layer in enumerate(analysed["layers"]):
        for repeat_slot, repeat in enumerate(analysed["repeats"]):
            labels = analysed["classification"][layer_slot][repeat_slot] if repeat_slot < len(analysed["classification"][layer_slot]) else []
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            histogram.append({
                "layer": int(layer),
                "repeat": int(repeat),
                "counts": counts,
            })

    # RQ3 secondary output: silhouette permutation null + chi-squared
    # independence test on the (layer, head) cluster panel at the final
    # repeat. Runs alongside Prediction-2 paired-t triangulation per the
    # §1.6 row 4 reconciliation; uses a fully separate result schema.
    rq3_silhouette_chi2 = rq3_silhouette_chi2_test(
        panel,
        repeats=analysed["repeats"],
        n_permutations=rq3_n_permutations,
        seed=rq3_seed,
    )

    payload = {
        "group": analysed["group"],
        "layers": analysed["layers"],
        "repeats": analysed["repeats"],
        "n_layer": n_layer,
        "n_repeat": n_repeat,
        "n_head": n_head,
        "seq_length": analysed["seq_length"],
        "validation_gate": validation_gate,
        "prediction2": {
            "paired_t_tests": paired,
            "per_layer_spearman": per_layer_spearman,
            "mixed_effects": mixed_effects,
            "verdict": verdict,
            "verdict_meta": verdict_meta,
        },
        "rq3_silhouette_chi2": rq3_silhouette_chi2,
        "per_head": per_head_records,
        "per_layer_repeat_histogram": histogram,
        "constants": {
            "sink_threshold": SINK_THRESHOLD,
            "prev_token_threshold": PREV_TOKEN_THRESHOLD,
            "uniform_gini_threshold": UNIFORM_GINI_THRESHOLD,
            "content_top5_threshold": CONTENT_TOP5_THRESHOLD,
            "top_k": TOP_K,
            "prediction2_alpha": PREDICTION2_ALPHA,
            "validation_entropy_gate_bits": VALIDATION_ENTROPY_GATE,
            "spearman_large_effect_threshold": SPEARMAN_LARGE_EFFECT,
            "mixed_effects_alpha": MIXED_EFFECTS_ALPHA,
            "rq3_chi2_alpha": RQ3_CHI2_ALPHA,
            "rq3_silhouette_alpha": RQ3_SILHOUETTE_ALPHA,
        },
    }
    return payload


# --------------------------------------------------------------
# Plotting
# --------------------------------------------------------------


_TAXONOMY_COLOURS = {
    "sink": "#d95f02",
    "previous-token": "#7570b3",
    "uniform": "#a6cee3",
    "content": "#1b9e77",
    "other": "#999999",
}


def _render_heatmap(payload: dict[str, Any], output_dir: str) -> str | None:
    """Render (n_layer, n_repeat) heatmap coloured by dominant taxonomy cell.

    Returns the absolute output path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    layers = payload["layers"]
    repeats = payload["repeats"]
    n_layer = payload["n_layer"]
    n_repeat = payload["n_repeat"]
    if n_layer == 0 or n_repeat == 0:
        return None

    # Collate per-(layer, repeat) dominant label.
    grid_colours = np.full((n_layer, n_repeat, 3), 0.6, dtype=np.float32)
    grid_labels = [[""] * n_repeat for _ in range(n_layer)]
    label_to_int = {label: i for i, label in enumerate(_TAXONOMY_COLOURS.keys())}
    int_to_colour = [_TAXONOMY_COLOURS[label] for label in _TAXONOMY_COLOURS]
    dominant_int = np.full((n_layer, n_repeat), -1, dtype=np.int32)
    for row in payload["per_layer_repeat_histogram"]:
        l_idx = layers.index(row["layer"])
        r_idx = repeats.index(row["repeat"])
        counts = row["counts"]
        if not counts:
            continue
        dominant_label = max(counts, key=counts.get)
        grid_labels[l_idx][r_idx] = dominant_label
        dominant_int[l_idx, r_idx] = label_to_int.get(dominant_label, -1)

    fig, ax = plt.subplots(figsize=(max(4, n_repeat * 0.8), max(4, n_layer * 0.3)))
    # Render as a coloured grid via pcolor-like mesh.
    for l_idx in range(n_layer):
        for r_idx in range(n_repeat):
            label = grid_labels[l_idx][r_idx]
            colour = _TAXONOMY_COLOURS.get(label, "#ffffff")
            ax.fill_between(
                [r_idx - 0.5, r_idx + 0.5],
                l_idx - 0.5,
                l_idx + 0.5,
                color=colour,
                edgecolor="white",
                linewidth=0.5,
            )
    ax.set_xticks(range(n_repeat))
    ax.set_xticklabels([f"r{r}" for r in repeats])
    ax.set_yticks(range(n_layer))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Repeat index")
    ax.set_ylabel("Mid-layer index")
    ax.set_title("Dominant attention-head taxonomy per (layer, repeat)")
    ax.set_xlim(-0.5, n_repeat - 0.5)
    ax.set_ylim(n_layer - 0.5, -0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_TAXONOMY_COLOURS[label])
        for label in _TAXONOMY_COLOURS
    ]
    ax.legend(handles, list(_TAXONOMY_COLOURS.keys()),
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    path = os.path.join(output_dir, "attention_taxonomy_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_sparsity_trajectory(payload: dict[str, Any], output_dir: str, panel: dict[str, np.ndarray]) -> str | None:
    """Per-layer mean Gini / entropy / top-k vs repeat curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n_layer = payload["n_layer"]
    n_repeat = payload["n_repeat"]
    if n_layer == 0 or n_repeat == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    repeats_range = np.arange(1, n_repeat + 1)
    for ax, metric_key, title in zip(
        axes,
        ("entropy", "gini", "top_k_mass"),
        ("Entropy (bits)", "Gini coefficient", f"Top-{TOP_K} mass"),
    ):
        mat = panel[metric_key].mean(axis=-1)  # (n_layer, n_repeat)
        for layer_idx in range(n_layer):
            ax.plot(
                repeats_range,
                mat[layer_idx],
                color=plt.cm.viridis(layer_idx / max(n_layer - 1, 1)),
                alpha=0.6,
                linewidth=0.8,
            )
        mean_across_layers = np.nanmean(mat, axis=0)
        ax.plot(repeats_range, mean_across_layers, color="black", linewidth=2.0, label="layer-mean")
        ax.set_xlabel("Repeat index")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="best")
    path = os.path.join(output_dir, "attention_sparsity_trajectory.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------
# CLI + main
# --------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """CLI for Protocol C.

    Required: ``--checkpoint`` (ckpt dir with summary.json), ``--workspace``
    (/scratch path for attn_weights caches), ``--output-dir`` (for JSON +
    PNG). Optional: ``--checkpoint-file``, ``--seq-length``,
    ``--batch-size``, ``--max-tokens``, ``--device``, ``--config-mode``,
    ``--module-path``, ``--skip-capture``, ``--seed``.
    """
    parser = argparse.ArgumentParser(
        description="Protocol C -- attention-head taxonomy + sparsity-across-repeat test"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Checkpoint directory containing summary.json + ckpt file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default ckpt.pt)",
    )
    parser.add_argument(
        "--workspace", type=str, required=True,
        help="Workspace directory for attn_weights_* .npy files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for attention_taxonomy_*.json / *.png",
    )
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Total tokens for the capture stage (default 2048)")
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip the non-flash capture stage; assume workspace is populated",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"k for the top-k mass metric (default {TOP_K})",
    )
    parser.add_argument(
        "--rq3-n-permutations", type=int, default=RQ3_DEFAULT_PERMUTATIONS,
        help=(
            "RQ3 silhouette permutation-null draw count "
            f"(default {RQ3_DEFAULT_PERMUTATIONS}; DEC-026 family)"
        ),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.workspace, exist_ok=True)

    # Run the capture stage unless explicitly skipped. The workspace may
    # already contain attn_weights_* if Protocol C was run previously;
    # in that case --skip-capture shortcuts directly to analysis.
    grid_info = _discover_layer_repeat_grid(args.workspace)
    has_cache = bool(grid_info["files"])
    if args.skip_capture and not has_cache:
        raise FileNotFoundError(
            "attention_taxonomy: --skip-capture requested but no "
            f"attn_weights_* files present in {args.workspace!r}"
        )
    if not args.skip_capture and not has_cache:
        _run_capture_stage(args)

    analysed = _analyse_workspace(
        workspace=args.workspace,
        seq_length_fallback=args.seq_length,
        top_k=args.top_k,
    )
    payload = _build_results_payload(
        analysed,
        rq3_n_permutations=args.rq3_n_permutations,
        rq3_seed=args.seed,
    )

    results_path = os.path.join(args.output_dir, "attention_taxonomy_results.json")
    with open(results_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    _render_heatmap(payload, args.output_dir)
    _render_sparsity_trajectory(payload, args.output_dir, analysed["panel"])


if __name__ == "__main__":
    main()
