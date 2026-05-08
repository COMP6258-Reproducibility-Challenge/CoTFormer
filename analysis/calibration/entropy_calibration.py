"""Protocol D-calibration entry point.

Scope
-----
Runs the three-tier substrate ladder (GPT-2-large -> ADM C5 -> 24L
standard, contingency only) across the two-condition synthetic
sequence set (Condition A sparse induction-head, Condition B
broad-integration multi-target) and computes the four-gate
validation ladder (baseline >= 0.5, sink-correction < 1 bit,
Tier 1 / Tier 2 triangulation, Spearman <= -0.5 AND classifier
>= 0.8).

Outputs
-------
- ``metrics.csv``       -- per-sequence entropy, H_no_sink, target
                            accuracy, attention-sink mass.
- ``spearman.json``     -- Spearman rho, 95 per cent bootstrap CI,
                            per-tier breakdown.
- ``classifier.json``   -- linear classifier accuracy on
                            (entropy, target_accuracy); the decision
                            boundary is reused for DV-2 at test time
                            when the PASS verdict holds.
- ``figure.png``        -- entropy-vs-accuracy scatter with decision
                            boundary overlay.

Falsifiability relevance
------------------------
If the Spearman gate fails (rho > -0.3), the monotone interpretation
is refuted on the calibrated substrate; DV-2 is replaced by the
TOP-K-MASS fallback (fraction of attention mass at repeat 5 landing
on context tokens surviving to the same repeat or deeper). See
`docs/extend-notes.md` §1.3 "D-cal four-gate validation ladder"
verdict table for the AMBIGUOUS and FAIL paths.

Ontological purpose
-------------------
Empirically grounds an otherwise unprincipled interpretation of
attention entropy: the published literature (Meister et al. [43],
Zhai et al. [44], Xiao et al. [26], Gu et al. [46]) does not support
a bidirectional monotone mapping a priori; the calibration is the
methodological prerequisite for any subsequent claim about DV-2.

Two execution modes
-------------------
1. Forward-pass mode (``--tier 1`` or ``--tier 2``, no
   ``--analysis-only``): loads the substrate, generates the two
   synthetic conditions, runs forward passes, and writes
   ``<out>/metrics.csv`` plus a per-tier ``meta.json``.
2. Analysis-only mode (``--analysis-only``): reads
   ``<out>/tier1/metrics.csv`` and ``<out>/tier2/metrics.csv``,
   applies the four-gate verdict ladder, and writes
   ``<out>/aggregate/{spearman,classifier,verdict}.json`` plus
   ``<out>/aggregate/figure.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np


# --------------------------------------------------------------
# Frozen pre-registered thresholds (preregister.md Section 1)
# --------------------------------------------------------------

GATE1_BASELINE_ACC = 0.5         # Tier-1 Condition-A target accuracy floor
GATE2_SINK_BITS = 1.0            # |H_full - H_no_sink| ceiling (bits)
GATE4_SPEARMAN_PASS = -0.5       # Spearman rho PASS ceiling
GATE4_SPEARMAN_FAIL = -0.3       # Spearman rho FAIL floor
GATE4_CLASSIFIER_PASS = 0.8      # 2-D classifier accuracy floor
GATE4_ENTROPY_ALONE_FAIL = 0.85  # 1-D (entropy-only) classifier ceiling

# Synthetic-substrate constants (mirror the protocol spec).
SEQ_LENGTH = 256
INF_VOCAB_SIZE = 100             # Condition B reserved subvocabulary
DEFAULT_VOCAB = 50257            # GPT-2-large == ADM C5 token space
EPS = 1e-12


# --------------------------------------------------------------
# Sink-aware Shannon entropy primitives
# --------------------------------------------------------------


def _shannon_entropy_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis (mirrors ``analysis.attention_taxonomy.shannon_entropy``).

    Caller is responsible for normalisation; the EPS clip handles the
    ``0 log 0 = 0`` limit.
    """
    q = np.clip(p, EPS, 1.0)
    return -np.sum(q * np.log2(q), axis=-1)


def _apply_sink_correction(p: np.ndarray) -> np.ndarray:
    """Zero position-0 mass, renormalise per row (pure-sink rows stay near-zero).

    Mirrors ``analysis.attention_taxonomy.apply_sink_correction``;
    see ``docs/extend-notes.md`` §1.3 sink-correction note.
    """
    corrected = p.copy()
    corrected[..., 0] = 0.0
    row_sums = np.sum(corrected, axis=-1, keepdims=True)
    safe = np.where(row_sums > EPS, row_sums, 1.0)
    return corrected / safe


# --------------------------------------------------------------
# Attention reduction at the query position
# --------------------------------------------------------------


def _reduce_attention_at_query(
    att: np.ndarray,
    target_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-sequence entropy / sink-mass / target-accuracy summaries.

    ``att`` is ``(n_seq, T_k)``: the attention distribution at the
    query position, averaged across (layer, head) of the upper half
    of the substrate's block stack -- this layer/head average is the
    standard "model-wide attention" reduction for the calibration
    ladder (per-head dispersion is Protocol C's job).
    ``target_positions`` is ``(n_seq,)`` (Condition A: 1 target) or
    ``(n_seq, K)`` (Condition B: K targets, mass summed).
    """
    if att.ndim != 2:
        raise ValueError(
            f"_reduce_attention_at_query: expected (n_seq, T_k); got {att.shape}"
        )

    entropy_full = _shannon_entropy_bits(att)
    sink_mass = att[:, 0].astype(np.float64)
    entropy_no_sink = _shannon_entropy_bits(_apply_sink_correction(att))

    n_seq = att.shape[0]
    idx = target_positions.astype(np.int64)
    if target_positions.ndim == 1:
        target_accuracy = att[np.arange(n_seq), idx].astype(np.float64)
    elif target_positions.ndim == 2:
        target_accuracy = np.take_along_axis(att, idx, axis=1).sum(axis=1).astype(np.float64)
    else:
        raise ValueError(
            f"_reduce_attention_at_query: target_positions ndim {target_positions.ndim}"
        )

    return {
        "entropy_full": entropy_full.astype(np.float64),
        "entropy_no_sink": entropy_no_sink.astype(np.float64),
        "sink_mass": sink_mass,
        "target_accuracy": target_accuracy,
    }


# --------------------------------------------------------------
# Tier 1 substrate: HuggingFace GPT-2-large
# --------------------------------------------------------------


def _forward_attention_gpt2(
    hf_model_id: str,
    tokens: np.ndarray,
    query_positions: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Tier-1 forward pass; returns ``(n_seq, T_k)`` attention at the query
    position, averaged across upper-half layers and heads. Uses
    ``cache_dir=$HF_HOME`` for offline-safe execution on compute nodes.
    """
    import torch
    from transformers import AutoModelForCausalLM  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        cache_dir=os.environ.get("HF_HOME"),
        attn_implementation="eager",  # eager backend exposes attention weights
    )
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    n_layer = int(model.config.n_layer)
    upper_layers = list(range(n_layer // 2, n_layer))

    n_seq, T = tokens.shape[0], tokens.shape[1]
    out = np.zeros((n_seq, T), dtype=np.float32)
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    qpos_t = np.asarray(query_positions, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            stop = min(start + batch_size, n_seq)
            batch = tokens_t[start:stop].to(device)
            outputs = model(batch, output_attentions=True, use_cache=False, return_dict=True)
            # outputs.attentions: tuple of (B, n_head, T, T) per layer.
            stacked = torch.stack(
                [outputs.attentions[layer] for layer in upper_layers], dim=0
            )
            mean_att = stacked.mean(dim=(0, 2))  # (B, T, T) -- mean over (layer, head)
            for offset in range(stop - start):
                qp = int(qpos_t[start + offset])
                out[start + offset] = mean_att[offset, qp, :].detach().to("cpu").float().numpy()
            del outputs, stacked, mean_att, batch

    del model
    return out


# --------------------------------------------------------------
# Tier 2 substrate: ADM C5 via the project's standard loader
# --------------------------------------------------------------


def _forward_attention_cotformer(
    ckpt_dir: str,
    ckpt_file: str,
    tokens: np.ndarray,
    query_positions: np.ndarray,
    device: str,
    batch_size: int,
    module_path: str,
) -> np.ndarray:
    """Tier-2 / Tier-3 forward pass via ``ActivationCollector`` (ATTN_WEIGHTS
    + ``non_flash=True``). Averages over upper-half (layer, repeat, head) and
    returns ``(n_seq, T_k)`` rows at the query position. The collector is
    re-entered per batch so buffers do not accumulate across n_seq (L4 RAM).
    """
    import torch

    from analysis.common.collector import ActivationCollector
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite

    # job-gpu.sh sometimes passes a full file path as ckpt_dir; tolerate it.
    if os.path.isfile(ckpt_dir):
        ckpt_dir, ckpt_file = os.path.split(ckpt_dir)

    model, _config = load_model_from_checkpoint(
        checkpoint_dir=ckpt_dir,
        checkpoint_file=ckpt_file,
        device=device,
        config_mode="raw",
        module_path=module_path,
    )
    model.eval()

    sites = [ActivationSite.ATTN_WEIGHTS]
    n_seq, T = tokens.shape[0], tokens.shape[1]
    out = np.zeros((n_seq, T), dtype=np.float32)
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    qpos_t = np.asarray(query_positions, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            stop = min(start + batch_size, n_seq)
            batch = tokens_t[start:stop].to(device)
            collector = ActivationCollector(
                model, sites, non_flash=True, module_path=module_path,
            )
            with collector:
                model(batch)
            # Collector buffers: keys ``attn_weights_<group>_l<L>_r<R>`` with
            # chunks of (B*T_q, n_head, T_k). Row i = query (i % T_q) of
            # batch element (i // T_q); see ActivationCollector docstring.
            stacked_rows: list[np.ndarray] = []
            for buf_key, chunks in collector._buffers.items():
                if not buf_key.startswith("attn_weights_") or not chunks:
                    continue
                stacked_rows.append(np.concatenate(chunks, axis=0))
            if not stacked_rows:
                raise RuntimeError(
                    "_forward_attention_cotformer: collector produced no "
                    "attn_weights buffers; check module_path / non_flash"
                )
            # Restrict to upper half of layers (analogous to GPT-2 path).
            stacked_rows = stacked_rows[len(stacked_rows) // 2:]

            B = stop - start
            T_local = batch.shape[1]
            mean_per_layer = [
                arr.reshape(B, T_local, arr.shape[1], arr.shape[2]).mean(axis=2)
                for arr in stacked_rows
            ]
            mean_lr = np.stack(mean_per_layer, axis=0).mean(axis=0)  # (B, T_q, T_k)
            for offset in range(B):
                qp = int(qpos_t[start + offset])
                row = mean_lr[offset, qp, :]
                # T_k may be < T (no padding mask); right-pad with zeros.
                if row.shape[0] < T:
                    out[start + offset, : row.shape[0]] = row
                else:
                    out[start + offset] = row[:T]
            del collector, stacked_rows, mean_per_layer

    del model
    return out


# --------------------------------------------------------------
# Forward-pass mode (Tier 1 / Tier 2)
# --------------------------------------------------------------


def _generate_substrate_tokens(
    n_per_condition: int,
    seed: int,
    vocab_size: int = DEFAULT_VOCAB,
    seq_length: int = SEQ_LENGTH,
) -> dict[str, Any]:
    """Generate Condition A and Condition B tokens with ground truth."""
    from analysis.calibration.synth_sequences import (
        generate_condition_A,
        generate_condition_B,
    )

    tok_a, qpos_a, tgt_a = generate_condition_A(
        n=n_per_condition, L=seq_length, vocab_size=vocab_size, seed=seed,
    )
    tok_b, qpos_b, tgt_b = generate_condition_B(
        n=n_per_condition, L=seq_length, vocab_size=vocab_size,
        inf_vocab=INF_VOCAB_SIZE, seed=seed + 1,
    )
    return {
        "A": {
            "tokens": tok_a.numpy().astype(np.int64),
            "query_positions": qpos_a.numpy().astype(np.int64),
            "target_positions": tgt_a.numpy().astype(np.int64),
        },
        "B": {
            "tokens": tok_b.numpy().astype(np.int64),
            "query_positions": qpos_b.numpy().astype(np.int64),
            "target_positions": tgt_b.numpy().astype(np.int64),
        },
    }


def _write_metrics_csv(
    out_dir: str,
    per_seq_a: dict[str, np.ndarray],
    per_seq_b: dict[str, np.ndarray],
) -> str:
    """Write per-sequence metrics.csv with columns:
    ``seq_id, condition, entropy_full, entropy_no_sink, sink_mass, target_accuracy``.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "seq_id", "condition", "entropy_full", "entropy_no_sink",
            "sink_mass", "target_accuracy",
        ])
        seq_id = 0
        for cond_label, per_seq in (("A", per_seq_a), ("B", per_seq_b)):
            for i in range(per_seq["entropy_full"].shape[0]):
                writer.writerow([
                    seq_id, cond_label,
                    float(per_seq["entropy_full"][i]),
                    float(per_seq["entropy_no_sink"][i]),
                    float(per_seq["sink_mass"][i]),
                    float(per_seq["target_accuracy"][i]),
                ])
                seq_id += 1
    return path


def _run_forward_pass(args: argparse.Namespace) -> None:
    """Forward-pass mode entry point; dispatches by ``--tier``."""
    os.makedirs(args.out, exist_ok=True)
    substrate = _generate_substrate_tokens(
        n_per_condition=args.n_per_condition, seed=args.seed,
    )

    if args.tier == 1:
        ckpt_arg = args.ckpt or "gpt2-large"

        def _fwd(cond: str) -> np.ndarray:
            return _forward_attention_gpt2(
                hf_model_id=ckpt_arg,
                tokens=substrate[cond]["tokens"],
                query_positions=substrate[cond]["query_positions"],
                device=args.device,
                batch_size=args.batch_size,
            )

        substrate_name = ckpt_arg
    elif args.tier in (2, 3):
        if not args.ckpt:
            raise ValueError(f"_run_forward_pass: --ckpt is required for tier {args.tier}")

        def _fwd(cond: str) -> np.ndarray:
            return _forward_attention_cotformer(
                ckpt_dir=args.ckpt,
                ckpt_file="ckpt.pt",
                tokens=substrate[cond]["tokens"],
                query_positions=substrate[cond]["query_positions"],
                device=args.device,
                batch_size=args.batch_size,
                module_path=args.module_path,
            )

        substrate_name = args.ckpt
    else:
        raise ValueError(f"_run_forward_pass: unknown tier {args.tier}")

    att_a, att_b = _fwd("A"), _fwd("B")

    per_seq_a = _reduce_attention_at_query(att_a, substrate["A"]["target_positions"])
    per_seq_b = _reduce_attention_at_query(att_b, substrate["B"]["target_positions"])
    csv_path = _write_metrics_csv(args.out, per_seq_a, per_seq_b)

    meta = {
        "tier": int(args.tier),
        "substrate": substrate_name,
        "module_path": args.module_path,
        "n_per_condition": int(args.n_per_condition),
        "seed": int(args.seed),
        "seq_length": SEQ_LENGTH,
        "vocab_size": DEFAULT_VOCAB,
        "inf_vocab_size": INF_VOCAB_SIZE,
        "metrics_csv": csv_path,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


# --------------------------------------------------------------
# Analysis-only mode (4-gate verdict)
# --------------------------------------------------------------


def _read_metrics_csv(path: str) -> dict[str, np.ndarray]:
    """Read a metrics.csv into per-column numpy arrays."""
    cond: list[str] = []
    e_full: list[float] = []
    e_ns: list[float] = []
    s_mass: list[float] = []
    t_acc: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cond.append(row["condition"])
            e_full.append(float(row["entropy_full"]))
            e_ns.append(float(row["entropy_no_sink"]))
            s_mass.append(float(row["sink_mass"]))
            t_acc.append(float(row["target_accuracy"]))
    return {
        "condition": np.asarray(cond),
        "entropy_full": np.asarray(e_full, dtype=np.float64),
        "entropy_no_sink": np.asarray(e_ns, dtype=np.float64),
        "sink_mass": np.asarray(s_mass, dtype=np.float64),
        "target_accuracy": np.asarray(t_acc, dtype=np.float64),
    }


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho via scipy; pure-numpy fallback when scipy missing."""
    try:
        from scipy import stats as _scipy_stats
        rho = float(_scipy_stats.spearmanr(a, b).correlation)
        return 0.0 if math.isnan(rho) else rho
    except ImportError:
        ra = _rankdata(a)
        rb = _rankdata(b)
        ra_c = ra - np.mean(ra)
        rb_c = rb - np.mean(rb)
        num = float(np.sum(ra_c * rb_c))
        den = float(math.sqrt(float(np.sum(ra_c ** 2) * np.sum(rb_c ** 2))))
        return num / den if den > 0.0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank ranking (mirrors taxonomy._rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    sorted_x = x[order]
    i, n = 0, x.shape[0]
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


def _fit_logistic_classifier(
    X: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Logistic regression accuracy + model info; sklearn with numpy GD fallback
    (analysis-only stage may run without sklearn -- portability requirement).
    """
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        acc = float(np.mean(clf.predict(X) == y))
        return acc, {
            "backend": "sklearn",
            "coef": [float(c) for c in clf.coef_.ravel()],
            "intercept": float(clf.intercept_[0]),
        }
    except ImportError:
        pass

    # Numpy fallback: standardised features + gradient descent.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + EPS
    Xn = (X - X_mean) / X_std
    n, d = Xn.shape
    Xb = np.concatenate([Xn, np.ones((n, 1))], axis=1)
    w = np.zeros(d + 1)
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30.0, 30.0)))
        w -= 0.5 * (Xb.T @ (p - y) / n)
    acc = float(np.mean((Xb @ w >= 0.0).astype(np.int64) == y))
    return acc, {
        "backend": "numpy-fallback",
        "coef_standardised": [float(c) for c in w[:-1]],
        "intercept_standardised": float(w[-1]),
        "feature_mean": [float(c) for c in X_mean],
        "feature_std": [float(c) for c in X_std],
    }


def _per_tier_verdict(
    metrics: dict[str, np.ndarray],
    operative_metric: str,
) -> dict[str, Any]:
    """Per-tier PASS / AMBIGUOUS / FAIL from Gate 4 (Spearman + 2-D classifier
    on pooled A union B). ``operative_metric`` selects ``entropy_full`` or
    ``entropy_no_sink`` depending on Gate-2 sink-shift violation.
    """
    e = metrics[operative_metric]
    a = metrics["target_accuracy"]
    y = (metrics["condition"] == "B").astype(np.int64)

    rho = _spearman_rho(e, a)
    classifier_acc, classifier_info = _fit_logistic_classifier(np.stack([e, a], axis=1), y)
    # Entropy-alone Gate-4 FAIL clause: a strong 1-D classifier (>= 0.85)
    # means entropy is acting as a condition-detector, not a quality measure
    # -- itself a refutation of the monotone interpretation.
    entropy_alone_acc, _ = _fit_logistic_classifier(e.reshape(-1, 1), y)

    if rho <= GATE4_SPEARMAN_PASS and classifier_acc >= GATE4_CLASSIFIER_PASS:
        verdict = "PASS"
    elif (rho > GATE4_SPEARMAN_FAIL) or (entropy_alone_acc >= GATE4_ENTROPY_ALONE_FAIL):
        verdict = "FAIL"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "spearman_rho": float(rho),
        "classifier_acc": float(classifier_acc),
        "classifier_info": classifier_info,
        "entropy_alone_acc": float(entropy_alone_acc),
        "operative_metric": operative_metric,
    }


def _gate_ladder(
    tier1: dict[str, np.ndarray],
    tier2: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Apply the four-gate ladder; return the aggregate verdict dict.

    Gates: (1) Tier-1 Condition-A baseline >= 0.5; (2) max sink-correction
    shift < 1 bit, else swap operative metric to entropy_no_sink (never
    aborts); (3) Tier-1 / Tier-2 triangulation -- Tier 2 governs on
    disagreement per DEC-029; (4) per-tier Spearman + classifier on the
    governing tier (PASS iff rho <= -0.5 AND clf >= 0.8).
    """
    flags: dict[str, Any] = {}

    # Gate 1 -- baseline
    tier1_a = tier1["target_accuracy"][tier1["condition"] == "A"]
    baseline_acc = float(np.mean(tier1_a)) if tier1_a.size > 0 else 0.0
    gate1 = bool(baseline_acc >= GATE1_BASELINE_ACC)
    flags["gate1_baseline_accuracy"] = baseline_acc
    if not gate1:
        return {
            "verdict": "FAIL_baseline",
            "operative_metric": "entropy_full",
            "spearman_rho": float("nan"),
            "classifier_acc": float("nan"),
            "tier1_verdict": "FAIL_baseline",
            "tier2_verdict": "not_evaluated",
            "gates": {
                "gate1_baseline": False,
                "gate2_sink": False,
                "gate3_triangulation": False,
                "gate4_spearman_classifier": False,
            },
            "flags": flags,
        }

    # Gate 2 -- sink correction (swaps metric, never aborts)
    def _max_sink_shift(m: dict[str, np.ndarray]) -> float:
        return float(np.max(np.abs(m["entropy_full"] - m["entropy_no_sink"])))

    flags["sink_shift_tier1_max_bits"] = _max_sink_shift(tier1)
    flags["sink_shift_tier2_max_bits"] = _max_sink_shift(tier2)
    sink_violation = bool(
        flags["sink_shift_tier1_max_bits"] >= GATE2_SINK_BITS
        or flags["sink_shift_tier2_max_bits"] >= GATE2_SINK_BITS
    )
    operative_metric = "entropy_no_sink" if sink_violation else "entropy_full"
    flags["sink_shift_violation"] = sink_violation

    # Gate 4 per-tier -> Gate 3 triangulation -> aggregate
    tier1_v = _per_tier_verdict(tier1, operative_metric)
    tier2_v = _per_tier_verdict(tier2, operative_metric)
    gate3 = tier1_v["verdict"] == tier2_v["verdict"]
    flags["triangulation_disagreement"] = not gate3
    governing = tier1_v if gate3 else tier2_v  # DEC-029: Tier 2 governs disagreement
    aggregate_verdict = governing["verdict"]

    return {
        "verdict": aggregate_verdict,
        "operative_metric": operative_metric,
        "spearman_rho": governing["spearman_rho"],
        "classifier_acc": governing["classifier_acc"],
        "entropy_alone_acc": governing["entropy_alone_acc"],
        "tier1_verdict": tier1_v["verdict"],
        "tier2_verdict": tier2_v["verdict"],
        "tier1": tier1_v,
        "tier2": tier2_v,
        "gates": {
            "gate1_baseline": gate1,
            "gate2_sink": True,  # gate 2 never aborts -- only swaps metric
            "gate3_triangulation": gate3,
            "gate4_spearman_classifier": aggregate_verdict == "PASS",
        },
        "flags": flags,
    }


def _render_calibration_figure(
    aggregate_dir: str,
    tier1: dict[str, np.ndarray],
    tier2: dict[str, np.ndarray],
    verdict: dict[str, Any],
) -> str | None:
    """2-panel calibration figure: scatter + per-condition histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    op_metric = verdict["operative_metric"]
    e_all = np.concatenate([tier1[op_metric], tier2[op_metric]])
    a_all = np.concatenate([tier1["target_accuracy"], tier2["target_accuracy"]])
    cond_all = np.concatenate([tier1["condition"], tier2["condition"]])
    mask_a = cond_all == "A"
    mask_b = cond_all == "B"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: scatter + regression line.
    axes[0].scatter(e_all[mask_a], a_all[mask_a], s=4, alpha=0.4, c="#1f77b4", label="Condition A")
    axes[0].scatter(e_all[mask_b], a_all[mask_b], s=4, alpha=0.4, c="#d62728", label="Condition B")
    if e_all.size > 1:
        slope, intercept = np.polyfit(e_all, a_all, 1)
        x_line = np.linspace(float(np.min(e_all)), float(np.max(e_all)), 100)
        axes[0].plot(x_line, slope * x_line + intercept, "k--", linewidth=1.0)
    axes[0].set_xlabel(f"Attention entropy ({op_metric}, bits)")
    axes[0].set_ylabel("Attention-target accuracy")
    axes[0].set_title(
        f"Pooled: rho={verdict['spearman_rho']:.3f} | "
        f"clf={verdict['classifier_acc']:.3f} | {verdict['verdict']}"
    )
    axes[0].legend(loc="best", fontsize=8)

    # Right: per-condition entropy histogram.
    axes[1].hist(e_all[mask_a], bins=40, alpha=0.55, color="#1f77b4", label="Condition A")
    axes[1].hist(e_all[mask_b], bins=40, alpha=0.55, color="#d62728", label="Condition B")
    axes[1].set_xlabel(f"Attention entropy ({op_metric}, bits)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Entropy distribution by condition")
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = os.path.join(aggregate_dir, "figure.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_analysis_only(args: argparse.Namespace) -> None:
    """Analysis-only mode entry point: 4-gate verdict and aggregate outputs."""
    tier1_csv = os.path.join(args.out, "tier1", "metrics.csv")
    tier2_csv = os.path.join(args.out, "tier2", "metrics.csv")
    for tier, path in ((1, tier1_csv), (2, tier2_csv)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"_run_analysis_only: missing {path}; run forward-pass mode with --tier {tier} first"
            )

    tier1 = _read_metrics_csv(tier1_csv)
    tier2 = _read_metrics_csv(tier2_csv)
    verdict = _gate_ladder(tier1, tier2)

    aggregate_dir = os.path.join(args.out, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    spearman_payload = {
        "operative_metric": verdict["operative_metric"],
        "tier1": {
            "rho": verdict["tier1"]["spearman_rho"],
            "n": int(tier1["entropy_full"].shape[0]),
        },
        "tier2": {
            "rho": verdict["tier2"]["spearman_rho"],
            "n": int(tier2["entropy_full"].shape[0]),
        },
        "governing_rho": verdict["spearman_rho"],
        "thresholds": {
            "pass": GATE4_SPEARMAN_PASS,
            "fail": GATE4_SPEARMAN_FAIL,
        },
    }
    with open(os.path.join(aggregate_dir, "spearman.json"), "w") as fh:
        json.dump(spearman_payload, fh, indent=2, sort_keys=True)

    classifier_payload = {
        "operative_metric": verdict["operative_metric"],
        "tier1": {
            "accuracy": verdict["tier1"]["classifier_acc"],
            "entropy_alone_accuracy": verdict["tier1"]["entropy_alone_acc"],
            "info": verdict["tier1"]["classifier_info"],
        },
        "tier2": {
            "accuracy": verdict["tier2"]["classifier_acc"],
            "entropy_alone_accuracy": verdict["tier2"]["entropy_alone_acc"],
            "info": verdict["tier2"]["classifier_info"],
        },
        "governing_accuracy": verdict["classifier_acc"],
        "thresholds": {
            "pass": GATE4_CLASSIFIER_PASS,
            "entropy_alone_fail": GATE4_ENTROPY_ALONE_FAIL,
        },
    }
    with open(os.path.join(aggregate_dir, "classifier.json"), "w") as fh:
        json.dump(classifier_payload, fh, indent=2, sort_keys=True)

    verdict_payload = {
        "verdict": verdict["verdict"],
        "operative_metric": verdict["operative_metric"],
        "spearman_rho": verdict["spearman_rho"],
        "classifier_acc": verdict["classifier_acc"],
        "tier1_verdict": verdict["tier1_verdict"],
        "tier2_verdict": verdict["tier2_verdict"],
        "gates": verdict["gates"],
        "flags": verdict["flags"],
        "thresholds": {
            "gate1_baseline_accuracy": GATE1_BASELINE_ACC,
            "gate2_sink_bits": GATE2_SINK_BITS,
            "gate4_spearman_pass": GATE4_SPEARMAN_PASS,
            "gate4_spearman_fail": GATE4_SPEARMAN_FAIL,
            "gate4_classifier_pass": GATE4_CLASSIFIER_PASS,
            "gate4_entropy_alone_fail": GATE4_ENTROPY_ALONE_FAIL,
        },
    }
    with open(os.path.join(aggregate_dir, "verdict.json"), "w") as fh:
        json.dump(verdict_payload, fh, indent=2, sort_keys=True)

    _render_calibration_figure(aggregate_dir, tier1, tier2, verdict)


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """CLI parser for Protocol D-calibration; per-flag docs live on each
    ``add_argument`` ``help=`` below.
    """
    parser = argparse.ArgumentParser(
        description="Protocol D-calibration -- 4-gate attention-entropy validator"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Tier 1: HuggingFace identifier ('gpt2-large') or local snapshot. "
             "Tier 2/3: path to the checkpoint dir or .pt file.",
    )
    parser.add_argument(
        "--tier", type=int, default=None, choices=(1, 2, 3),
        help="Substrate tier (1 GPT-2-large, 2 ADM C5, 3 24L standard).",
    )
    parser.add_argument(
        "--n-per-condition", type=int, default=1000,
        help="Sequences per condition (1000 for the primary ladder, "
             "4000 for the AMBIGUOUS retry).",
    )
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output directory. In forward-pass mode receives "
             "metrics.csv + meta.json. In --analysis-only mode this "
             "is the parent of tier1/ + tier2/; aggregate/ is written here.",
    )
    parser.add_argument(
        "--module-path", type=str, default="model.transformer.h_mid",
        help="Tier 1 'model.transformer.h'; Tier 2/3 'model.transformer.h_mid'.",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Skip the forward pass; aggregate tier1/ + tier2/ metrics.csv "
             "and write the 4-gate verdict.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.analysis_only:
        _run_analysis_only(args)
    else:
        if args.tier is None:
            raise ValueError("main: --tier is required unless --analysis-only is set")
        _run_forward_pass(args)


if __name__ == "__main__":
    main()
