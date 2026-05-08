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
    """Shannon entropy in bits, summed over the last axis.

    Mirrors ``analysis.attention_taxonomy.shannon_entropy``: the EPS
    floor handles the ``0 log 0 = 0`` limit. Input is NOT
    renormalised; callers must pass a probability distribution that
    already sums to one within floating-point tolerance.
    """
    q = np.clip(p, EPS, 1.0)
    return -np.sum(q * np.log2(q), axis=-1)


def _apply_sink_correction(p: np.ndarray) -> np.ndarray:
    """Zero out position-0 mass and renormalise the remainder per row.

    Mirrors ``analysis.attention_taxonomy.apply_sink_correction``.
    Pure-sink rows (mass outside position 0 near zero) are kept
    near-zero and contribute zero to entropy under the EPS-clipped
    log; this is the documented behaviour per
    ``docs/extend-notes.md`` §1.3 sink-correction note.
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
    """Compute per-sequence entropy / sink-mass / target-accuracy summaries.

    Parameters
    ----------
    att : np.ndarray
        Shape ``(n_seq, T_k)`` -- per-sequence attention distribution
        at the query position, AVERAGED across (layer, head) of the
        upper half of the substrate's block stack. The average across
        layers/heads is the standard "model-wide attention" reduction
        used by the calibration ladder; per-head dispersion is
        captured by Protocol C, not here.
    target_positions : np.ndarray
        Integer ground-truth target positions, shape either
        ``(n_seq,)`` (Condition A) or ``(n_seq, K)`` (Condition B).

    Returns
    -------
    dict
        Keys: ``entropy_full``, ``entropy_no_sink``, ``sink_mass``,
        ``target_accuracy``. Each is shape ``(n_seq,)``.
    """
    if att.ndim != 2:
        raise ValueError(
            f"_reduce_attention_at_query: expected (n_seq, T_k); "
            f"got shape {att.shape}"
        )

    entropy_full = _shannon_entropy_bits(att)
    sink_mass = att[:, 0].astype(np.float64)
    att_ns = _apply_sink_correction(att)
    entropy_no_sink = _shannon_entropy_bits(att_ns)

    n_seq = att.shape[0]
    if target_positions.ndim == 1:
        # Condition A: single target per sequence.
        idx = target_positions.astype(np.int64)
        target_accuracy = att[np.arange(n_seq), idx].astype(np.float64)
    elif target_positions.ndim == 2:
        # Condition B: K targets per sequence; sum mass over them.
        idx = target_positions.astype(np.int64)
        target_accuracy = np.take_along_axis(att, idx, axis=1).sum(axis=1).astype(np.float64)
    else:
        raise ValueError(
            f"_reduce_attention_at_query: target_positions has "
            f"unsupported ndim {target_positions.ndim}"
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
    """Forward Tier-1 GPT-2-large on synthetic tokens; return query-row attention.

    Returns a ``(n_seq, T_k)`` attention distribution at each query
    position averaged across the upper half of the layer stack and
    across all heads. The HuggingFace model is loaded with
    ``cache_dir=$HF_HOME`` so the call is offline-safe on compute
    nodes (the login-node helper ``cache_huggingface_model.py``
    pre-populates the cache).
    """
    import torch
    from transformers import AutoModelForCausalLM  # type: ignore

    cache_dir = os.environ.get("HF_HOME")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        cache_dir=cache_dir,
        attn_implementation="eager",  # eager backend exposes attention weights
    )
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    n_layer = int(model.config.n_layer)
    upper_layers = list(range(n_layer // 2, n_layer))

    n_seq = tokens.shape[0]
    T = tokens.shape[1]
    out = np.zeros((n_seq, T), dtype=np.float32)

    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    qpos_t = np.asarray(query_positions, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            stop = min(start + batch_size, n_seq)
            batch = tokens_t[start:stop].to(device)
            outputs = model(
                batch,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            # outputs.attentions: tuple of (B, n_head, T, T) per layer
            stacked = torch.stack(
                [outputs.attentions[layer] for layer in upper_layers], dim=0
            )
            # Mean across (layer, head) for the AVERAGED attention
            # distribution per (batch, query, key) cell.
            mean_att = stacked.mean(dim=(0, 2))  # (B, T, T)
            for offset in range(stop - start):
                qp = int(qpos_t[start + offset])
                row = mean_att[offset, qp, :].detach().to("cpu").float().numpy()
                # The post-softmax row sums to 1 per (layer, head); the
                # average across layer/head is also a valid distribution.
                out[start + offset] = row
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
    """Forward Tier-2 / Tier-3 CoTFormer-style ckpt; return query-row attention.

    Uses the project's ``ActivationCollector`` with
    ``ATTN_WEIGHTS`` + ``non_flash=True`` to recover per-(layer,
    repeat, head) attention. Averages across (layer, repeat, head)
    of the upper half of mid-block stack to yield a
    ``(n_seq, T_k)`` row at the query position per sequence.
    """
    import torch

    from analysis.common.collector import ActivationCollector
    from analysis.common.loader import load_model_from_checkpoint
    from analysis.common.sites import ActivationSite

    # Convention: ckpt_dir is a directory; ckpt_file lives inside it.
    # The protocol passes a single absolute path with the file name in
    # job-gpu.sh; tolerate either input by splitting if the path points
    # at an existing file.
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
    n_seq = tokens.shape[0]
    T = tokens.shape[1]
    out = np.zeros((n_seq, T), dtype=np.float32)

    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    qpos_t = np.asarray(query_positions, dtype=np.int64)

    # The collector buffers attention per (layer, repeat). We forward
    # batch-by-batch and aggregate the captured weights at the query
    # position. The collector is re-entered per batch so its buffers
    # do not accumulate across the full n_seq (RAM economy on L4).
    seq_idx = 0
    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            stop = min(start + batch_size, n_seq)
            batch = tokens_t[start:stop].to(device)
            collector = ActivationCollector(
                model, sites, non_flash=True, module_path=module_path,
            )
            with collector:
                model(batch)
            # Buffers: ``attn_weights_<group>_l<L>_r<R>`` keys.
            # Each buffer is a list of (B*T_q, n_head, T_k) numpy arrays.
            stacked_rows: list[np.ndarray] = []
            n_layer_tot = 0
            for buf_key, chunks in collector._buffers.items():
                if not buf_key.startswith("attn_weights_"):
                    continue
                if not chunks:
                    continue
                arr = np.concatenate(chunks, axis=0)  # (B*T_q, n_head, T_k)
                stacked_rows.append(arr)
                n_layer_tot += 1
            if not stacked_rows:
                raise RuntimeError(
                    "_forward_attention_cotformer: collector produced no "
                    "attn_weights buffers; check module_path / non_flash"
                )
            # Restrict to upper half of layers (analogous to GPT-2 path).
            upper_start = n_layer_tot // 2
            stacked_rows = stacked_rows[upper_start:]

            # Each chunk has shape (B*T_q, n_head, T_k); the convention
            # is that row i corresponds to query position i % T_q within
            # batch element i // T_q (see ActivationCollector docstring).
            # Mean across heads first; then across (layer, repeat).
            B = stop - start
            T_local = batch.shape[1]
            mean_per_layer = []
            for arr in stacked_rows:
                # (B*T_q, n_head, T_k) -> (B, T_q, n_head, T_k)
                arr_re = arr.reshape(B, T_local, arr.shape[1], arr.shape[2])
                head_mean = arr_re.mean(axis=2)  # (B, T_q, T_k)
                mean_per_layer.append(head_mean)
            mean_lr = np.stack(mean_per_layer, axis=0).mean(axis=0)  # (B, T_q, T_k)
            for offset in range(B):
                qp = int(qpos_t[start + offset])
                row = mean_lr[offset, qp, :]
                # Tk may be < T (no padding mask present); pad with zero.
                if row.shape[0] < T:
                    padded = np.zeros((T,), dtype=np.float32)
                    padded[: row.shape[0]] = row
                    out[start + offset] = padded
                else:
                    out[start + offset] = row[:T]
            seq_idx += B
            # Free the collector's buffers immediately.
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
    """Generate Condition A and Condition B token tensors and ground truth."""
    from analysis.calibration.synth_sequences import (
        generate_condition_A,
        generate_condition_B,
    )

    tok_a, qpos_a, tgt_a = generate_condition_A(
        n=n_per_condition, L=seq_length, vocab_size=vocab_size, seed=seed,
    )
    tok_b, qpos_b, tgt_b = generate_condition_B(
        n=n_per_condition,
        L=seq_length,
        vocab_size=vocab_size,
        inf_vocab=INF_VOCAB_SIZE,
        seed=seed + 1,
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
    """Write the per-sequence metrics.csv.

    Columns: ``seq_id, condition, entropy_full, entropy_no_sink,
    sink_mass, target_accuracy`` per the directive's contract.
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
            n = per_seq["entropy_full"].shape[0]
            for i in range(n):
                writer.writerow([
                    seq_id,
                    cond_label,
                    float(per_seq["entropy_full"][i]),
                    float(per_seq["entropy_no_sink"][i]),
                    float(per_seq["sink_mass"][i]),
                    float(per_seq["target_accuracy"][i]),
                ])
                seq_id += 1
    return path


def _run_forward_pass(args: argparse.Namespace) -> None:
    """Forward-pass mode entry point. Dispatches by ``--tier``."""
    os.makedirs(args.out, exist_ok=True)

    substrate = _generate_substrate_tokens(
        n_per_condition=args.n_per_condition,
        seed=args.seed,
    )

    if args.tier == 1:
        ckpt_arg = args.ckpt or "gpt2-large"
        att_a = _forward_attention_gpt2(
            hf_model_id=ckpt_arg,
            tokens=substrate["A"]["tokens"],
            query_positions=substrate["A"]["query_positions"],
            device=args.device,
            batch_size=args.batch_size,
        )
        att_b = _forward_attention_gpt2(
            hf_model_id=ckpt_arg,
            tokens=substrate["B"]["tokens"],
            query_positions=substrate["B"]["query_positions"],
            device=args.device,
            batch_size=args.batch_size,
        )
        substrate_name = ckpt_arg
    elif args.tier in (2, 3):
        if not args.ckpt:
            raise ValueError(
                f"_run_forward_pass: --ckpt is required for tier {args.tier}"
            )
        att_a = _forward_attention_cotformer(
            ckpt_dir=args.ckpt,
            ckpt_file="ckpt.pt",
            tokens=substrate["A"]["tokens"],
            query_positions=substrate["A"]["query_positions"],
            device=args.device,
            batch_size=args.batch_size,
            module_path=args.module_path,
        )
        att_b = _forward_attention_cotformer(
            ckpt_dir=args.ckpt,
            ckpt_file="ckpt.pt",
            tokens=substrate["B"]["tokens"],
            query_positions=substrate["B"]["query_positions"],
            device=args.device,
            batch_size=args.batch_size,
            module_path=args.module_path,
        )
        substrate_name = args.ckpt
    else:
        raise ValueError(f"_run_forward_pass: unknown tier {args.tier}")

    per_seq_a = _reduce_attention_at_query(
        att_a, substrate["A"]["target_positions"]
    )
    per_seq_b = _reduce_attention_at_query(
        att_b, substrate["B"]["target_positions"]
    )

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
    """Spearman rank correlation; pure-numpy fallback when scipy missing."""
    try:
        from scipy import stats as _scipy_stats
        rho = float(_scipy_stats.spearmanr(a, b).correlation)
        if math.isnan(rho):
            return 0.0
        return rho
    except ImportError:
        ra = _rankdata(a)
        rb = _rankdata(b)
        ra_c = ra - np.mean(ra)
        rb_c = rb - np.mean(rb)
        num = float(np.sum(ra_c * rb_c))
        den = float(math.sqrt(float(np.sum(ra_c ** 2) * np.sum(rb_c ** 2))))
        if den == 0.0:
            return 0.0
        return num / den


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank ranking (mirrors taxonomy._rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=np.float64)
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


def _fit_logistic_classifier(
    X: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Fit logistic regression and return (accuracy, model-info dict).

    Falls back to a numpy gradient-descent logistic if sklearn is
    unavailable; both produce the same accuracy on this 2D problem.
    """
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        pred = clf.predict(X)
        acc = float(np.mean(pred == y))
        return acc, {
            "backend": "sklearn",
            "coef": [float(c) for c in clf.coef_.ravel()],
            "intercept": float(clf.intercept_[0]),
        }
    except ImportError:
        pass

    # Numpy fallback: standardised features + closed-form-ish IRLS.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + EPS
    Xn = (X - X_mean) / X_std
    n, d = Xn.shape
    Xb = np.concatenate([Xn, np.ones((n, 1))], axis=1)
    w = np.zeros(d + 1)
    for _ in range(200):
        z = Xb @ w
        # numerically stable sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad = Xb.T @ (p - y) / n
        w -= 0.5 * grad
    pred = (Xb @ w >= 0.0).astype(np.int64)
    acc = float(np.mean(pred == y))
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
    """Compute a per-tier PASS / AMBIGUOUS / FAIL verdict from Gate 4.

    Spearman + classifier evaluated on the pooled (Condition A union
    Condition B) population; the operative entropy column is selected
    by ``operative_metric`` ("entropy_full" or "entropy_no_sink").
    """
    e = metrics[operative_metric]
    a = metrics["target_accuracy"]
    cond = metrics["condition"]
    y = (cond == "B").astype(np.int64)  # binary classifier label

    rho = _spearman_rho(e, a)
    X = np.stack([e, a], axis=1)
    classifier_acc, classifier_info = _fit_logistic_classifier(X, y)
    # Entropy-alone failure check (Gate-4 FAIL clause): >= 0.85 means
    # classifier-on-entropy alone separates the conditions strongly,
    # which is itself a refutation of the monotone interpretation
    # (the metric becomes a condition-detector instead of a quality
    # measure).
    Xe = e.reshape(-1, 1)
    entropy_alone_acc, _ = _fit_logistic_classifier(Xe, y)

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
    """Apply the four-gate ladder and return the aggregate verdict dict."""
    flags: dict[str, Any] = {}

    # ---------------- Gate 1: baseline ----------------
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

    # ---------------- Gate 2: sink correction ----------------
    def _max_sink_shift(m: dict[str, np.ndarray]) -> float:
        return float(np.max(np.abs(m["entropy_full"] - m["entropy_no_sink"])))

    sink_shift_t1 = _max_sink_shift(tier1)
    sink_shift_t2 = _max_sink_shift(tier2)
    flags["sink_shift_tier1_max_bits"] = sink_shift_t1
    flags["sink_shift_tier2_max_bits"] = sink_shift_t2

    sink_violation = bool(
        sink_shift_t1 >= GATE2_SINK_BITS or sink_shift_t2 >= GATE2_SINK_BITS
    )
    operative_metric = "entropy_no_sink" if sink_violation else "entropy_full"
    flags["sink_shift_violation"] = sink_violation
    gate2 = True  # gate 2 never aborts; a violation only swaps the metric

    # ---------------- Gate 4 first (per-tier) ----------------
    tier1_v = _per_tier_verdict(tier1, operative_metric)
    tier2_v = _per_tier_verdict(tier2, operative_metric)

    # ---------------- Gate 3: triangulation ----------------
    # PASS if both PASS, FAIL if both FAIL; otherwise disagreement.
    if tier1_v["verdict"] == tier2_v["verdict"]:
        gate3 = True
        flags["triangulation_disagreement"] = False
    else:
        gate3 = False
        flags["triangulation_disagreement"] = True
        # Tier 2 governs per DEC-029.

    # The aggregate verdict is governed by Tier 2 (per DEC-029) when
    # disagreement is reported; when both tiers agree, either tier's
    # verdict is the aggregate.
    governing = tier2_v if not gate3 else tier1_v
    aggregate_verdict = governing["verdict"]

    gate4 = bool(aggregate_verdict == "PASS")

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
            "gate2_sink": gate2,
            "gate3_triangulation": gate3,
            "gate4_spearman_classifier": gate4,
        },
        "flags": flags,
    }


def _render_calibration_figure(
    aggregate_dir: str,
    tier1: dict[str, np.ndarray],
    tier2: dict[str, np.ndarray],
    verdict: dict[str, Any],
) -> str | None:
    """Write a 2-panel calibration figure: scatter + per-condition histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    op_metric = verdict["operative_metric"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: scatter + regression line, colour by condition.
    e_all = np.concatenate([tier1[op_metric], tier2[op_metric]])
    a_all = np.concatenate([tier1["target_accuracy"], tier2["target_accuracy"]])
    cond_all = np.concatenate([tier1["condition"], tier2["condition"]])
    mask_a = cond_all == "A"
    mask_b = cond_all == "B"
    axes[0].scatter(
        e_all[mask_a], a_all[mask_a],
        s=4, alpha=0.4, c="#1f77b4", label="Condition A",
    )
    axes[0].scatter(
        e_all[mask_b], a_all[mask_b],
        s=4, alpha=0.4, c="#d62728", label="Condition B",
    )
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

    # Right panel: per-condition entropy histogram.
    axes[1].hist(
        e_all[mask_a], bins=40, alpha=0.55, color="#1f77b4", label="Condition A",
    )
    axes[1].hist(
        e_all[mask_b], bins=40, alpha=0.55, color="#d62728", label="Condition B",
    )
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
    if not os.path.isfile(tier1_csv):
        raise FileNotFoundError(
            f"_run_analysis_only: missing {tier1_csv}; run forward-pass mode "
            f"with --tier 1 first"
        )
    if not os.path.isfile(tier2_csv):
        raise FileNotFoundError(
            f"_run_analysis_only: missing {tier2_csv}; run forward-pass mode "
            f"with --tier 2 first"
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
    """Return the CLI parser for the calibration entry point.

    Expected inputs: ``--ckpt`` (Tier 1 path to GPT-2-large snapshot;
    Tier 2 path to ADM C5; Tier 3 contingency), ``--tier`` (1 / 2 / 3),
    ``--n-per-condition`` (default 1000 per condition for the primary
    ladder; 4000 per condition for the AMBIGUOUS-verdict retry),
    ``--seed``, ``--out`` (output directory), ``--module-path``
    (forwarded to the common loader; Tier 1 uses
    ``model.transformer.h``, Tiers 2 and 3 use
    ``model.transformer.h_mid``).
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
