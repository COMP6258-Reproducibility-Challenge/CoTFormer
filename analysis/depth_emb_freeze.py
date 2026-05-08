"""Protocol I -- Depth-Embedding 8-Condition Freeze Ablation (RQ8).

Scope
-----
Addresses RQ8 (causal necessity of the depth embedding). Runs the
LN-CoTFormer (C3 at 60k) under 8 freeze conditions produced by the
Cartesian product of {freeze / unfreeze} x {zero / preserve / random}
applied to the ``depth_embedding`` entry and two matched controls
(per DEC-026's zero-vector condition addition). Per-condition
perplexity and loss are reported with bootstrap 95 per cent CI.

Falsifiability relevance
------------------------
H0 "depth embedding is non-causal for the recurrent convergence
signature" is rejected if perplexity under the zero-condition is
significantly worse than the preserve-condition at paired bootstrap
p < 0.01 AND the effect exceeds the seed-variance baseline (per
``docs/extend-notes.md`` §1.2 RQ8 "Statistical test"). The
8-condition ANOVA is per-condition rather than the earlier
4-condition layout per DEC-026.

Ontological purpose
-------------------
Isolates depth-embedding causality from init-order confounds
(pre-B6 vs post-B6 checkpoints have different init-order RNG). A
non-causal depth embedding is evidence that the paper's C3-vs-C2
gain is attributable to ``ln_mid`` alone, which bears on RQ10's
Table-2-unconfound agenda.

Implementation notes
--------------------
The model's forward loop at
``models/cotformer_full_depth_lnmid_depthemb.py`` selects the
depth-embedding entry by reverse index (``self.n_repeat - rep_idx``
with ``rep_idx in [1, n_repeat]``). The freeze hook installs on the
``depth_emb`` module's forward and either (a) overrides the
``indices`` argument to a frozen value (``preserve``-mode at a fixed
target index), or (b) replaces the additive embedding with a chosen
tensor (``zero`` and ``random`` modes). The hook reads the active
mid-block repeat counter via the same forward-pre / forward-post
mechanism used by ``analysis/counting_dv4_causal.py`` so the per-
repeat counter is always in sync with the forward loop.

Counter-registration order
--------------------------
Per ``docs/extend-technical.md`` §"Counter-registration order in
collector + DV-4 ablation" (and as enforced verbatim in
``analysis/counting_dv4_causal.py``), the per-repeat increment hook
MUST be registered AFTER the per-block / per-module ablation hooks.
PyTorch fires forward post-hooks in registration order; for V1/V2
architectures (no ``ln_mid``) the increment lands on the last
``h_mid`` block which is also a hook target -- registering increment
first would shift every ablation activation by one repeat. With
ablation hooks first, they read the pre-bump counter and activate at
the correct target_repeat. RQ8 only runs on V3 (LN-CoTFormer with
``ln_mid``), but the order discipline is preserved unconditionally
because the script is also smoke-runnable on V1/V2 stub checkpoints.

Spec choice flags (inline ``# Spec choice:`` comments)
------------------------------------------------------
The CLI surface defined by the docstring uses ``--freeze-mode`` for
the kind of freeze (``zero`` / ``preserve`` / ``random``) and
``--freeze-target`` for the target index. The 8-condition Cartesian
product additionally varies a ``freeze`` / ``unfreeze`` axis: a
``--freeze-active`` flag (default ``true``) selects between
intervening (``true``) and a no-op control with the same hook
overhead (``false``). The HPC sweep at
``iridis/analyze-lncot+adm/job-gpu.sh`` currently does NOT pass this
flag; both the freeze and unfreeze rows therefore default to
``--freeze-active true``. Per the task's "Spec ambiguity discipline"
this is flagged for the user; resolution requires either updating
the HPC script to pass ``--freeze-active`` or accepting that the
8-condition matrix collapses to 6 distinct interventions plus 2
duplicate target-index entries until the script is updated.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


VALID_FREEZE_MODES = ("zero", "preserve", "random")


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol I.

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``, ``--freeze-mode``
    (``zero`` / ``preserve`` / ``random``), ``--freeze-target``
    (``depth_emb`` entry index; 1..n_repeat using the reverse-index
    convention of the forward loop), ``--n-bootstrap``.
    """
    parser = argparse.ArgumentParser(
        description="Protocol I -- Depth-Embedding 8-Condition Freeze Ablation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Checkpoint directory containing summary.json + checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename (default ckpt.pt)",
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Optional workspace dir on /scratch (parity with sibling protocols).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False,
        help="Directory for depth_emb_freeze_results.json + figure",
    )
    parser.add_argument(
        "--seed", type=int, default=2357,
        help="Forward-pass seed; also seeds the random freeze-kind RNG",
    )
    parser.add_argument(
        "--freeze-mode", type=str, default="preserve",
        choices=list(VALID_FREEZE_MODES),
        help="Kind of freeze: zero (replace with 0 vector), preserve "
             "(keep the actual learned embedding -- null intervention), "
             "random (deterministic Gaussian per repeat per seed).",
    )
    parser.add_argument(
        "--freeze-target", type=int, default=1,
        help="1-indexed depth_emb target index using the reverse-index "
             "forward-loop convention. Active when --freeze-active true.",
    )
    parser.add_argument(
        "--freeze-active", type=str, default="true",
        choices=["true", "false"],
        help="Spec choice: 'true' -> intervene (freeze the depth_emb at "
             "--freeze-target with --freeze-mode); 'false' -> no-op control "
             "with the same hook overhead. Default 'true' so the existing "
             "HPC sweep (which does not pass this flag) selects the "
             "intervention rows of the 8-condition matrix.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap resamples for delta CI (default 1000; HPC uses 10000).",
    )
    parser.add_argument("--n-batches", type=int, default=4,
                        help="Inter-batch robustness batch count (default 4)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Tokens per batch (default 2048)")
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a pure-numpy smoke test on the bootstrap "
                             "+ Holm-Bonferroni helpers and exit.")
    return parser


# ---------------------------------------------------------------------------
# Hook context manager (torch-deferred; mirrors counting_dv4_causal.py)
# ---------------------------------------------------------------------------


_COUNTER_ATTR = "_analysis_depth_emb_freeze_repeat_counter"


@contextmanager
def freeze_hooks(
    model: "Any",
    freeze_mode: str,
    freeze_target: int,
    freeze_active: bool,
    seed: int,
    module_path: str,
) -> Iterator[None]:
    """Install depth-embedding freeze hooks for one (mode, target) condition.

    Parameters
    ----------
    model : nn.Module
        Loaded LN-CoTFormer (or compatible). Must expose
        ``transformer`` and a non-``None`` ``depth_emb`` attribute on
        the model root (per
        ``models/cotformer_full_depth_lnmid_depthemb.py``). Aborts
        with a clear ``AttributeError`` if absent so the orchestrator
        can ESC instead of silently continuing.
    freeze_mode : {"zero", "preserve", "random"}
        Kind of replacement applied at the target repeat:
          ``zero`` -- replace the additive depth_emb output with 0;
          ``preserve`` -- identity (no-op intervention; same overhead);
          ``random`` -- replace with a deterministic Gaussian draw
          seeded by ``(seed, repeat_idx)``.
    freeze_target : int
        1-indexed target repeat. The forward loop's reverse indexing
        is handled internally; the user supplies the natural 1..n
        repeat number and the hook fires when the repeat counter hits
        it. ``freeze_target <= 0`` together with ``freeze_active=False``
        means no-op (baseline pass).
    freeze_active : bool
        ``False`` disables the per-target intervention but still
        installs the counter machinery so any downstream consumer
        sees the same overhead and timing characteristics
        (no-op control).
    seed : int
        Determinism seed for the ``random`` freeze-mode.
    module_path : str
        Dotted path to the h_mid block list (parity with
        ``counting_dv4_causal.py`` -- used only for the counter
        increment fallback when ``ln_mid`` is absent).

    Yields
    ------
    None
        Control to the caller's forward loop. Hooks are installed on
        entry and removed on exit (via ``try/finally``).
    """
    import torch  # noqa: F401

    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError(
            "freeze_hooks: model has no transformer attribute"
        )

    depth_emb = getattr(model, "depth_emb", None)
    if depth_emb is None:
        # Spec choice: surface a clear error rather than silently
        # falling back to a no-op so the orchestrator can ESC if a
        # future variant relocates the attribute.
        raise AttributeError(
            "freeze_hooks: model.depth_emb is None or absent. RQ8 "
            "requires a checkpoint with a learned depth embedding "
            "(LN-CoTFormer / V3 / V4 only)."
        )

    if freeze_mode not in VALID_FREEZE_MODES:
        raise ValueError(
            f"freeze_hooks: freeze_mode must be one of "
            f"{VALID_FREEZE_MODES} (got {freeze_mode!r})"
        )

    # Resolve h_mid via module_path suffix (mirrors counting_dv4_causal).
    mid_path = module_path
    if mid_path.startswith("model."):
        mid_path = mid_path[len("model.") :]
    cursor = model
    for name in mid_path.split("."):
        cursor = getattr(cursor, name)
    if not hasattr(cursor, "__len__") or len(cursor) == 0:
        raise RuntimeError(
            "freeze_hooks: model has an empty h_mid list at "
            f"{module_path!r}; cannot install counter hooks."
        )
    h_mid = cursor

    n_repeat = int(getattr(model, "n_repeat", 1) or 1)
    if freeze_active and freeze_target > 0 and freeze_target > n_repeat:
        raise ValueError(
            f"freeze_hooks: freeze_target={freeze_target} exceeds "
            f"model.n_repeat={n_repeat}"
        )

    handles: list = []

    def _reset_counter(module, inputs):
        setattr(model, _COUNTER_ATTR, 1)

    def _increment_counter(module, inputs, output):
        current = int(getattr(model, _COUNTER_ATTR, 1))
        setattr(model, _COUNTER_ATTR, current + 1)

    def _make_freeze_hook():
        def _hook(module, inputs, output):
            # ``inputs`` is ``(x, indices)``; ``output`` is the
            # additive ``x + emb`` from the encoder's forward.
            if not freeze_active:
                return output
            current = int(getattr(model, _COUNTER_ATTR, 1))
            current = max(1, min(current, n_repeat))
            if current != freeze_target:
                return output

            x_in = inputs[0]
            if freeze_mode == "preserve":
                # Identity intervention: keep the original output.
                # This branch exists so the timing / hook-fire path
                # is identical across the three kinds (avoids a
                # confound where ``preserve`` short-circuits).
                return output
            if freeze_mode == "zero":
                # True signal ablation: ``output = x + 0`` so the
                # additive embedding contribution is removed.
                return x_in
            if freeze_mode == "random":
                # Deterministic per-repeat per-seed Gaussian; matches
                # the spec's "random" condition (per RQ8 row).
                generator = torch.Generator(device=x_in.device)
                generator.manual_seed(int(seed) * 31 + int(current))
                # The forward emits ``x + embs(indices)``; we replace
                # only the additive embedding component, preserving
                # ``x`` so the residual stream's pre-emb structure is
                # intact.
                emb_shape = output.shape
                random_emb = torch.randn(
                    emb_shape, generator=generator,
                    device=x_in.device, dtype=output.dtype,
                )
                # Match the trained embedding's per-row scale (std
                # ~0.02 from the GPT-2 init) so the intervention is
                # in-distribution rather than gross-over-magnitude.
                random_emb = random_emb * 0.02
                return x_in + random_emb
            return output

        return _hook

    # 1. Reset counter at the start of each forward pass.
    handles.append(model.register_forward_pre_hook(_reset_counter))

    # 2. Install the freeze hook on depth_emb FIRST. This is the
    # ablation hook; it must read the counter BEFORE the per-repeat
    # increment fires (counter-registration order discipline; see
    # docs/extend-technical.md and counting_dv4_causal.py:218-228).
    handles.append(depth_emb.register_forward_hook(_make_freeze_hook()))

    # 3. Register the counter increment hook AFTER the freeze hook.
    # PyTorch fires forward post-hooks in registration order; on V3
    # the increment lands on ln_mid (a different module from
    # depth_emb) so order is irrelevant for V3. We preserve the
    # discipline unconditionally so the script is safe on V1/V2 stub
    # checkpoints where ln_mid is absent and the increment falls back
    # to the last h_mid block. See counting_dv4_causal.py for the
    # full rationale.
    ln_mid = getattr(transformer, "ln_mid", None)
    if ln_mid is not None and not isinstance(
        ln_mid, _identity_marker_type()
    ):
        handles.append(ln_mid.register_forward_hook(_increment_counter))
    else:
        last_mid = h_mid[len(h_mid) - 1]
        handles.append(last_mid.register_forward_hook(_increment_counter))

    setattr(model, _COUNTER_ATTR, 1)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(model, _COUNTER_ATTR):
            delattr(model, _COUNTER_ATTR)


def _identity_marker_type():
    """Return ``nn.Identity`` for the ln_mid disable check."""
    import torch.nn as nn
    return nn.Identity


# ---------------------------------------------------------------------------
# Per-batch metrics
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FreezeRunMetrics:
    """Per-condition forward-pass metrics."""

    n_tokens: int
    cross_entropy: float
    top1_accuracy: float
    # Per-position CE retained for paired bootstrap on the delta
    # against the baseline condition (token-aligned across runs).
    per_token_ce: list[float]
    per_token_correct: list[int]
    # Mean log-softmax over the vocabulary, kept token-wise so the
    # downstream KL-vs-baseline comparison can be computed without
    # re-running the forward.
    per_token_log_softmax_top: list[float]


def _ce_and_accuracy_from_logits(
    logits: "Any",  # torch.Tensor (B, T, V)
    targets: "Any",  # torch.Tensor (B, T)
) -> dict[str, "Any"]:
    """Compute per-token CE, top-1 correctness, and KL-friendly cache."""
    import torch
    import torch.nn.functional as F

    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V).float()
    targets_flat = targets.reshape(-1)
    per_token_ce = F.cross_entropy(
        logits_flat, targets_flat, reduction="none", ignore_index=-1
    )  # (B*T,)
    preds = logits_flat.argmax(dim=-1)
    correct = (preds == targets_flat).long()

    # Cache the log-softmax distribution for KL computation upstream.
    # Storing the FULL vocab is too large; we keep the predicted-token
    # log-softmax row (target token's log-prob) which lets us compute
    # an empirical CE-aligned KL approximation. Full-distribution KL
    # is computed inline at the caller for accuracy.
    log_softmax = F.log_softmax(logits_flat, dim=-1)
    target_log_prob = log_softmax[
        torch.arange(log_softmax.shape[0], device=log_softmax.device),
        targets_flat.clamp(min=0),
    ]

    return {
        "per_token_ce": per_token_ce,
        "per_token_correct": correct,
        "per_token_target_log_prob": target_log_prob,
        "log_softmax": log_softmax,
        "B": B, "T": T, "V": V,
    }


def _run_one_condition(
    model,
    config,
    args: argparse.Namespace,
    freeze_mode: str,
    freeze_target: int,
    freeze_active: bool,
    device: str,
    baseline_log_softmax: "Any | None" = None,
) -> tuple[FreezeRunMetrics, "Any", "Any"]:
    """Run ``args.n_batches`` forward passes under one condition.

    Returns
    -------
    metrics : FreezeRunMetrics
        Aggregated per-token CE / accuracy + per-token CE list for
        downstream paired bootstrap.
    log_softmax_concat : torch.Tensor
        Concatenated per-token log-softmax over the vocab (for KL).
    targets_concat : torch.Tensor
        Concatenated per-token target ids (for sanity-checking the
        paired alignment across conditions).
    """
    import torch
    from analysis.common.data import iterate_owt2_val

    model.eval()
    per_token_ce_list: list[float] = []
    per_token_correct_list: list[int] = []
    per_token_target_log_prob_list: list[float] = []
    log_softmax_chunks: list["Any"] = []
    targets_chunks: list["Any"] = []
    n_tokens = 0

    ctx = freeze_hooks(
        model,
        freeze_mode=freeze_mode,
        freeze_target=freeze_target,
        freeze_active=freeze_active,
        seed=int(args.seed),
        module_path=args.module_path,
    )
    with torch.no_grad(), ctx:
        for batch_idx in range(int(args.n_batches)):
            start_offset = batch_idx * int(args.max_tokens)
            data_iter = iterate_owt2_val(
                args.data_dir,
                config,
                seq_length=int(args.seq_length),
                batch_size=int(args.batch_size),
                total_tokens=int(args.max_tokens),
                device=device,
                start_offset=start_offset,
                split="val",
            )
            for x, y in data_iter:
                outputs = model(x, targets=y, get_logits=True)
                logits = outputs["logits"]
                stats = _ce_and_accuracy_from_logits(logits, y)
                per_token_ce_list.extend(
                    stats["per_token_ce"].detach().cpu().tolist()
                )
                per_token_correct_list.extend(
                    stats["per_token_correct"].detach().cpu().tolist()
                )
                per_token_target_log_prob_list.extend(
                    stats["per_token_target_log_prob"].detach().cpu().tolist()
                )
                log_softmax_chunks.append(stats["log_softmax"].detach().cpu())
                targets_chunks.append(y.reshape(-1).detach().cpu())
                n_tokens += stats["B"] * stats["T"]

    ce_array = np.asarray(per_token_ce_list, dtype=np.float64)
    correct_array = np.asarray(per_token_correct_list, dtype=np.int64)

    metrics = FreezeRunMetrics(
        n_tokens=int(n_tokens),
        cross_entropy=float(ce_array.mean()) if ce_array.size > 0 else float("nan"),
        top1_accuracy=float(correct_array.mean()) if correct_array.size > 0 else float("nan"),
        per_token_ce=per_token_ce_list,
        per_token_correct=[int(v) for v in per_token_correct_list],
        per_token_log_softmax_top=per_token_target_log_prob_list,
    )

    log_softmax_concat = (
        log_softmax_chunks[0] if len(log_softmax_chunks) == 1
        else _concat_chunks(log_softmax_chunks)
    )
    targets_concat = (
        targets_chunks[0] if len(targets_chunks) == 1
        else _concat_chunks(targets_chunks)
    )
    return metrics, log_softmax_concat, targets_concat


def _concat_chunks(chunks: list["Any"]) -> "Any":
    """Concatenate a list of CPU tensors along dim 0 (defensive helper)."""
    import torch
    return torch.cat(chunks, dim=0)


def _kl_divergence(
    log_softmax_a: "Any",  # torch.Tensor (N, V)
    log_softmax_b: "Any",  # torch.Tensor (N, V)
) -> float:
    """Mean ``D_KL(P_a || P_b)`` averaged over the N token positions."""
    import torch
    softmax_a = log_softmax_a.exp()
    kl_per_token = (softmax_a * (log_softmax_a - log_softmax_b)).sum(dim=-1)
    return float(kl_per_token.mean().item())


# ---------------------------------------------------------------------------
# Bootstrap CI + Holm-Bonferroni
# ---------------------------------------------------------------------------


def bootstrap_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap CI on ``mean(a) - mean(b)``.

    Returns a dict with ``mean_delta``, ``ci_low``, ``ci_high``,
    ``p_value`` (two-sided fraction of bootstrap deltas with sign
    opposite to the observed mean delta).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "bootstrap_delta_ci: a and b must have the same length "
            f"(got {a.shape[0]} vs {b.shape[0]})"
        )
    n = a.shape[0]
    if n < 2:
        return {
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }

    observed_delta = float((a - b).mean())
    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = float((a[idx] - b[idx]).mean())

    ci_low = float(np.quantile(boot_deltas, alpha / 2.0))
    ci_high = float(np.quantile(boot_deltas, 1.0 - alpha / 2.0))

    # Two-sided p-value: fraction of bootstrap deltas at-or-beyond
    # zero on the side opposite to the observed mean. Multiplied by
    # two for two-sidedness; clamped to [0, 1].
    if observed_delta >= 0:
        p = float((boot_deltas <= 0.0).mean())
    else:
        p = float((boot_deltas >= 0.0).mean())
    p = min(1.0, 2.0 * p)

    return {
        "mean_delta": observed_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment for a family of p-values.

    Returns the adjusted p-values in the original order. Accepts
    ``nan`` inputs (passed through; they consume a rank but do not
    affect downstream comparisons).
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = p_arr.shape[0]
    if n == 0:
        return []

    order = np.argsort(p_arr, kind="mergesort")
    adjusted = np.empty(n, dtype=np.float64)
    running_max = 0.0
    for rank, original_idx in enumerate(order):
        raw_p = p_arr[original_idx]
        if np.isnan(raw_p):
            adjusted[original_idx] = raw_p
            continue
        adj = (n - rank) * raw_p
        adj = min(1.0, adj)
        adj = max(adj, running_max)
        running_max = adj
        adjusted[original_idx] = adj
    return [float(v) for v in adjusted]


# ---------------------------------------------------------------------------
# Per-repeat CE figure
# ---------------------------------------------------------------------------


def _plot_per_repeat_ce(
    per_condition: dict[str, dict[str, Any]],
    baseline_key: str,
    output_dir: str,
    n_repeat: int,
) -> str | None:
    """Save the per-repeat CE delta with 95 per cent CI band per condition.

    Each (mode, target) condition contributes one line; the y-axis is
    the mean per-token CE delta vs the baseline (``preserve``,
    target=baseline). The figure is grouped by the freeze target
    index on the x-axis. Returns the saved path on success;
    ``None`` if no condition has a target axis (e.g. baseline-only).
    """
    from analysis.common.plotting import setup_figure, savefig, palette_for_repeats

    target_axis = list(range(1, n_repeat + 1))
    if not target_axis:
        return None

    fig, ax = setup_figure(1, 1, size=(9.0, 5.0))
    palette = palette_for_repeats(max(2, len(VALID_FREEZE_MODES)))

    # Group by mode; one line per mode across freeze_target on the
    # x-axis. Conditions whose target is not in target_axis are
    # plotted with NaN (gap) so the line is still drawn for the
    # available targets.
    for mode_idx, mode in enumerate(VALID_FREEZE_MODES):
        means = np.full(len(target_axis), np.nan, dtype=np.float64)
        ci_low = np.full(len(target_axis), np.nan, dtype=np.float64)
        ci_high = np.full(len(target_axis), np.nan, dtype=np.float64)
        for t_idx, target in enumerate(target_axis):
            key = _condition_key(mode, target, freeze_active=True)
            if key not in per_condition:
                continue
            cell = per_condition[key]
            ci = cell.get("ce_delta_vs_baseline_ci", {})
            means[t_idx] = ci.get("mean_delta", float("nan"))
            ci_low[t_idx] = ci.get("ci_low", float("nan"))
            ci_high[t_idx] = ci.get("ci_high", float("nan"))

        colour = palette[mode_idx % len(palette)]
        ax.plot(target_axis, means, marker="o", color=colour, label=mode)
        ax.fill_between(
            target_axis, ci_low, ci_high, color=colour, alpha=0.18,
        )

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Freeze target (1-indexed forward-loop repeat)")
    ax.set_ylabel("Per-token CE delta vs baseline (preserve, t=1)")
    ax.set_title(
        "Depth-embedding freeze: per-repeat CE delta with 95 per cent CI"
    )
    ax.set_xticks(target_axis)
    ax.legend(title="Freeze mode")
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "depth_emb_freeze_per_repeat.png")
    savefig(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------


def _condition_key(mode: str, target: int, freeze_active: bool) -> str:
    """Stable key for the per-condition results dict."""
    if not freeze_active:
        return f"unfreeze:{mode}:t{target}"
    return f"freeze:{mode}:t{target}"


def _bool_from_str(s: str) -> bool:
    """Parse the ``--freeze-active`` argument value."""
    return str(s).strip().lower() == "true"


def analyse_one_condition(args: argparse.Namespace) -> dict[str, Any]:
    """Per-invocation entry point: run ONE condition + write JSON.

    The HPC sweep invokes this script once per condition with a
    distinct ``--output-dir``. The aggregate ANOVA across the 8
    conditions is computed CPU-side from the per-condition JSON
    artefacts in a downstream step (per ``iridis/analyze-lncot+adm/
    job-cpu.sh`` post-processing). This function therefore writes
    the per-condition data plus a small comparison block against the
    in-call baseline (``preserve`` at the same target) so each
    artefact is self-contained.
    """
    import torch  # noqa: F401
    from analysis.common.loader import load_model_from_checkpoint

    device = (
        args.device
        if (args.device == "cpu" or torch.cuda.is_available())
        else "cpu"
    )
    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode=args.config_mode,
        device=device,
        module_path=args.module_path,
    )
    n_repeat = int(getattr(model, "n_repeat", 1) or 1)

    freeze_active = _bool_from_str(args.freeze_active)
    freeze_mode = str(args.freeze_mode)
    freeze_target = int(args.freeze_target)

    # Run the requested condition.
    target_metrics, target_log_softmax, target_targets = _run_one_condition(
        model, config, args,
        freeze_mode=freeze_mode,
        freeze_target=freeze_target,
        freeze_active=freeze_active,
        device=device,
    )

    # Run the in-call baseline (preserve, same target, freeze_active=True).
    # This is the comparator for the per-condition delta. The full
    # 8-condition cross-comparison is left to the CPU post-step.
    baseline_metrics, baseline_log_softmax, baseline_targets = _run_one_condition(
        model, config, args,
        freeze_mode="preserve",
        freeze_target=freeze_target,
        freeze_active=True,
        device=device,
    )

    # Sanity: targets must be aligned across the two runs (same data
    # iterator, same seed). If not, abort -- the paired delta
    # would be meaningless.
    if int(target_targets.shape[0]) != int(baseline_targets.shape[0]):
        raise RuntimeError(
            "analyse_one_condition: target / baseline forward passes "
            "yielded different token counts "
            f"({int(target_targets.shape[0])} vs "
            f"{int(baseline_targets.shape[0])}); cannot pair."
        )

    # CE delta (paired bootstrap on per-token CE).
    ce_delta_ci = bootstrap_delta_ci(
        np.asarray(target_metrics.per_token_ce, dtype=np.float64),
        np.asarray(baseline_metrics.per_token_ce, dtype=np.float64),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    # Accuracy delta (paired bootstrap on 0/1 correctness).
    acc_delta_ci = bootstrap_delta_ci(
        np.asarray(target_metrics.per_token_correct, dtype=np.float64),
        np.asarray(baseline_metrics.per_token_correct, dtype=np.float64),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 1,
    )
    # Full-distribution KL vs baseline.
    kl_vs_baseline = _kl_divergence(target_log_softmax, baseline_log_softmax)

    condition_key = _condition_key(
        freeze_mode, freeze_target, freeze_active=freeze_active
    )

    per_condition = {
        condition_key: {
            "freeze_mode": freeze_mode,
            "freeze_target": freeze_target,
            "freeze_active": freeze_active,
            "n_tokens": target_metrics.n_tokens,
            "cross_entropy": target_metrics.cross_entropy,
            "top1_accuracy": target_metrics.top1_accuracy,
            "ce_delta_vs_baseline_ci": ce_delta_ci,
            "acc_delta_vs_baseline_ci": acc_delta_ci,
            "kl_vs_baseline": kl_vs_baseline,
        },
        "baseline:preserve:t" + str(freeze_target): {
            "freeze_mode": "preserve",
            "freeze_target": freeze_target,
            "freeze_active": True,
            "n_tokens": baseline_metrics.n_tokens,
            "cross_entropy": baseline_metrics.cross_entropy,
            "top1_accuracy": baseline_metrics.top1_accuracy,
            "ce_delta_vs_baseline_ci": None,
            "acc_delta_vs_baseline_ci": None,
            "kl_vs_baseline": 0.0,
        },
    }

    # Holm-Bonferroni applied across the two delta tests of THIS
    # invocation (CE delta + accuracy delta). The full 8-condition
    # family-wise correction is the post-step's responsibility.
    p_values = [ce_delta_ci["p_value"], acc_delta_ci["p_value"]]
    p_adj = holm_bonferroni(p_values)

    # H0 falsification call (per RQ8 spec, post-W3 wording): reject if
    # this is a freeze=True, kind=zero condition AND the CE delta is
    # positive AND the Holm-corrected paired bootstrap p < 0.01. This
    # is the per-condition view; the cross-condition Holm correction
    # over the 8-condition family lives in the CPU post-step.
    h0_rejected = bool(
        freeze_active
        and freeze_mode == "zero"
        and ce_delta_ci["mean_delta"] > 0.0
        and not np.isnan(p_adj[0])
        and p_adj[0] < 0.01
    )

    return {
        "checkpoint": args.checkpoint,
        "checkpoint_file": args.checkpoint_file,
        "seed": int(args.seed),
        "n_bootstrap": int(args.n_bootstrap),
        "n_batches": int(args.n_batches),
        "max_tokens": int(args.max_tokens),
        "seq_length": int(args.seq_length),
        "batch_size": int(args.batch_size),
        "n_repeat": n_repeat,
        "freeze_mode": freeze_mode,
        "freeze_target": freeze_target,
        "freeze_active": freeze_active,
        "module_path": args.module_path,
        "per_condition": per_condition,
        "h0_per_condition": {
            "rejected": h0_rejected,
            "rule": (
                "freeze=true, kind=zero, CE_delta>0 vs preserve, "
                "Holm-corrected paired-bootstrap p<0.01"
            ),
            "p_values_raw": p_values,
            "p_values_holm": p_adj,
        },
    }


# ---------------------------------------------------------------------------
# Smoke test (pure-numpy: bootstrap + Holm sanity)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Sanity-check the bootstrap + Holm-Bonferroni helpers."""
    rng = np.random.default_rng(2357)
    a = rng.normal(loc=0.5, scale=1.0, size=128)
    b = rng.normal(loc=0.0, scale=1.0, size=128)

    ci = bootstrap_delta_ci(a, b, n_bootstrap=500, seed=7)
    assert ci["mean_delta"] > 0.0, f"expected positive delta, got {ci}"
    assert ci["ci_low"] < ci["ci_high"], f"degenerate CI: {ci}"
    # With n=128 and a 0.5-sigma effect, the 500-bootstrap p-value is
    # typically <0.05; we soft-check that the bootstrap returned a
    # finite p in [0, 1].
    assert 0.0 <= ci["p_value"] <= 1.0, f"out-of-range p: {ci}"

    null = bootstrap_delta_ci(a, a, n_bootstrap=500, seed=7)
    assert abs(null["mean_delta"]) < 1e-12, f"null delta nonzero: {null}"

    p_raw = [0.001, 0.01, 0.04, 0.5]
    p_adj = holm_bonferroni(p_raw)
    # Holm: smallest p multiplied by family size = 4*0.001 = 0.004;
    # next: 3*0.01 = 0.03; etc. Adjusted values are non-decreasing.
    assert p_adj[0] <= p_adj[1] <= p_adj[2] <= p_adj[3], (
        f"Holm output not non-decreasing: {p_adj}"
    )
    assert abs(p_adj[0] - 0.004) < 1e-9, f"smallest Holm-adj wrong: {p_adj}"

    # NaN handling.
    p_adj_nan = holm_bonferroni([0.01, float("nan"), 0.5])
    assert np.isnan(p_adj_nan[1]), f"NaN dropped: {p_adj_nan}"

    print("RQ8 smoke test PASS")
    print(f"  signal CI: {ci}")
    print(f"  null CI:   {null}")
    print(f"  Holm adj:  {p_adj}")


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

    import torch
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    os.makedirs(args.output_dir, exist_ok=True)
    results = analyse_one_condition(args)

    out_json = os.path.join(args.output_dir, "depth_emb_freeze_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"RQ8 condition results written to {out_json}")

    # Per-repeat CE figure: plotted from the in-call data only (one
    # mode/target pair). The cross-condition figure is the post-step's
    # responsibility (it has access to all 8 JSON artefacts).
    fig_path = _plot_per_repeat_ce(
        results["per_condition"],
        baseline_key="baseline:preserve:t" + str(int(args.freeze_target)),
        output_dir=args.output_dir,
        n_repeat=int(results["n_repeat"]),
    )
    if fig_path is not None:
        print(f"RQ8 per-repeat figure written to {fig_path}")


if __name__ == "__main__":
    main()
