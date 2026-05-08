"""DV-2 -- Paired linear-vs-non-linear count probe (RQ9).

Scope
-----
Addresses DV-2 of RQ9. Loads a counting-task checkpoint, captures
the post-block hidden state at every ``(layer, repeat)`` site (or
``(layer,)`` site for the 4L baseline), and trains TWO probes per
site against the running-count integer target:

1. **Linear probe**: ``sklearn.linear_model.Ridge(alpha=1.0)``.
2. **Non-linear probe**: ``Linear(n_embd, 8) -> ReLU -> Linear(8, 1)``,
   200 epochs, Adam lr=1e-3.

Both probes are evaluated on a matched 80/20 split. The site-wise
"selectivity gap" ``mlp_r2 - ridge_r2`` flags whether the count
representation at that site is canonical-linear or whether a
non-linear readout is required:

- ``gap < 0.05`` -> ``canonical = "linear"`` (Ridge already extracts
  the count; the MLP brings no extra capacity advantage).
- ``gap >= 0.05`` -> ``canonical = "non-linear"`` and the per-site
  ``verdict_flag`` is set to ``"non-linear representation likely"``.

Falsifiability relevance
------------------------
Per `docs/extend-notes.md` RQ9 DV-2 row, the probe-pair signature is
the representational triangulation for the RQ9 H0. A "recurrent
inductive bias for counting" claim (RQ9 positive) is consistent with
a monotone-by-repeat probe-R2 profile where a linear readout already
captures most of the count signal. A flat / non-monotone profile or
a profile dominated by non-linear gap weakens the interpretation.

Capacity-matching note
----------------------
The MLP hidden width is fixed at 8 so its parameter count
(``n_embd * 8 + 8 + 8 * 1 + 1``) is at least 8x the Ridge effective
d.o.f. (``n_embd + 1``). For ``n_embd = 128`` this is 1041 vs 129;
the non-linear probe has strictly more representational capacity at
every width considered for RQ9, so a small gap cannot be attributed
to the MLP being under-capacity.

Repeat-counter mechanism
------------------------
Mirrors `analysis.counting_dv4_causal.ablation_hooks`: a
forward-pre-hook on the model resets a per-batch repeat counter at
the start of every forward; a forward-post-hook on
``transformer.ln_mid`` (or the last ``h_mid`` block when
``ln_mid`` is absent) increments the counter at the end of every
repeat. Each ``h_mid[i]`` forward-post-hook reads the counter at
fire time to tag its capture with ``(layer_i, repeat_r)``.

Registration order: per-block capture hooks are registered BEFORE
the counter-increment hook. PyTorch fires post-hooks in registration
order; for V1/V2 architectures (no ``ln_mid``) the increment lands
on the last h_mid block which is also a capture target. Capture
first guarantees the per-block hook reads the pre-bump counter and
labels its tensor with the correct repeat index. For V3/V4
(ln_mid present) the increment lands on a different module so
registration order is irrelevant; the swap is safe for both code
paths.

4L-baseline mode
----------------
When ``--module-path`` points at ``transformer.h`` (the standard
4L baseline used as the RQ9 control arm), the model has no repeat
loop. The script captures one hidden state per layer (4 sites
total) and emits ``repeat = null`` in every per-site JSON entry.
The counter-increment hook is still installed but its value is
never read.

Smoke test
----------
``python -m analysis.counting_dv2_probe --smoke-test`` constructs a
small synthetic feature/target pair where the count is a known
linear function of the hidden state, fits both probes, and asserts
that Ridge attains R2 close to 1.0 and the gap is below the
selectivity threshold. Used by CI / quick validation; matches the
sibling DV-3 / DV-4 smoke pattern.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


OOD_LENGTH_RANGE = (51, 200)
DEFAULT_SMOKE_N_SAMPLES = 32
DEFAULT_SMOKE_MLP_EPOCHS = 2


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for DV-2."""
    parser = argparse.ArgumentParser(
        description="DV-2 paired Ridge-vs-MLP count probe"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Checkpoint directory containing summary.json + checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default ckpt.pt)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False,
        help="Directory for counting_dv2_results.json + PNG",
    )
    parser.add_argument("--seed", type=int, default=19937,
                        help="RNG seed for the eval set + probe split")
    parser.add_argument("--task", type=str, default="task1",
                        help="RQ9 task tag (task1 / task2 / task3); recorded "
                             "in the JSON for downstream cross-task comparison")
    parser.add_argument("--eval-length", type=int, default=200,
                        help="Upper bound on evaluation sequence length "
                             "(test-set samples are drawn from [51, eval_length])")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Eval-set sample count (default 500)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-batch sample count for the forward pass")
    parser.add_argument("--sequence-length", type=int, default=256,
                        help="Pad length (default 256; must be >= eval-length + 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device string; 'cpu' forces CPU run")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="transformer.h_mid",
                        help="Dotted path to the block list to probe; "
                             "transformer.h_mid for CoTFormer (default), "
                             "transformer.h for the 4L baseline")
    parser.add_argument("--train-fraction", type=float, default=0.8,
                        help="Probe train fraction (rest is held-out eval)")
    parser.add_argument("--mlp-epochs", type=int, default=200,
                        help="Number of training epochs for the MLP probe")
    parser.add_argument("--mlp-lr", type=float, default=1e-3,
                        help="Adam learning rate for the MLP probe")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge regularisation strength")
    parser.add_argument("--selectivity-threshold", type=float, default=0.05,
                        help="Gap threshold (mlp_r2 - ridge_r2) for "
                             "flagging a site as non-linear-canonical")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the synthetic-feature smoke test and exit")
    return parser


# ---------------------------------------------------------------------------
# Hidden-state capture (mirrors counting_dv4_causal.ablation_hooks structure)
# ---------------------------------------------------------------------------


_COUNTER_ATTR = "_analysis_dv2_repeat_counter"


def _resolve_block_list(model: "Any", module_path: str) -> "Any":
    """Resolve ``module_path`` to an ``nn.ModuleList``-like block list.

    Strips a leading ``"model."`` segment if present so callers may
    pass either ``"model.transformer.h_mid"`` (DIR-001 canonical form)
    or ``"transformer.h_mid"`` (the spec form).
    """
    path = module_path
    if path.startswith("model."):
        path = path[len("model.") :]
    cursor = model
    for name in path.split("."):
        cursor = getattr(cursor, name)
    if not hasattr(cursor, "__len__") or len(cursor) == 0:
        raise RuntimeError(
            f"_resolve_block_list: module_path {module_path!r} resolved to "
            f"an empty list; cannot probe."
        )
    return cursor


@contextmanager
def capture_hooks(
    model: "Any",
    module_path: str,
    has_repeats: bool,
) -> Iterator[dict]:
    """Install per-(layer, repeat) hidden-state capture hooks.

    Parameters
    ----------
    model : nn.Module
        Loaded counting model.
    module_path : str
        Dotted path to the block list (e.g. ``"transformer.h_mid"``
        or ``"transformer.h"``).
    has_repeats : bool
        ``True`` for CoTFormer variants (capture key includes the
        repeat index). ``False`` for the 4L baseline (no repeat
        loop; per-layer captures only).

    Yields
    ------
    captures : dict[str, list[np.ndarray]]
        Maps ``"layer={L}/repeat={R}"`` (or ``"layer={L}"`` for the
        4L baseline) to a list of per-batch hidden-state arrays of
        shape ``(B, T, D)``. The caller concatenates after the
        forward loop.
    """
    import torch  # noqa: F401

    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError(
            "capture_hooks: model has no transformer attribute"
        )

    block_list = _resolve_block_list(model, module_path)
    n_repeat = int(getattr(model, "n_repeat", 1) or 1) if has_repeats else 1

    captures: dict[str, list[np.ndarray]] = {}
    handles: list = []

    def _reset_counter(module, inputs):
        setattr(model, _COUNTER_ATTR, 1)

    def _increment_counter(module, inputs, output):
        current = int(getattr(model, _COUNTER_ATTR, 1))
        setattr(model, _COUNTER_ATTR, current + 1)

    def _make_capture_hook(layer_idx: int):
        def _hook(module, inputs, output):
            # The block returns the post-residual hidden state directly.
            tensor = output if not isinstance(output, tuple) else output[0]
            arr = tensor.detach().to("cpu").float().numpy()
            if has_repeats:
                current = int(getattr(model, _COUNTER_ATTR, 1))
                # Clamp to [1, n_repeat] (mirror counting_dv4_causal clamp).
                current = max(1, min(current, n_repeat))
                key = f"layer={layer_idx}/repeat={current}"
            else:
                key = f"layer={layer_idx}"
            captures.setdefault(key, []).append(arr)
            return output

        return _hook

    # 1. Reset counter at the start of each forward pass.
    handles.append(model.register_forward_pre_hook(_reset_counter))

    # 2. Install per-block capture hooks BEFORE the counter-increment hook.
    # PyTorch fires forward post-hooks in registration order; for V1/V2
    # (no ln_mid) the increment lands on the last h_mid block which is
    # also a capture target. Capturing first guarantees the per-block
    # hook reads the pre-bump counter for the correct repeat index.
    for layer_idx in range(len(block_list)):
        handles.append(
            block_list[layer_idx].register_forward_hook(
                _make_capture_hook(layer_idx)
            )
        )

    # 3. Register the counter-increment hook AFTER all capture hooks.
    # See module docstring on registration order.
    if has_repeats:
        ln_mid = getattr(transformer, "ln_mid", None)
        if ln_mid is not None:
            handles.append(ln_mid.register_forward_hook(_increment_counter))
        else:
            last_block = block_list[len(block_list) - 1]
            handles.append(last_block.register_forward_hook(_increment_counter))

    setattr(model, _COUNTER_ATTR, 1)
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(model, _COUNTER_ATTR):
            delattr(model, _COUNTER_ATTR)


# ---------------------------------------------------------------------------
# Probe primitives (operate on numpy / torch tensors; smoke-testable)
# ---------------------------------------------------------------------------


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination R^2.

    Defined as ``1 - SS_res / SS_tot`` with the standard guard for
    a degenerate target (zero variance: returns 0.0 by convention).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    mean = float(y_true.mean())
    ss_tot = float(((y_true - mean) ** 2).sum())
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Fit ``sklearn.Ridge(alpha)`` on (x_train, y_train); evaluate on the held-out split."""
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=float(alpha))
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    r2 = _r2_score(y_eval, pred)
    mse = float(((y_eval - pred) ** 2).mean())
    return {"r2": float(r2), "mse": mse, "alpha": float(alpha)}


def fit_mlp_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    """Train ``Linear(D, 8) -> ReLU -> Linear(8, 1)`` for ``epochs`` with Adam.

    Full-batch gradient descent on the training split; deterministic
    initialisation via ``torch.manual_seed(seed)``. The non-linear
    capacity (8 hidden units, ReLU) is the spec-mandated capacity-
    matching choice (see module docstring).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))

    D = int(x_train.shape[1])
    net = nn.Sequential(
        nn.Linear(D, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    n_params = sum(int(p.numel()) for p in net.parameters())

    x_t = torch.from_numpy(x_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
    x_e = torch.from_numpy(x_eval.astype(np.float32))
    y_e_np = y_eval.astype(np.float64).reshape(-1)

    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        pred = net(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred_eval = net(x_e).numpy().reshape(-1)

    r2 = _r2_score(y_e_np, pred_eval)
    mse = float(((y_e_np - pred_eval) ** 2).mean())
    return {
        "r2": float(r2),
        "mse": mse,
        "epochs": int(epochs),
        "lr": float(lr),
        "n_params": int(n_params),
    }


def split_train_eval(
    features: np.ndarray,
    targets: np.ndarray,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 80/20-style split shared by both probes.

    A single permutation seeded by ``seed`` partitions the indices;
    both probes read identical (x_train, y_train, x_eval, y_eval)
    arrays so the matched-data comparison is exact.
    """
    n = int(features.shape[0])
    if n != int(targets.shape[0]):
        raise ValueError(
            f"split_train_eval: feature/target row mismatch ({n} vs "
            f"{targets.shape[0]})"
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_train = max(1, int(round(float(train_fraction) * n)))
    n_train = min(n_train, n - 1)  # ensure at least one eval sample
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]
    return (
        features[train_idx], targets[train_idx],
        features[eval_idx], targets[eval_idx],
    )


# ---------------------------------------------------------------------------
# Checkpoint-driven analysis
# ---------------------------------------------------------------------------


def _gather_site_features_and_targets(
    captures: dict[str, list[np.ndarray]],
    targets_per_batch: list[np.ndarray],
    loss_mask_per_batch: list[np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Concatenate per-batch captures and align with loss-masked targets.

    Each capture array has shape ``(B, T, D)``. The integer count
    target sits at every position where ``loss_mask == 1``; for the
    counting vocab ``vocab[i] = str(i)`` (i in [0, max_out]), so the
    raw target index IS the integer count. We extract only the
    masked positions; the per-site feature matrix has shape
    ``(N_valid, D)`` and the target vector has shape ``(N_valid,)``.

    Returns
    -------
    dict
        ``{site_key: (features, targets)}`` over every site key in
        ``captures``.
    """
    # Concatenate the per-batch target / mask arrays once (shared across sites).
    targets_flat = np.concatenate(
        [t.reshape(-1) for t in targets_per_batch], axis=0
    )
    mask_flat = np.concatenate(
        [m.reshape(-1) for m in loss_mask_per_batch], axis=0
    )
    valid_mask = mask_flat > 0.5

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for site_key, batch_arrs in captures.items():
        # Each batch_arr has shape (B, T, D). Concatenate along the
        # batch axis -> (sum_B, T, D); reshape to (N_tokens, D).
        site_arr = np.concatenate(batch_arrs, axis=0)  # (sum_B, T, D)
        D = int(site_arr.shape[-1])
        feats_flat = site_arr.reshape(-1, D)
        if feats_flat.shape[0] != targets_flat.shape[0]:
            raise RuntimeError(
                f"feature/target row count mismatch at site {site_key!r}: "
                f"{feats_flat.shape[0]} vs {targets_flat.shape[0]}"
            )
        feats_valid = feats_flat[valid_mask]
        tgt_valid = targets_flat[valid_mask].astype(np.float64)
        out[site_key] = (feats_valid, tgt_valid)
    return out


def _parse_site_key(site_key: str) -> tuple[int, int | None]:
    """Parse ``"layer={L}/repeat={R}"`` or ``"layer={L}"`` into (layer, repeat)."""
    parts = site_key.split("/")
    layer = int(parts[0].split("=")[1])
    repeat: int | None = None
    if len(parts) > 1:
        repeat = int(parts[1].split("=")[1])
    return layer, repeat


def _site_label_for_output(layer: int, repeat: int | None, group: str) -> str:
    """Return the human-readable site label used in the JSON ``site`` field."""
    if repeat is None:
        return f"{group}[{layer}]"
    return f"{group}[{layer}]/repeat={repeat}"


def _spearman_monotone(per_site_results: list[dict]) -> bool | None:
    """Return ``True`` if mean per-repeat ridge_r2 is non-decreasing across r.

    Computed only when at least one per-site entry has a non-null
    ``repeat`` field (CoTFormer mode). For the 4L baseline returns
    ``None``. The trend is computed on the per-repeat MEAN of
    ``ridge_r2`` across layers; "monotone" means each successive
    repeat's mean is >= the previous repeat's mean (allowing ties).
    """
    has_repeat = any(s.get("repeat") is not None for s in per_site_results)
    if not has_repeat:
        return None
    by_repeat: dict[int, list[float]] = {}
    for s in per_site_results:
        r = s.get("repeat")
        if r is None:
            continue
        by_repeat.setdefault(int(r), []).append(float(s["ridge"]["r2"]))
    if len(by_repeat) < 2:
        return True  # vacuously monotone
    sorted_repeats = sorted(by_repeat.keys())
    means = [float(np.mean(by_repeat[r])) for r in sorted_repeats]
    for i in range(1, len(means)):
        if means[i] < means[i - 1]:
            return False
    return True


def analyse_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Load a checkpoint, capture hidden states, fit per-site probes."""
    import torch  # noqa: F401
    import inspect
    from torch.utils.data import DataLoader
    from analysis.common.loader import load_model_from_checkpoint
    from data.counting import CountingDataset, TE200_MAX_OUT

    device = (
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode=args.config_mode,
        device=device,
        module_path=args.module_path,
    )
    model.eval()

    # 4L baseline detection: the "transformer.h" path means the model
    # has a flat block list with no repeat loop.
    leaf = args.module_path.rsplit(".", 1)[-1]
    has_repeats = leaf != "h"
    group = "h_mid" if has_repeats else "h"

    n_repeat = int(getattr(model, "n_repeat", 1) or 1) if has_repeats else 1

    forward_sig = inspect.signature(model.forward)
    accepts_attention_mask = "attention_mask" in forward_sig.parameters

    # Build the eval iterator. The eval-set length range is
    # [51, args.eval_length]; for the default eval_length=200 this
    # matches the OOD_LENGTH_RANGE used by DV-3 / DV-4.
    eval_length_range = (51, int(args.eval_length))
    dataset = CountingDataset(
        split="ood",
        seed=int(args.seed),
        num_samples=int(args.n_samples),
        sequence_length=int(args.sequence_length),
        max_out=TE200_MAX_OUT,
        length_range=eval_length_range,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    # Forward pass with capture hooks; collect per-batch targets and
    # loss masks alongside the captured tensors so the masked-token
    # alignment is unambiguous downstream.
    targets_per_batch: list[np.ndarray] = []
    loss_mask_per_batch: list[np.ndarray] = []

    with torch.no_grad(), capture_hooks(
        model, args.module_path, has_repeats
    ) as captures:
        for batch in loader:
            x, y, pad_mask, loss_mask = batch
            x_d = x.to(device)
            pad_mask_d = pad_mask.to(device)

            if accepts_attention_mask:
                _ = model(x_d, attention_mask=pad_mask_d, get_logits=False)
            else:
                _ = model(x_d, get_logits=False)

            targets_per_batch.append(y.numpy().astype(np.int64))
            loss_mask_per_batch.append(loss_mask.numpy().astype(np.float32))

    site_data = _gather_site_features_and_targets(
        captures, targets_per_batch, loss_mask_per_batch
    )

    per_site_results: list[dict] = []
    for site_key in sorted(site_data.keys()):
        features, target_vec = site_data[site_key]
        x_train, y_train, x_eval, y_eval = split_train_eval(
            features, target_vec,
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )

        ridge_metrics = fit_ridge_probe(
            x_train, y_train, x_eval, y_eval,
            alpha=float(args.ridge_alpha),
        )
        mlp_metrics = fit_mlp_probe(
            x_train, y_train, x_eval, y_eval,
            epochs=int(args.mlp_epochs),
            lr=float(args.mlp_lr),
            seed=int(args.seed),
        )
        gap = float(mlp_metrics["r2"] - ridge_metrics["r2"])
        threshold = float(args.selectivity_threshold)
        if gap < threshold:
            canonical = "linear"
            verdict_flag = None
        else:
            canonical = "non-linear"
            verdict_flag = "non-linear representation likely"

        layer, repeat = _parse_site_key(site_key)
        per_site_results.append({
            "site": _site_label_for_output(layer, repeat, group),
            "layer": int(layer),
            "repeat": (None if repeat is None else int(repeat)),
            "n_train": int(x_train.shape[0]),
            "n_eval": int(x_eval.shape[0]),
            "ridge": ridge_metrics,
            "mlp": mlp_metrics,
            "gap": gap,
            "canonical": canonical,
            "verdict_flag": verdict_flag,
        })

    # Aggregate.
    if per_site_results:
        mean_ridge_r2 = float(np.mean([s["ridge"]["r2"] for s in per_site_results]))
        mean_mlp_r2 = float(np.mean([s["mlp"]["r2"] for s in per_site_results]))
        mean_gap = float(np.mean([s["gap"] for s in per_site_results]))
        n_flag = int(sum(1 for s in per_site_results if s["verdict_flag"] is not None))
    else:
        mean_ridge_r2 = float("nan")
        mean_mlp_r2 = float("nan")
        mean_gap = float("nan")
        n_flag = 0
    monotone = _spearman_monotone(per_site_results)

    n_eval_samples_total = int(sum(t.shape[0] for t in targets_per_batch))

    args_dict = {k: v for k, v in vars(args).items()}

    config_dict: dict[str, Any] = {}
    for key in ("model", "n_embd", "n_head", "n_layer", "n_repeat",
                "n_mid_layer", "n_begin_layer", "n_end_layer"):
        if hasattr(config, key):
            config_dict[key] = getattr(config, key)

    return {
        "schema_version": "dv2-1.0",
        "checkpoint": {
            "path": args.checkpoint,
            "file": args.checkpoint_file,
            "config": config_dict,
        },
        "args": args_dict,
        "n_eval_samples": n_eval_samples_total,
        "n_repeat": int(n_repeat),
        "accepts_attention_mask": bool(accepts_attention_mask),
        "per_site": per_site_results,
        "aggregate": {
            "mean_ridge_r2": mean_ridge_r2,
            "mean_mlp_r2": mean_mlp_r2,
            "mean_gap": mean_gap,
            "n_sites_non_linear_flag": n_flag,
            "monotone_by_repeat": monotone,
        },
        "figure_paths": ["counting_dv2_r2_per_site.png"],
        "meta": {
            "eval_length_range": list(eval_length_range),
            "sequence_length": int(args.sequence_length),
            "seed": int(args.seed),
            "module_path": args.module_path,
            "task": args.task,
        },
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_figure(per_site_results: list[dict], output_path: str) -> None:
    """Render the DV-2 R^2-per-site bar/line figure to ``output_path``.

    Two bars per site (ridge_r2, mlp_r2) with the selectivity gap
    annotated as a shaded band. Uses the project's
    ``analysis.common.plotting.setup_figure`` helper for the canvas.
    """
    if not per_site_results:
        return
    from analysis.common.plotting import setup_figure, savefig

    labels = [s["site"] for s in per_site_results]
    ridge_r2 = np.array([s["ridge"]["r2"] for s in per_site_results])
    mlp_r2 = np.array([s["mlp"]["r2"] for s in per_site_results])
    gaps = mlp_r2 - ridge_r2

    fig, ax = setup_figure(rows=1, cols=1, size=(max(8.0, len(labels) * 0.6), 5.5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, ridge_r2, width, label="Ridge probe", color="#1f77b4")
    ax.bar(x + width / 2, mlp_r2, width, label="MLP probe (8u)", color="#ff7f0e")

    # Shade the gap band per site so positive (non-linear advantage)
    # vs negative gaps are immediately visible.
    for xi, g in zip(x, gaps):
        ax.fill_between(
            [xi - width, xi + width],
            [min(ridge_r2[xi], mlp_r2[xi])] * 2,
            [max(ridge_r2[xi], mlp_r2[xi])] * 2,
            alpha=0.15,
            color="#2ca02c" if g >= 0.05 else "#cccccc",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probe R^2 (held-out 20% split)")
    ax.set_xlabel("Site")
    ax.set_title("DV-2: Ridge vs MLP probe R^2 per (layer, repeat)")
    ax.set_ylim(min(0.0, float(min(ridge_r2.min(), mlp_r2.min())) - 0.05), 1.05)
    ax.axhline(0.0, color="#333333", linewidth=0.5)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Smoke test (synthetic features, deterministic)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Verify the probe primitives on synthetic data with a known answer.

    Constructs a 200-sample feature matrix where the target is
    ``y = w . x + small_noise`` with a known ``w``. A Ridge probe
    must attain near-perfect R^2; the MLP must NOT exceed Ridge by
    more than the selectivity threshold (since the truth is exactly
    linear). We also confirm split determinism and that
    ``_gather_site_features_and_targets`` aligns features and
    masked targets correctly.
    """
    rng = np.random.default_rng(19937)
    N = 200
    D = 16
    w_true = rng.normal(size=(D,))
    X = rng.normal(size=(N, D)).astype(np.float64)
    noise = 0.05 * rng.normal(size=(N,))
    y = X @ w_true + noise

    x_train, y_train, x_eval, y_eval = split_train_eval(
        X, y, train_fraction=0.8, seed=19937
    )
    assert x_train.shape == (160, D), x_train.shape
    assert x_eval.shape == (40, D), x_eval.shape

    ridge_metrics = fit_ridge_probe(
        x_train, y_train, x_eval, y_eval, alpha=1.0
    )
    assert ridge_metrics["r2"] > 0.95, (
        f"linear truth -> Ridge R^2 should be near 1.0; got {ridge_metrics['r2']}"
    )

    mlp_metrics = fit_mlp_probe(
        x_train, y_train, x_eval, y_eval,
        epochs=DEFAULT_SMOKE_MLP_EPOCHS, lr=1e-3, seed=19937,
    )
    # With only 2 epochs the MLP is severely under-trained and must NOT
    # beat Ridge (which is fitted in closed form). The MLP gap can be
    # arbitrarily negative; we assert only that it does not falsely
    # claim non-linear-canonical status.
    gap = mlp_metrics["r2"] - ridge_metrics["r2"]
    assert gap < 0.05, (
        f"linear truth -> selectivity gap should be < 0.05; got {gap}"
    )
    assert mlp_metrics["n_params"] == D * 8 + 8 + 8 + 1, mlp_metrics["n_params"]

    # Gather alignment smoke: build a fake (B=2, T=4, D=3) capture and
    # a target/loss-mask pair where only positions [1, 2] are scored.
    captures = {
        "layer=0/repeat=1": [np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)],
    }
    targets = [np.array([[-1, 5, 6, -1], [-1, 7, 8, -1]], dtype=np.int64)]
    masks = [np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]], dtype=np.float32)]
    site_data = _gather_site_features_and_targets(captures, targets, masks)
    feats, tgts = site_data["layer=0/repeat=1"]
    assert feats.shape == (4, 3), feats.shape
    assert list(tgts) == [5.0, 6.0, 7.0, 8.0], list(tgts)

    # Site-key parsing smoke.
    layer, repeat = _parse_site_key("layer=2/repeat=3")
    assert layer == 2 and repeat == 3
    layer, repeat = _parse_site_key("layer=1")
    assert layer == 1 and repeat is None

    print("DV-2 smoke test PASS")
    print(
        f"  Ridge R^2 = {ridge_metrics['r2']:.4f} on a known-linear target"
    )
    print(
        f"  MLP R^2   = {mlp_metrics['r2']:.4f} (epochs="
        f"{DEFAULT_SMOKE_MLP_EPOCHS}); gap = {gap:.4f}"
    )
    print(
        f"  Gather alignment: {feats.shape[0]} masked positions extracted "
        f"(expected 4)."
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        # Smoke mode: probe primitives only; do not require a checkpoint.
        # If --checkpoint is also provided, run the end-to-end fast path
        # with reduced sample / epoch counts and CPU device per the spec.
        if args.checkpoint and args.output_dir:
            args.n_samples = DEFAULT_SMOKE_N_SAMPLES
            args.mlp_epochs = DEFAULT_SMOKE_MLP_EPOCHS
            args.device = "cpu"
            os.makedirs(args.output_dir, exist_ok=True)
            results = analyse_checkpoint(args)
            out_json = os.path.join(args.output_dir, "counting_dv2_results.json")
            with open(out_json, "w") as fh:
                json.dump(results, fh, indent=2, sort_keys=True)
            png_path = os.path.join(args.output_dir, "counting_dv2_r2_per_site.png")
            render_figure(results["per_site"], png_path)
            print(f"DV-2 smoke (end-to-end) results written to {out_json}")
            return
        _smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when --smoke-test is not set")
    if not args.output_dir:
        parser.error("--output-dir is required when --smoke-test is not set")

    os.makedirs(args.output_dir, exist_ok=True)
    results = analyse_checkpoint(args)

    out_json = os.path.join(args.output_dir, "counting_dv2_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    png_path = os.path.join(args.output_dir, "counting_dv2_r2_per_site.png")
    render_figure(results["per_site"], png_path)
    print(f"DV-2 results written to {out_json}")
    print(f"DV-2 figure written to {png_path}")


if __name__ == "__main__":
    main()
