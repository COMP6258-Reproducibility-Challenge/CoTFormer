import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, device="cpu"):
    x, y = next(dataloader)
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def get_counting_batch(dataloader, device="cpu"):
    """BLOCKER 4 counting batch fetcher.

    Unpacks a 4-tuple ``(tokens, targets, pad_mask, loss_mask)`` produced by
    ``data.counting.CountingDataset``. ``pad_mask`` is passed to the model as
    ``attention_mask`` (BLOCKER 6); ``loss_mask`` is consumed by the
    training-loop counting branch for the per-position CE scatter
    (control-row 3 label-side masking hook).
    """
    x, y, pad_mask, loss_mask = next(dataloader)
    if "cuda" in torch.device(device).type:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        pad_mask = pad_mask.pin_memory().to(device, non_blocking=True)
        loss_mask = loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        pad_mask = pad_mask.to(device)
        loss_mask = loss_mask.to(device)
    return x, y, pad_mask, loss_mask


@torch.no_grad()
def eval(model, data_val_iter, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    avg_depth_list = []

    for _ in range(max_num_batches):
        x, y = get_batch(data_val_iter, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['cross_entropy_loss'] if 'cross_entropy_loss' in outputs else outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        if 'average_depth' in outputs:
            avg_depth_list.append(outputs['average_depth'])

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss
    avg_depth = torch.stack(avg_depth_list).float().mean().item() if avg_depth_list else None

    return val_acc, val_loss, val_perplexity, avg_depth


@torch.no_grad()
def eval_counting(model, data_val_iter, device='cpu', max_num_batches=24, ctx=nullcontext()):
    """BLOCKER 4 counting-specific eval helper.

    Consumes the 4-tuple batch produced by ``get_counting_batch``, forwards the
    model with ``attention_mask=pad_mask`` (BLOCKER 6), and computes both the
    masked per-position CE loss and the count-position accuracy using
    ``loss_mask``. Returns ``(val_acc, val_loss, val_perplexity, avg_depth)``
    matching the OWT2 ``eval`` contract; ``avg_depth`` is always None because
    counting models do not report a router depth.

    OOD-length override (RQ9 DV-1)
    ------------------------------
    This helper is length-agnostic; the per-sample sequence length is
    determined at dataset-construction time by ``length_range`` in
    ``data.counting.CountingDataset``. To run the eval at an OOD length
    L, the caller passes ``ood_length=L`` to
    ``data.counting.get_counting_data`` (or constructs a
    ``CountingDataset`` directly with ``length_range=(L, L)`` and a
    seed disjoint from the in-distribution val split). The DataLoader
    built over that dataset is then iterated here unchanged; no
    signature or branch change is required in ``eval_counting``.
    """
    assert model.training is False

    loss_list_val, acc_list = [], []
    for _ in range(max_num_batches):
        x, y, pad_mask, loss_mask = get_counting_batch(data_val_iter, device=device)
        with ctx:
            outputs = model(x, targets=y, attention_mask=pad_mask, get_logits=True)

        logits = outputs['logits']
        B, T, V = logits.shape
        per_pos_ce = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1),
            reduction='none', ignore_index=-1,
        ).reshape(B, T)
        masked = per_pos_ce * loss_mask
        n_valid = loss_mask.sum().clamp(min=1.0)
        val_loss = masked.sum() / n_valid
        loss_list_val.append(val_loss)

        preds = logits.argmax(dim=-1)
        correct = ((preds == y) & (loss_mask > 0.5)).float().sum()
        total = loss_mask.sum().clamp(min=1.0)
        acc_list.append(correct / total)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss
    return val_acc, val_loss, val_perplexity, None

def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    # Save to local /tmp first, then atomic-copy to target (Lustre).
    # torch.save() uses zip serialization that buffers the entire state dict
    # before flushing — a single ~2 GB write to Lustre can stall for 30+ min
    # under cluster I/O contention, causing the post-save DDP barrier to
    # timeout (see reprod-notes B9).
    import shutil, tempfile, time

    t0 = time.time()
    with tempfile.NamedTemporaryFile(dir='/tmp', suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(checkpoint, tmp_path)

    ckpt_tmp = ckpt_path + '.tmp'
    shutil.copy2(tmp_path, ckpt_tmp)
    os.rename(ckpt_tmp, ckpt_path)
    os.remove(tmp_path)

    elapsed = time.time() - t0
    print(f"  checkpoint saved in {elapsed:.1f}s ({os.path.getsize(ckpt_path) / 1e6:.0f} MB)")
