import os
import sys
import math
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import logging
import time
from contextlib import nullcontext

from tqdm import tqdm

import config
import models
from data.utils import get_dataset, prepare_dataset
from optim.base import train_base
import distributed
from optim.utils import get_batch

def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_args(args=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=none_or_str, required=True)
    parser.add_argument('--config_format', type=str, required=False)
    parser.add_argument('--output_dir', type=none_or_str, default=None,
                        help='Directory for eval_summary JSON output. Defaults to checkpoint dir.')
    parser.add_argument('--eval-length', '--eval_length', dest='eval_length',
                        type=int, default=None,
                        help='Optional OOD-length override for the counting '
                             'task. When set, the val split is rebuilt at the '
                             'fixed sequence length L=eval_length drawn from '
                             'data.counting.TE200_OOD_LENGTH_RANGE; output is '
                             'written to eval_summary_ood_L<L>.json. When '
                             'unset (default), behaviour is byte-identical to '
                             'the pre-DIR eval contract.')

    args, rem_args = parser.parse_known_args(args)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            args.checkpoint, args.checkpoint_filename = os.path.split(args.checkpoint)
        else:
            args.checkpoint_filename = "ckpt.pt"

        with open(os.path.join(args.checkpoint, "summary.json")) as f:
            summary = json.load(f)

        for k, v in summary['args'].items():
            if k == "config_format" and args.config_format is not None:
                continue
            if k not in ["device", "dtype", "output_dir", "eval_length"]:
                setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format, base_parser=argparse.ArgumentParser(allow_abbrev=False), args=rem_args, namespace=args)


def get_as_batch(data, seq_length, batch_size, device='cpu', sample_size=None):
    all_ix = list(range(0, len(data), seq_length))
    assert all_ix[-1] + seq_length + 1 > len(data)
    all_ix.pop()
    if sample_size is not None:
        all_ix = np.random.choice(all_ix, size=sample_size // seq_length, replace=False).tolist()
    
    idx = 0
    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, iterations, acc_steps, batch_size, sequence_length, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {}

    num_substeps_per_epoch = len(data['val']) // (batch_size * sequence_length)
    
    if extra_args.compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        # torchdynamo.config.log_level = logging.DEBUG
        model = torch.compile(model) # requires pytorch 2.0+

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []
    avg_depth_list = []
    seq_length = extra_args.eval_seq_length or extra_args.sequence_length
    with torch.no_grad():
        torch.set_printoptions(sci_mode=False)
        t0 = time.time()
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=extra_args.device, 
                    sample_size=extra_args.eval_sample_size
                )
            ),
            total=iceildiv(
                extra_args.eval_sample_size // seq_length if extra_args.eval_sample_size is not None else 
                iceildiv(len(data['val']), seq_length), 
                batch_size
            )
        ):
            cnt = 0
            with type_ctx:
                outputs = model(x, targets=y, get_logits=True)
        
            val_loss = outputs['cross_entropy_loss'] if 'cross_entropy_loss' in outputs else outputs['loss']
            acc = ((outputs['logits'].argmax(-1) == y).float().mean())
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
            if 'average_depth' in outputs:
                avg_depth_list.append(outputs['average_depth'])
        t1 = time.time()
        dt = t1 - t0
        eval_per_batch_time = dt * 1000 / len(acc_list)
        

    loss_t = torch.as_tensor(loss_list_val)
    acc_t = torch.as_tensor(acc_list)

    stats['val_acc'] = acc_t.mean().item()
    stats['val_loss'] = loss_t.mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['eval_per_batch_time'] = eval_per_batch_time
    if avg_depth_list:
        stats['average_depth'] = torch.as_tensor(avg_depth_list).float().mean().item()

    # Per-batch statistics for confidence intervals.
    # NOTE: batches are contiguous windows of the val set (not i.i.d. draws),
    # so the CI is approximate -- the true effective sample size is smaller.
    n = len(loss_list_val)
    stats['n_batches'] = n
    stats['val_loss_std'] = loss_t.std().item() if n > 1 else 0.0
    stats['val_acc_std'] = acc_t.std().item() if n > 1 else 0.0
    loss_se = stats['val_loss_std'] / (n ** 0.5) if n > 1 else 0.0
    stats['val_loss_ci95'] = [stats['val_loss'] - 1.96 * loss_se,
                              stats['val_loss'] + 1.96 * loss_se]
    ppl_lo = math.e ** stats['val_loss_ci95'][0]
    ppl_hi = math.e ** stats['val_loss_ci95'][1]
    stats['val_perplexity_ci95'] = [ppl_lo, ppl_hi]

    # Raw per-batch arrays (for downstream analysis)
    stats['_per_batch_losses'] = loss_list_val
    stats['_per_batch_accs'] = acc_list
    if avg_depth_list:
        stats['_per_batch_depths'] = [d.item() if isinstance(d, torch.Tensor) else d
                                      for d in avg_depth_list]

    return stats

def evaluate_counting(model, data, batch_size, sequence_length, extra_args):
    """Counting-task eval branch (BLOCKER 4 4-tuple loader pathway).

    Used when ``args.dataset == 'counting'``: bypasses the OWT2-style
    token-stream chunking in ``evaluate()`` and instead iterates the
    pre-built ``CountingDataset`` (val split) once via a DataLoader,
    consuming the 4-tuple ``(tokens, targets, pad_mask, loss_mask)``
    contract. Returns the same ``stats`` dict shape as ``evaluate``
    (val_loss / val_acc / val_perplexity / per-batch arrays / CIs)
    plus a per-position accuracy curve along the count window
    (positions 1..L) for the RQ9 DV-1 OOD sweep aggregator.

    The override that selects the OOD eval length L lives one level
    up: ``data.counting.get_counting_data(args, ood_length=L)``
    re-builds the val split with ``length_range=(L, L)`` so every
    sample in this loader has the SAME count window length. The
    per-position accuracy returned here is therefore a clean function
    of position index t in [1, L].
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype
    )
    model.eval()

    val_dataset = data['val']
    loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    forward_sig = inspect.signature(model.forward)
    accepts_attention_mask = "attention_mask" in forward_sig.parameters

    seq_pad = int(sequence_length)
    per_position_correct = np.zeros(seq_pad, dtype=np.float64)
    per_position_total = np.zeros(seq_pad, dtype=np.float64)
    per_sample_exact_correct = 0
    per_sample_total = 0

    loss_list_val: list[float] = []
    acc_list: list[float] = []

    t0 = time.time()
    with torch.no_grad(), type_ctx:
        for batch in tqdm(loader, total=len(loader)):
            x, y, pad_mask, loss_mask = batch
            x = x.to(extra_args.device)
            y = y.to(extra_args.device)
            pad_mask = pad_mask.to(extra_args.device)
            loss_mask = loss_mask.to(extra_args.device)

            if accepts_attention_mask:
                outputs = model(x, targets=y, attention_mask=pad_mask, get_logits=True)
            else:
                outputs = model(x, targets=y, get_logits=True)

            logits = outputs['logits']
            B, T, V = logits.shape

            per_pos_ce = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1),
                reduction='none', ignore_index=-1,
            ).reshape(B, T)
            masked_ce = per_pos_ce * loss_mask
            n_valid = loss_mask.sum().clamp(min=1.0)
            val_loss = (masked_ce.sum() / n_valid).item()

            preds = logits.argmax(dim=-1)
            correct = ((preds == y) & (loss_mask > 0.5)).float()
            total_pp = loss_mask.sum().clamp(min=1.0)
            val_acc = (correct.sum() / total_pp).item()

            loss_list_val.append(val_loss)
            acc_list.append(val_acc)

            # Per-position accumulation along the pad axis.
            correct_np = correct.detach().to('cpu').numpy().astype(np.float64)
            mask_np = (loss_mask > 0.5).detach().to('cpu').numpy().astype(np.float64)
            per_position_correct += correct_np.sum(axis=0)
            per_position_total += mask_np.sum(axis=0)

            # Per-sample exact match.
            per_sample_valid = (loss_mask > 0.5).float().sum(dim=1)
            per_sample_correct = correct.sum(dim=1)
            sample_correct = (
                (per_sample_valid > 0.5) & (per_sample_correct == per_sample_valid)
            ).float()
            per_sample_exact_correct += int(sample_correct.sum().item())
            per_sample_total += int(B)

    dt = time.time() - t0

    stats = {}
    n = len(loss_list_val)
    loss_t = torch.as_tensor(loss_list_val)
    acc_t = torch.as_tensor(acc_list)
    stats['val_loss'] = loss_t.mean().item() if n > 0 else float('nan')
    stats['val_acc'] = acc_t.mean().item() if n > 0 else float('nan')
    stats['val_perplexity'] = math.e ** stats['val_loss']
    stats['eval_per_batch_time'] = dt * 1000 / max(n, 1)
    stats['n_batches'] = n
    stats['val_loss_std'] = loss_t.std().item() if n > 1 else 0.0
    stats['val_acc_std'] = acc_t.std().item() if n > 1 else 0.0
    loss_se = stats['val_loss_std'] / (n ** 0.5) if n > 1 else 0.0
    stats['val_loss_ci95'] = [stats['val_loss'] - 1.96 * loss_se,
                              stats['val_loss'] + 1.96 * loss_se]
    stats['val_perplexity_ci95'] = [math.e ** stats['val_loss_ci95'][0],
                                    math.e ** stats['val_loss_ci95'][1]]

    stats['exact_match_acc'] = (per_sample_exact_correct / per_sample_total
                                if per_sample_total > 0 else float('nan'))
    stats['n_eval_samples'] = per_sample_total
    pp_acc = np.where(per_position_total > 0,
                      per_position_correct / np.maximum(per_position_total, 1.0),
                      0.0)
    stats['per_position_acc'] = pp_acc.tolist()
    stats['per_position_total'] = per_position_total.tolist()

    stats['_per_batch_losses'] = loss_list_val
    stats['_per_batch_accs'] = acc_list
    return stats


def main(args):

    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")

    if distributed_backend.is_master_process():
        prepare_dataset(args)
    distributed_backend.sync()

    # OOD-length override: when --eval-length is set on the counting
    # task, build the val split at the fixed OOD sequence length L via
    # data.counting.get_counting_data(args, ood_length=L). Falls back
    # to the default get_dataset(args) path for OWT2 / non-counting
    # tasks (the flag is silently ignored there to keep --help simple
    # and behaviour identical for non-counting callers).
    eval_length = getattr(args, 'eval_length', None)
    if eval_length is not None and getattr(args, 'dataset', None) == 'counting':
        from data.counting import get_counting_data
        data = get_counting_data(args, ood_length=int(eval_length))
    else:
        data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}

    if isinstance(data['train'], np.ndarray):
        print(f"Num training tokens: {len(data['train'])}")
        print(f"Num validation tokens: {len(data['val'])}")
    else:
        print(f"Num training samples: {len(data['train'])}")
        print(f"Num validation samples: {len(data['val'])}")

    model = models.make_model_from_args(args).to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
        model.load_state_dict({x: y for x, y in checkpoint['model'].items() if "attn.bias" not in x and "wpe" not in x}, strict=False)

    model = distributed_backend.transform_model(model)

    print(f"Evaluating model={args.model}\n{vars(args)}\n")

    is_counting = getattr(args, 'dataset', None) == 'counting'
    if is_counting:
        stats = evaluate_counting(model, data, args.batch_size, args.sequence_length,
                                  extra_args=args)
    else:
        stats = evaluate(model, data, args.iterations, args.acc_steps, args.batch_size, args.sequence_length,
                      distributed_backend=distributed_backend,
                      extra_args=args)

    # Print summary (excludes per-batch arrays for readability)
    summary = {k: v for k, v in stats.items() if not k.startswith('_')}
    print(summary)

    # Write eval_summary JSON with full per-batch data
    if distributed_backend.is_master_process():
        output_dir = getattr(args, 'output_dir', None) or args.checkpoint
        os.makedirs(output_dir, exist_ok=True)

        ckpt_stem = os.path.splitext(args.checkpoint_filename)[0]
        if eval_length is not None and is_counting:
            summary_basename = f"eval_summary_ood_L{int(eval_length)}.json"
        else:
            summary_basename = f"eval_summary_{ckpt_stem}.json"
        summary_path = os.path.join(output_dir, summary_basename)

        json_out = {
            'checkpoint': args.checkpoint_filename,
            'checkpoint_dir': args.checkpoint,
            'model': args.model,
            'batch_size': args.batch_size,
            'sequence_length': args.sequence_length,
            'eval_length': eval_length,
        }
        json_out.update(stats)
        # Recursively convert tensors/numpy types for JSON serialisation
        def _convert(obj):
            if isinstance(obj, (torch.Tensor, np.generic)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        json_out = {k: _convert(v) for k, v in json_out.items()}

        with open(summary_path, 'w') as f:
            json.dump(json_out, f, indent=2)
        print(f"\nEval summary written to: {summary_path}")

    distributed_backend.finalize()
    return stats


if __name__ == "__main__":
    args = get_args()
    main(args)
