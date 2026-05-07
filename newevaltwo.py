import os
import sys
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
            if k not in ["device", "dtype"]:
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

    # NEW: Diagnostic metric accumulators    
    b_sim_list, v_in_list, v_out_list = [], [], []
    dynamic_diag_metrics = {}
    
    stepwise_cos_dist = {}  
    stepwise_var = {}
    # NEW DIAGNOSTIC METRIC ACCUMULATORS END

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
            with type_ctx:
                outputs = model(x, targets=y, get_logits=True, log_metrics=True)
            
            

            raw_model = distributed_backend.get_raw_model(model)
            if hasattr(raw_model, 'forward_metrics'):
                for key, val in raw_model.forward_metrics.items():
                    if 'cos_dist' in key:
                        if key not in stepwise_cos_dist: stepwise_cos_dist[key] = []
                        stepwise_cos_dist[key].append(val)
                    elif 'var_end_rep' in key:
                        if key not in stepwise_var: stepwise_var[key] = []
                        stepwise_var[key].append(val)
                        
                raw_model.forward_metrics.clear()  # Crucial: clear it so it doesn't leak memory
        
            val_loss = outputs['cross_entropy_loss'] if 'cross_entropy_loss' in outputs else outputs['loss']
            acc = ((outputs['logits'].argmax(-1) == y).float().mean())
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
            if 'average_depth' in outputs:
                avg_depth_list.append(outputs['average_depth'])
                
            # --- NEW: Extract and accumulate diagnostic metrics ---
            b_sim_list.append(outputs.get('sim_of_xs', 0.0) or 0.0)
            v_in_list.append(outputs.get('var_into', 0.0) or 0.0)
            v_out_list.append(outputs.get('var_outof', 0.0) or 0.0)
            
            d_metrics = outputs.get('diag_metrics', {})
            if d_metrics:
                # 1. Process standard metrics (macro, etc)
                for k, v in d_metrics.items():
                    if k.startswith('jsd') or k.startswith('macro'):
                        if k not in dynamic_diag_metrics: dynamic_diag_metrics[k] = []
                        dynamic_diag_metrics[k].append(v)
                
                # 2. Process Hidden States (Logit Lens)
                hidden_states = d_metrics.get('hidden_states')
                final_logits = outputs.get('logits') # [B, T, V] or [B, 1, V]
                
                if hidden_states and final_logits is not None:
                    raw_model = distributed_backend.get_raw_model(model)
                    final_probs = torch.nn.functional.softmax(final_logits, dim=-1)
                    final_preds = final_logits.argmax(dim=-1) # [B, T] or [B, 1]
                    
                    for step_name, h in hidden_states.items():
                        # Temporal Alignment
                        if final_logits.shape[1] == 1:
                            h = h[:, [-1], :] # Match the [B, 1, V] shape of final_logits
                        
                        # Project to Vocabulary
                        with torch.no_grad():
                            # Project: h -> LN -> Head
                            step_logits = raw_model.lm_head(raw_model.transformer.ln_f(h))
                            step_probs = torch.nn.functional.softmax(step_logits, dim=-1)
                            
                            # A. JSD / KL Divergence (Convergence Map)
                            m = 0.5 * (step_probs + final_probs)
                            jsd = 0.5 * torch.nn.functional.kl_div(m.log(), step_probs, reduction='batchmean') + \
                                  0.5 * torch.nn.functional.kl_div(m.log(), final_probs, reduction='batchmean')
                            
                            jsd_key = f'jsd_to_final_{step_name}'
                            if jsd_key not in dynamic_diag_metrics: dynamic_diag_metrics[jsd_key] = []
                            dynamic_diag_metrics[jsd_key].append(jsd.item())
                            
                            # B. Confidence (Trajectory Map)
                            # Get the probability of the *final* predicted token at this intermediate step
                            # final_preds is [B, T], step_probs is [B, T, V]
                            conf = torch.gather(step_probs, -1, final_preds.unsqueeze(-1)).mean().item()
                            conf_key = f'conf_to_final_{step_name}'
                            if conf_key not in dynamic_diag_metrics: dynamic_diag_metrics[conf_key] = []
                            dynamic_diag_metrics[conf_key].append(conf)
                            
                            # C. Rank (Rank Map)
                            # How many tokens have higher logits than the final chosen token?
                            target_logits = torch.gather(step_logits, -1, final_preds.unsqueeze(-1)) # [B, T, 1]
                            rank = (step_logits > target_logits).sum(dim=-1).float().mean().item()
                            rank_key = f'rank_of_final_{step_name}'
                            if rank_key not in dynamic_diag_metrics: dynamic_diag_metrics[rank_key] = []
                            dynamic_diag_metrics[rank_key].append(rank)
                            
                            # Cleanup heavy tensors
                            del step_logits, step_probs, m, target_logits
            # ----------------------------------------------------

        t1 = time.time()
        dt = t1 - t0
        eval_per_batch_time = dt * 1000 / len(acc_list)

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['eval_per_batch_time'] = eval_per_batch_time
    if avg_depth_list:
        stats['average_depth'] = torch.as_tensor(avg_depth_list).float().mean().item()
    if stepwise_cos_dist:
        for key, val in stepwise_cos_dist.items():
            stats[f'diag/{key}'] = np.mean(val)
    if stepwise_var:
        for key, val in stepwise_var.items():
            stats[f'diag/{key}'] = np.mean(val)
    
    # --- NEW: Average the diagnostic metrics into stats ---
    stats['diag/b_sim'] = np.mean(b_sim_list) if b_sim_list else 0.0
    stats['diag/v_in'] = np.mean(v_in_list) if v_in_list else 0.0
    stats['diag/v_out'] = np.mean(v_out_list) if v_out_list else 0.0
    
    for k, v_list in dynamic_diag_metrics.items():
        if len(v_list) > 0:
            if isinstance(v_list[0], np.ndarray):
                stats[f'diag/{k}'] = np.mean(v_list, axis=0)
            else:
                stats[f'diag/{k}'] = np.mean(v_list)
    # ------------------------------------------------------

    return stats


# --- NEW: Qualitative Sweep Function ---
def run_qualitative_sweep(model, device, type_ctx, save_dir):
    print("\n" + "="*50)
    print("Running Qualitative Logit Lens Sweep...")
    import tiktoken
    import pickle
    tokenizer = tiktoken.get_encoding("gpt2")
    
    test_prompts = [
        "The capital of France is Paris. The capital of Germany is",
        "Sometimes, when people say plasma, they mean a state of matter. Other times, when people say",
        "John has 3 apples. He eats 1. How many apples are left? The answer is",
        "Because of the extreme weather conditions, the flight to New York was"
    ]
    
    model.eval()
    logit_lens_results = []
    
    from distributed import single
    # Helper to get the raw model
    if hasattr(model, 'module'):
        raw_model = model.module
    else:
        raw_model = model

    for i, prompt in enumerate(test_prompts):
        idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with type_ctx:
                # get_logits=True and log_metrics=True triggers the data collection
                outputs = model(idx, get_logits=True, log_metrics=True)
                
        d_metrics = outputs.get('diag_metrics', {})
        hidden_states = d_metrics.get('hidden_states', {})
        final_logits = outputs.get('logits') # [1, 1, V] since it's inference
        
        if hidden_states and final_logits is not None:
            print(f"  [+] Captured Logit Lens data for Prompt {i}")
            
            final_token_id = final_logits.argmax(dim=-1).item()
            final_word = tokenizer.decode([final_token_id])
            final_probs = torch.nn.functional.softmax(final_logits, dim=-1)
            
            step_data = {}
            
            for step_name, h in hidden_states.items():
                # Focus on the last token for the trajectory map
                h_last = h[:, [-1], :]
                
                # Project
                s_logits = raw_model.lm_head(raw_model.transformer.ln_f(h_last))
                s_probs = torch.nn.functional.softmax(s_logits, dim=-1)
                
                # 1. Trajectory Data (Top token & Confidence)
                top_prob, top_idx = s_probs.max(dim=-1)
                top_word = tokenizer.decode([top_idx.item()])
                
                # 2. Rank Data (Where is the final answer ranked?)
                target_logit = s_logits[0, 0, final_token_id]
                rank = (s_logits > target_logit).sum().item()
                
                # 3. KL Divergence (Convergence)
                m = 0.5 * (s_probs + final_probs)
                jsd = 0.5 * torch.nn.functional.kl_div(m.log(), s_probs, reduction='batchmean') + \
                      0.5 * torch.nn.functional.kl_div(m.log(), final_probs, reduction='batchmean')

                step_data[step_name] = {
                    "top_word": top_word,
                    "confidence": top_prob.item(),
                    "rank": rank,
                    "jsd": jsd.item(),
                    "matches_final": top_idx.item() == final_token_id
                }
                
                del s_logits, s_probs, m

            logit_lens_results.append({
                "prompt_idx": i,
                "prompt_text": prompt,
                "final_prediction": final_word,
                "steps": step_data,
                "appendix_f_heatmap": d_metrics.get('appendix_f_heatmap') 
            })
            
    # Save the extracted data
    save_path = os.path.join(save_dir, "logit_lens_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(logit_lens_results, f)
    print(f"Logit Lens data saved to: {save_path}")
    print("="*50 + "\n")
# ---------------------------------------

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
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
        
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = models.make_model_from_args(args).to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
        model.load_state_dict({x: y for x, y in checkpoint['model'].items() if "attn.bias" not in x and "wpe" not in x}, strict=False)

    model = distributed_backend.transform_model(model)
    
    print(f"\nEvaluating model={args.model} \n{vars(args)}\n")

    stats = evaluate(model, data, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  distributed_backend=distributed_backend,
                  extra_args=args)

    print(stats)
    print("\n--- Evaluation Complete ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if distributed_backend.is_master_process():
        raw_model = distributed_backend.get_raw_model(model) 
        # Using "." saves it directly to your SLURM RUN_DIR
        run_qualitative_sweep(raw_model, args.device, type_ctx, save_dir=".")


    distributed_backend.finalize()
    return stats


if __name__ == "__main__":
    args = get_args()
    main(args)
