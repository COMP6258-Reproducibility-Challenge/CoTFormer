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

import config
import models
from data.utils import get_dataset, prepare_dataset, get_dataloader
from optim.base import train_base
import distributed


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def main(args): 

    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")

    if distributed_backend.is_master_process():
        prepare_dataset(args)
    distributed_backend.sync()
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
    if args.data_in_ram:
        print("Moving dataset to ram")
        data = {'train': np.array(data['train']), 'val': np.array(data['val'])}
        
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")

    data["train"] = get_dataloader(
        data["train"],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.data_seed,
        distributed_backend=distributed_backend,
    )
    
    data["val"] = get_dataloader(
        data["val"],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.data_seed,
    )  
    
    model = models.make_model_from_args(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'

    model = distributed_backend.transform_model(model)
    
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt/1e6,))
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    elif args.opt == 'adafactor':
        from optim.adafactor import Adafactor
        opt = Adafactor(group_specs, lr=args.lr)
    else:
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                                                            pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=args.final_div_factor)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    args.world_size = distributed_backend.get_world_size()
    exp_name = args.exp_name
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy, entity=args.wandb_entity)
    
    extending = False
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    elif os.path.isfile(f"{ckpt_path}/summary.json"):
        with open(f"{ckpt_path}/summary.json") as f:
            prev_iters = json.load(f)["args"].get("iterations", 0)
        if args.iterations <= prev_iters:
            print(f"Already completed experiment '{ckpt_path}' ({prev_iters} iters).\nSkipping.")
            sys.exit(0)
        else:
            extending = True
            print(f"Extending experiment '{ckpt_path}' from {prev_iters} to {args.iterations} iters.")
    distributed_backend.sync()

    itr = 0
    rng_state_dict = None
    if args.use_pretrained == "auto":
        checkpoints = [file for file in os.listdir(ckpt_path) if 'ckpt_' in file]
        if checkpoints:
            last_iter_found = -1
            for ckpt in checkpoints:
                ckpt_iter = int(ckpt.split('ckpt_')[1].split('.pt')[0])
                if ckpt_iter > last_iter_found:
                    args.use_pretrained = ckpt
                    last_iter_found = ckpt_iter
            if last_iter_found == -1:
                print("WARN: Could not find last checkpoint")
                args.use_pretrained = sorted(checkpoints)[-1]
        else:
            args.use_pretrained = None
            print(f"WARN: No checkpoint found in {ckpt_path}, starting fresh")
    
    if args.use_pretrained is not None:
        last_ckpt_path = args.use_pretrained
        print(f"Resuming from {last_ckpt_path}")
        try:
            # weights_only=False: checkpoint contains numpy RNG state (np.random.get_state())
            # pickled via numpy._core.multiarray._reconstruct, which PyTorch 2.6+ rejects under
            # the new weights_only=True default. RNG restore is load-bearing for resume
            # reproducibility (see optim/base.py:65-80) — we trust our own checkpoints.
            checkpoint = torch.load(os.path.join(ckpt_path, last_ckpt_path), map_location=args.device, weights_only=False)
        except (RuntimeError, EOFError, TypeError, ImportError, ModuleNotFoundError, AttributeError) as e:
            # RuntimeError/EOFError → bytes corruption (Drive FUSE partial write).
            # TypeError/ImportError/ModuleNotFoundError/AttributeError → version drift
            # between save and load env (numpy minor-version pickle incompat is the
            # canonical case). Either way: graceful fresh-start beats hard crash.
            print(f"WARN: Could not load checkpoint {last_ckpt_path}: {type(e).__name__}: {e}")
            print("Starting fresh.")
            args.use_pretrained = None

    if args.use_pretrained is not None:
        model_state_dict = {distributed_backend.translate_model_parameter_name_for_node(k.replace("_orig_mod.", ""))[0]:v for k,v in checkpoint['model'].items()}
        # FIXME checkpoints from compiled model have _orig_mod keyword

        optimizer_state_dict = checkpoint['optimizer']
        _rng_keys = [
            "cpu_rng_state",
            "gpu_rng_state",      # legacy single-GPU checkpoints
            "gpu_rng_states",     # new per-rank DDP checkpoints
            "numpy_rng_state",
            "py_rng_state",
            "train_sampler_state",
        ]
        rng_state_dict = {k: checkpoint[k] for k in _rng_keys if k in checkpoint}
        if "gpu_rng_state" in rng_state_dict:
            rng_state_dict["gpu_rng_state"] = rng_state_dict["gpu_rng_state"].type(torch.ByteTensor)
        if "cpu_rng_state" in rng_state_dict:
            rng_state_dict["cpu_rng_state"] = rng_state_dict["cpu_rng_state"].cpu().type(torch.ByteTensor)
        if "gpu_rng_states" in rng_state_dict:
            rng_state_dict["gpu_rng_states"] = [
                s.cpu().type(torch.ByteTensor) if isinstance(s, torch.Tensor) else s
                for s in rng_state_dict["gpu_rng_states"]
            ]
        # Same ByteTensor coercion for the sampler RNG: torch.load demotes the
        # legacy ByteTensor subtype to plain uint8, which optim/base.py:35
        # train_sampler.generator.set_state() rejects on resume. The DDP path
        # saves None here (DistributedSampler has set_epoch), so the guard
        # short-circuits and DDP resumes still work.
        if "train_sampler_state" in rng_state_dict and rng_state_dict["train_sampler_state"] is not None:
            rng_state_dict["train_sampler_state"] = rng_state_dict["train_sampler_state"].cpu().type(torch.ByteTensor)

        model.load_state_dict(model_state_dict)
        opt.load_state_dict(optimizer_state_dict)
        itr = checkpoint['itr']
        if scheduler is not None:
            if extending:
                # Extending training: scheduler was created with new total_steps.
                # Don't restore old state (has old total_steps that would block).
                # Fast-forward the fresh scheduler to the resume iteration.
                for _ in range(itr):
                    scheduler.step()
                print(f"Scheduler rebuilt for {args.iterations} total steps, "
                      f"fast-forwarded to itr {itr} (lr={scheduler.get_last_lr()[0]:.6f})")
            else:
                scheduler_state_dict = checkpoint['scheduler']
                scheduler.load_state_dict(scheduler_state_dict)

    if True: # all train functions have the same interface
        train = train_base
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"\nTraining model={args.model} \n{vars(args)}\n")

    stats = train(model, opt, data, args.data_seed, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  eval_freq=args.eval_freq, 
                  distributed_backend=distributed_backend,
                  ckpt_path=ckpt_path, itr=itr, rng_state_dict=rng_state_dict, extra_args=args)
    
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
