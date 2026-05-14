# diag/ -- 12L_5R underperformance diagnostics

Three read-only checks comparing the failing `BaseCot_12L_5R` checkpoint to a
working sibling (default `BaseCot_24L_5R`, same `n_repeat`). Goal: rule out
diagnosable causes empirically before committing to a retrain.

## Setup

```bash
# On iridis login node (or any compute node with GPU for Diag B)
cd ~/CoTFormer
conda activate /scratch/ab3u21/cotformer-env
```

## Run all three

```bash
bash iridis/base-cots-eval/diag/run_all.sh
```

Outputs `./diag_<timestamp>.log` in CWD. rsync that file back for review.

## Run individually

```bash
# Diag A -- args diff (instant, no GPU)
python iridis/base-cots-eval/diag/diag_a_args_diff.py

# Diag C -- state_dict structural audit (~30s, no GPU)
python iridis/base-cots-eval/diag/diag_c_state_dict_audit.py

# Diag B -- n_head hot-swap PPL probe (~1 min, CUDA required)
python iridis/base-cots-eval/diag/diag_b_nhead_hotswap.py
```

## Decision tree

| Output                                            | Interpretation                                                  |
|---------------------------------------------------|-----------------------------------------------------------------|
| Diag A reports any suspicious field difference    | That field is a candidate root cause; investigate it first      |
| Diag A reports identical args                     | Cause is NOT in summary['args']; proceed to B and C             |
| Diag C reports unique keys or shape mismatches    | Structural drift between trained models; root cause located     |
| Diag C reports identical structure                | Architecture matches; not a structural bug                      |
| Diag B ratio > 1.5 (claimed PPL << alt PPL)       | Weights genuinely trained at claimed n_head; not an n_head bug  |
| Diag B ratio < 0.7 (alt PPL << claimed PPL)       | summary.json lies about n_head; retrain required                |
| Diag B ratio between 0.7 and 1.5                  | Inconclusive on this sample; increase --n-batches               |

If all three diagnostics come back clean (identical args, identical structure,
n_head verified) the PPL gap is genuinely a training-quality issue that cannot
be post-hoc verified. At that point retrain is the only path forward, justified
by the negative diagnostic results.

## Overrides

All three accept `--ckpt-root`, `--failing`, `--working`. Examples:

```bash
# Compare 12L_5R against 12L_3R instead (different n_repeat, same n_layer)
python iridis/base-cots-eval/diag/diag_a_args_diff.py --working BaseCot_12L_3R

# Probe a different n_head alternate (default 24)
python iridis/base-cots-eval/diag/diag_b_nhead_hotswap.py --alt-n-head 6
```
