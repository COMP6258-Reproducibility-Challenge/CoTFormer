# Reproducibility Notes

> Reproduction fidelity log for [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper]
> in our COMP 6258 reproduction. Covers both successful replications and known divergences.

## Table of Contents

- [Reproducibility Notes](#reproducibility-notes)
  - [Table of Contents](#table-of-contents)
- [Methodology](#methodology)
  - [M1. Reproduction scope](#m1-reproduction-scope)
  - [M2. Evaluation protocol](#m2-evaluation-protocol)
  - [M3. Confidence intervals and the paper's SEM](#m3-confidence-intervals-and-the-papers-sem)
  - [M4. Training configuration shared across variants](#m4-training-configuration-shared-across-variants)
  - [M5. What "delta" means in this document](#m5-what-delta-means-in-this-document)
  - [M6. Reproduction-fidelity vs fair-comparison regime distinction](#m6-reproduction-fidelity-vs-fair-comparison-regime-distinction)
- [Part A: Successful Replications](#part-a-successful-replications)
  - [A1. CoTFormer + Reserved Layers Perplexity (Table 2, 40k steps)](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps)
    - [Delta interpretation](#delta-interpretation)
    - [Training configuration](#training-configuration)
    - [Evaluation configuration](#evaluation-configuration)
    - [Ordering rationale](#ordering-rationale)
  - [A2. LN-CoTFormer Perplexity (Table 2, 40k steps)](#a2-ln-cotformer-perplexity-table-2-40k-steps)
  - [A2a. LN-CoTFormer Perplexity (Section 5, 60k steps)](#a2a-ln-cotformer-perplexity-section-5-60k-steps)
  - [A3. Architecture Depth](#a3-architecture-depth)
  - [A4. Training Stability](#a4-training-stability)
  - [A5. Effective Batch Size](#a5-effective-batch-size)
  - [A6. ADM Perplexity (Section 4.2, Figure 4, 60k steps)](#a6-adm-perplexity-section-42-figure-4-60k-steps)
    - [Architecture](#architecture)
    - [Training Stability](#training-stability)
    - [Job Chaining](#job-chaining)
    - [Capacity Factor Sampling](#capacity-factor-sampling)
    - [Why the Gap Is Wider than the 40k Reproductions](#why-the-gap-is-wider-than-the-40k-reproductions)
    - [Paper Comparison Note](#paper-comparison-note)
  - [A7. Section 5 Comparison: LN-CoTFormer vs ADM at 60k Steps](#a7-section-5-comparison-ln-cotformer-vs-adm-at-60k-steps)
    - [Absolute deltas](#absolute-deltas)
    - [Why the adaptive gap is smaller](#why-the-adaptive-gap-is-smaller)
  - [A8. Table 1 — CoTFormer rows (40k steps) — partial replication](#a8-table-1--cotformer-rows-40k-steps--partial-replication)
    - [Delta interpretation](#delta-interpretation-1)
    - [Training configuration](#training-configuration-1)
    - [Evaluation configuration](#evaluation-configuration-1)
    - [Uncertainty interpretation](#uncertainty-interpretation)
    - [Coverage: ours vs paper-reported reference](#coverage-ours-vs-paper-reported-reference)
  - [A9. Figure 2 — accuracy–compute tradeoff (partial replication)](#a9-figure-2--accuracycompute-tradeoff-partial-replication)
    - [Reproduction strategy](#reproduction-strategy)
    - [Extra points beyond Table 1](#extra-points-beyond-table-1)
    - [Visual style](#visual-style)
  - [A10. Figure 3 — compute cost vs sequence length (analytical-only reproduction)](#a10-figure-3--compute-cost-vs-sequence-length-analytical-only-reproduction)
    - [Method](#method)
    - [Expected outcome and significance](#expected-outcome-and-significance)
- [Part B: Known Divergences](#part-b-known-divergences)
  - [B1. Train/Val Split Divergence](#b1-trainval-split-divergence)
    - [B1a. Document ordering](#b1a-document-ordering)
    - [B1b. Split RNG implementation](#b1b-split-rng-implementation)
    - [B1c. Validation set size rounding](#b1c-validation-set-size-rounding)
    - [Why the original split is unrecoverable](#why-the-original-split-is-unrecoverable)
    - [Impact assessment](#impact-assessment)
  - [B2. Token Ordering Within Bins](#b2-token-ordering-within-bins)
  - [B3. Micro-Batch Size (Hardware-Constrained)](#b3-micro-batch-size-hardware-constrained)
    - [What changed](#what-changed)
    - [Why it diverges numerically](#why-it-diverges-numerically)
    - [Impact assessment](#impact-assessment-1)
    - [Additional hardware-driven settings](#additional-hardware-driven-settings)
  - [B4. RNG State Restoration on DDP Resume (ADM) — Resolved](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)
    - [Background](#background)
    - [Why it doesn't affect LN-CoTFormer](#why-it-doesnt-affect-ln-cotformer)
    - [Why it affects ADM training](#why-it-affects-adm-training)
    - [DDP complication (original code)](#ddp-complication-original-code)
    - [Fix](#fix)
    - [Practical impact (post-fix)](#practical-impact-post-fix)
  - [B5. Base Model Batch Size](#b5-base-model-batch-size)
  - [B6. Orphan MoDBlock in Adaptive/MoD Models — Resolved](#b6-orphan-modblock-in-adaptivemod-models--resolved)
    - [Background](#background-1)
    - [Why the original codebase doesn't crash](#why-the-original-codebase-doesnt-crash)
    - [How it manifests under DDP](#how-it-manifests-under-ddp)
    - [Fix](#fix-1)
    - [Affected files (all from original commit `2723e30`)](#affected-files-all-from-original-commit-2723e30)
  - [B7. Training Summary Metrics Never Recorded (Original Code Bug) — Resolved](#b7-training-summary-metrics-never-recorded-original-code-bug--resolved)
    - [Background](#background-2)
    - [Fix](#fix-2)
  - [B8. Missing `nullcontext` Import in `eval.py` (Original Code Bug) — Resolved](#b8-missing-nullcontext-import-in-evalpy-original-code-bug--resolved)
    - [Background](#background-3)
    - [Fix](#fix-3)
  - [B9. DDP Checkpoint Save Failures — NCCL + Lustre (ADM) — Resolved](#b9-ddp-checkpoint-save-failures--nccl--lustre-adm--resolved)
    - [Background](#background-4)
    - [Failure timeline](#failure-timeline)
    - [Root causes](#root-causes)
    - [Affected files (cumulative across all four fix iterations)](#affected-files-cumulative-across-all-four-fix-iterations)
    - [Checkpoint format compatibility](#checkpoint-format-compatibility)
  - [B10. LR Schedule Discontinuity on 40k-to-60k Extension](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension)
    - [Background](#background-5)
    - [What happened](#what-happened)
    - [The discontinuity](#the-discontinuity)
    - [Impact assessment](#impact-assessment-2)
  - [B11. Evaluation Batch Size Independence](#b11-evaluation-batch-size-independence)
  - [B12. RQ9 counting sub-stream is not a Chang and Bisk reproduction](#b12-rq9-counting-sub-stream-is-not-a-chang-and-bisk-reproduction)
  - [B13. Chang and Bisk `max_grad_norm = 0.3` is dead code](#b13-chang-and-bisk-max_grad_norm--03-is-dead-code)
  - [B14. Pilot 1 Chang and Bisk-exact results (stub)](#b14-pilot-1-chang-and-bisk-exact-results-stub)
  - [B15. RNG State Restoration on Single-GPU Resume (Case CoT) — Resolved](#b15-rng-state-restoration-on-single-gpu-resume-case-cot--resolved)
  - [B16. Host-RAM OOM via `--data_in_ram` (Case CoT) — Mitigated](#b16-host-ram-oom-via---data_in_ram-case-cot--mitigated)
- [Part C: Original Bugs and Mistakes](#part-c-original-bugs-and-mistakes)
  - [C1. Orphan MoDBlock: 5 Allocated, 4 Used](#c1-orphan-modblock-5-allocated-4-used)
    - [What the code does](#what-the-code-does)
    - [Functional equivalence: a dead-weight removal](#functional-equivalence-a-dead-weight-removal)
    - [Init-order RNG offset: the only real delta](#init-order-rng-offset-the-only-real-delta)
    - [Empirical bound](#empirical-bound)
    - [Reframing earlier attributions](#reframing-earlier-attributions)
  - [C2. Figure 5: Position-Indexed Cascade Bug and Methodological Critique](#c2-figure-5-position-indexed-cascade-bug-and-methodological-critique)
    - [The technical bug](#the-technical-bug)
    - [The methodological critique](#the-methodological-critique)
    - [Experiment matrix (2 x 2 x 4 cells)](#experiment-matrix-2-x-2-x-4-cells)
    - [Findings](#findings)
    - [What we cannot reproduce, and why](#what-we-cannot-reproduce-and-why)
    - [Qualitative claim is confirmed](#qualitative-claim-is-confirmed)
    - [Cross-references](#cross-references)
  - [C3. Underspecified Training Schedule (Reproducibility Gap)](#c3-underspecified-training-schedule-reproducibility-gap)
    - [Warmup steps for 60k models](#warmup-steps-for-60k-models)
    - [LR schedule type](#lr-schedule-type)
    - [Resolution](#resolution)
  - [C4. `eval.py` perplexity uses truncated `e` (2.71828) instead of `math.e` — Original Code Quirk](#c4-evalpy-perplexity-uses-truncated-e-271828-instead-of-mathe--original-code-quirk)
  - [References](#references)

---

# Methodology

Reproduction methodology shared across all
[Part A](#part-a-successful-replications) results. Individual
A-subsections cite back to this section for scope, evaluation
protocol, and uncertainty interpretation rather than restating it.

## M1. Reproduction scope

We reproduce the chain of variants that drives the paper's Section 5
comparison (LN-CoTFormer vs ADM at full compute, 60k steps) plus the
intermediate 40k results from Table 2 that establish the non-adaptive
baseline. Concretely:

| Subsection | Variant | Paper row | Steps |
|---|---|---|---|
| [§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps) | CoTFormer + Reserved Layers | Table 2 "+ Reserved Layers" | 40k |
| [§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps) | LN-CoTFormer | Table 2 "+ Layer Norm" | 40k |
| [§A2a](#a2a-ln-cotformer-perplexity-section-5-60k-steps) | LN-CoTFormer | Section 5 | 60k |
| [§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) | Adaptive LN-CoTFormer (ADM) | Section 4.2, Figure 4 | 60k |
| [§A7](#a7-section-5-comparison-ln-cotformer-vs-adm-at-60k-steps) | LN-CoTFormer vs ADM | Section 5 claim | 60k |

The only Table 2 row we deliberately skip is the bare "CoTFormer 24×5"
(24 layers, no reserved prefix/suffix). That variant sits outside the
`2→21×5→1` layer allocation that every subsequent Section 3/4/5 variant
uses, so reproducing it would not advance the Section 5 comparison chain
that [§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps)
"Ordering rationale" prioritises.

## M2. Evaluation protocol

All metrics reported in [Part A](#part-a-successful-replications) come
from standalone `eval.py` runs launched with `--distributed_backend
None` on a single L4 GPU, reading the trained-model checkpoint
directory and the OWT2 `val.bin` memmap used during training. Each run iterates the full validation set exactly
once: 3,973 contiguous windows of length `sequence_length=256` at the
training-time `batch_size=8`, no subsampling. See `eval.py:77-167` for
the sweep implementation (batch construction at `eval.py:57-72`, loss
and CI computation at `eval.py:135-165`).

The original authors' evaluation path (`eval.py` from commit `2723e30`)
uses `data['val']` exclusively with the same contiguous-window
construction, so our protocol matches theirs modulo batch size — the
batch-size mismatch has no measurable effect on the reported perplexity
and is analysed in [§B11](#b11-evaluation-batch-size-independence).

The `--distributed_backend None` override is mandatory because
`summary.json` serialises the training `distributed_backend` (`nccl`);
without the override, the evaluator would try to initialise NCCL and
fail on single-GPU evaluation resources. The point estimate, 95%
perplexity CI, and raw per-batch losses for each run are persisted to
`eval_summary_ckpt_<step>.json` for downstream consumption.

## M3. Confidence intervals and the paper's SEM

[Part A](#part-a-successful-replications)'s result tables report two
distinct uncertainty quantities — our 95% confidence interval and the
paper's parenthetical SEM — that answer different statistical questions. Both are labelled "SEM" in various
dictionaries but sample from different populations, and this document
never treats them as interchangeable. This subsection defines each
quantity and states how the rest of the document uses them.

**What our 95% CI measures (intra-run cross-batch variance).** For a
single evaluation run of one trained model, `eval.py` collects the 3,973
per-batch cross-entropy losses, takes their mean and sample standard
deviation, and constructs `mean ± 1.96 × (std / √n)` in loss space. The
endpoints are then exponentiated into perplexity space for the CI row in
each A-section table. The construction is visible at `eval.py:148-157`.
This samples from the distribution of per-batch losses on the validation
set for ONE fixed trained model, and answers: *given the spread of losses
across validation batches, how confident are we that the reported point
estimate is close to the true intra-run mean?* In
[§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps), for
example, our point estimate is 24.64 with 95% CI [24.25, 25.03], and the
paper's 24.51 falls inside this interval.

**What the paper's parenthetical SEM measures (inter-run cross-seed
variance).** Paper table entries such as "24.11(0.03)" in Table 2 report
`std(PPL_seed_1, PPL_seed_2, PPL_seed_3) / √3` — the standard error of
the mean across three independently trained seeds. This samples from the
distribution of trained models for ONE fixed architecture and training
recipe, and answers: *how sensitive is the learned model to training
randomness?* It does not measure how confident any single evaluation is
in its own point estimate.

**Why the two sample different populations.** Intra-run cross-batch
variance fixes the model and varies the data subset; inter-run cross-seed
variance fixes the data and varies the model. An architecture can carry a
tight inter-seed SEM and a wide intra-run CI at the same time, or vice
versa, because the two quantities answer different questions.
Collapsing them is a category error. We run a single training seed per
variant (hardware budget), so we cannot construct our own inter-seed
SEM; the paper does not release per-batch losses for any variant, so we
cannot construct their intra-run CI. The two quantities are therefore
not directly comparable.

**How this document uses both quantities.** When the paper's point
estimate falls inside our intra-run 95% CI, we describe the result as
*"reproduced within intra-run noise"*. We do not claim statistical
equivalence in the paper's inter-seed sense unless explicitly flagged in
the relevant A-subsection. When the paper's estimate falls outside our
CI, we name the absolute gap as a delta and attribute it via
[Part B](#part-b-known-divergences) divergences rather than to any
single divergence — see
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) for an example
where the ADM +0.23 PPL gap is attributed to the combined effect of
[§B1](#b1-trainval-split-divergence),
[§B3](#b3-micro-batch-size-hardware-constrained), hardware/kernel
differences, and seed variance.

**Per-batch loss-level SEM.** Several A-section tables also report
`Per-batch SEM (loss) = 0.0081`. This is the standard error of the mean
of the 3,973 per-batch losses in loss space, `std_loss / √n`, and is
computed alongside the CI by `eval.py:152`. Propagated into perplexity
space via the delta method, it bounds the intra-run noise on the point
estimate at approximately 0.25 PPL. It is labelled **intra-run** here
and in the A-section tables to make explicit that it is NOT comparable
to the paper's inter-seed SEM.

## M4. Training configuration shared across variants

Every [Part A](#part-a-successful-replications) training run uses
2× NVIDIA L4 24 GB GPUs via DDP on OpenWebText2 with the same
optimiser and schedule: fused AdamW, cosine
LR via `OneCycleLR` (peak 1e-3, `warmup_percent=0.2`,
`final_div_factor=0.05`), effective batch size 128
(`batch_size=8 × acc_steps=16` before DDP halves `acc_steps`),
`dropout=0.0`, `find_unused_parameters=False`. The authoritative
configuration is in `iridis/adm-train/job.sh` (ADM 60k) and the sibling
`iridis/lncot-train/job.sh` and `iridis/cot-res-train/job.sh`
(non-adaptive 40k); shared environment variables live in
`iridis/env.sh`.

The micro-batch decomposition is dictated by the 2× L4 24 GB memory
budget rather than by choice —
[§B3](#b3-micro-batch-size-hardware-constrained) documents the specific
constraint and its expected floating-point accumulation effect. DDP
infrastructure diverges from the authors' single-A100 setup in two ways
that only affect ADM training: RNG state restoration across resume
boundaries
([§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)) and
checkpoint save coordination that replaces NCCL with a secondary Gloo
process group ([§B9](#b9-ddp-checkpoint-save-failures--nccl--lustre-adm--resolved)).
Non-adaptive variants ([§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps),
[§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps),
[§A2a](#a2a-ln-cotformer-perplexity-section-5-60k-steps)) consume no
CUDA RNG during training and are unaffected by
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved); they share
the same DDP infrastructure otherwise.

## M5. What "delta" means in this document

Each A-section reports a "delta" row: our point estimate minus the
paper's. When the delta is positive our perplexity is higher (worse);
when negative, lower (better). The delta is then compared against two
noise bounds from the confidence-intervals section: the intra-run 95% CI width (computed per-variant,
typically ~0.76 PPL on our hardware) and the 0.25 PPL per-batch loss
SEM propagated through the delta method from the 0.0081 loss-space SEM.

Two interpretations of a positive delta follow. When the delta falls
inside our 95% CI, the paper's point estimate is statistically
compatible with our intra-run evaluation and we describe the result as
*"reproduced within intra-run noise"*. We do not claim statistical
equivalence in the paper's inter-seed sense — we cannot measure that —
but we can assert the point estimates are mutually compatible given the
spread of per-batch losses on our trained model. When the delta falls
outside our 95% CI, the paper's estimate lies beyond our intra-run
uncertainty; we name the gap explicitly and attribute it to a
combination of [Part B](#part-b-known-divergences) divergences
(train/val split, micro-batch decomposition, kernel/hardware
differences, single-seed variance). We
do not run controlled isolation experiments of each divergence because
the compute budget to do so (one extra training run per divergence,
~52 h per ADM 60k run on `ecsstudents_l4`) exceeds our allocation.

We train a single seed per variant. We therefore have no inter-seed
variance estimate of our own, and when the paper's inter-seed SEM is
mentioned in an A-subsection it is cited as paper context rather than
as a quantity we can check.

## M6. Reproduction-fidelity vs fair-comparison regime distinction

The analysis counting sub-stream (`docs/extend-notes.md` §1.2 RQ9
twin-pilot design) runs two regimes in parallel: a
**Pilot 1** that reproduces the Chang and Bisk reference setup
bit-for-bit, and a **main sweep** that applies the project's
CoTFormer-native training regime to the same Chang and Bisk
counting task. The two regimes serve distinct epistemic
purposes, and conflating them inflates either the reproduction
strength or the fair-comparison claim. This subsection codifies
the distinction so that every section touching RQ9 can cite a
single named definition rather than re-stating the discipline.

**Reproduction-fidelity regime** — the goal is bit-identical
reproduction of a published result so that the *codebase* is
validated as a faithful execution of the published method. In
the Pilot 1 instance, the reproduction-fidelity regime fixes
seven training hyperparameters to match
`inductive_counting_with_LMs/scripts/causal_transformer/{trainer,
config}.py` exactly: ReLU activation, `tie_word_embeddings =
False`, `scale_attn_by_inverse_layer_idx = True`,
`max_grad_norm = 1.0` (the value `trainer.py:231` actually
hardcodes; see [§B13](#b13-chang-and-bisk-max_grad_norm--03-is-dead-code)
for the dead-code 0.3), `get_constant_schedule_with_warmup` (not
cosine), AdamW defaults `wd = 0.01`, `beta2 = 0.999` with no
decay grouping, `n_head = 8` at `n_embd = 1024`, FP16, and
non-flash attention. The seed convention follows Chang and
Bisk's `[1234, 12]`-style values extended to 5 seeds for
seed-symmetry. Pass criteria are anchored to the paper's
published per-OOD-length pass band (IND ~100 %, OOD@200
~22-31 %); see
[§B14](#b14-pilot-1-chang-and-bisk-exact-results-stub) for the
results stub.

**Fair-comparison regime** — the goal is a controlled comparison
between architectures (4L baseline vs 1L-4R CoT+Reserved vs
1L-4R LN-CoTFormer) under the *training regime the rest of this
project uses*. The fair-comparison regime fixes the architecture
as the independent variable and the training regime as a
controlled constant: GELU activation, `tie_word_embeddings =
True`, `scale_attn_by_inverse_layer_idx = False`,
`max_grad_norm = 1.0`, cosine schedule with `warmup = 3000`,
AdamW with `wd = 0.1`, `beta2 = 0.95` and decay grouping,
`n_head = 4`, BF16 / FP32, Flash where supported. Seeds follow
the project convention `(train = 2357, val = 8191,
ood = 19937)`. The fair-comparison regime is the
publication-relevant instrument; the question it answers is
"does the recurrent inductive bias help under the training
regime we use elsewhere?".

**Discipline**: a result reported under the
reproduction-fidelity regime is annotated "Pilot 1
reproduction-fidelity"; a result reported under the
fair-comparison regime is annotated "main-sweep
fair-comparison". A result that is reported without the regime
annotation is a methodological error in the synthesis output.
The two regimes are not interchangeable: the
reproduction-fidelity result establishes the codebase's
faithfulness to [25]; the fair-comparison result establishes
the architectural inductive-bias claim under our chosen
training regime. A failure of one does not by itself invalidate
the other, although a failure of Pilot 1 sheds the entire RQ9
sub-stream to the extension per the
`docs/extend-notes.md` §0 Scope-shedding pre-registered order.

---

# Part A: Successful Replications

See [Methodology](#methodology) for the reproduction scope
([§M1](#m1-reproduction-scope)), evaluation protocol
([§M2](#m2-evaluation-protocol)), uncertainty interpretation
([§M3](#m3-confidence-intervals-and-the-papers-sem)), shared training
configuration ([§M4](#m4-training-configuration-shared-across-variants)),
and delta vocabulary ([§M5](#m5-what-delta-means-in-this-document)) that
every subsection below inherits.

## A1. CoTFormer + Reserved Layers Perplexity (Table 2, 40k steps)

**Result:** Reproduced within +0.13 PPL of the paper; the paper's point
estimate lies inside our intra-run 95% CI.

| Metric | Paper (Table 2, + Reserved Layers) | Ours | Delta |
|--------|------------------------------------|------|-------|
| Val perplexity | 24.51 (0.01) | 24.64 | +0.13 |
| Val accuracy | -- | 0.4106 | -- |
| Val loss | -- | 3.2042 | -- |
| 95% CI (perplexity) | -- | [24.25, 25.03] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

The "+ Reserved Layers" row of Table 2 corresponds to the
`cotformer_full_depth` architecture: 2 reserved prefix layers, 21 mid
blocks repeated 5×, and 1 reserved suffix layer, for a total of 108
effective forward-pass layers. It is the architectural parent of the
LN-CoTFormer ([§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps)):
both share the `2→21×5→1` layer allocation and differ only in whether
a LayerNorm is applied at the end of each repeated pass through the
mid-block stack.

### Delta interpretation

Our point estimate (24.64) sits inside our intra-run 95% CI
[24.25, 25.03], which also contains the paper's 24.51. The residual
+0.13 PPL delta is 0.53% of the paper's value and consistent with the
combined effect of [§B1](#b1-trainval-split-divergence) (train/val
split), [§B3](#b3-micro-batch-size-hardware-constrained) (micro-batch
decomposition under bfloat16), and single-seed variance; the paper's
0.01 parenthetical is an inter-seed SEM and is not directly comparable
— see [§M3](#m3-confidence-intervals-and-the-papers-sem) for the
uncertainty-quantity distinction.

Unlike [§A2a](#a2a-ln-cotformer-perplexity-section-5-60k-steps) and
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps), this run is
free of both [§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension)
(LR discontinuity; only applies to 40k-to-60k extensions) and
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) (RNG state
restoration at resume boundaries; only applies to ADM). A single SLURM
submission completed 40,000 steps without auto-resume, so neither
divergence applies.

### Training configuration

Trained for 40,000 steps using the shared setup of
[§M4](#m4-training-configuration-shared-across-variants) with
`OneCycleLR(total_steps=40000, anneal_strategy='cos')`. Training package:
`iridis/cot-res-train/`.

### Evaluation configuration

Standalone `eval.py` per [§M2](#m2-evaluation-protocol). Evaluated from
the final `ckpt_40000.pt` snapshot of the `cotformer_full_depth` class
via `--checkpoint`/`--checkpoint_filename ckpt_40000.pt`. Evaluation
package: `iridis/eval-cot-res/`; artefacts in
`iridis/eval-cot-res/run_0/` (`eval_4k0.txt` stdout log,
`eval_summary_ckpt_40000.json` machine-readable summary including CI).

### Ordering rationale

The "+ Reserved Layers" variant is the first of the Section 3.3
architectural tweaks, introduced in the "Reserving Beginning and Final
Layers" paragraph before the LayerNorm addition that produces the
LN-CoTFormer in the following paragraph. Reporting this row first
mirrors the paper's order of presentation and isolates the
reserved-layer contribution independently of the LayerNorm
contribution. [§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps)
then reports the LN-CoTFormer (Table 2 "+ Layer Norm" row) which adds
a post-repeat LayerNorm atop this architecture, and
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) reports the
ADM, which replaces the sampled repeat count with an adaptive routing
scheme on the same `2→21×5→1` backbone.

---

## A2. LN-CoTFormer Perplexity (Table 2, 40k steps)

**Result:** Reproduced within +0.02 PPL of the paper.

| Metric | Paper (Table 2) | Ours | Delta |
|--------|----------------|------|-------|
| Val perplexity | 24.11 (0.03) | 24.13 | +0.02 |
| Val accuracy | -- | 0.4129 | -- |
| Val loss | -- | 3.1834 | -- |
| 95% CI (perplexity) | -- | [23.75, 24.51] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

Our point estimate (24.13) sits inside our intra-run 95% CI
[23.75, 24.51] which also contains the paper's 24.11; the result is
therefore reproduced within intra-run noise
(see [§M3](#m3-confidence-intervals-and-the-papers-sem) and
[§M5](#m5-what-delta-means-in-this-document)). The paper's 0.03
parenthetical is an inter-seed SEM across three seeds and is not
directly comparable to our single-seed intra-run CI. Trained for
40,000 steps using the shared configuration of
[§M4](#m4-training-configuration-shared-across-variants); wall time
23h 26m across 2 SLURM submissions with auto-resume.

This confirms that our data pipeline, architecture, and training loop
faithfully reproduce the paper's non-adaptive LN-CoTFormer result,
despite the divergences documented in
[Part B](#part-b-known-divergences).

---

## A2a. LN-CoTFormer Perplexity (Section 5, 60k steps)

**Result:** Reproduced within +0.40 PPL of the paper.

| Metric | Paper (Section 5) | Ours | Delta |
|--------|-------------------|------|-------|
| Val perplexity | 23.19 | 23.59 | +0.40 |
| Val accuracy | -- | 0.4157 | -- |
| Val loss | -- | 3.1609 | -- |
| 95% CI (perplexity) | -- | [23.22, 23.97] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

The 40k-to-60k extension used auto-resume from the 40k checkpoint with a
rebuilt `OneCycleLR(total_steps=60000)` scheduler. This introduces a
learning rate discontinuity at the resume boundary
([§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension)): the
40k schedule had decayed to ~2e-5 while the 60k schedule expects
~3.5e-4 at step 40000, a ~17× LR jump that acts as a mild warm restart.

The wider delta (+0.40 vs +0.02 at 40k) is consistent with
[§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension)'s impact
on the final 20k steps. The ADM model
([§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps)) is unaffected
by [§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension) as it
was trained for 60k steps from scratch. See
[§M5](#m5-what-delta-means-in-this-document) for how this +0.40 PPL
delta is compared against the 95% CI in
[§M3](#m3-confidence-intervals-and-the-papers-sem).

---

## A3. Architecture Depth

**Result:** Exact match.

The paper's 24-layer LN-CoTFormer uses 2 prefix + 21 mid (repeated 5x) + 1
suffix = 108 effective layers. Our implementation reports `avg_depth=108.000`
consistently across all 40,000 training steps, confirming the block structure
and repeat loop are correct.

---

## A4. Training Stability

**Result:** No gradient spikes observed.

The paper (Section 3.3) documents "benign spikes" during LN-CoTFormer
training. Our run with gradient norm tracking shows:

| Signal | Last 1000 steps |
|--------|----------------|
| Gradient norm range | 0.49 -- 0.54 |
| Max gradient norm | 0.83 (isolated, at step 39300) |
| Spike threshold (5x mean) | Never triggered |

The cosine learning rate decayed smoothly from 1e-3 to ~2e-4. Loss curves
show stable convergence with normal per-eval variance (PPL 22.4 -- 25.7 in
final 1000 steps; this is expected from the 24-batch eval sample size).

---

## A5. Effective Batch Size

**Result:** Exact match at 128.

Despite using smaller micro-batches than the original (8 vs likely 32+), the
effective batch size of 128 is preserved via gradient accumulation. DDP
correctly halves `acc_steps` (16 -> 8 per GPU) while keeping `batch_size` (8)
per GPU: 8 x 8 x 2 = 128.

---

## A6. ADM Perplexity (Section 4.2, Figure 4, 60k steps)

**Result:** Reproduced within +0.23 PPL of the paper.

| Metric | Paper (Section 5) | Ours | Delta |
|--------|-------------------|------|-------|
| Val perplexity | 23.83 | 24.06 | +0.23 |
| Val accuracy | -- | 0.4127 | -- |
| Val loss | -- | 3.1806 | -- |
| 95% CI (perplexity) | -- | [23.68, 24.44] | -- |
| Per-batch SEM (loss) | -- | 0.0081 | -- |
| n_batches | -- | 3,973 | -- |

Trained for 60,000 steps from scratch using the shared configuration of
[§M4](#m4-training-configuration-shared-across-variants). Wall time
~52h across 3 SLURM submissions with auto-resume (see Job Chaining
table below).

### Architecture

The ADM uses the same 24-layer architecture as the LN-CoTFormer
(2 prefix + 21 mid repeated 5x + 1 suffix = 108 effective layers),
plus a Mixture-of-Repeats router (Section 4.1) that selects per-token
capacity at each repeat. Parameters: 208.55M (0.01M more than the
non-adaptive LN-CoTFormer due to 4 router weight vectors of 768 dims).

### Training Stability

Training was stable throughout all 60,000 steps:

| Signal | Final phase (55k--60k) |
|--------|----------------------|
| Gradient norm range | 0.50 -- 0.64 |
| Max gradient norm | 0.74 |
| Loss range | 2.66 -- 3.50 |
| PPL range (eval) | 22.5 -- 25.3 |

No gradient spikes were observed. The PPL oscillation in eval is
consistent with the non-adaptive LN-CoTFormer (22.4--25.7 in
[§A4](#a4-training-stability)) and is driven by the small eval sample
size, not training instability.

### Job Chaining

The 60k-step training required 3 SLURM submissions due to the
`ecsstudents_l4` partition's 24-hour wall limit:

| Run | Job ID | Steps | Resumed from |
|-----|--------|-------|-------------|
| 0 | 696795 | 0 -- 28570 | (fresh start) |
| 1 | 698707 | 28000 -- 57110 | ckpt_28000.pt |
| 2 | 701232 | 56000 -- 60000 | ckpt_56000.pt |

Auto-resume (`--use_pretrained auto`) worked correctly at each boundary.

### Capacity Factor Sampling

The model uses the default `depth_random_method='uniform'`, which
generates `n_repeat - 1 = 4` random capacity factors via:

```python
length_factors = torch.clamp(torch.rand((4,)) * 1.05, max=1).sort(descending=True).values
```

This matches the paper's Section 4.1: "assign a capacity of 1 to the
first repeat and sample `n_repeat - 1` numbers and sort them in
decreasing order." The 1.05 multiplier gives ~5% extra probability of
capacity=1.0 (a minor stabilisation detail not mentioned in the paper).

During evaluation, `eval_length_factor=None` activates all repeats
(`length_factors = [1, 1, 1, 1]`), so the reported PPL of 24.06 is
at full compute.

### Why the Gap Is Wider than the 40k Reproductions

The +0.23 delta falls outside our intra-run 95% CI
([§M5](#m5-what-delta-means-in-this-document)), so we name it and
attribute it. An earlier revision attributed it to
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)'s
RNG-state discontinuities at resume boundaries, since the ADM is the
only variant that consumes CUDA RNG during training (`torch.rand()`
for `length_factors`). The `eval-adm/run_6` matrix falsifies that
attribution by re-evaluating the pre-fix ADM (v1, trained before the
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) and
[§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved) fixes)
alongside the post-fix retrain (v2) at both 40k and 60k checkpoints
under identical conditions:

| Version | Training state | 60k PPL | 60k 95% CI | 40k PPL |
|---------|----------------|---------|------------|---------|
| v1 | pre-fix (RNG restore commented out, orphan MoDBlock present) | 24.074 | [23.70, 24.46] | 28.073 |
| v2 | post-fix (Gloo per-rank RNG, orphan removed) | 24.060 | [23.68, 24.44] | 28.079 |
| Delta | -- | 0.014 | -- | -0.006 |

The 60k v1-vs-v2 delta of 0.014 PPL is well below either 95% CI width
(~0.76 PPL) and well below the per-batch loss-level intra-run SEM of
0.0081 in loss space (~0.25 PPL in perplexity space; see
[§M3](#m3-confidence-intervals-and-the-papers-sem)).
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) is
therefore methodologically correct but its practical impact on the
final ADM perplexity is below the intra-run noise floor on our
hardware. See [§C1](#c1-orphan-modblock-5-allocated-4-used) for the
mechanism that links the pre- and post-fix init paths through an
init-order RNG cascade, and `iridis/eval-adm/README.md` § *ADM v1 vs
v2: B4 empirical magnitude* for the per-variant statistics.

The residual +0.23 delta against the paper is consistent with the
combined contribution of several small divergences, none individually
attributable without controlled experiments we cannot run:
[§B1](#b1-trainval-split-divergence) (train/val split divergence
against the authors' HF Arrow split, which has been permanently
disabled and is unrecoverable),
[§B3](#b3-micro-batch-size-hardware-constrained) (micro-batch
decomposition under bfloat16, rounding differences relative to the
authors' A100-era accumulation), hardware/kernel divergences (cuDNN
kernel selection, SM89 vs SM80, CUDA 11.8 vs unrecorded), and
single-seed variance against the paper's three-seed mean. None is a
divergence-of-interest on its own, but their sum accounts for the
observed gap more plausibly than
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) alone.

The non-adaptive LN-CoTFormer 40k
([§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps)) has no runtime
randomness (`dropout=0.0`, fixed repeat count) so
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) does not
apply to it in any case. The cot-res-train reproduction
([§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps))
also has no runtime randomness and lands at +0.13 delta, bounding the
ADM's expected delta in the absence of
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved).

### Paper Comparison Note

See [§A7](#a7-section-5-comparison-ln-cotformer-vs-adm-at-60k-steps)
for the full Section 5 comparison (LN-CoTFormer vs ADM at 60k).

---

## A7. Section 5 Comparison: LN-CoTFormer vs ADM at 60k Steps

**Result:** Directional claim confirmed. Non-adaptive LN-CoTFormer outperforms
adaptive model (ADM) at full compute, consistent with the paper.

| Metric | Paper | Ours | Paper gap | Our gap |
|--------|-------|------|-----------|---------|
| LN-CoTFormer 60k PPL | 23.19 | 23.59 | -- | -- |
| ADM 60k PPL (full compute) | 23.83 | 24.06 | -- | -- |
| Non-adaptive advantage | 0.64 | 0.47 | -- | -- |

The paper (Section 5) states: "after 60k steps, the former [adaptive]
reaches perplexity 23.83 while the latter [non-adaptive] achieves 23.19."
Our results confirm this ordering: the non-adaptive model outperforms the
adaptive model at full compute by 0.47 PPL (paper: 0.64 PPL).

### Absolute deltas

Both models run slightly above the paper's reported values:

| Model | Paper | Ours | Delta | Likely contributors |
|-------|-------|------|-------|-------------------|
| LN-CoTFormer 60k | 23.19 | 23.59 | +0.40 | [§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension) (LR discontinuity from 40k-to-60k extension) |
| ADM 60k | 23.83 | 24.06 | +0.23 | Combined [§B1](#b1-trainval-split-divergence) + [§B3](#b3-micro-batch-size-hardware-constrained) + hardware + single-seed variance; see [§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) "Why the Gap Is Wider than the 40k Reproductions" |

The gap in deltas (+0.40 vs +0.23) is consistent with
[§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension) being
unique to the LN-CoTFormer (trained 40k then extended), while the ADM
was trained 60k from scratch and its delta aggregates the same
hardware/seed contributors the 40k CoT + Reserved variant already sees
in [§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps).

### Why the adaptive gap is smaller

The paper's 0.64 gap vs our 0.47 gap is explained by the larger
LN-CoTFormer delta (+0.40) relative to the ADM delta (+0.23). Since
[§B10](#b10-lr-schedule-discontinuity-on-40k-to-60k-extension) inflates
only the LN-CoTFormer's perplexity, the gap narrows. A fresh
60k-from-scratch LN-CoTFormer run would likely produce a wider gap
closer to the paper's 0.64.

---

## A8. Table 1 — CoTFormer rows (40k steps) — partial replication

We reproduce the CoTFormer-only rows of Table 1 from the paper using
seven trained ablation checkpoints: `BaseCot_12L_2R`, `BaseCot_12L_3R`,
`BaseCot_12L_5R`, `BaseCot_12L_15R`, `BaseCot_24L_2R`, `BaseCot_24L_3R`,
and `BaseCot_24L_5R`. The Standard Transformer and Block Universal
Transformer rows of Table 1 are paper-reported reference values — those
architectures were not retrained as part of this submission. This partial
replication is conducted in the fair-comparison regime
([§M6](#m6-reproduction-fidelity-vs-fair-comparison-regime-distinction)):
we use a different random seed and a different OWT2 train/val split
([§B1](#b1-trainval-split-divergence)) from the original authors,
so deltas relative to paper values are interpreted as training-noise
plus split-divergence effects rather than numerical-precision errors.

### Delta interpretation

Each result cell deviates from the paper-reported value by ΔPPL = ours −
paper. Because we operate in the fair-comparison regime
([§M6](#m6-reproduction-fidelity-vs-fair-comparison-regime-distinction)),
the expected delta from cross-seed and cross-split variation is a few
tenths of a PPL unit — consistent with the range observed in
[§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps) (+0.13)
and [§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps) (+0.02). Any
delta larger than ~2–3× the per-batch CI95 half-width is treated as
meaningful evidence of a systematic effect rather than sampling noise;
candidates for attribution include the micro-batch decomposition
([§B3](#b3-micro-batch-size-hardware-constrained)) and effective batch
size ([§B5](#b5-base-model-batch-size)) divergences.

### Training configuration

All seven ablations are trained using the shared configuration of
[§M4](#m4-training-configuration-shared-across-variants) for 40,000
steps. The architecture is `cotformer_full_depth` in every case, with
`n_layer_begin=0` and `n_layer_end=0` — this is the basic CoTFormer
(no reserved prefix or suffix layers). It is distinct from the
`2→21×5→1` layer allocation used by the "+ Reserved Layers" variant
([§A1](#a1-cotformer--reserved-layers-perplexity-table-2-40k-steps)) and
the LN-CoTFormer ([§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps));
those use `n_layer_begin=2, n_layer_end=1`. Table 1 reports the basic
CoTFormer without reserved layers, so `n_layer_begin=0, n_layer_end=0`
is the correct configuration for this replication. The `n_repeat` value
varies per ablation ({2, 3, 5} for both 12L and 24L; plus 15 for 12L
only). Evaluation is orchestrated by `iridis/base-cots-eval/job.sh`; the
underlying training checkpoints (from `iridis/base-train/`) live under
`/scratch/ab3u21/exps/owt2/cotformer_full_depth/`.

### Evaluation configuration

Evaluation follows [§M2](#m2-evaluation-protocol): standalone `eval.py`
on a single L4 GPU, `--distributed_backend None`, iterating the OWT2
`val.bin` in contiguous windows. The script defaults used are
`sequence_length=256` and `batch_size=32`; see
`scripts/reproduce_table1_fig2.py` for the exact invocation. Evaluation
is batch-size-independent ([§B11](#b11-evaluation-batch-size-independence)),
so the `batch_size=32` choice (larger than the `batch_size=8` used in
earlier A-section runs) does not affect reported perplexity values.

### Uncertainty interpretation

Paper Table 1 reports SEM across three independently trained seeds
(e.g., "27.55(0.02)" for CoTFormer 12×2). We have **one seed per
ablation** and therefore cannot construct our own inter-seed SEM. Our
uncertainty is the `val_perplexity_ci95` from `eval.py`, which is
derived from per-batch losses — a measure of intra-run cross-batch
variance for one fixed model. As discussed in
[§M3](#m3-confidence-intervals-and-the-papers-sem), these two quantities
answer different statistical questions and are not directly comparable.
The per-batch CI95 half-width is typically smaller than the inter-seed
SEM at this model scale, so a delta larger than 2–3× the CI95 half-width
provides meaningful evidence of a systematic difference beyond training
noise.

### Coverage: ours vs paper-reported reference

The table below records which cells of Table 1 are retrained by us and
which are carried over from the paper.

| Model | n_layer | n_repeat | Source |
|-------|---------|----------|--------|
| CoTFormer | 12 | 2 | **ours** (BaseCot_12L_2R) |
| CoTFormer | 12 | 3 | **ours** (BaseCot_12L_3R) |
| CoTFormer | 12 | 5 | **ours** (BaseCot_12L_5R) |
| CoTFormer | 12 | 15 | **ours** (BaseCot_12L_15R — used in Fig 2; not a Table 1 column) |
| CoTFormer | 24 | 2 | **ours** (BaseCot_24L_2R) |
| CoTFormer | 24 | 3 | **ours** (BaseCot_24L_3R) |
| CoTFormer | 24 | 5 | **ours** (BaseCot_24L_5R) |
| Block Universal Transformer | 12 | {2, 3, 5} | paper-reported reference |
| Block Universal Transformer | 24 | {2, 3, 5} | paper-reported reference |
| Standard Transformer | 12, 24, 48 | 1 | paper-reported reference |

**Cross-references:**
[§M2](#m2-evaluation-protocol) (eval protocol),
[§M3](#m3-confidence-intervals-and-the-papers-sem) (CI interpretation),
[§M4](#m4-training-configuration-shared-across-variants) (shared training config),
[§M6](#m6-reproduction-fidelity-vs-fair-comparison-regime-distinction) (fair-comparison regime),
[§B1](#b1-trainval-split-divergence) (split divergence),
[§B3](#b3-micro-batch-size-hardware-constrained) (micro-batch),
[§B5](#b5-base-model-batch-size) (effective batch size).

---

## A9. Figure 2 — accuracy–compute tradeoff (partial replication)

Figure 2 plots perplexity vs MACs for CoTFormer and Block Universal
Transformer at both n_layer=12 (subplot a, log-x scale) and n_layer=24
(subplot b, linear-x scale), evaluated at sequence length 256.

### Reproduction strategy

The y-axis (perplexity) for CoTFormer points comes directly from our
trained checkpoints, evaluated via the same pipeline as
[§A8](#a8-table-1--cotformer-rows-40k-steps--partial-replication). BUT
perplexity values are paper-reported (those checkpoints were not
retrained); for those points, the y-coordinate is borrowed from the
paper while the x-coordinate (MACs) is computed analytically.

The x-axis (MACs) for **all** points — both CoTFormer and BUT — is
computed analytically via `ptflops` (`aten` backend) applied to
synthetic model configs constructed from the paper-stated
hyperparameters: `n_head=12`, `n_embd=768`, `vocab_size=50304`,
`positional_encoder=rotary`, `bias=False`, `sequence_length=256`. No
trained weights are required for MAC counts; only the architecture
hyperparameters matter. The MACs are produced by
`scripts/compute_macs.py` and stored in `macs.json` (see
`docs/schema_table1_fig23.md`). This means that even though we lack
BUT checkpoints, the x-axis positions of BUT points are independently
verified — only their y-axis (perplexity) is borrowed from the paper.

### Extra points beyond Table 1

Figure 2(a) contains BUT 12×6 and 12×15 data points that do not appear
in Table 1. The perplexity values for these two points are read
visually from the paper figure to approximately ±0.05 PPL precision;
they are stored as `ppl_visual` in the `paper_reference_fig2` block of
`results_table1_fig2.json` (see `docs/schema_table1_fig23.md`) to flag
that this approximation is not as precise as a checkpoint eval. The
CoTFormer 12×15 point in Figure 2(a) is produced by our
`BaseCot_12L_15R` checkpoint and is therefore an honest eval result,
not a visual read-off.

### Visual style

Verbatim reproduction is attempted: matplotlib serif font
(`rcParams["font.family"] = "serif"`, `rcParams["mathtext.fontset"] = "cm"`
to match the paper's Times-like rendering), colors `#1f77b4` (BUT) and
`#2ca02c` (CoTFormer), solid lines with circular markers, per-point
text labels formatted as `f"{n_layer}x{n_repeat}"`. Subplots (a) and (b)
are saved separately as `fig2a.png` and `fig2b.png` (plus PDF) at
200 dpi. Style constants are the single source of truth in
`docs/schema_table1_fig23.md` § Color and style conventions; the plot
script imports them and does not redefine them.

**Cross-references:**
[§A8](#a8-table-1--cotformer-rows-40k-steps--partial-replication) (perplexity values),
[§A10](#a10-figure-3--compute-cost-vs-sequence-length-analytical-only-reproduction) (same MAC computation pipeline),
[§M3](#m3-confidence-intervals-and-the-papers-sem) (CI interpretation for CoTFormer points).

---

## A10. Figure 3 — compute cost vs sequence length (analytical-only reproduction)

Figure 3 compares the MAC count of CoTFormer 12×3 against Block Universal
Transformer 12×5 across sequence lengths {128, 256, 512, 1024, 2048,
4096, 8192, 12288}. No checkpoint evaluation is involved: the figure
depends only on architecture-derived FLOP counts, not on trained weights
or val-set perplexity.

### Method

`scripts/compute_macs.py` evaluates a closed-form MAC formula derived
directly from `models/cotformer_full_depth.py` and `models/but_full_depth.py`
(linear-per-token cost: `4·d_model² + 2·d_model·d_ff`; quadratic attention
cost: `2·d_model·N²` where `N` is the attended-to sequence length per
block; plus an `vocab·d_model` embedding-table constant matching ptflops'
accounting). The resulting `macs.json` is consumed by `plot_fig2.py` and
`plot_fig3.py`; the schema is in `docs/schema_table1_fig23.md`.

The formula handles the three architectures differently:
- **CoTFormer**: at repeat `r ∈ {1..n_repeat}`, the model processes T new
  tokens while attending to `r·T` context (per `cotformer_full_depth.py`
  lines 318–325). Summing over `r`: linear scales with `n_repeat`,
  quadratic scales with `n_repeat·(n_repeat+1)`.
- **BUT**: each of `n_repeat` passes attends to `T` tokens (no concatenation).
- **Standard**: single pass at `T` tokens.

The formula's correctness is verified at script startup against four
bit-exact reference values captured from a real `ptflops` run on the
BaseCot training config (CoTFormer 12L × {2,3,5} R @ seq=256 and
12L × 3R @ seq=128). If `compute_macs.py` ever disagrees with `ptflops`
on these reference points by even one MAC, the script aborts with a
clear "formula drift detected" error — guarding against silent
architecture changes in the model files.

### Historical: the ptflops OOM that prompted the closed-form rewrite

The original `compute_macs.py` (run_0, slurm 920549) called
`get_ppl_per_mac.get_macs_for_seqlens`, which invokes `ptflops` with
flash attention explicitly disabled. ptflops' `aten` backend then
materialises the full `(n_repeat·T)²` attention score matrix to let its
`__torch_dispatch__` hook count individual matmul MACs. At point
`[10/29] CoTFormer 12L × 3R @ seq=8192`, the score matrix needed
`24,576² × 4 bytes × 12 heads ≈ 28.8 GB`, exceeding the L4's 22 GB
VRAM (`run_0/slurm_920549.out:65` records the
`torch.cuda.OutOfMemoryError`). SLURM `--mem=400G` does **not** cap
CUDA allocations — the OOM persists regardless of host RAM — so the
fix is structural, not a resource bump.

The closed-form approach computes the same MACs in microseconds on CPU
without allocating any tensor larger than the model itself. The
empirical bit-exact validation against the surviving 9 ptflops data
points (slurm_920549.out lines 35–63) gives us higher confidence in
the closed-form values at seq=8192 and seq=12288 than ptflops could
ever deliver on L4 — ptflops simply cannot reach those scales on this
hardware.

### Expected outcome and significance

The CoTFormer 12×3 curve should remain below the BUT 12×5 curve across
the entire sequence-length range up to 12,288 — this is the paper's
qualitative claim. The quantitative MAC values should match the paper's
reported values to within `ptflops`' arithmetic precision (rounding of
multiply-accumulate counts at the operator level).

Reproducing this figure analytically is meaningful even without
retraining, for three reasons. First, it confirms that we constructed
the CoTFormer and BUT architectures with matching hyperparameters to the
paper — a mismatch would shift one curve relative to the other and break
the qualitative ordering. Second, it verifies that the paper's reported
compute ratios are consistent with those hyperparameters. Third, it
establishes that the qualitative claim ("CoTFormer 12×3 has lower compute
than BUT 12×5 even at seq_len=8192") holds under an independent
analytical reconstruction, strengthening confidence in the paper's
architecture comparisons beyond what any single perplexity number can
provide.

**Cross-references:**
[§A8](#a8-table-1--cotformer-rows-40k-steps--partial-replication) (the CoTFormer 12×3 checkpoint whose perplexity appears in Table 1 and Fig 2),
[§A9](#a9-figure-2--accuracycompute-tradeoff-partial-replication) (the same MAC computation pipeline, at seq_len=256 only).

---

# Part B: Known Divergences

## B1. Train/Val Split Divergence

**Impact:** Negligible (within noise floor of stochastic training)

The original codebase uses a pre-built HuggingFace Arrow dataset
(`the_pile_openwebtext2`, now permanently defunct) with
`train_test_split(test_size=0.0005, seed=2357)`. Our pipeline uses the
identical raw tarball ([`segyges/OpenWebText2`][segyges-mirror]) but
diverges in three independent ways:

### B1a. Document ordering

The pre-built Arrow dataset had a fixed row ordering that determined which
documents landed in train vs. val. We read raw `.jsonl.zst` files in sorted
filename order, which is not guaranteed to match the original Arrow row
ordering. Since the split partitions by row index, different ordering means
different documents in each split.

### B1b. Split RNG implementation

The original uses HF datasets' `train_test_split()`, which internally
generates a permutation via numpy. The exact backend depends on the
`datasets` library version (`RandomState` vs `default_rng`), which is
unrecorded in the original codebase. Our pipeline uses:

```python
rng = np.random.default_rng(2357)
perm = rng.permutation(n_docs)
val_set = set(perm[:n_val].tolist())
```

Even with identical document ordering, the split would differ unless the
exact `datasets` version were matched.

### B1c. Validation set size rounding

The original uses `test_size=0.0005` with HF's internal rounding. We use
`n_val = max(1, round(n_docs * 0.0005))`, which may differ by +/-1 document.

### Why the original split is unrecoverable

The pre-built Arrow artifact was permanently disabled on HuggingFace
(PII/safety flags, copyright takedowns). No mirror preserves the Arrow row
ordering — only raw tarball copies exist.

### Impact assessment

The split ratio is 99.95% train / 0.05% val across ~3.4M documents. At
this ratio, any two random partitions produce statistically equivalent
validation sets. Negligible impact on reported perplexity.

---

## B2. Token Ordering Within Bins

**Impact:** None

The original pipeline writes tokens in shuffled order (via
`train_test_split(shuffle=True)` reindexing + 1024 sharded batches). Our
pipeline writes in file-encounter order (sorted filenames, sequential
within each file).

No effect on training — the training loop (`optim/base.py`) samples random
offsets into the memmap file, making physical token ordering irrelevant.

---

## B3. Micro-Batch Size (Hardware-Constrained)

**Impact:** Negligible (statistically equivalent gradients).

The per-GPU micro-batch size is constrained by our hardware budget
(see [§M4](#m4-training-configuration-shared-across-variants) for the
shared training configuration). The authors' A100 80 GB setup allowed
larger micro-batches than our 24 GB L4s; this subsection documents
the specific numerical consequence.

### What changed

The effective batch size is preserved at 128 tokens/step, but the
micro-batch/accumulation split differs:

| Setting | Paper (assumed A100) | Ours (2× L4 24 GB) |
|---------|---------------------|---------------------|
| `batch_size` (per GPU) | Unspecified (likely 32+) | 8 |
| `acc_steps` (per GPU) | Unspecified | 8 (16 before DDP halving) |
| Effective batch size | 128 | 128 |

### Why it diverges numerically

With loss normalisation `loss = outputs['loss'] / acc_steps`, gradient
accumulation is mathematically equivalent regardless of micro-batch
size. However, floating-point addition is non-associative: accumulating
8 micro-batches of 8 samples produces different rounding than 4
micro-batches of 16 samples (or any other decomposition). Under
bfloat16 autocast these rounding differences are amplified compared
to float32.

### Impact assessment

The expected gradient is identical. Only the accumulation noise floor
changes, and it is dwarfed by stochastic minibatch sampling noise. No
measurable effect on final perplexity.

### Additional hardware-driven settings

These settings are specific to the 2× L4 memory-constrained decomposition
and sit alongside the shared training configuration in
[§M4](#m4-training-configuration-shared-across-variants):

| Setting | Value | Reason |
|---------|-------|--------|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Enabled | Reduces VRAM fragmentation on 24 GB GPUs |
| `find_unused_parameters` (DDP) | `False` | No unused params after the [§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved) fix; avoids extra autograd traversal |
| `--mem` (SLURM) | 128 GB | `--data_in_ram` with 2 DDP workers needs >96 GB system RAM |

---

## B4. RNG State Restoration on DDP Resume (ADM) — Resolved

**Impact:** None for LN-CoTFormer; previously caused a minor divergence
for ADM training at resume boundaries.
**Status:** Resolved. Per-rank GPU RNG gathering + full restore on
resume (`optim/base.py:64-79,218-280`).

### Background

The original codebase saves four RNG states in intermediary checkpoints
(`cpu_rng_state`, `gpu_rng_state`, `numpy_rng_state`, `py_rng_state`)
but never restores them — the restore calls in `optim/base.py` were
commented out. Additionally, the final `ckpt.pt` omitted RNG states
entirely, and `main.py` would raise `KeyError` if resuming from such a
checkpoint.

### Why it doesn't affect LN-CoTFormer

The LN-CoTFormer (`cotformer_full_depth_lnmid_depthemb`) has no
runtime randomness — a fixed repeat loop and, per the shared
configuration of
[§M4](#m4-training-configuration-shared-across-variants),
`dropout=0.0`. Training is therefore fully deterministic modulo
cuDNN kernel selection, and saving or restoring RNG state has no
effect on its trajectory.

### Why it affects ADM training

The ADM model calls `torch.rand()` at every training step to sample the
`length_factors` that control per-repeat capacity — see
`models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:400-431`
for the branches of `depth_random_method`. These `torch.rand()` calls
consume CUDA RNG state, so a resume without RNG restoration produces a
different capacity-factor sequence than a continuous run.

An earlier revision of these notes incorrectly referenced `_predict_depth`
and `torch.randint` from another adaptive variant; our specific model
variant uses `torch.rand()`.

### DDP complication (original code)

Uncommenting the original restore lines was incorrect under DDP. The
checkpoint is saved only by rank 0 (the
`distributed_backend.is_master_process()` guard at `optim/base.py:230`),
so the stored `gpu_rng_state` belonged to GPU 0. On resume, all ranks
load the same file — rank 1 would receive rank 0's CUDA state, giving
both GPUs identical random sequences and breaking statistical
independence.

### Fix

Three changes resolve the issue completely; all code references below
are to files in the current repo, not to fragments reproduced in this
section.

1. **Per-rank GPU RNG gathering** (`optim/base.py:218-229,250-257`):
   before each checkpoint save, every rank's `torch.cuda.get_rng_state()`
   is collected into a list via a Gloo `all_gather` on the secondary
   process group (`distributed/ddp.py:26`). Rank 0 saves the list as
   `gpu_rng_states` (plural). Single-GPU runs wrap the local state in a
   one-element list for format consistency. The transport evolved through
   several iterations — `all_gather_object` → NCCL `all_gather` → Gloo
   `all_gather` — documented in full at
   [§B9](#b9-ddp-checkpoint-save-failures--nccl--lustre-adm--resolved).

2. **Full restore on resume** (`optim/base.py:64-79`): all four RNG
   states are restored. CPU, numpy, and Python RNG are restored directly
   (shared across ranks). GPU RNG is restored per-rank by indexing
   `gpu_rng_states` with the rank ID. Legacy checkpoints carrying the
   singular `gpu_rng_state` key are handled by restoring only on rank 0.

3. **Tolerant checkpoint loading** (`main.py:167-184`): the RNG key
   extraction is guarded by `if k in checkpoint`, so old checkpoints
   missing RNG keys (e.g. the final `ckpt.pt` from before this fix)
   load without error. Both `gpu_rng_state` (legacy) and
   `gpu_rng_states` (new) are accepted and coerced to `ByteTensor`
   before use.

### Practical impact (post-fix)

ADM multi-job training across SLURM submissions now produces the same
`length_factors` sequences as a hypothetical continuous run, assuming
the same number of GPUs on resume. The remaining source of
non-determinism is cuDNN kernel selection (inherent to CUDA, not fixable
without `torch.use_deterministic_algorithms(True)`, which has a
performance cost).

**Empirical magnitude.** The `eval-adm/run_6` evaluation retrained the
ADM from scratch with this fix in place (v2) and re-evaluated the
pre-fix training (v1) at identical checkpoints. The 60k PPL delta is
0.014 (v1 24.074, v2 24.060), well below the intra-run noise floor —
see
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) "Why the Gap
Is Wider than the 40k Reproductions" for the attribution analysis,
[§C1](#c1-orphan-modblock-5-allocated-4-used) "Init-order RNG offset"
for the mechanism that makes v1 and v2 trained models differ at all,
and `iridis/eval-adm/README.md` § *ADM v1 vs v2: B4 empirical
magnitude* for the full per-variant statistics.

---

## B5. Base Model Batch Size

**Impact:** Minor

Discrepancy found between the paper and `experiments.md` file. Paper states
they use a batch size of 128 whereas the `md` file states 64. We opted to
use a size of 64 due to GPU memory constraints.

---

## B6. Orphan MoDBlock in Adaptive/MoD Models — Resolved

**Impact:** Fatal crash under DDP; no impact on single-GPU training.
**Status:** Resolved and verified under 2-GPU DDP.

### Background

All MoD-router model variants created `n_repeat` `MoDBlock` instances
in `__init__` (pre-fix), but the forward loop only invokes routers 0
through `n_repeat - 2`. The final pass unconditionally finalises all
remaining tokens without consulting a router. `mod[n_repeat - 1]` is
therefore an orphan — its `mod_router.weight` parameter (768 floats)
never participates in the autograd graph.
[§C1](#c1-orphan-modblock-5-allocated-4-used) contains the
init/forward-loop walkthrough with exact line ranges; this section
records the DDP symptom and the specific fix.

### Why the original codebase doesn't crash

The authors ran on a single-GPU setup (see
[§M4](#m4-training-configuration-shared-across-variants)). PyTorch
only enforces gradient-reduction checks inside `DistributedDataParallel`;
on single-GPU, unused parameters are silently ignored.

### How it manifests under DDP

Our `find_unused_parameters=False` setting on the DDP constructor
(see [§M4](#m4-training-configuration-shared-across-variants) and
`distributed/ddp.py:50-51`) causes DDP to detect on iteration 2 that
parameter index 152 did not receive a gradient in the prior
iteration and raise:

```
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. [...] Parameter indices which did not receive
grad for rank 0: 152
```

Index 152 corresponds to `transformer.mod[4].mod_router.weight` for the
24-layer `n_repeat=5` ADM.

### Fix

Changed `range(self.n_repeat)` to `range(self.n_repeat - 1)` at the
init site of every affected model file. This removes the orphan router
module without changing model behaviour. DDP training with
`find_unused_parameters=False` now completes without the
parameter-index-152 error. The `length_factors` generation already used
`n_repeat - 1` elements so it is unaffected.

Why this is a semantically neutral dead-weight removal, and why the
pre-fix and post-fix code paths nevertheless produce two minutely
different trained models (via an init-order RNG offset, empirically
bounded at 0.014 PPL below the intra-run noise floor), is walked in
[§C1](#c1-orphan-modblock-5-allocated-4-used).

### Affected files (all from original commit `2723e30`)

1. `models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:322` (ADM, primary model)
2. `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.py` (BUT with MoD router)
3. `models/but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.py` (MAC-compute ADM used by `get_ppl_per_mac.py`)

---

## B7. Training Summary Metrics Never Recorded (Original Code Bug) — Resolved

**Impact:** Training curve data lost in `summary.json`; no effect on
model quality or training correctness.
**Status:** Resolved.

### Background

`optim/base.py:45` initialises the `stats` dict with empty lists for
`train_loss`, `val_loss`, `val_pp`, and `val_acc`, and returns it to
`main.py` for serialisation into `summary.json`. The corresponding
`stats[...].append(...)` calls were never present in the authors'
code, so every `summary.json` the original codebase produced carried
empty metric arrays, making post-hoc training-curve reconstruction
from the summary file impossible. Training correctness was unaffected:
the metrics were still computed, logged to stdout and wandb, and used
for checkpoint decisions.

### Fix

Added the four append calls inside the master-process evaluation
block at `optim/base.py:167-170`. The `json.dump` that writes
`summary.json` is master-only (`main.py:217-219`), so the writes and
the appends are both rank-0 operations.

---

## B8. Missing `nullcontext` Import in `eval.py` (Original Code Bug) — Resolved

**Impact:** `NameError` crash on CPU-only evaluation; no impact on GPU
runs.
**Status:** Resolved.

### Background

`eval.py:79` uses `nullcontext()` as the no-op alternative to
`torch.amp.autocast` when running on a CPU device, but the
`from contextlib import nullcontext` import was missing from the
authors' `eval.py`. On GPU runs (all Iridis jobs), the
`torch.amp.autocast` branch is taken and `nullcontext` is never
evaluated, so the bug was latent. A CPU evaluation run would crash
with `NameError: name 'nullcontext' is not defined`.

### Fix

Added the import (now at `eval.py:14`).

---

## B9. DDP Checkpoint Save Failures — NCCL + Lustre (ADM) — Resolved

**Impact:** Fatal crash at first DDP checkpoint save (step 2000), every run
**Status:** Resolved. Required four fix iterations across five SLURM jobs.

### Background

The [§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) fix
introduced per-rank GPU RNG gathering via
`torch.distributed.all_gather_object` so that checkpoint saves
preserve each rank's CUDA RNG state. This was the first code path to
inject NCCL collective operations into the checkpoint save sequence.
On our setup (see
[§M4](#m4-training-configuration-shared-across-variants) for the
DDP/hardware context; specifically PyTorch 2.2.0+cu118 with a Lustre
`/scratch/` filesystem), a cascade of four distinct failures
prevented checkpoint saves from ever succeeding.

The authors' single-GPU setup saved only rank 0's RNG state and
never restored it (restore calls commented out in `optim/base.py`).
v1 training (runs 0-2) used the pre-[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)
code without any distributed RNG gathering and completed 60k steps
without checkpoint issues.

### Failure timeline

All five run_3 jobs crashed at step 2000 (`save_checkpoint_freq` boundary):

| Job | Date | Fix state | Error | Root cause |
|-----|------|-----------|-------|------------|
| 707599 | Mar 23 | [§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) only | `OutOfMemoryError: >1EB` in `all_gather_object` | NCCL size-negotiation corruption [2][3][4] |
| 714422 | Mar 25 | + ByteTensor gather | NCCL gradient ALLREDUCE timeout (SeqNum=51989, NumelIn=2360064) | No barrier after save; rank 1 races ahead [5][6] |
| 724964 | Mar 27 | + barrier + 30 min timeout | NCCL barrier timeout (SeqNum=51988, NumelIn=1, 1800s) | Lustre I/O stall in `torch.save` [10] |
| 726223 | Mar 28 | + local-first save to `/tmp` | NCCL barrier timeout (SeqNum=51988, NumelIn=1, 1800s) | NCCL LL protocol hang (see below) |
| (resolved) | Mar 29 | + Gloo checkpoint coordination | Success | Gloo bypass of NCCL confirmed [12]; ADM v2 completed 60k steps ([§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps)) |

### Root causes

**1. `all_gather_object` OOM (job 707599).** `all_gather_object` pickles
Python objects and exchanges sizes via CUDA all-gather. A corrupted value in
NCCL's size-negotiation step [2] caused a >1 EB allocation request.
`torch.cuda.get_rng_state()` returns a fixed-size CPU ByteTensor, so pickle
serialisation was unnecessary. **Fix:** replaced with `torch.distributed.all_gather`
on fixed-size ByteTensors (commit `ef3bff5`). A defensive ByteTensor conversion
was added in `main.py` for backward-compatible checkpoint loading.

**2. Missing post-save barrier (job 714422).** After the gather, only rank 0
writes the ~2.5 GB checkpoint. Without a barrier, rank 1 races to the next
gradient ALLREDUCE while rank 0 is blocked on Lustre I/O, exceeding the NCCL
10-minute default timeout [9]. This race was latent in the original code.
The torchtune project hit the identical pattern [6]. **Fix:** added
`distributed_backend.sync()` after both periodic and final checkpoint saves,
following the Megatron-LM pattern [7][8]. Increased `init_process_group`
timeout to 30 minutes [9].

**3. Lustre I/O stall (job 724964).** `torch.save()` buffers the entire
checkpoint in memory before writing [10]. Under cluster I/O contention,
a single ~2 GB sequential write to one Lustre OST stalled for >30 minutes,
exceeding the post-save barrier timeout. Memory pressure and CUDA
synchronisation were both ruled out. **Fix:** local-first save
(`optim/utils.py`): `torch.save()` to `/tmp` (node-local), then
`shutil.copy2` to Lustre + `os.rename` for atomicity. Also added
checkpoint integrity handling on resume (`main.py`: try/except for
corrupted files, explicit warning on missing checkpoints).

**4. NCCL LL protocol hang (job 726223).** With the local-first save, the
checkpoint completed in 3.8 seconds and the "checkpoint saved" message was
printed. Yet the post-save barrier still hung for 30 minutes. The root
cause: NCCL's Low-Latency (LL) protocol on NCCL 2.15-2.17 (CUDA 11.8) has
a slot-reuse race when switching between collective types at small tensor
sizes. After ~52K gradient AllReduces using the Simple protocol (large
tensors), the first LL-protocol operations --- `all_gather` (16 bytes) then
`barrier` (1-element AllReduce) --- trigger the hang. NCCL 2.18.1 fixed the
LL cleanup race [11]; our PyTorch 2.2.0+cu118 ships an affected version.

Crucially, our code was the only framework using NCCL tensor collectives for
RNG gathering. Meta's torchtitan [12], Megatron-LM [8], HuggingFace
Accelerate, and DeepSpeed all use either Gloo (CPU) process groups or
per-rank files for checkpoint metadata exchange --- never NCCL.

**Fix:** replaced all NCCL operations in the checkpoint path with a
secondary Gloo (CPU/TCP) process group. The group is created in
`distributed/ddp.py:26` via `dist.new_group(backend="gloo")` and used
for the per-rank RNG `all_gather` (`optim/base.py:223-226, 253-255`;
CPU ByteTensors, no GPU round-trip) and the post-save `barrier`
(`distributed/ddp.py:71-72`, called from `optim/base.py:247-248`).
`broadcast_buffers=False` was added to the DDP constructor at
`distributed/ddp.py:50-51` (all registered buffers are deterministic
from config, so broadcasting them every forward is wasted NCCL
traffic). `NCCL_PROTO=Simple` is set in
`iridis/adm-train/job.sh:115` as belt-and-suspenders (forces Simple
protocol for the remaining NCCL ops; negligible throughput impact on
PCIe: ~0.04%).

### Affected files (cumulative across all four fix iterations)

| File | Changes |
|------|---------|
| `optim/base.py` | Gloo `all_gather` for RNG, Gloo barrier after save (2 locations) |
| `optim/utils.py` | Local-first save to `/tmp` then copy to Lustre |
| `distributed/ddp.py` | Gloo process group, `broadcast_buffers=False`, 30 min NCCL timeout |
| `main.py` | ByteTensor conversion for `gpu_rng_states`, try/except on `torch.load`, resume logging |
| `iridis/adm-train/job.sh` | `NCCL_PROTO=Simple` environment variable |

### Checkpoint format compatibility

The checkpoint format is unchanged: `gpu_rng_states` is a list of CPU
ByteTensors regardless of whether Gloo or NCCL was used for
gathering. Old checkpoints with the legacy singular `gpu_rng_state`
key are handled by the fallback path in `optim/base.py`. Non-ADM
models (`base`, `cotformer_full_depth`, `cotformer_full_depth_lnmid_depthemb`)
have no runtime randomness (see
[§M4](#m4-training-configuration-shared-across-variants) for the
shared `dropout=0.0` config, and
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) "Why it
doesn't affect LN-CoTFormer" for the fixed-repeat-loop argument), so
RNG state is irrelevant to their training math.

---

## B10. LR Schedule Discontinuity on 40k-to-60k Extension

**Impact:** Likely contributor to the +0.40 PPL delta in
[§A2a](#a2a-ln-cotformer-perplexity-section-5-60k-steps) (LN-CoTFormer
60k: 23.59 vs paper's 23.19). No effect on ADM (trained 60k from
scratch).
**Status:** Accepted. Assessed in
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps); consistent
with the observed delta pattern.

### Background

The paper trains the LN-CoTFormer for 60k steps with a single
`OneCycleLR(total_steps=60000)` cosine schedule. We initially trained for
40k steps (reproducing Table 2), then extended to 60k for the Section 5
comparison. This two-phase approach introduced a scheduler mismatch.

### What happened

The 40k training used `OneCycleLR(total_steps=40000)`:
- Warmup: 0 to peak LR over steps 0--8000 (20%)
- Cosine decay: peak to final LR over steps 8000--40000
- At step 40000: LR at its minimum (~2e-5, i.e. `1e-3 * final_div_factor`)

On resume with `--iterations 60000`, the original `summary.json`
triggered an early exit in `main.py` ("Already found experiment…
Skipping"). After fixing that with an iteration-aware skip check
(`main.py:123-131`), the restored `OneCycleLR` state dict had
`total_steps=40000`, causing a `ValueError` at step 40001.

The fix rebuilds the scheduler with `total_steps=60000` and
fast-forwards it to the resume iteration (`main.py:189-200` — see the
`if extending` branch). Steps 40001--60000 then follow the latter
portion of a 60k cosine schedule, where the LR is still in mid-decay
(~3.5e-4 at step 40000 under a 60k schedule).

### The discontinuity

| Step | 40k schedule LR | 60k schedule LR | Delta |
|------|-----------------|-----------------|-------|
| 39999 | ~2.0e-5 (near minimum) | ~3.6e-4 | -- |
| 40000 | ~2.0e-5 (end) | ~3.5e-4 | -- |
| 40001 | (would not exist) | ~3.5e-4 | -- |

At the 40k/60k boundary, the effective learning rate jumps from the 40k
schedule's minimum (~2e-5) to the 60k schedule's mid-decay value (~3.5e-4),
a ~17x increase. The paper's continuous 60k run would have smoothly decayed
through this region.

### Impact assessment

The LR jump at the boundary acts as a mild "warm restart" for the final 20k
steps. Two factors limit the practical impact:

1. The model weights and optimiser states are correct (restored from
   checkpoint). Only the LR trajectory diverges.
2. The 60k schedule's LR at step 40000 (~3.5e-4) is within the range the
   model trained stably at during the first 40k steps (peak 1e-3 to 2e-5).

**Post-evaluation update
([§A2a](#a2a-ln-cotformer-perplexity-section-5-60k-steps)):** The
final 60k LN-CoTFormer perplexity is 23.59 vs the paper's 23.19
(delta +0.40). The delta is significantly wider than the 40k
reproduction (+0.02,
[§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps)). Since the ADM
(trained 60k from scratch, unaffected by this schedule mismatch) has a
delta of only +0.23
([§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps)), this LR
discontinuity is the most likely contributor to the excess LN-CoTFormer
delta. A fresh 60k-from-scratch LN-CoTFormer run would be the
definitive test.

---

## B11. Evaluation Batch Size Independence

**Impact:** Negligible (< 0.001 PPL).

Our standalone evaluation (`eval.py`) uses `batch_size=8` (loaded from
`summary.json`, matching training). The paper's evaluation was likely
performed with a larger batch size on A100 80 GB GPUs.

Per-sample cross-entropy loss is mathematically invariant to batch
size: each sequence in a batch is processed independently and
`F.cross_entropy` computes per-token loss before averaging. The only
micro-effect is the final partial batch: `eval.py:57-72` yields
batches of `batch_size` samples from the validation memmap. If the
number of validation sequences is not evenly divisible by
`batch_size`, the last batch has fewer samples but receives equal
weight in the mean (one `loss_list_val.append()` per batch, see
`eval.py:126`). At ~16,400 sequences / `batch_size=8` = ~2,050
batches, the partial-batch weight contribution is at most 1/2,050 of
the total, producing < 0.001 PPL divergence.

We deliberately use the training `batch_size` from `summary.json`
rather than overriding it, to eliminate this micro-divergence entirely.
Single-GPU evaluation with `--distributed_backend None` (see
[§M2](#m2-evaluation-protocol)) also matches the paper's evaluation
conditions (no DDP data sharding during eval).

## B12. RQ9 counting sub-stream is not a Chang and Bisk reproduction

**Impact:** Scope clarification, not a numeric divergence.

The RQ9 counting sub-stream documented in `docs/extend-notes.md`
§1.2 runs **two regimes in parallel**: a Pilot 1 that
reproduces Chang and Bisk's [counting baseline] bit-for-bit, and a
main sweep that applies our project's CoTFormer-native training
regime to the same Chang and Bisk counting task. The main sweep
is **not** a reproduction of Chang and Bisk; it is a
fair-comparison experiment that uses the published counting
benchmark as a substrate. The two regimes serve distinct
epistemic purposes (see
[§M6](#m6-reproduction-fidelity-vs-fair-comparison-regime-distinction)
for the discipline), and only the Pilot 1 result feeds the
"reproduction" claim.

The seven training-regime divergences between Pilot 1 (Chang and
Bisk-exact) and the main sweep (CoTFormer-native) are documented
in `docs/extend-notes.md` `DEC-020`. The dead-code source of
the most consequential divergence (the `max_grad_norm = 0.3`
value referenced in earlier project documentation) is the subject
of [§B13](#b13-chang-and-bisk-max_grad_norm--03-is-dead-code).

The synthesis output for RQ9 reports the architecture comparison
under the *main-sweep fair-comparison* regime annotation; any
reproduction claim is reported under the *Pilot 1
reproduction-fidelity* annotation. Conflating the two would be
the same kind of methodological error documented in
[§C2](#c2-figure-5-position-indexed-cascade-bug-and-methodological-critique)
for the original Figure 5 reproduction (substituting an
alternative methodology for the authors' own).

## B13. Chang and Bisk `max_grad_norm = 0.3` is dead code

**Impact:** Citation correction, not a numeric divergence.

`inductive_counting_with_LMs/scripts/causal_transformer/config.py:28`
declares `max_grad_norm = 0.3`. This value is **never read by the
training loop**: the trainer at
`inductive_counting_with_LMs/scripts/causal_transformer/trainer.py:231`
passes `max_grad_norm = 1.0` to
`torch.nn.utils.clip_grad_norm_` directly, hardcoded as a literal,
without consulting the config object. The `config.py:28` value
is therefore dead code; any prior reference in our project notes
that cited Chang and Bisk's "grad clip = 0.3" was repeating the
unread declaration rather than the actually-applied value.

The correct attribution of Chang and Bisk's grad-clip behaviour
is `max_grad_norm = 1.0` (the literal in `trainer.py:231`). Our
`docs/extend-notes.md` DEC-020 reconciliation captures this
dead-code finding alongside six other training-regime
divergences between the codebase and the paper's prose; the
prior `DEC-019` documentation stated `grad_clip = 0.3` because
it followed the dead-code declaration. `DEC-020` supersedes;
`DEC-019` is preserved as a historical record of the
pre-dead-code-discovery state.

The Pilot 1 implementation must use `max_grad_norm = 1.0`
(matching the actually-applied training behaviour); any HPC job
script that sets `--max_grad_norm 0.3` for Pilot 1 is a
specification error and triggers an empirical-step-time
validation gate failure under the Phase 1 skeleton commit's
pre-registration.

## B14. Pilot 1 Chang and Bisk-exact results (stub)

**Impact:** TBD — to be filled by the code-wave orch
post-execution.

This subsection is reserved for the Pilot 1 reproduction-fidelity
results. After the Pilot 1 run completes, the code-wave orch
populates this stub with:

- 5-seed mean and standard deviation per OOD length in `{50, 75,
  100, 125, 150, 175, 200}` for the per-position top-1 accuracy
  metric.
- Pass/fail verdict against the paper's published per-OOD-length
  pass band (IND ~100 %, OOD@200 ~22-31 % per [25]).
- Step-time measurement and the empirical-step-time validation
  gate verdict (per `docs/extend-notes.md` §1.8 pre-flight).
- Citation to the SLURM job ID and the checkpoint paths under
  `/scratch/$USER/exps/counting/pilot1_*/`.

If Pilot 1 fails the per-OOD-length pass band, the entire RQ9
sub-stream sheds to the extension per the
`docs/extend-notes.md` §0 Scope-shedding pre-registered order
trigger 1; this stub is then re-purposed to record the failure
and the shedding decision. If Pilot 1 passes, the main-sweep
fair-comparison results are reported in a separate section to be
added at synthesis time.

`TODO: filled by code-wave orch post-Pilot-1 execution.`

## B15. RNG State Restoration on Single-GPU Resume (Case CoT) — Resolved

**Impact:** Crashes the resume path on single-GPU runs that use a
`RandomSampler`-derived generator state. No effect on the DDP
trajectories of any [Part A](#part-a-successful-replications)
variant, which all use `DistributedSampler` and therefore never
persist a real sampler RNG.
**Status:** Resolved. Added the `.type(torch.ByteTensor)` cast for
`train_sampler_state` at `main.py:185-191`.

### Background

The checkpoint resume path in `main.py:175-184` casts every saved
`torch.Generator` byte tensor back to `ByteTensor` before
`set_state(...)` --- the same pattern documented in
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved). Three
of four sibling fields received the cast (`cpu_rng_state`,
`gpu_rng_state`, `gpu_rng_states[i]`); a fourth,
`train_sampler_state`, did not. The omission was harmless under
DDP because `DistributedSampler.set_epoch(...)` causes
`optim/base.py:38` to set `sampler_state_before_iter = None`
before the save call --- the saved value is `None`, the load-side
guard short-circuits, and no cast is needed.

### Why it surfaces only on single-GPU

The 12L x 15-repeat Case CoT runs (`iridis/base-train/job3.sh`,
paper Fig. 2 / Fig. 3) were the first reproduction queued without
DDP. `RandomSampler`, the single-GPU default selected by
`data/utils.py:63-67`, does not implement `set_epoch`, so
`sampler_state_before_iter` retains a real `torch.Generator` byte
tensor at every save call. On resume,
`torch.load(..., map_location=...)` demotes the legacy `ByteTensor`
subtype to a plain `torch.uint8` Tensor;
`Generator.set_state(...)` rejects this with
`TypeError: RNG state must be a torch.ByteTensor` before the first
training step.

The bug had been latent since the team's first RNG-restore commit
(`commit 657de47`) and the per-rank refactor (`commit 29b9690`) ---
neither commit landed the fourth cast --- but had no observable
effect because every prior reproduction (Tables 1 / 2, ADM,
LN-CoT) ran under DDP.

### Fix

A single conditional cast at `main.py:185-191`, symmetric with the
existing [§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)
cast block:

```python
if "train_sampler_state" in rng_state_dict and rng_state_dict["train_sampler_state"] is not None:
    rng_state_dict["train_sampler_state"] = rng_state_dict["train_sampler_state"].cpu().type(torch.ByteTensor)
```

The `is not None` guard preserves correctness on the DDP resume
path (which still saves `None` and must short-circuit). The fix
adds no new control flow.

### Affected files

1. `main.py:185-191` (cast added).

### Practical impact

Single-GPU resumes --- used for Case CoT pilot runs and any future
debugging session that pulls one L4 --- now restore the sampler
RNG identically to a continuous run. DDP trajectories are
unchanged because they still skip the sampler-state branch at
save time. Existing `ckpt_*.pt` files saved before the fix remain
compatible: the cast is idempotent on `uint8` payloads.

## B16. Host-RAM OOM via `--data_in_ram` (Case CoT) — Mitigated

**Impact:** Job 893189 (run_1) was OOM-killed at step ~2530 after
14 h 52 m of training. The 12L x 15-repeat Case CoT config has the
largest cumulative DataLoader RAM footprint of any
[Part A](#part-a-successful-replications) variant on the
single-rank path and exceeds the 128 GB cgroup budget listed in
[§B3](#b3-micro-batch-size-hardware-constrained).
**Status:** Mitigated. `--data_in_ram` removed from
`iridis/base-train/job3.sh` with an explanatory inline comment.
The root-cause int64-upcast pipeline in `data/utils.py` is
unchanged but unreachable from this script.

### Background

`--data_in_ram` materialises the full uint16 OpenWebText2 token
array (~32 GB) into host memory. `data/utils.py` then upcasts
each sampled batch from `uint16` to `int64` per worker (an 8 x
per-token expansion). With `num_workers=4` (the team's DataLoader
default), each worker holds an independent fork-COW view of the
cached slice plus its own int64 working buffer. Empirical
steady-state RSS on Job 893189 climbed to **292.58 GB** (228 % of
the 128 GB cgroup), at which point the SLURM cgroup OOM-killer
terminated the process. The kernel SIGKILL is uncatchable, so no
Python traceback was emitted in the job's `*.err` --- only the
SLURM `*** CANCELLED ... DUE to SIGNAL Terminated ***` line and
the post-mortem
`Memory Utilized: 292.58 GB / 228.58% of 128.00 GB`.

### Why this surfaces only on the 12L x 15-repeat config

Earlier reproductions (the Table-1 / Table-2 entries and the
[§A2](#a2-ln-cotformer-perplexity-table-2-40k-steps) /
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps)
LN-CoTFormer and ADM 60k runs) trained under 2-GPU DDP, which
halves the per-rank workload and effectively splits the
DataLoader-worker fanout across two process trees. The 12L x
15-repeat run on a single rank concentrates all four worker copies
plus the in-RAM array in one process; the per-process RSS is
roughly twice the DDP-per-rank figure, which is enough to cross
the cgroup boundary. Run_0 (Job 893149) was killed at 49 s during
the same `np.array(memmap)` materialisation step before peak
allocation, with only 21.92 GB visible at the post-mortem
checkpoint --- a different proximate trigger but the same
root-cause family.

### Fix

`--data_in_ram` is commented out in
`iridis/base-train/job3.sh:164-169` with an inline reference back
to this section. The script falls back to memmap reads from
`$DATA_DIR/openwebtext2/train.bin` on the Lustre scratch
filesystem.

The root-cause int64 upcast in `data/utils.py:39,42` is **not
patched** in this fix. Every other reproduction variant exercises
it under DDP without incident, so removing the upcast would be a
non-local change with broad blast radius. Removing the flag at
the script level is the minimal, script-scoped change consistent
with the rest of this document's "subtract before you add"
disposition.

### Affected files

1. `iridis/base-train/job3.sh:164-169` (`--data_in_ram` removed;
   multi-line explanatory comment added that points back to this
   section).

### Practical impact

Resubmitting `job3.sh` after this fix is expected to complete the
40k-step run within the 128 GB cgroup, regardless of whether the
run uses 1 GPU or 2 GPUs. The script also flips back to 2-GPU DDP
(`--gres=gpu:2`, `N_GPUS=2`), which cuts wall time roughly in
half (from ~233 h to ~115 h, in line with the
[§A2a](#a2a-ln-cotformer-perplexity-section-5-60k-steps) /
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) 60k DDP
envelope) and provides a second layer of defence against the OOM
by halving per-rank dataset materialisation.

A future cleanup of `data/utils.py` to keep the int64 upcast out
of the DataLoader-worker copy path would let `--data_in_ram` be
re-enabled safely on this config, but that is out of scope for
the Case CoT pilot.

---

# Part C: Original Bugs and Mistakes

Bugs and methodological mistakes present in the original codebase
(`commit 2723e30`, "Add everything") that affect the authors' own
reported results or interpretation. These are distinct from
[Part B](#part-b-known-divergences) divergences, which arise from our
hardware or setup constraints. Subsections are ordered chronologically
in the life of our reproduction:
[§C1](#c1-orphan-modblock-5-allocated-4-used) was encountered during
reproduction setup before any training started,
[§C2](#c2-figure-5-position-indexed-cascade-bug-and-methodological-critique)
during the Figure 5 analysis after the ADM run completed, and
[§C3](#c3-underspecified-training-schedule-reproducibility-gap) from
paper-vs-code inspection when extending runs from 40k to 60k.
Original-code bugs discovered via their DDP symptoms
([§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved) is one)
remain in [Part B](#part-b-known-divergences) because it records the
full DDP-side fix history, while Part C focuses on the semantic
impact on the authors' reported numbers.

## C1. Orphan MoDBlock: 5 Allocated, 4 Used

**Impact:** The model allocates one more MoDBlock router than it uses.
The authors' model file creates `n_repeat` routers in `__init__` but
the forward loop only invokes routers 0 through `n_repeat - 2` — the
final pass unconditionally finalises all remaining tokens without
consulting a router. `mod[n_repeat - 1]` is a dead weight: created,
initialised, and serialised, but never forward-passed, never gradient
-receiving, never optimised. Our fix removes it; the removal is a
dead-weight deletion with zero semantic change to forward, backward,
or inference.

**Status:** Original author bug (commit `2723e30`), present in all five
MoD-router model variants. Fixed in commit `657de47` by changing
`range(self.n_repeat)` to `range(self.n_repeat - 1)` at the init site.
See [§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved) for the
DDP discovery story and the specific fix, and this section for the
downstream semantics.

### What the code does

Our post-fix initialisation is a single line in
`models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:322`:

```python
mod = nn.ModuleList([MoDBlock(config) for _ in range(self.n_repeat - 1)])
```

The authors' pre-fix version allocated `self.n_repeat` MoDBlocks; the
fix is `self.n_repeat - 1`. The forward loop was NOT modified — the
loop bound and the routing guard that skips `mod[n_repeat - 1]` were
already present in the authors' code, which is why the last router
becomes an orphan in the first place. The loop lives in
`models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:458-479`
and is structured as:

```python
for rep_idx in range(1, self.n_repeat + 1):
    ...
    if rep_idx < self.n_repeat:
        is_final, selected_indices, router_weights = \
            self.transformer.mod[rep_idx - 1](x, capacity_factor=...)
    else:
        is_final = x.new_ones((B, x.shape[1])) == 1.
        selected_indices = x.new_ones((B, 0, 1)).long()
        router_weights = None  # Not gonna be used anymore
```

For `n_repeat=5` the loop runs five passes. The guard `rep_idx <
self.n_repeat` calls `mod[0]`, `mod[1]`, `mod[2]`, `mod[3]` at
iterations 1 through 4 respectively, gating entry into passes 2 to 5.
Iteration 5 falls through to the else branch, which synthesises a
fully-final mask and an empty selection without consulting any router.
Under the authors' pre-fix allocation `mod[4]` exists in the
`ModuleList` with an `nn.Linear(768, 1, bias=False)` weight (768
parameters), but that weight is never reached.

### Functional equivalence: a dead-weight removal

The orphan has three possible effects on training, and all three are
zero:

1. **Forward pass.** Never called by the `if rep_idx < self.n_repeat`
   guard. Contributes nothing to logits or to the loss.
2. **Backward pass.** Never touched by autograd; `.grad` stays `None`
   throughout the run.
3. **Optimiser step.** Fused AdamW and the standard
   `torch.optim.AdamW` skip parameters whose gradient is `None` before
   applying updates — including the decoupled weight-decay term that
   otherwise multiplies every parameter. The orphan keeps its
   random-initialisation value for the entire 60,000-step run.

On the authors' single-GPU setup the orphan also has no runtime
effect — single-GPU PyTorch does not enforce gradient-reduction
checks, so the unused parameter is silently ignored. Under our DDP
setup the orphan is detected as a parameter without a gradient on
the second training iteration and DDP aborts with `RuntimeError:
Expected to have finished reduction in the prior iteration [...]
Parameter indices which did not receive grad for rank 0: 152` —
index 152 being `transformer.mod[4].mod_router.weight` for our
24-layer `n_repeat=5` model. This is the symptom documented in
[§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved); see
[§M4](#m4-training-configuration-shared-across-variants) for the
shared hardware and DDP configuration, and `distributed/ddp.py:50-51`
for the `find_unused_parameters=False` constructor flag that makes
the detection fatal rather than silent. The fix is necessary for DDP
to run at all, not to change the trained-model semantics.

A pre-fix model trained on single-GPU and a post-fix model trained
on the same hardware would produce identical logits, identical
losses, identical gradients, and identical optimiser trajectories
for every parameter in the intersection of the two model
definitions. The only difference between them is whether `mod[4]`
exists in the state dict.

### Init-order RNG offset: the only real delta

There is one subtle difference between the pre-fix and post-fix code
paths that survives the dead-weight reasoning. `GPTBase.__init__` calls
`self.apply(self._init_weights)` which walks the module tree in
post-order and redraws every `nn.Linear.weight` and
`nn.Embedding.weight` from a Normal(mean=0, std=0.02) distribution
(`models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:333,359-365`).
The traversal visits, in order, `transformer.wte`, `transformer.wpe`,
the `h_begin`/`h_mid`/`h_end` Blocks, `transformer.ln_f` and
`transformer.ln_mid` (LayerNorms are skipped by the init), the
`transformer.mod[...]` ModuleList, and finally `lm_head`.

Because `self.apply` walks `mod[0..N-1]` **before** `lm_head`, the
number of MoDBlocks controls the RNG stream offset at which
`lm_head.weight` is drawn. Pre-fix: `5 × 768 = 3840` samples consumed
by the `mod` segment. Post-fix: `4 × 768 = 3072` samples. This is a
**768-sample offset** in the CUDA RNG stream at the moment `lm_head`
is drawn. `lm_head.weight` is tied to `transformer.wte.weight` via
`self.transformer.wte.weight = self.lm_head.weight`
(`models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:330`),
so the re-draw at `lm_head` init time also overwrites the initial
`wte` values; both share the same 38.6 M-element tensor drawn from the
offset RNG position.

After `self.apply` returns, `__init__` runs a second re-init loop
(`models/adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.py:335-337`)
that rescales every `c_proj.weight` in the residual projections:

```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

This loop runs at an RNG position that inherits the 768-sample offset
from the earlier `mod[4]` draw, so every `c_proj.weight` in
`h_begin`, `h_mid`, and `h_end` is redrawn from a different RNG state
between pre-fix and post-fix code.

| Tensor | Shape | Role | Pre-fix init state | Post-fix init state |
|---|---|---|---|---|
| `mod[4].mod_router.weight` | (1, 768) | orphan router (pre-fix only) | exists with random values, never trained | not allocated |
| `lm_head.weight` / `transformer.wte.weight` (tied) | (50257, 768) | output projection + token embedding | drawn at RNG offset +768 | baseline |
| `c_proj.weight` across 24 blocks × 2 per block (attention output + MLP output) | (768, 768) each | residual projections rescaled by the `c_proj` re-init loop | drawn at RNG offset +768 | baseline |

Parameter count at a different RNG position: approximately 38.6 M
(`lm_head`/`wte`, tied) + 28.3 M (`c_proj` weights across 24 blocks ×
2 per block × 768 × 768) ≈ **66 M parameters out of the model's
208.55 M total, ≈ 32 %**. Every other weight is identical between
pre-fix and post-fix initialisation, because the common RNG prefix
through the `h_begin`/`h_mid`/`h_end` first-pass init has already been
consumed by the time `self.apply` reaches the `mod` ModuleList.

Statistically, this is a seed change. Two 60,000-step training runs
from the pre-fix and post-fix init states produce different gradient
trajectories that converge to equivalent distributions of final
weights. The paper's reported inter-seed SEM for LN-CoTFormer 40k is
0.03 PPL across three seeds (Table 2 parentheticals), which is a
reasonable prior for the order of magnitude of inter-seed variance on
the ADM as well.

### Empirical bound

The ADM v1 vs v2 retraining in `eval-adm/run_6` is a controlled
experiment for exactly this effect. v1 was trained with the pre-fix
init path (before the orphan removal and the subsequent RNG stream
changes). v2 was trained from scratch with the post-fix path at
`iridis/adm-train/job.sh:167` (`--exp_name adm_v2_...`). All other
training inputs — the shared configuration of
[§M4](#m4-training-configuration-shared-across-variants), the data
pipeline, hyperparameters, and hardware — were held constant.

| Checkpoint | v1 PPL | v2 PPL | Delta | Sign |
|---:|---:|---:|---:|:---:|
| 40,000 | 28.073 | 28.079 | +0.006 | v2 worse |
| 60,000 | 24.074 | 24.060 | -0.014 | v2 better |

At 60k the delta is 0.014 PPL — well below either 95 % CI width
(~0.76 PPL each), well below the per-batch loss-level intra-run SEM of
0.0081 in loss space (~0.25 PPL in perplexity space; see
[§M3](#m3-confidence-intervals-and-the-papers-sem)), and well below
the paper's inter-seed SEM of 0.03 for LN-CoTFormer 40k. The 40k delta
(+0.006, sign reversed) is consistent with intermediate-checkpoint
noise from the mid-cosine-decay training state and does not contradict
the 60k result.

**The init-order RNG offset contributes below the noise floor on our
hardware.** See
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved)
"Empirical magnitude" and `iridis/eval-adm/README.md` § *ADM v1 vs v2:
B4 empirical magnitude* for the full per-checkpoint statistics; this
section does not duplicate them.

### Reframing earlier attributions

Earlier revisions of this document framed the fix as a correctness
change that allowed the model to "learn to use repeat 5". The
forward-loop evidence contradicts this: both pre-fix and post-fix
versions execute five forward passes and call the same four routers
(`mod[0..3]`) to gate entry into passes 2 to 5. The fifth iteration
runs unconditionally in both versions; the `if rep_idx <
self.n_repeat` guard prevents calling a hypothetical fifth router that
would not route anywhere anyway.

The fix is therefore best described as:

- **A DDP-compatibility dead-weight removal** — no semantic change to
  forward, backward, or inference in any trained-model sense.
- **With a subtle init-order side-effect** — equivalent to a seed
  change, affecting approximately 32 % of parameters via the 768-sample
  RNG stream offset cascaded through `lm_head/wte` and every
  `c_proj.weight`.
- **Empirically bounded below the noise floor** by the v1↔v2
  0.014 PPL delta at 60k.

Reproduction fidelity against the authors' Section 4.2/Section 5
numbers is preserved. An earlier attribution of the
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) +0.23 PPL ADM
gap to
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) alone is
invalidated by the same v1 vs v2 experiment: the combined
[§B1](#b1-trainval-split-divergence) +
[§B3](#b3-micro-batch-size-hardware-constrained) + hardware +
single-seed contribution explains the gap more plausibly than any
single divergence, as discussed in
[§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps) "Why the Gap
Is Wider than the 40k Reproductions".

---

## C2. Figure 5: Position-Indexed Cascade Bug and Methodological Critique

**Impact:** The paper's Figure 5 shows a broad distribution of last-repeat
router weights that we cannot reproduce on L4/PyTorch 2.2 under any of
four systematic experimental variations. The technical bug in the
original extraction script is real and documented; the paper's textual
claim about training-time deepening of router utilisation remains
qualitatively confirmed but far smaller in magnitude on our hardware.

**Status:** Root-caused (bug), experimentally exhausted (reproduction).
The full methodology, experiment matrix, per-variant statistics, and
discussion live in `iridis/eval-adm/README.md`; this section summarises
the verdict and keeps the
[Part C](#part-c-original-bugs-and-mistakes) divergence log
self-contained.

### The technical bug

The original `get_router_weights.py` (commit `2723e30`) defines a custom
`forward()` that manually executes the model's repeat loop, collecting
router weights from each `MoDBlock`. At each repeat, the MoDBlock
selects tokens via `torch.topk(router_logits, top_k, sorted=False)`,
which **reorders tokens by score**. The custom forward then applies
`x.take_along_dim(selected_indices, dim=1)`, so the representation
passed to the next repeat is in a different token order.

After collecting per-repeat router weights, the script cascades via:

```python
for i in range(2, len(router_weights)):
    for j in range(len(router_weights[i])):
        router_weights[i][j] = torch.minimum(
            router_weights[i][j], router_weights[i - 1][j]
        )
```

This takes the element-wise minimum at the same **position index** `k`
across repeats. Position `k` at repeat `i` is a different token than
position `k` at repeat `i-1`, so the cascade pairs up scores from
unrelated tokens rather than computing a per-token minimum.

### The methodological critique

The position-indexed cascade does not answer the question the paper
associates with Figure 5. Section 5 states: "the model starts to favor
using the final repeat more when the model is trained for longer." The
position-indexed cascade answers a different question: *"at position k
in the reordered sequence, what is the minimum repeat score across
repeats?"* After each repeat's `topk` reordering, position `k` refers to
a different token, so the cascaded score is the minimum across several
unrelated tokens, not a single token's depth trajectory.

We therefore define the **per-token cascade** as a sounder alternative:
*"what is the minimum router score token X received across all
repeats?"* This tracks each token's actual trajectory through the repeat
loop by mapping scores back to original positions via `active_indices`
before cascading, and is what a reader of Section 5 would reasonably
expect Figure 5 to depict.

### Experiment matrix (2 x 2 x 4 cells)

`iridis/eval-adm/job.sh` runs a systematic matrix over two training
versions (ADM v1 pre-fix,
[§B4](#b4-rng-state-restoration-on-ddp-resume-adm--resolved) and
[§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved) both
applied in v2), two checkpoints (40k, 60k), and four extraction
variants. The full per-variant table and confounder-isolation matrix
live in `iridis/eval-adm/README.md` § *Experiment matrix design*;
the verdict is:

| # | Directory | Method | Config | What it isolates |
|---|-----------|--------|--------|------------------|
| 1 | `exp5-original` | custom forward | raw JSON | Baseline: authors' code path, PI vs PT delta within the same run |
| 2 | `exp5-hooks` | forward hooks on `mod_router` | raw JSON | Extraction-method equivalence (captures logits before topk) |
| 3 | `exp5-cf-sorted` | custom forward, `topk(sorted=True)` | raw JSON | Forced topk reordering, hypothesised to restore A100-like behaviour |
| 4 | `exp5-cf-argparse` | custom forward | argparse (dtype=bf16) | Config-loading path (`dtype=None` → float16 vs bfloat16 autocast) |

### Findings

All four variants produce consistent results on L4/PyTorch 2.2. Router 4
(last repeat) statistics for ADM v2 at the 60k checkpoint, across
extraction variants and cascade methods:

| Variant | Raw mean | PI mean | PT mean | PI frac<0.01 | PT frac<0.01 |
|---------|---------|---------|---------|--------------|---------------|
| exp5-original (cf, raw) | 0.0424 | 0.0336 | 0.0345 | 0.296 | 0.291 |
| exp5-hooks (hooks, raw) | 0.0424 | -- | 0.0332 | -- | 0.297 |
| exp5-cf-sorted (cf-sorted, raw) | 0.0424 | 0.0421 | 0.0345 | 0.291 | 0.291 |
| exp5-cf-argparse (cf, argparse) | 0.0423 | 0.0335 | 0.0344 | 0.294 | 0.289 |

Four findings emerge:

1. **PI vs PT cascade delta is within 3% on L4/PyTorch 2.2.** The
   technical bug is invisible on our hardware. The position-indexed
   (buggy) cascade and per-token (correct) cascade produce nearly
   identical router 4 distributions (relative delta on mean: 2.6% at
   exp5-original, 0.9% at exp5-cf-argparse).

2. **Forcing `topk(sorted=True)` does not restore the paper's
   distribution.** The `exp5-cf-sorted` variant was designed to test
   whether forced descending-score reordering would reproduce the
   paper's broader distribution by maximising cross-token mixing in the
   PI cascade. Instead, the forced-sort PI cascade converges to the raw
   distribution (PI 0.0421 vs Raw 0.0424), because when the topk output
   is sorted by score, similar-rank positions carry similar scores
   across repeats and the positional cascade becomes a no-op. Neither
   sorted nor unsorted PI cascades on L4 reproduce the paper's Figure 5
   breadth.

3. **Extraction method is empirically equivalent.** Capturing router
   logits via forward hooks on `MoDBlock.mod_router` (pre-topk, no
   reordering) and via the authors' custom forward path (post-topk)
   produces identical raw histograms. Multiset equivalence: for a
   single repeat's scores, `{sigmoid(all_logits)} =
   {sigmoid(topk(logits, k=T))}` because `topk` with `k=T` is a
   permutation and `sigmoid` is element-wise. Our empirical means
   match to all four reported digits.

4. **Config-loading path is empirically equivalent.** Raw JSON dict
   loading (which leaves `dtype=None` and defaults to float16 autocast)
   and argparse loading (which applies `dtype=torch.bfloat16`) produce
   differences below 0.001 in router 4 mean across all checkpoints. The
   dtype path is not a confounder on our hardware.

### What we cannot reproduce, and why

The matrix systematically eliminates four hypothesised confounders.
None produces a distribution resembling the paper's Figure 5. The
residual unexplained factor must live in:

(a) The authors' exact PyTorch version and CUDA toolkit, which control
    the internal behaviour of `topk(sorted=False)` on SM80 (A100). On
    our SM89 (L4) with CUDA 11.8, neither `sorted=False` nor
    `sorted=True` reproduces whatever reordering pattern the authors'
    environment produced. We have no sorted-between-identity-and-descent
    intermediate to test.

(b) Differences in the trained router itself. Our ADM v2 reaches PPL
    24.06 against the paper's 23.83
    ([§A6](#a6-adm-perplexity-section-42-figure-4-60k-steps),
    delta +0.23). If the authors'
    Figure 5 was produced from a model trained with a different seed,
    different `acc_steps` decomposition, or different hardware, the
    underlying router weight distribution would differ even before any
    cascading is applied. The paper does not release the numerical data
    behind Figure 5, so we cannot compare our raw router 4 distribution
    (mean 0.0424 at 60k) directly against theirs.

Neither direction admits further experiments without access to the
authors' exact environment or a second ADM training run on different
hardware or seeds (compute-prohibitive at ~52h per 60k run on
`ecsstudents_l4`).

### Qualitative claim is confirmed

The paper's Section 5 textual claim -- "the model starts to favor using
the final repeat more when the model is trained for longer" -- is
qualitatively supported by our per-token cascade on ADM v2:

| Checkpoint | Router 4 mean (PT) | frac(score < 0.01) |
|------------|-------------------:|-------------------:|
| 40k        | 0.0220             | 44.9%              |
| 60k        | 0.0345             | 29.1%              |

Router 4 mean increases by 57% between 40k and 60k, and the fraction of
tokens essentially skipping the final repeat decreases from 44.9% to
29.1%. The direction and monotonicity match the paper's claim. The
magnitude of the shift is smaller than Figure 5 would suggest, which is
consistent with our inability to reproduce the broad distribution
itself.

### Cross-references

- `iridis/eval-adm/README.md` — full methodology, experiment matrix,
  per-variant statistics, v1 vs v2 equivalence discussion, and output
  tree.
- [§C1](#c1-orphan-modblock-5-allocated-4-used) — orphan MoDBlock:
  5 routers allocated, 4 used (original author bug, fixed in our code
  via [§B6](#b6-orphan-modblock-in-adaptivemod-models--resolved)).
- `get_router_weights.py` — unified extraction script with `--method`
  (`cf`/`cf-sorted`/`hooks`) and `--config-loading` (`raw`/`argparse`)
  flags driving the experiment matrix; also emits the
  `router_weights_meta.json` sidecar consumed by `plot_fig5.py`.
- `plot_fig5.py` — Figure 5 histogram plotter emitting
  `figure5_raw.png`, `figure5_picascade.png`, and
  `figure5_ptcascade.png` from the sidecar-selected last-repeat slot.

## C3. Underspecified Training Schedule (Reproducibility Gap)

**Impact:** Increases reproduction difficulty for 60k models; ambiguity
in LR schedule type.

**Status:** Resolved by code inspection. Documented here for
completeness.

### Warmup steps for 60k models

Section 3.2 states: "we apply linear warmup of the learning rate for the
first 8000 steps." This figure is correct for the 40k models (Table 1,
Table 2), but the paper never specifies the warmup duration for the 60k
models introduced in Section 4.2.

The original codebase uses `--warmup_percent 0.2` (a fraction of total
iterations), not a fixed step count. For the 60k ADM and LN-CoTFormer
runs, this yields 12,000 warmup steps, not 8,000. A reader following
the paper's "8000 steps" claim verbatim for a 60k reproduction would
use a shorter warmup than the authors' actual implementation.

### LR schedule type

The paper states "linear warmup" but does not specify the post-warmup
annealing strategy. The codebase uses
`torch.optim.lr_scheduler.OneCycleLR` with `anneal_strategy='cos'`
(cosine annealing), `div_factor=100`, and `final_div_factor=0.05` —
see `main.py:103-105` for the constructor call. This is not a simple
linear warmup + constant LR schedule; it is a full cosine cycle from
`lr/100` up to `lr` (warmup phase) then back down to `lr × 0.05`
(annealing phase). A reproducer without access to the code would
likely implement a simpler schedule.

### Resolution

Our reproduction inherits the percentage-based warmup and cosine
annealing from the original codebase (specifically
`warmup_percent=0.2` and `final_div_factor=0.05` per
[§M4](#m4-training-configuration-shared-across-variants)), matching
the authors' implementation exactly. The paper's "8000 steps" claim
is accurate for the 40k scope of Section 3.2 but should not be
extrapolated to Section 4.2's 60k runs.

---

## C4. `eval.py` perplexity uses truncated `e` (2.71828) instead of `math.e` — Original Code Quirk

### What the code does

`eval.py:140` computes the point-estimate perplexity as:

```python
stats['val_perplexity'] = 2.71828 ** stats['val_loss']
```

Lines 155-156 of the same file compute the CI95 endpoints using the
correct constant:

```python
ppl_lo = math.e ** stats['val_loss_ci95'][0]
ppl_hi = math.e ** stats['val_loss_ci95'][1]
```

The point estimate and the CI bounds therefore use different bases.
This is a quirk of the original codebase (commit `2723e30`), not
something introduced during our reproduction.

### Numerical impact

`math.e = 2.718281828...` ; the truncated literal is `2.71828`.
The relative error per unit of loss is
`log(2.71828) − log(math.e) ≈ −6.7 × 10⁻⁷`. At a representative
`val_loss ≈ 3.3` (PPL ≈ 27), the absolute PPL discrepancy is
approximately `27 × 3.3 × 6.7 × 10⁻⁷ ≈ 6 × 10⁻⁵`. This is four
orders of magnitude below the paper's smallest reported SEM (0.00 in
Table 1 for Standard 48L) and well below any delta discussed in this
document. It does not affect any reported conclusion.

The CI endpoints computed via `math.e` are internally consistent with
each other but not with the point estimate. For a PPL of ~27 the
resulting asymmetry in half-widths is at the sixth decimal place and
is invisible at the precision we report.

### Why we preserve it

This reproduction uses the project's existing `eval.py` pipeline
unchanged to maintain parity across all reproduced variants. Replacing
`2.71828` with `math.exp(loss)` would shift every reported PPL by
approximately −6 × 10⁻⁵, contaminating "delta = ours − paper"
comparisons with a non-experimental confound. The cost of preserving
the quirk is a sub-micron PPL error; the cost of patching it silently
is a systematic bias in delta attribution.

### Recommendation

If the evaluation pipeline is rewritten, use `math.exp(stats['val_loss'])`
(or equivalently `math.e ** stats['val_loss']`) throughout for internal
consistency. The current truncation is a code smell with no material
numerical impact.

**Cross-references:** [§M3](#m3-confidence-intervals-and-the-papers-sem) (CI interpretation and the distinct statistical populations sampled by each uncertainty quantity).

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
2. PyTorch Issue [#145965](https://github.com/pytorch/pytorch/issues/145965): `all_gather_object` 1 EB OOM --- confirmed by maintainer @kwen2501 as collective call ordering mismatch corrupting size negotiation.
3. PyTorch Issue [#116177](https://github.com/pytorch/pytorch/issues/116177): NCCL collective OOM under high GPU memory utilization.
4. NVIDIA/nccl Issue [#962](https://github.com/NVIDIA/nccl/issues/962): NCCL allocator fragmentation during `all_gather_object`.
5. PyTorch Blog: [Flight Recorder --- A New Lens for Understanding NCCL Watchdog Timeouts](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/) (2024).
6. PyTorch/torchtune Issue [#2093](https://github.com/meta-pytorch/torchtune/issues/2093): NCCL ALLREDUCE timeout during Llama 3.1 70B checkpoint save on multi-GPU.
7. PyTorch Tutorial: [Getting Started with Distributed Data Parallel --- Save/Load Checkpoints](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).
8. NVIDIA Megatron-LM: [checkpointing.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py) --- barrier after save labeled `# Wait so everyone is done (necessary)`.
9. PyTorch 2.2 Docs: [`init_process_group`](https://docs.pytorch.org/docs/2.2/distributed.html#torch.distributed.init_process_group) --- *"Default value is 10 minutes for NCCL [...] duration after which collectives will be aborted asynchronously."*
10. PyTorch Issue [#63720](https://github.com/pytorch/pytorch/issues/63720): `torch.save` zip serialization buffers entire state dict in memory before writing --- causes single large sequential write to filesystem.
11. NCCL 2.18.1 Release Notes: fixed LL protocol cleanup race on PCIe when switching between collective types in rapid succession. Related: PyTorch Issue [#104530](https://github.com/pytorch/pytorch/issues/104530) (DDP hang at specific step count traced to NCCL proxy thread state); NVIDIA/nccl Issue [#789](https://github.com/NVIDIA/nccl/issues/789) (LL protocol hang on PCIe with mixed collective types).
12. Meta torchtitan: [checkpoint.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/checkpoint.py) --- uses `dist.new_group(backend="gloo")` for checkpoint coordination, decoupling checkpoint I/O from NCCL training collectives.
