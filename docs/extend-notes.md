# Extension Notes

> Scientific investigation and architectural extension of the CoTFormer
> (Mohtashami et al., ICLR 2025). This document records the team's
> methodology, research questions, experimental designs, and extension
> proposals across three streams: reproduction, mechanistic analysis,
> and architectural extension.
>
> For reproduction divergences, see `docs/reprod-notes.md`.
> For training commands, see `experiments.md`.

## Table of Contents

- [0. Methodology](#0-methodology)
- [1. Mechanistic Analysis](#1-mechanistic-analysis)
  - [1.1 Ontological Framework](#11-ontological-framework)
  - [1.2 Research Questions](#12-research-questions)
  - [1.3 Experimental Protocols](#13-experimental-protocols)
  - [1.4 Theoretical Predictions](#14-theoretical-predictions)
  - [1.5 Analysis Matrix](#15-analysis-matrix)
  - [1.6 Reliability Discipline and Triangulation](#16-reliability-discipline-and-triangulation)
  - [1.7 Code and Infrastructure Strategy](#17-code-and-infrastructure-strategy)
  - [1.8 Compute Budget and Sequencing](#18-compute-budget-and-sequencing)
  - [1.9 Decision Log](#19-decision-log)
- [2. Architectural Extensions](#2-architectural-extensions)
  - [2.1 Multi-Head Latent Attention (MLA)](#21-multi-head-latent-attention-mla)
  - [2.2 Manifold-Constrained HyperConnections (mcHC)](#22-manifold-constrained-hyperconnections-mchc)
  - [2.3 Causal Cross-Repeat Mask for ADM](#23-causal-cross-repeat-mask-for-adm)
  - [2.4 Residual Attention](#24-residual-attention)
- [Appendix A: MLA Dimensionality Reference](#appendix-a-mla-dimensionality-reference)
- [References](#references)

---

## 0. Methodology

### Three Streams

| # | Stream | Goal | Status |
|---|--------|------|--------|
| 1 | Reproduce | Verify paper claims (Tables 1-2, Figures 2-5, Section 5) | Near-complete |
| 2 | Unpick and Analyse | Understand what iterative computation computes | Active |
| 3 | Extend | Improve architecture informed by analysis findings | Blocked on the analysis |

The analysis has two sub-streams:

- **Analysis main** (analytic): 10 protocols plus synthesis on frozen C1-C5
  checkpoints. Eval-only. Compute: ~2.3 h GPU + ~5.5 h CPU. Single
  HPC envelope split into a GPU job and a CPU job.
- **Analysis counting** ([RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting)):
  fresh training of two baseline architectures and a 1L-4R CoTFormer
  across a width sweep of n_embd in {128, 256, 512, 1024}. ~4-5
  L4-days total. Sequential queue after the analysis main stream; Aras
  Kavuncu (tak1e25) is the human reviewer.

### Scientific Ethos

The authors write (Appendix F): "a thorough discussion around
understanding how these models operate is outside the scope of the
current work, we hope these results encourage such investigations."
Prior work on depth-recurrent transformers (Lu et al. [27] on
Huginn-3.5B [16] applies Logit Lens and Coda Lens; Chen et al. [28]
on looped transformers measures cross-loop representation
degradation) does not cover the specific combination of cross-repeat
KV attention, weight-tied mid-block with reserved begin/end layers,
and canonical Belrose 2023 [24] Tuned Lens at five virtual layers
that CoTFormer exhibits. This narrower gap is the contribution.
Knupp et al. [37] is a contemporary architectural comparator
(depth-recurrent attention mixtures with FLOP/parameter/memory
matching) but does not perform the lens-trajectory or cross-repeat
KV measurement programme proposed here. See
[§1.9 novel contributions](#novel-contributions-confirmed-narrow-reframings)
for the per-claim novelty audit.

Each RQ states: H0, falsification criterion, IV/DV, confounders.

### Reliability Discipline

> **note**: Lecturer raised concern that interpretability tools may not give "meaningful or reliable" results; addressed by the three practices below.

1. **Triangulation rule**: every interpretive claim must be supported
   by at least two independent measurements. "Independent" means
   different protocols with different operational definitions, not
   the same metric computed twice. Claims supported by one measurement
   only are flagged as "single-measurement" and explicitly marked
   preliminary. See
   [§1.6](#16-reliability-discipline-and-triangulation) for the
   canonical mechanistic-analysis application with the cross-validation matrix.
2. **Non-claim list**: every protocol's output includes an explicit
   list of conclusions it does NOT support. This pre-empts
   over-claiming and protects against reviewer pushback.
3. **Tuned Lens as active triangulation partner**: Tuned Lens [24]
   runs alongside unweighted Logit Lens from the start of the analysis, not
   as a conditional fallback. Quantitative agreement thresholds are
   specified in
   [RQ1](#rq1-prediction-convergence-across-repeats-sq1).

### Scope Discipline

The analysis operates under a **2-week elapsed ceiling** from Abe's
approval to the analysis deliverable. Whatever protocols produce triangulated results by
the ceiling are reported; the remainder go to "future work".

> **note**: Lecturer cautioned the analysis "could become an entire project in its own right"; addressed by the 2-week ceiling and scope-shedding triggers below.

##### Scope-shedding pre-registered order

Three triggers, pre-registered before HPC submission, fire in precedence order if the ceiling is at risk:

1. **Pilot 1 validation fail (~day 3)**: if [RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting) Pilot 1 does not reproduce Chang and Bisk [25] within the [§B13](../docs/reprod-notes.md#b13-pilot-1-cb-exact-results-stub) pass band, the entire RQ9 sub-stream sheds to the extension; RQ1-RQ8 continue.
2. **Calibration inconclusive (~day 5)**: if [Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation) returns AMBIGUOUS on both Tier 1 and Tier 2, DV-2 is flagged single-measurement-preliminary and [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4) downgrades to DV-1 + DV-3.
3. **Day-10 progress review under 60% complete**: shed in fixed order — [RQ8](#rq8-depth-embedding-freeze-ablation-sq1), then [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4), then RQ9 (if not already shed).

Shedding is one-way within the analysis.

### Pre-registration venue

All H0 statements, validation gates, threshold values, and
falsification criteria documented in this section and
[§1.2](#12-research-questions) are pre-registered at the **Phase 1
skeleton commit** (see [§1.8 sequencing](#18-compute-budget-and-sequencing)).
The skeleton commit freezes the protocol contract before any
training or analysis run begins. Subsequent revisions require a new
DEC-NNN entry in [§1.9](#19-decision-log) with rationale, and
the corresponding commit message carries a `pre-registration:
DEC-NNN` marker so the protocol-history audit is reconstructable
from `git log --grep`. Calibration-specific gates (Protocol
D-calibration: Spearman threshold, classifier-accuracy threshold,
sink-shift threshold) are additionally pre-registered in
`analysis/calibration/preregister.md` per the Code-wave deliverable
documented in [§1.3](#13-experimental-protocols).

### Checkpoint Matrix

| ID | Architecture | Config | Steps | PPL | Purpose |
|----|-------------|--------|-------|-----|---------|
| C1 | CoT+Reserved | cotformer_full_depth, 2+21*5+1 | 40k | TBD | Baseline (no ln_mid, no depth_emb) |
| C2 | LN-CoT | cotformer_full_depth_lnmid_depthemb | 40k | 24.13 | Isolate ln_mid + depth_emb effect |
| C3 | LN-CoT | cotformer_full_depth_lnmid_depthemb | 60k | 23.59 | Effect of additional training |
| C4 | ADM v2 | adaptive_cotformer_..._single_final | 40k | TBD | Isolate router effect |
| C5 | ADM v2 | adaptive_cotformer_..._single_final | 60k | 24.06 | Router maturation |
| C4' | ADM v1 | Same architecture, pre-fix (no B4/B6) | 40k | -- | RNG divergence control |
| C5' | ADM v1 | Same architecture, pre-fix (no B4/B6) | 60k | -- | RNG divergence control |

Comparison C2-vs-C1 isolates `ln_mid`. C4-vs-C2 isolates the router.
C5-vs-C4 isolates training duration. C5'-vs-C5 isolates the
init-order RNG offset from the orphan-MoDBlock fix (see
`docs/reprod-notes.md` §C1 for the mechanism).

**C4 provenance footnote**: C4 is an in-run 40k snapshot extracted
from the same ADM training trajectory that produced C5 at 60k
(checkpoint at `/scratch/ab3u21/exps/owt2/adaptive_cotformer_.../adm_v2_lr...`),
not an independent end-of-training run. C4 and C5 therefore share
the same optimiser trajectory, learning-rate schedule, RNG history,
and DDP communication record up to step 40k. The C4-vs-C2
comparison consequently carries a **learning-rate trajectory
confounder**: C2 was trained on the LN-CoTFormer schedule and C4
on the ADM schedule, so any C4-vs-C2 differential includes both the
router contribution and the schedule contribution. The C4-vs-C5
comparison is unaffected because both endpoints inherit the same
trajectory from step 0 to step 40k, with the only contrast being
the additional 20k steps; the LR schedule discontinuity documented
in `docs/reprod-notes.md` §B10 applies to the 40k-to-60k extension
and is the same for C5 and C5'.

---

## 1. Mechanistic Analysis

### 1.1 Ontological Framework

The central question is: **"What does iterative computation in
CoTFormer actually compute?"**

This decomposes into five specific sub-questions, each targeting a
distinct aspect of the model's internal dynamics:

| ID | Sub-Question | What Truth We Seek |
|----|-------------|-------------------|
| SQ1 | Does the model refine predictions monotonically across repeats, or does it restructure representations non-monotonically? | Whether repeats implement iterative refinement (converging toward a fixed point) or multi-phase processing (qualitative restructuring). This determines whether the "chain-of-thought" analogy is accurate or misleading. |
| SQ2 | Do attention heads specialise by repeat, and does this specialisation vary with layer depth within the mid-block? | Whether the weight-tied transformer blocks develop emergent functional specialisation despite identical parameters. If yes, the cross-repeat KV cache carries structurally different information at different repeats. |
| SQ3 | Does the ADM router learn a meaningful proxy for token difficulty, or does it approximate random pruning? | Whether the budget-adaptive claim is scientifically supported. The init-order RNG divergence (v1 vs v2 routing patterns; see `docs/reprod-notes.md` §C1) makes this urgent -- if the router converges to different patterns under minor training perturbations, it may not learn a robust difficulty signal. |
| SQ4 | When only a fraction of tokens survive to repeat 5, do the surviving tokens lose contextual information because their sequence neighbours halted earlier? | Whether the cross-repeat KV cache (containing mixed-depth frozen representations from halted tokens) provides meaningful context, or whether surviving tokens become informationally isolated. No existing literature addresses this. |
| SQ5 | Does the absence of a proper causal mask in the ADM's cross-repeat attention degrade routing quality? | Whether the ADM's routing decisions are less informed than they could be. The current mask (`k_indices <= indices`) enforces temporal ordering but does not account for repeat-depth relationships. |

##### SQ5 deferral to the extension

SQ5 is decomposed here for ontological completeness but operationalised as Extension §2.3. Protocol C reports a descriptive baseline of the current ADM mask's attention-mass allocation as a secondary output (~30 min extra eval compute) without committing analysis compute to a counterfactual mask ablation.

### 1.2 Research Questions

Each sub-question generates specific, falsifiable research questions
with full variable isolation.

#### RQ1: Prediction Convergence Across Repeats (SQ1)

**Statement**: During evaluation on OWT2 validation data, does the
LN-CoTFormer (C3, 60k) converge monotonically toward the correct
next-token prediction across repeats 1 to 5, AND do the unweighted
Logit Lens and Tuned Lens [24] agree on the convergence trajectory?

- **IV**: Repeat index r in {1, 2, 3, 4, 5} (plus the `.5` halfway
  sub-sites from the activation capture framework)
- **DV**: Top-1 accuracy of `lm_head(ln_f(h_r))` (unweighted) and
  `lm_head(T_l(h_r))` (tuned), KL divergence
  `D_KL(P_r || P_{r-1})`, entropy `H(P_r)`
- **Controls**: Same checkpoint (C3), same data (OWT2 val), same
  forward pass
- **Confounders**:
  - `ln_mid` normalisation distorts the unweighted projection
    differently at each repeat (norm varies). Mitigation: dual
    measurement with and without `ln_mid` projection.
  - Early-repeat projections have a systematic baseline disadvantage
    because `lm_head` was trained only against the final residual.
    Mitigation: the trajectory SHAPE is what we measure, not absolute
    values; the disadvantage is constant across repeats.
  - **Geometric alignment confounder** (unique to unweighted lens):
    intermediate residuals live in a geometrically different subspace
    than final-layer residuals. Tuned Lens eliminates this via
    learned affine translation.
  - **LR-trajectory confounder for cross-checkpoint comparisons**:
    C1-vs-C3 and C2-vs-C3 both span the 40k-to-60k extension whose
    LR schedule was reconstructed rather than restored from the
    original optimiser state, see `docs/reprod-notes.md` §B10. The
    schedule discontinuity affects the late-training trajectory and
    therefore the lens-based "convergence" claim at sites whose
    trained representation depended on the post-40k LR sequence.
    Mitigation: report C3-vs-C2 as the primary cross-checkpoint
    comparison (both LN-CoTFormer; the discontinuity is identical
    in shape across them) and treat any C1-vs-C3 differential as
    confounded by both `ln_mid` and the LR trajectory.
- **H0**: Top-1 accuracy is flat across repeats AND Spearman rho
  between lenses >= 0.80 on the trajectory.
- **Falsifies H0**: Paired t-test p < 0.01 between repeats 1 and 5
  (accuracy monotonic increase) AND the two lenses agree.
- **What truth**: the RATE of convergence reveals whether most useful
  work happens in repeats 1-2 (diminishing returns) or is distributed
  (each repeat adds substantial value).
- **Cross-checkpoint**: repeat on C1 (no `ln_mid`) and C5 (ADM) as
  separate invocations. C1-vs-C3 reveals whether `ln_mid` changes
  the convergence trajectory. C5-vs-C3 reveals whether the router
  alters which tokens converge early.

##### Tuned Lens triangulation

Agreement thresholds (from the operational specification):

- **Agree** if: Spearman rho >= 0.80 (Belrose 2023 [24] empirical agreement floor between unweighted and translator lenses on standard transformers; we adopt as the binding cross-lens-agreement threshold for the LN-CoTFormer) AND sign agreement on all consecutive-repeat KL pairs AND cross-lens KL <= 0.5 nats averaged over sites.
- **Disagree** if: Spearman rho < 0.60 OR sign disagreement on >= 2
  consecutive pairs OR cross-lens KL > 1.0 nats OR top-1
  disagreement rate > 30 %.
- **Ambiguous** (0.60 <= rho < 0.80): report both trajectories, draw
  no strong conclusion.

Interpretation:

- Agreement: convergence trajectory is a genuine informational signal.
- Disagreement: interpretability measurement is inconclusive; fall
  back to non-lens RQ1 metrics (cosine distance between repeat hidden
  states, per-token CE loss) plus
  [RQ2](#rq2-representation-structural-similarity-sq1) (CKA) plus
  [RQ6](#rq6-effective-dimensionality-sq1) (effective
  dimensionality).

**Critical nuance**: Tuned Lens trains its affine translator to match
the model's own final-layer distribution, NOT the ground-truth next
token. "Convergence" under Tuned Lens means "convergence toward what
the final layer would have predicted." For the "does the model refine
toward the correct answer" interpretation, condition the trajectory
analysis on tokens where the final-layer probability of the
ground-truth token exceeds 0.5.

**Narrow novelty framing** (per [DEC-023](#19-decision-log)):
the contribution is the first canonical Belrose 2023 [24] Tuned Lens
(affine + bias, SGD + Nesterov, forward-KL, identity-init) on a
CoTFormer-style weight-tied depth-recurrent transformer with
cross-repeat KV attention. The closest prior art is Paulo et al.
[30], which transferred Tuned Lens to RNN architectures (Mamba,
RWKV) but not to weight-tied recurrent transformers; Lu et al.
[27] used Logit Lens and Coda Lens (not canonical Tuned Lens) on
Huginn-3.5B [16]. The combination of cross-repeat KV cache, five
weight-shared mid-block repeats, and per-(layer, repeat) translator
fitting is the specific gap. Conclusions are scoped accordingly:
results validate the Tuned Lens for this architectural family,
they do not generalise to depth-recurrent models without the
weight-tying constraint.

##### Tuned Lens identity diagnostic (per-site applicability gate)

For each of the 105 trained translators T_l, report the Frobenius
distance from the identity `||A_l - I||_F` and the bias norm
`||b_l||_2` after the 250-step optimisation has converged
(see [Tuned Lens convergence gate](#tuned-lens-convergence-gate-dec-017)).
A site whose translator stays close to the identity contributes
nothing the unweighted Logit Lens does not already capture, and
the redundant translator should not be over-interpreted as
evidence of "tuned-lens-confirmed convergence" at that site.

| Diagnostic band | Threshold | Interpretation |
|---|---|---|
| Identity-equivalent | `||A - I||_F < 0.1` and `||b||_2 < 0.1` | Translator is empirically the identity. Report unweighted Logit Lens as the operative measurement; flag the Tuned Lens row as redundant rather than confirmatory. |
| Mild adjustment | `0.1 <= ||A - I||_F < 1.0` | Translator applies a small affine correction. Report both lenses; cross-lens KL is informative. |
| Substantive translation | `||A - I||_F >= 1.0` | Translator implements a non-trivial reorientation; the lens is doing meaningful work. The cross-lens KL becomes the canonical convergence signal. |

The diagnostic is reported per (layer, repeat) site as a 21x5 grid
in the synthesis output. Sites in the identity-equivalent band do
not contribute to the agreement-vs-disagreement counts in the
[Tuned Lens triangulation](#tuned-lens-triangulation) gate;
they are reported as "lens-redundant" in the synthesis.

##### Tuned Lens failure modes (non-claims)

Three documented failure modes that Protocol A does NOT conclude
through:

1. Early-site predictions are inherently noisy -- report trajectory
   shape, do not over-interpret individual points.
2. No a priori "best layer" -- report the full trajectory across all
   21 mid-layers and 5 repeats.
3. "Overthinking" is a known phenomenon (intermediate layers
   outperform final) [24]. If convergence is non-monotonic, this is a
   model property, not a lens failure.

##### Tuned Lens training specification

Key parameters for `analysis/tuned_lens.py`:

| Parameter | Value |
|-----------|-------|
| Architecture | `nn.Linear(768, 768, bias=True)` (pure affine) |
| Init | `nn.init.eye_(weight)`, `nn.init.zeros_(bias)` |
| Loss | Forward KL: D_KL(p_final, p_lens) |
| Optimiser | SGD + Nesterov momentum, lr=1.0 (DEC-017: fallback to Muon if convergence is poor) |
| Schedule | Linear decay over 250 steps |
| Batch size | 256 tokens per SGD step (matches Belrose 2023 [24]) |
| Translators | 105 (21 layers x 5 repeats) |
| Translator input (per-site) | Residual AFTER layer add, BEFORE `ln_mid` (captured via `RESIDUAL_POST_MID`) |
| Translator target (`p_final`) | `softmax(lm_head(ln_f(h_pre_ln_f)))` where `h_pre_ln_f` is captured via `RESIDUAL_PRE_LN_F` pre-hook; `ln_f` is applied exactly once downstream, matching canonical Belrose 2023 target-residual convention |
| Compute | ~5 h CPU for all translators |

Post-training diagnostic: inspect the singular-value spectrum of each
learned translator A_l. A sharp spectral drop cross-validates
[RQ6](#rq6-effective-dimensionality-sq1) dimensionality findings.

##### Tuned Lens convergence gate (DEC-017)

After 250 SGD steps, the Tuned Lens forward KL must be lower than
the unweighted Logit Lens KL at the same site for >= 95 of the 105
sites. If the Tuned Lens fails to improve over identity (the logit
lens IS the identity translator), the training has failed. Failure
triggers the Muon fallback: re-train all failing sites with Muon
optimiser for 250 steps (same batch size, same identity init), then
re-check the same >= 95/105 gate. If Muon also fails, report the
Tuned Lens as "unconverged" and rely on unweighted logit lens plus
[RQ2](#rq2-representation-structural-similarity-sq1) CKA plus
[RQ6](#rq6-effective-dimensionality-sq1) dimensionality for the
convergence claim (degraded triangulation, explicitly flagged).

##### Inter-batch robustness

Per [§1.6](#16-reliability-discipline-and-triangulation) reliability discipline, RQ1 runs on 4 independent OWT2 validation batches with inter-batch CV reported per metric.

#### RQ2: Representation Structural Similarity (SQ1)

**Statement**: Across the 105 (layer, repeat) activation sites in the
LN-CoTFormer mid-block, which pairs produce structurally similar
representations as measured by centred kernel alignment (CKA)?

- **IV**: Pair of (layer_i, repeat_j) activation sites
- **DV**: CKA (linear + RBF kernels) over 2048 validation tokens;
  additionally, per-token cosine similarity
  `sim(x^(r), x^(r+1))` as a subsidiary metric measuring
  token-level representation decay across consecutive repeats
- **Controls**: Same checkpoint (C3), same data batch
- **Confounders**: Linear CKA saturates at high similarity for wide
  representations (mitigation: RBF kernel cross-check); finite-sample
  CKA has positive bias at small N (mitigation: Chun et al. [9]
  correction at 2048 tokens).
- **Estimator** (per [DEC-026](#19-decision-log) and
  Murphy et al. [32]): debiased linear CKA via the unbiased HSIC
  U-statistic. The biased estimator returns spuriously high
  similarity in the low-data high-dimensionality regime that
  applies to per-(layer, repeat) site comparisons (105 sites x 2048
  tokens vs `n_embd=768` features). Davari et al. [31] further
  document that biased CKA fails to identify functionally
  equivalent networks, motivating the switch.
- **Three-zone interpretation** (pre-registered, **empirical
  quantiles** of the 1000-permutation null distribution rather than
  fixed numeric cut-offs): the null distribution is generated by
  permuting one side of each comparison's token order 1000 times
  and recomputing the debiased CKA on each shuffle. The thresholds
  are set as quantiles of the resulting null:
  - **Redundant**: observed CKA above the 99th-percentile of the
    permutation null. Cross-repeat KV is compressible with minimal
    information loss at this site.
  - **Partially distinct**: observed CKA between the 90th- and
    99th-percentile of the null. Report as "partial refinement"
    with the layer-resolved map showing WHERE distinctiveness
    emerges.
  - **Structurally distinct**: observed CKA below the
    90th-percentile of the null. Each repeat contributes unique
    information at this site.
- **RBF kernel cross-check** uses bandwidth `sigma = median pairwise
  distance / sqrt(2)` (median heuristic). The RBF result is reported
  as a robustness column; the linear debiased estimator is primary.
- **H0**: observed CKA above the 99th-percentile of the per-pair
  permutation null at every cross-repeat pair within each layer
  (redundant representations through same weights).
- **Falsifies H0**: observed CKA below the 90th-percentile of the
  null for any adjacent-repeat pair.
- **What truth**: the distribution of CKA values across all 105
  sites and the fraction falling into each zone determines whether
  cross-repeat KV is redundant, partially refined, or structurally
  unique. Every outcome has a pre-registered interpretation.
- **Confounders update**: the biased-CKA inflation documented by
  Davari et al. [31] and Murphy et al. [32] is now a controlled
  variable rather than a residual confounder; the permutation null
  also absorbs the finite-sample bias documented by Chun et al. [9].
  The original Kornblith et al. [40] CKA definition (linear,
  orthogonal-invariant, isotropic-scaling-invariant) is the
  reference; the debiased estimator preserves both invariances
  while removing the small-N bias.

#### RQ3: Attention Head Specialisation (SQ2)

**Statement**: In the LN-CoTFormer (C3), do attention heads partition
into functionally distinct classes based on cross-repeat attention
patterns, and does the partition differ between early (layers 3-7)
and late (layers 17-21) mid-layers?

- **IV**: (Layer index, head index) -- 252 heads
- **DV**: Final-token attention distribution at repeat 5 over all
  (token, repeat) pairs, clustered via k-means (k=4-6, silhouette
  selection from k in {2, 3, 4, 5, 6, 7, 8}; **silhouette quality
  gate**: the observed best silhouette must exceed the 95th-percentile
  of a 1000-permutation null distribution constructed by independently
  shuffling cluster labels and recomputing the silhouette. The
  permutation null replaces the prior fixed 0.25 cut-off (which was
  a categorical convention in Kaufman and Rousseeuw [51] not
  derived from the data distribution); a one-tailed p < 0.05 against
  the empirical null is required to claim "meaningful clustering
  detected". The Kaufman-Rousseeuw 0.25 reference is retained as a
  sanity-check secondary diagnostic. Additionally, 7
  attention-geometry metrics (`macro_budget`, `macro_rep_entropy`,
  `macro_in_entropy`, `macro_same_pos`, `head_budget`,
  `head_rep_entropy`, `appendix_f_heatmap`) per forward pass
- **Controls**: Same checkpoint, averaged over full validation set
- **Confounders**: Flash Attention does not return weights (mitigation:
  non-flash forward pass; verify overall PPL matches the flash-pass
  value to 4 decimal places).
- **H0**: Head-cluster assignments uniformly distributed across
  layers.
- **Falsification**: Layer-vs-cluster contingency chi-squared
  p < 0.01.
- **Ontological purpose**: determine whether the weight-tied
  transformer develops emergent functional specialisation despite
  identical parameters.
- **Implementation**: port the flash/causal switch and reshape
  pipeline as a runtime monkey-patch on
  `CausalSelfAttention.forward` -- no new model class, no
  checkpoint-loading mismatch. See
  [§1.7](#17-code-and-infrastructure-strategy) for the porting
  specification.

##### Zheng taxonomy gap

Zheng et al.'s taxonomy [7] fixes head function by layer; weight-tying makes the same physical head occupy 5 virtual layers, so functional role may vary across repeats — uncharted by any existing taxonomy.

##### Cross-repeat attention sinks

Cross-repeat KV creates novel sink opportunities (Xiao et al. [26]): repeat-1 KV entries are visible to all later repeats and may receive disproportionate mass; Protocol C measures per-position attention mass aggregated across repeats to detect this.

#### RQ4: Router Validity (SQ3)

**Statement**: In the ADM (C5), do tokens halted at early repeats
have systematically lower next-token CE loss than tokens surviving to
repeat 5?

- **IV**: Token halting depth (repeat at which cascaded score drops
  below 0.5)
- **DV**: Per-token cross-entropy loss at final output; additionally,
  router-score vs state-delta correlation
  `corr(s_i, ||B(x^(i)) - x^(i)||)` measuring whether the router
  score tracks the magnitude of the block's actual transformation
- **Controls**: Same checkpoint (C5), same data (OWT2 val)
- **Confounders**: Position-in-sequence correlates with both loss and
  halting. Mitigation: partial correlation controlling for position.
- **H0**: No correlation between halting depth and per-token CE
  loss after partial-correlation position control. Operationalised
  via a non-parametric **bootstrap 95 % confidence interval**
  (B = 1000 resamples with replacement at the token level). Claim
  "router learns difficulty proxy" only if the CI excludes zero AND
  the point estimate `|r|` exceeds **0.05** (a smaller threshold
  than the Cohen 1988 [49] |r| < 0.1 trivially-small cut-off
  because N exceeds 10^6 tokens; per Curran-Everett 2008 [50] the
  practical-significance threshold should track the achievable
  precision rather than a categorical convention from a textbook
  that addressed N << 10^4). Additionally compute Spearman rho; if
  Pearson |r| < 0.05 but Spearman |rho| >= 0.2, report "non-linear
  monotonic association detected" and follow up with
  quintile-binned analysis (ANOVA across halting-depth quintiles
  with Holm-Bonferroni correction across the four contrasts).
- **What truth**: if H0 rejected, router learns difficulty proxy. If
  H0 holds, router decisions are not meaningfully related to
  difficulty. C5'-vs-C5 tests robustness to training perturbations.

#### RQ5: Context Preservation Under Adaptive Depth (SQ4)

**Statement**: In the ADM (C5), do tokens surviving to repeat 5
produce higher-quality predictions when their context tokens also
survived deep, compared to when most context halted at repeat 1-2?

- **IV**: Fraction of context tokens surviving to same depth
- **DV-1**: Per-token CE loss for the target token (logit-derived;
  measures output quality).
- **DV-2** (attention-derived; per
  [DEC-026](#19-decision-log) and the calibration verdict
  documented in
  [§1.3 Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation)):
  the attention distribution at the final repeat for surviving
  tokens, binned by context survival fraction, is reduced to an
  **empirically-calibrated isolation score** rather than to raw
  attention entropy. The calibration is performed via the
  three-tier substrate (Tier 1 GPT-2-large, Tier 2 ADM C5, Tier 3
  contingent 24L standard; see
  [DEC-029](#19-decision-log)) on synthetic induction-head
  and broad-integration sequences whose ground-truth attention
  targets are known by construction. The classifier's decision
  boundary in (entropy, attention-target-accuracy) space labels
  each natural test-time bin as "isolation" or "integration"; the
  classifier-accuracy validation gate is `>= 0.8` at test time.
  When tiers 1 and 2 disagree, the **ADM C5 tier governs** because
  the target architecture, not the published-checkpoint reference
  point, determines DV-2's validity for an adaptive-depth
  measurement.
  - **Sink correction is mandatory**: both raw `H_full` and
    sink-excluded `H_no_sink` (initial-position attention mass
    masked and the distribution renormalised) are reported.
    `H_no_sink` is the primary metric whenever the per-bin
    sink-attention shift exceeds 1 bit, in line with the
    attention-sink phenomenology of Xiao et al. [26] and Gu et al.
    [46]. Sink-uncorrected entropy is never reported as a
    standalone DV-2 number.
  - **Failure mode**: if the calibration classifier accuracy at
    test time falls below 0.8 (the
    [Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation)
    fourth-gate threshold), DV-2 is flagged
    single-measurement-preliminary and the contextual-isolation
    claim is downgraded accordingly. On calibration FAIL, the top-k-mass fallback applies (see [Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation)).
- **DV-3** (representation-derived; new triangulation partner per
  [DEC-026](#19-decision-log) responding to the pre-Phase-1
  hostile review's operational-independence concern): cosine
  similarity between the surviving target token's
  final-mid-block hidden state and the **unweighted mean** of
  surviving-context tokens' final-mid-block hidden states, both
  evaluated at repeat 5 and at the residual stream **PRE-`ln_mid`
  application** so the comparison is on the unnormalised
  representation rather than the post-LN unit-norm projection.
  The unweighted mean is chosen because the alternative
  attention-weighted mean would import DV-2's attention signal and
  break operational independence.
  - **Inclusion criterion**: only tokens with `halt_depth(j) >= 5`
    (strict deep-survival) are included in the context mean. The
    sample-size gate (N >= 200 per context-fraction bin) applies
    to DV-3 separately from DV-1/DV-2; under-resourced bins are
    reported as "DV-3 deferred" rather than merged.
  - **Operational independence**: DV-1 is logit-derived, DV-2 is
    attention-derived, DV-3 is hidden-state-derived. All three are
    extracted from the same forward pass (no extra GPU cost) but
    they are three operationally distinct quantities (probability
    distribution over vocab; probability distribution over KV
    positions; cosine in `n_embd`-dimensional residual space).
    This is the W-18 operational-independence bar; it does not
    require sampling independence.
  - **H0**: per-token Pearson `|r|` between DV-3 cosine and
    context-survival-fraction is below 0.1, evaluated under the
    Holm-Bonferroni-corrected family of RQ5 tests (DV-1, DV-2, and
    DV-3 jointly; see
    [§1.6 Concern 3](#concern-3-multiple-comparisons-correction)).
  - **Falsifies H0**: bootstrap 95 % CI on `r` excludes zero AND
    one-tailed p < 0.01 against a 1000-permutation null
    constructed by shuffling the survival-fraction labels.
- **Controls**: Conditioned on target surviving to repeat 5
- **Confounders**: (a) Sequence difficulty -- mitigation: z-score
  loss per sequence. (b) Position-in-sequence -- mitigation: partial
  correlation controlling for position (matching
  [RQ4](#rq4-router-validity-sq3) treatment).
- **Sample-size gate**: N >= 200 surviving tokens per
  context-fraction bin. If any bin has fewer, merge adjacent bins
  until the minimum is met. The ADM uses MoD-style top-k selection
  with capacity factors (not per-token halting thresholds): survival
  at repeat 5 is approximately `product(c_i for i in 1..4)` of the
  sequence. With typical capacity factors ~0.6, approximately
  `0.6^4 = 13 %` of tokens survive -- well above the gate. If a
  specific configuration yields less than 5 % survival, report
  "insufficient data" and defer.
- **H0** (joint, three-DV): target CE loss independent of context
  survival fraction AND calibrated isolation score independent of
  context survival fraction AND DV-3 cosine independent of context
  survival fraction. Rejection requires Holm-Bonferroni-corrected
  p-values across the three DVs at family-wise alpha = 0.01 (per
  [§1.6 Concern 3](#concern-3-multiple-comparisons-correction)).
- **What truth**: if H0 rejected on at least two of the three DVs,
  surviving tokens suffer "contextual isolation" -- a fundamental
  limitation of adaptive depth (triangulated claim with operational
  independence across logit/attention/hidden-state derivation). If
  only one DV rejects, the claim is "preliminary,
  single-measurement". If DV-2 calibration fails the
  classifier-accuracy gate, the claim is reported under DV-1 + DV-3
  triangulation only, with DV-2 explicitly downgraded.

**Novelty**: no prior MoD/MoR work measures information-propagation under cross-repeat KV; closest comparators do not address contextual isolation under adaptive depth.

#### RQ6: Effective Dimensionality (SQ1)

**Statement**: Does intrinsic dimensionality decrease across repeats
1 to 5 in the LN-CoTFormer (C3)?

- **IV**: Repeat index r in {1, 2, 3, 4, 5}
- **DV**: Participation ratio with finite-sample correction (Chun et
  al. [9]) at each (layer, repeat) site; additionally,
  residual-stream scale `||x||_2` per repeat as a pre-check for
  exponential norm growth
- **Controls**: Same checkpoint, same 2048 tokens
- **H0**: per-layer slope of `log(d_eff)` regressed against repeat
  index `r in {1,2,3,4,5}` is non-negative, evaluated by
  Holm-Bonferroni-corrected one-tailed t-test across the 21 mid-
  layer regressions. The Mann-Kendall test originally proposed at
  this site is **dropped as the primary statistic**: at n = 5
  observations per regression Mann-Kendall is severely
  underpowered (Wang et al. [34] document a power floor below
  approximately 0.6 at small n even for monotone signals well
  separated from noise). The per-layer trend regression on 105
  observations per (layer, repeat) site provides direct
  log-linear-slope evidence with a per-layer confidence interval.
  The Mann-Kendall test is retained as a **secondary
  robustness-check** with explicit power caveat; agreement between
  the two methods strengthens the claim, disagreement is reported.
- **Falsification**: per-layer log-linear slope significantly
  negative (one-tailed t-test p < 0.01 with Holm-Bonferroni across
  21 layers) **for at least 16 of 21 layers** (a non-arbitrary
  threshold of 75 % of layers; the remaining layers are reported
  with their slope CI for transparency).
- **Ontological purpose**: the truth sought is whether
  iterative computation in CoTFormer compresses representations
  into a low-rank manifold across repeats (a structural property
  of how the recurrent architecture allocates representational
  capacity). If d_eff decreases monotonically, later repeats
  operate in lower-dimensional subspaces; if d_eff is flat or
  increasing, the recurrent architecture preserves or expands
  representational dimensionality despite identical weights.

##### Rogue dimensions and three-way cross-validation

Compute participation ratio on raw hidden states AND after removing
the top-2 principal components. If the two diverge significantly,
rogue dimensions (outlier directions carrying disproportionate
variance) are present and must be flagged. Standard transformers show
an inverted-U pattern within each forward pass (d_eff increases to
mid-depth, decreases toward output; evidence from Chun et al. [9]
Figure 3b). Expect this within each repeat; across repeats, expect
monotonic decrease if iterative refinement compresses
representations.

Three independent rank metrics provide cross-validation:

| Metric | Source | What It Measures |
|--------|--------|-----------------|
| Chun gamma_both (participation ratio) | [9] | Effective dimensionality with finite-sample correction |
| Kobayashi pseudo-rank (95 % SV threshold) | [11] | Number of singular values above 5 % of max |
| KV-CoRE NER (Shannon entropy of normalised SVs) | [8] | Information-theoretic compressibility |

Agreement across all three strengthens claims; disagreement reveals
which aspect of the spectrum matters for KV compression.

#### RQ7: Interpolation-Mechanism Validity (SQ3)

**Statement**: In the ADM v2 (C5, 60k), when a token's router score
`s_i` is small (< 0.1), does the interpolated update
`x^(i+1) := (1 - s_i) x^(i) + s_i B(x^(i))` keep
`||x^(i+1) - x^(i)||` close to zero as the paper claims, or does
the persistence of near-duplicate hidden states cause attention at
later repeats to distribute mass across the clones, raising per-head
attention entropy on polluted positions?

- **IV**: Router score bin
  `s_i in {[0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9), [0.9, 1.0]}`
- **DV-1**: Relative update magnitude
  `||x^(i+1) - x^(i)||_2 / ||x^(i)||_2`
- **DV-2**: Mean attention entropy at repeat 5 over the KV-cache
  entries whose source tokens had `s_{i-1} < 0.1` (the "clone set")
- **Controls**: Same checkpoint, same data, same forward pass, same
  seed
- **Confounders**: Position correlates with loss and halting
  (mitigation: partial correlation); easy-token bias (mitigation:
  control for hidden-state norm at entry).
- **Three-zone interpretation** for DV-2 (pre-registered with
  **empirical zone bounds** rather than fixed numeric cut-offs;
  the original 0.05 / 0.2 thresholds had no literature
  derivation): the bounds are derived from a control-group
  calibration on an OWT2 forward pass that does NOT contribute to
  RQ7 (a held-out validation slice processed with the same model
  but without the clone-set selection), and additionally
  cross-checked against CoTFormer training-run inter-step variance.
  Per [DEC-026](#19-decision-log) (S6 resolution), both
  control-group calibration and training-run variance are computed
  and reported; the smaller of the two is the operational zone
  boundary.
  - **No pollution** band: entropy delta within `1 sigma` of the
    control-group non-clone inter-token entropy variance --
    interpolation works as claimed.
  - **Mild pollution** band: entropy delta between `1 sigma` and
    `3 sigma` of the same control variance -- statistically
    detectable, practically modest. Report exact delta and
    permutation test p-value.
  - **Significant pollution** band: entropy delta beyond `3 sigma`
    of the control variance -- KV cache quality degraded by
    clones.
- **H0**: DV-1 independent of `s_i` bin (per RQ4 bootstrap CI
  procedure) AND clone-set attention entropy delta within the "no
  pollution" band.
- **Falsification**: Attention entropy delta beyond the
  significant-pollution band, evaluated against a
  **10000-permutation null** (raised from the previous 1000 to
  ensure the minimum achievable p-value of 0.0001 is well below
  the family-wise alpha = 0.01 of
  [§1.6 Concern 3](#concern-3-multiple-comparisons-correction)).
- **Ontological purpose**: the truth sought is the mechanical
  behaviour of the ADM's interpolation formula at low router
  scores -- whether the small-`s_i` update preserves the previous
  hidden state (as the paper [3] presents as a "safety net") or
  injects clones whose later KV-cache attention degrades the
  per-(layer, repeat) attention distribution. Identifying which of
  these two mechanical regimes operates is itself the
  contribution; it determines whether ADM's failure mode (when
  it fails) is router miscalibration or interpolation pollution.
- **Priority**: HIGH -- feeds the extension mcHC design decision (router
  replacement vs complement).
- **Relation to [RQ4](#rq4-router-validity-sq3)**: orthogonal. RQ4
  measures input-side prediction-quality; RQ7 measures output-side
  mechanical behaviour. Both can be true, both false, or split.

##### Router evolution calibration

The CoTFormer paper (Section 5, [3]) shows that at 40k steps, most
tokens receive LOW `s_i` for the final repeat (KV cache pollution
maximal). At 60k, the distribution shifts rightward (more tokens use
deep processing). This implies: (a) the interpolation formula IS
being actively used, not bypassed; (b) our C5 (60k) should show the
rightward shift; (c) the clone-set entropy (DV-2) should be elevated
specifically at low-`s_i` positions.

Additionally, if K/V activations are low-rank (per
[RQ6](#rq6-effective-dimensionality-sq1) and Kobayashi et al. [11]),
representations across repeats become more similar. This creates
competing effects: beneficial (interpolation robust to weight errors)
vs detrimental (router cannot distinguish repeats, exacerbating the
clone problem). Protocol G's per-repeat-index rank measurement is
essential to disentangle these.

#### RQ8: Depth-Embedding Freeze Ablation (SQ1)

**Statement**: In the LN-CoTFormer (C3, 60k), if the learned depth
embedding is frozen to a constant index at evaluation time, how does
the per-repeat logit-lens distribution deviate from its natural state?

- **IV**: `depth_idx` condition in {actual, frozen-to-0,
  frozen-to-1, frozen-to-2, frozen-to-3, frozen-to-4, random,
  **zero-vector**} -- 8 conditions. The `zero-vector` condition
  replaces the looked-up depth embedding with a length-`n_embd`
  zero vector (true signal ablation: `depth_emb` is removed rather
  than substituted). The 8-condition design separates "frozen at a
  specific learned vector" from "no signal at all"; both are
  necessary because the paper architecture commits the model to
  consume some `depth_emb` input and the model's behaviour under
  zero input is the ablation reference, while frozen-to-`k`
  measures sensitivity to substituted-but-trained signal.
- **DV-1**: Per-repeat logit-lens top-1 accuracy against ground truth
- **DV-2**: Per-repeat KL divergence
  `D_KL(P_{r, condition} || P_{r, actual})`
- **DV-3**: Per-repeat overall PPL
- **Controls**: Same checkpoint, same val batch, same forward-pass
  code path
- **Confounders**: `ln_mid` may mask changes in the underlying
  residual (mitigation: dual measurement without `ln_mid`);
  counterfactual probe limitations (mitigation: state findings as
  "the model's LEARNED use of `depth_emb` at inference").
- **H0**: Freezing has no measurable PPL or top-1-accuracy impact across all 8 conditions.
- **Falsification**: zero/freeze ablation produces a measurable PPL increase at repeats 3-5 vs the unfrozen control (paired t-test p < 0.01, with Holm-Bonferroni correction over the 8-condition family).
- **Ontological purpose**: the truth sought is whether the learned
  `depth_emb` lookup encodes meaningful "repeat identity" that the
  model exploits at inference, or whether the embedding has been
  trained but the model has learned to ignore it. The 8-condition
  IV separates three logical possibilities: (a) the model uses
  the embedding's signal (large delta vs zero-vector), (b) the
  model uses a *specific* trained vector (large delta between
  frozen-to-`k` conditions), (c) the model treats the embedding
  as a no-op (no condition produces a delta). All three are
  experimentally distinguishable with this IV and form the
  ontological output.
- **Priority**: MEDIUM.

#### RQ9: Recurrent Inductive Bias for Algorithmic Counting

**Statement**: Does a 1-layer-4-repeat CoTFormer with weight tying
and recurrent attention learn to count from a random starting
integer, generalising to OOD sequence lengths (train <= 50,
test <= 200), at higher accuracy than a standard 4-layer Transformer
with RoPE trained on the same data with the same compute?

##### Twin-pilot design for RQ9 reproduction validity

Per [DEC-021](#19-decision-log) (Abe 2026-04-18, addressing
the pre-Phase-1 hostile review's reproduction-fidelity concern),
RQ9 runs two regimes in parallel before any cross-architecture
comparison is reported. The twin-pilot architecture cleanly
isolates the *training-regime* effect from the *architecture*
effect, so that subsequent CoTFormer-vs-baseline comparisons in
the main sweep cannot be confounded by undocumented configuration
drift between our setup and Chang and Bisk [25]'s setup. Per
[DEC-033](#19-decision-log) (Abe 2026-04-22), the main
sweep is expanded from two CoTFormer variants to four (adding
`cotformer_full_depth` without reserved layers, and adding the
ADM variant), and the training-step budget reverts from the
match-samples 78K-at-BS-128 resolution to the step-matched
312.5K-at-BS-32 regime matching Chang and Bisk [25]; the
architecture list is prioritised as *breadth-over-depth* (more
architectures, fewer seeds per cell).

| Arm | Substrate | Hyperparameter regime | Seeds | Compute |
|---|---|---|---|---|
| Pilot 1 (Chang and Bisk-exact reproduction, CONTROL) | 4L standard Transformer at `n_embd=1024`, 4 heads, `d_ff=4096`, ReLU, `tie_word_embeddings=False`, `scale_attn_by_inverse_layer_idx=True`, `max_grad_norm=1.0`, `get_constant_schedule_with_warmup`, AdamW defaults (`wd=0.01`, `beta2=0.999`), no decay grouping, `n_head=8`, `fp16`, `enable_flash=False` | Bit-identical to [25]'s defaults verified through `inductive_counting_with_LMs/scripts/causal_transformer/{trainer,config}.py` (see [DEC-020](#19-decision-log) and `docs/reprod-notes.md` §B12 for the seven divergences from the paper's prose to the code). The Chang and Bisk 4L standard Transformer is the **only** strict-reproduction control in RQ9; all other cells are experimental. | 5 (`[1234, 12, ...]`-style following [25]'s seed convention) | ~1-1.5 L4-hours per seed; ~5-7.5 L4-hours total |
| Main sweep (CoTFormer-native, breadth-over-depth; initial wave at `n_embd=1024`) | Four CoTFormer variants at `n_embd=1024` (initial wave) with widths `{128, 256, 512}` as budget-contingent secondary expansion: (V1) `cotformer_full_depth` without reserved layers (`n_layer_begin=0, n_layer_end=0`, no `ln_mid`, no `depth_emb`); (V2) `cotformer_full_depth` with reserved layers (CoT+Reserved, `n_layer_begin=2, n_layer_end=1`, no `ln_mid`, **7 effective layers per [DEC-033](#19-decision-log) Option (c)**); (V3) `cotformer_full_depth_lnmid_depthemb` (LN-CoTFormer, with `ln_mid` and `depth_emb`); (V4) `adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final` (ADM, with `ln_mid`, `depth_emb`, and Mixture-of-Repeats router with `min_repeat=n_repeat=4` for comparability with non-adaptive variants). | CoTFormer-native regime applies to all four variants: GELU, `tie_word_embeddings=True` (the CoTFormer architecture's hardcoded tying), `scale_attn_by_inverse_layer_idx=False`, `max_grad_norm=1.0`, cosine schedule with `warmup=3000`, AdamW (`wd=0.1`, `beta2=0.95`), decay grouping, `n_head=4`, `bf16`/FP32, `enable_flash=True` where supported. **Training budget: 312.5K steps at BS=32 per cell** (matching Chang and Bisk [25]'s step count; step-matched not sample-matched). | **Initial wave: 3 seeds per architecture at `n_embd=1024`** = 12 cells (4 variants x 3 seeds). Secondary expansion to widths `{128, 256, 512}` at reduced per-cell seed count is contingent on the initial-wave results and remaining budget (per [DEC-033](#19-decision-log)). | Initial wave: 12 cells x ~24-36 L4-hours per cell = ~288-432 L4-hours = ~6-9 L4-days on 2x L4; secondary-width expansion adds ~24-96 L4-hours at reduced seed count. |
| Arm C (C&B-adapted CoTFormers, CONTINGENT) | Same four CoTFormer variants (V1-V4) at the width(s) where Arm B shows interesting divergences from expected recurrence behaviour. | Arm-B architectures with **C&B-adapted training regime** injected: ReLU activation, `max_grad_norm=0.3`, `scale_attn_by_inverse_layer_idx` (using CoTFormer-specific semantics; see `docs/extend-technical.md` §3.3). All other hyperparameters remain CoTFormer-native. | Contingent on Arm B results; seed count TBD at the point of triggering. | TBD at the point of triggering. |

##### Depth-matching and the reserved-layer variant

The four CoTFormer variants V1-V4 differ in effective depth at
1L-4R:

| Variant | `n_layer_begin` | `n_layer_end` | Effective layers at 1L-4R | Depth-match to 4L standard |
|---|---|---|---|---|
| V1 (full_depth, no reserved) | 0 | 0 | 4 | Yes |
| V2 (CoT+Reserved) | 2 | 1 | 7 | No (7-eff vs 4L) |
| V3 (LN-CoTFormer, no reserved for depth-match) | 0 | 0 | 4 | Yes |
| V4 (ADM, no reserved for depth-match) | 0 | 0 | 4 | Yes |

V2 runs at its paper-canonical 7 effective layers and is compared against the 4L standard with the depth asymmetry reported as a secondary finding ([DEC-033](#19-decision-log) Option (c)).

**Pilot 1 RNG seed protocol**: Pilot 1 uses Chang and Bisk [25]'s
`[1234, 12]`-style seed convention extended to 5 seeds for
seed-symmetry parity with the main sweep at `n_embd=1024`. The
main sweep uses the project's standard seed convention `(train =
2357, val = 8191, ood = 19937)` plus per-seed perturbations
documented at training-job submission. Both regimes record their
seed list in `summary.json` under `--seed_list` so that the
post-hoc audit can attribute any seed-driven variance correctly.

- **IV-1**: Architecture in {4-layer standard Transformer
  (depth-matched), 1-layer-4-repeat CoT+Reserved CoTFormer (no
  `ln_mid`, no `depth_emb`), 1-layer-4-repeat LN-CoTFormer (with
  `ln_mid`, no `depth_emb`)}
- **IV-2**: Test sequence length in {50, 100, 150, 200}
- **IV-3**: Counting task variant in {Task 1 sequential shifted-start
  (PRIMARY), Task 2 mod-10 (SECONDARY), Task 3 selective counting
  (SECONDARY)}
- **IV-4**: Model width n_embd in {128, 256, 512, 1024} (width sweep
  isolates width as a confounder; see
  [width sweep design](#width-sweep-design-dec-012) below and
  DEC-012 in [§1.9](#19-decision-log))
- **DV-1**: Per-position next-token top-1 accuracy, per-sequence
  exact-match accuracy, accuracy-by-position curve
- **DV-2** (mechanistic triangulation): paired
  linear-and-non-linear probe R^2 for predicting the ground-truth
  count value from the hidden state at each repeat/layer. The
  paired design follows Hewitt and Liang [35] selectivity discipline:
  a non-linear probe with capacity above the linear probe's must
  beat the linear probe's R^2 by a quantitatively significant
  margin to claim "non-linear count representation"; otherwise the
  linear probe's R^2 is the operative result.
  - **Linear probe**: `sklearn.linear_model.Ridge(alpha=1.0)` on
    frozen hidden states. One probe per (architecture, width,
    repeat/layer, seed).
  - **Non-linear probe**: a small MLP `Linear(n_embd, 8) -> ReLU
    -> Linear(8, 1)` capacity-matched to the Ridge's effective
    degrees of freedom (8-unit hidden width keeps the parameter
    count below `n_embd + 8 + 1` so the non-linear probe does not
    out-capacity its linear counterpart by more than one order of
    magnitude). Trained for 200 epochs with Adam at lr=1e-3 on the
    same 80/20 split as the Ridge; same seed assignment.
  - **Selectivity rule**: if the MLP probe's R^2 exceeds the Ridge
    probe's R^2 by less than 0.05 absolute, report the linear
    probe as the canonical DV-2 number; if the gap exceeds 0.05,
    report both with the gap size and flag "non-linear
    representation likely".
  - Accuracy (DV-1) measures "does the model output the right
    number?"; the probe (DV-2) measures "does the model internally
    represent the count?" These can diverge: a model might
    represent the count but fail to decode it, or output correctly
    via a shortcut without a clean internal counter. Agreement
    between probe and accuracy strengthens the "inductive bias"
    claim.
  - Implementation: for CoTFormer (1L-4R), one probe pair per
    repeat exit (4 pairs); for the 4L baseline, one probe pair per
    layer exit (4 pairs). Input: full `n_embd`-dimensional hidden
    state. Target: ground-truth integer. ~800 tiny fits total
    (400 Ridge + 400 MLP), less than 5 min compute on a single L4.
- **DV-3** (attention-pattern analysis at penultimate repeat; per
  [DEC-027](#19-decision-log) and S4): structure of the
  attention distribution at the penultimate decoding step (just
  before the model emits its prediction) over the input positions
  carrying the running count tokens. For the CoTFormer (1L-4R),
  this is the layer-1 repeat-3 attention; for the 4L baseline, the
  layer-3 attention. Operationalised via two summary statistics:
  (a) **count-position attention mass** — total attention assigned
  to the input positions whose tokens are counting numbers (as
  opposed to filler `a` tokens or BOS); (b) **per-position
  attention slope** — slope of attention assigned to position `p`
  as a function of `count_at_p`, fit by linear regression. A
  recurrent-bias hypothesis predicts both a high count-position
  attention mass and a clean monotone slope; a baseline that
  succeeds via a non-counting shortcut shows neither.
- **DV-4** (causal intervention via zero-ablation; per
  [DEC-027](#19-decision-log) and S4): at the penultimate
  decoding step, replace the hidden state at repeat `r in {1, 2,
  3, 4}` with a zero vector and re-run the remaining
  forward-pass; record the per-position accuracy delta and the
  change in cross-entropy at the target position. For CoTFormer,
  this is one ablation per repeat (4 conditions); for the 4L
  baseline, one ablation per layer (4 conditions). The
  intervention is *causal* in the sense that the change in
  prediction is attributable to the absent computation, modulo
  the standard caveat that zero-ablation is not the same as
  removing-the-mechanism (it injects a specific signal: the
  zero vector). Compute: 4 conditions x 36 main-sweep runs x 7
  OOD lengths x 1 forward pass each = ~1000 extra forward passes
  on already-trained models; ~30 min on a single L4 (4x the
  current eval compute, per Abe's S4 budget allocation).
- **Controls**: 25-item checklist below
- **Confounders**: Parameter count (mitigation: the width sweep
  naturally provides param-matched comparisons -- see table below);
  RoPE confound (mitigation: both models use identical PE; if Task 1
  result is positive, the RECURRENCE overcomes RoPE's limit, not the
  absence of RoPE).
- **H0**: CoTFormer OOD accuracy at length 200 on Task 1 <= baseline
  OOD accuracy at length 200 (either baseline).
- **Falsification**: CoTFormer accuracy >= baseline + 5 pp, paired
  McNemar test p < 0.01, across >= 3 seeds.
- **Ontological purpose**: whether depth-recurrent architecture
  provides a sufficient inductive bias for algorithmic counting that
  depth-non-recurrent architectures lack. If positive, novel evidence
  that CoTFormer's vertical recurrence partially substitutes for RNN
  horizontal recurrence.

##### Factorial sub-analysis structure

RQ9 has four IVs. Each is analysed in a dedicated sub-analysis with
all other IVs fixed:

| Sub-analysis | Fixed | Varied | Purpose |
|---|---|---|---|
| Architecture comparison | n\_embd=256, Task 1, length=200 | 3 architectures | Core hypothesis test |
| Width scaling | 4L baseline, Task 1, length=200 | 4 widths | Confounder isolation |
| OOD generalisation | n\_embd=256, CoT+Res, Task 1 | 7 test lengths | Fragility profile |
| Task generality | n\_embd=256, CoT+Res, length=200 | 3 tasks | Robustness check |

##### Xu and Sato 2025 framing (formal prior)

Per [DEC-024](#19-decision-log) (S2 resolution), RQ9 is
framed as an **empirical test of Xu and Sato 2025 [29] Theorem 1**
on the approximate-counting expressiveness gap between latent
(weight-tied depth-recurrent) and horizontal (token-generating)
chain-of-thought. Xu and Sato prove that latent-thought reasoning
admits more efficient parallel computation than sequential CoT
under bounded budgets while CoT retains an *approximate-counting
gap*: the horizontal regime can implement counting via explicit
intermediate token generation that the latent regime cannot
trivially mirror. Theorem 1 provides the formal prediction; our
RQ9 design measures whether the gap is empirically narrowed,
preserved, or widened in the CoTFormer architecture's specific
weight-tied recurrent setting.

##### Width sweep design (DEC-012)

Chang and Bisk [25] use `n_embd=1024, n_head=8, d_ff=4096` for all
counting experiments. They never test below 1024-dim. At 1024-dim, 1L
and 2L Transformers fail; 4L is required. A single-point experiment
at `n_embd=128` risks producing an uninformative null result where
both models fail. The width sweep isolates width as a confounder:

| n_embd | d_k | d_ff | 4L params (approx.) | 1L-4R unique params |
|--------|-----|------|--------------------|--------------------|
| 128 | 32 | 512 | ~0.8M | ~0.2M |
| 256 | 64 | 1024 | ~3.2M | ~0.8M |
| 512 | 128 | 2048 | ~12.6M | ~3.1M |
| 1024 | 256 | 4096 | ~50M | ~12.6M |

**Natural param-matching via the sweep**: the sweep subsumes both
depth-matched and param-matched comparisons without additional runs:

| CoTFormer (unique params) | Approx. param-matches | Standard 4L at width |
|---------------------------|----------------------|---------------------|
| 1L-4R @ 256 (~0.8M) | ~ | 4L @ 128 (~0.8M) |
| 1L-4R @ 512 (~3.1M) | ~ | 4L @ 256 (~3.2M) |
| 1L-4R @ 1024 (~12.6M) | ~ | 4L @ 512 (~12.6M) |

**Reproduction check**: at n_embd=1024, our 4L Transformer is
architecturally identical to Chang and Bisk's [25]. If our 4L results
match theirs, our counting setup is validated. If they diverge, the
implementation requires debugging before any CoTFormer conclusions are
drawn.

**Mandatory validation gate** (DEC-012a): for each width, confirm
the 4L baseline achieves >= 80 % IND accuracy on vanilla counting
before running the full CoTFormer comparison. If the baseline fails at
a given width, the comparison at that width is uninformative and is
reported as such.

**External citation flag**: the baseline design is inspired by Chang
and Bisk [25] (local copy: `docs/InductiveBiasCheng&Bisk.pdf`). Our
hyperparameters follow their published configuration where verifiable;
deviations are documented in [§1.9](#19-decision-log).

##### RoPE fragility calibration

Expected 4L OOD accuracy based on Chang and Bisk [25]:

| Our test length | Ratio vs train | Expected 4L RoPE OOD accuracy |
|-----------------|----------------|-------------------------------|
| 50 (IND) | 1x | ~100 % |
| 100 | 2x | ~85-99 % |
| 150 | 3x | ~40-60 % (interpolated) |
| 200 | 4x | ~22-31 % |

##### 25-control checklist

| # | Control | Treatment |
|---|---------|-----------|
| 1 | Tokenisation | One-to-one integer-to-token; never compositional (DEC-007) |
| 2 | Vocabulary size | Task-specific, matching Chang and Bisk [25] codebase: `vocab = [str(i) for i in range(max_output + 1)] + ['<pad>', 'a']`. For `tr50_te200`: V=203 (integers 0-200, pad, a). Token IDs follow enumeration order: 0="0", ..., 200="200", 201="<pad>", 202="a". No BPE/SentencePiece; the tokeniser is a trivial `w2i` dictionary constructed at init. Larger V wastes embedding capacity. |
| 3 | Placeholder `a` loss masking | Exclude positions where input=`a` from CE loss; compute CE only at output positions (DEC-014b) |
| 4 | Train sequence length distribution | start ~ Uniform[0, V - L - 1], L ~ Uniform[1, 50]; fix generator seed |
| 5 | Per-integer frequency balance | Verify histogram of integer appearances as targets is uniform across [0, 199]; log before training |
| 6 | OOD test sequence lengths | {50, 75, 100, 125, 150, 175, 200} -- all held-out, never in training |
| 7 | Compute-matching convention | Both depth-matched and param-matched baselines via the width sweep (DEC-010, DEC-012) |
| 8 | Baseline hyperparameters | n_layer=4, n_head=4, n_embd in {128, 256, 512, 1024}, d_ff = 4 x n_embd |
| 9 | CoTFormer hyperparameters | n_repeat=4 ([DEC-015](#19-decision-log); depth-matched to the 4L standard for V1/V3/V4, depth-mismatched for V2 pending Abe's resolution in [DEC-033](#19-decision-log)), min_repeat=4 (disable adaptive routing for comparability with non-adaptive variants), depth_embedding=None for V1/V2, depth_embedding=linear_learned for V3/V4. **Four variants** per [DEC-033](#19-decision-log): (V1) `cotformer_full_depth` without reserved layers (n_layer_begin=0, n_layer_end=0; no `ln_mid`; no `depth_emb`); (V2) `cotformer_full_depth` with reserved layers (n_layer_begin=2, n_layer_end=1; no `ln_mid`); (V3) `cotformer_full_depth_lnmid_depthemb` (with `ln_mid` and `depth_emb`; no reserved for depth-match); (V4) `adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final` (with `ln_mid`, `depth_emb`, and router; no reserved for depth-match; routing disabled via min_repeat=n_repeat=4). All four variants trained under the CoTFormer-native regime. |
| 10 | Positional encoding | RoPE for both (primary); NoPE condition deferred (DEC-014a) |
| 11 | Optimiser + schedule | AdamW, lr=1e-4, weight_decay=0.1, beta1=0.9, beta2=0.95, cosine schedule with warmup\_steps=3000. Activation: **ReLU** (not GELU; matching Chang and Bisk codebase `config.py:27`). Gradient clipping: **max\_norm=0.3** (codebase `config.py:28`; NOT 1.0). `scale_attn_by_inverse_layer_idx=True` (GPT-2-specific attention regularisation). Fixed lr=1e-4 across all widths (matching [25]); if validation gate fails at n\_embd=128, re-run with lr=3e-4 as a secondary check. |
| 12 | Training step budget | 312.5K gradient steps at BS=32 (DEC-016), matching Chang and Bisk [25]; report wall-clock separately |
| 13 | Batch size | batch_size=32, acc_steps=4 (effective 128) for both |
| 14 | Context length | sequence_length >= 200 for both at train and test time |
| 15 | Data leakage | Train/val/test from disjoint generator seeds; test includes (start, length) pairs not seen in train |
| 16 | Validation set | Fixed seed; use for checkpoint selection only |
| 17 | Loss function | Token-level CE, mean reduction over non-masked positions |
| 18 | Evaluation mode | Greedy argmax decoding; parallel next-token prediction (single forward pass, no autoregressive generation) |
| 19 | Reported metrics | Per-position top-1, per-OOD-length exact-match, per-task-variant accuracy breakdown |
| 20 | Seeds and reporting | >= 3 per config; **primary**: mean +/- SE across seeds; **secondary**: best-of-3 (for direct Chang and Bisk [25] comparison, who report best-of-5); McNemar's paired test p < 0.01 per OOD length (methodologically stronger than [25], who use no significance test) |
| 21 | Statistical test | McNemar's paired, p < 0.01 per OOD length |
| 22 | Pre-registration | H0 committed before any training run |
| 23 | Start-value distribution adequacy | Verify empirically that all integers in [0, 199] appear as targets at least N times |
| 24 | BOS token handling | Identical across both models; same token ID, same attention inclusion (DEC-012b) |
| 25 | Fixed vs adaptive capacity | Fixed only (adaptive routing disabled via min_repeat=n_repeat=4) |

##### Additional controls from Chang and Bisk

- **BOS token** (DEC-012b): include in sequence format. RoPE
  depends on BOS for certain counting tasks.
- **Shifted PEs** (DEC-012c): randomise starting position 1..P
  for OOD-position robustness.
- **Loss at every position**: compute loss across all output
  positions, not just the last token.
- **Evaluation**: per-position top-1 accuracy, reported separately
  for IND (count <= 50) and OOD (count > 50). No sequence-level
  exact match.

##### Codebase implementation details (from `inductive_counting_with_LMs/`)

The Chang and Bisk reference codebase [25] (local copy:
`../inductive_counting_with_LMs/`) reveals implementation details the
paper does not state. These must be matched for the reproduction check
at n_embd=1024 to be valid.

**Data format** (from `data/rasp_primitives/counting_samesymbol_shiftedstart3__tr50_te200__/`):

```
Input:  ["k", "a", "a", ..., "a"]   (start integer k as FIRST token, then L 'a' tokens)
Output: ["-1", "k+1", "k+2", ...]   ("-1" = mask position, then count from k+1)
```

The starting integer k is an **explicit input token**, not inferred
from positional encoding. For the OOD test, start + L = max\_count
(200 for the te200 variant), so both start and length vary together.

**Tokeniser** (from `scripts/causal_transformer/dataset.py:45`):
a trivial string-to-index dictionary, NOT a BPE/SentencePiece
tokeniser. Constructed as `w2i = {w: i for i, w in enumerate(vocab)}`
where `vocab` is task-specific. Our `data/counting.py` replicates
this construction identically.

**Shifted PE mechanism** (from `dataset.py:10-12`):
`shift_value = random.randint(0, max_position_embeddings - seq_len)`,
then position IDs = `range(shift_value, shift_value + seq_len)`.
Contiguous integers from a random offset, not randomised.
`max_position_embeddings = 256` for the te200 task.

**Configuration divergences from standard GPT-2** (from `config.py`):

| Parameter | Chang and Bisk codebase | Standard GPT-2 | Our CoTFormer |
|-----------|------------------------|----------------|---------------|
| Activation | ReLU | GELU | GELU |
| Gradient clipping | max\_norm = 0.3 | 1.0 | 1.0 |
| `scale_attn_by_inverse_layer_idx` | True | False | False |
| Linear layer | Conv1D (transposed) | nn.Linear | nn.Linear |
| `tie_word_embeddings` | False | True (GPT-2) | True (hardcoded in `but_full_depth.py::GPTBase.__init__`; requires code change for the 4L baseline) |
| Dropout | 0.0 | 0.1 | 0.0 |

Our counting models adopt the Chang and Bisk configuration (ReLU,
grad\_clip=0.3, `scale_attn_by_inverse_layer_idx=True`) for the 4L
baseline to match their setup, and apply the same activation and
clipping to the CoTFormer variants for a fair comparison. The Conv1D
vs nn.Linear difference is mathematically equivalent (transposed
weight matrix) and does not affect model behaviour.

**Code changes required** (Phase 1 skeleton orch deliverables):

1. **BLOCKER 1 — `--activation` flag**: parameterise the MLP activation so counting jobs can use ReLU instead of GELU. Required to match Chang and Bisk [25] codebase (`inductive_counting_with_LMs/scripts/causal_transformer/config.py:27`). Implementation: extend-technical.md §3.1.
2. **BLOCKER 2 — `--tie_word_embeddings` flag**: parameterise the hardcoded `wte.weight = lm_head.weight` tie. Required because the Chang and Bisk 4L baseline trains without weight tying, whereas CoTFormer hardcodes the tie. Implementation: extend-technical.md §3.2.
3. **BLOCKER 3 — `--scale_attn_by_inverse_layer_idx` flag**: parameterise per-layer attention-logit rescaling, enabled only in Pilot 1 to match Chang and Bisk's exact regime per [DEC-020](#19-decision-log). The flag is disabled for the main sweep because the `layer_idx` semantics are ambiguous on weight-tied recurrent blocks. Implementation: extend-technical.md §3.3.
4. **BLOCKER 4 — `data/counting.py` module + dispatcher refactor**: introduce the synthetic counting dataset and wire it through the existing training loop. Required because the project's data pipeline is OWT2-specific (numpy memmap), and the counting task needs a separate dataset class plus a `constant_with_warmup` scheduler option to match Chang and Bisk's `trainer.py:72`. Implementation: extend-technical.md §3.4.
5. **BLOCKER 5 — `indices` threading for shifted PEs**: thread per-sample random position offsets through every block's forward path so the OOD-position robustness regime (DEC-012c) actually reaches the rotary embedding. Required because the existing `Block.forward` declared `indices=None` but never forwarded it to `self.attn`. Implementation: extend-technical.md §3.5.
6. **BLOCKER 6 — Pad-mask-aware attention**: thread a per-sample `attention_mask` through the block forward so variable-length counting batches do not (a) leak attention onto padded keys or (b) bias the per-batch loss with padding mass. Required because the counting task uses variable-length sequences whereas OWT2 uses fixed-length contiguous chunks. Implementation: extend-technical.md §3.6.

**Compute update**: the main sweep retains the 36-run baseline
matrix (4 widths x 3 architectures x 3 seeds at small widths),
**increased to 42 runs** by [DEC-031](#19-decision-log)'s
seed-symmetry resolution at `n_embd in {256, 1024}` (5 seeds at
each, +6 runs over 3-seed). Pilot 1 adds 5 seeds at
`n_embd=1024` (4L Chang and Bisk-exact regime), ~5-7.5 L4-hours.
Estimated wall-clock on 2x L4 for the main sweep at the
match-samples 78K-step budget: ~3-4 L4-days; Pilot 1 finishes
within the first half-day of the schedule.

##### RQ9b — `ln_mid` effect on counting (pre-registered interaction test)

Per [DEC-028](#19-decision-log) (S7 resolution), RQ9b pre-registers the architecture x `ln_mid` interaction as its own falsifiable claim alongside the main-effect architecture comparison. CoT+Reserved (no `ln_mid`) and LN-CoTFormer (with `ln_mid`) are trained at all four widths in the main sweep, so RQ9b reuses that data — only the analysis is new.

- **IV-1**: Architecture in {4L standard, 1L-4R CoT+Reserved, 1L-4R LN-CoTFormer}.
- **IV-2**: `ln_mid` presence in {absent (CoT+Reserved), present (LN-CoTFormer)}; the 4L baseline lacks `ln_mid` and is excluded from the 2x2 factorial, reported separately as cross-check.
- **DV**: Per-position OOD accuracy at length 200 on Task 1.
- **Controls**: widths {128, 256, 512, 1024}; same Task 1; same training-step budget (78K at eff BS 128).
- **Statistical test**: two-way ANOVA on the CoT+Reserved/LN-CoTFormer x `n_embd` panel with bootstrap 95% CI on interaction eta-squared. Implementation: `statsmodels.formula.api.ols('accuracy ~ C(arch) + C(width) + C(arch):C(width)', data=df).fit()` plus `anova_lm(model, typ=2)`.
- **H0**: no architecture x `ln_mid` interaction effect on OOD accuracy.
- **Falsification**: ANOVA interaction p < 0.05 AND eta-squared > 0.02 (small effect-size lower bound per Cohen 1988 [49]); both conditions must hold, otherwise reported as "preliminary suggestion".
- **Ontological purpose**: tests whether `ln_mid` is a neutral architectural ingredient or a gating ingredient that changes how recurrent bias interacts with width; bears on interpretation of the reproduction's C2-vs-C1 comparison ([§0 Checkpoint Matrix](#checkpoint-matrix)).
- **Cross-references**: shares training data with RQ9; Pilot 1 has no `ln_mid` analogue and is excluded from RQ9b.

### 1.3 Experimental Protocols

| Protocol | Serves | Implementation | Compute |
|----------|--------|---------------|---------|
| A: Logit Lens | [RQ1](#rq1-prediction-convergence-across-repeats-sq1) | `analysis/logit_lens.py` | ~10 min GPU |
| A-ext: Tuned Lens | [RQ1](#rq1-prediction-convergence-across-repeats-sq1) triangulation | `analysis/tuned_lens.py` | ~5 h CPU (training) + ~10 min CPU (application) |
| B: CKA Matrix | [RQ2](#rq2-representation-structural-similarity-sq1) | `analysis/cka.py` | ~20 min CPU |
| C: Attention Taxonomy | [RQ3](#rq3-attention-head-specialisation-sq2) | `analysis/attention_taxonomy.py` | ~9 min GPU |
| D: Router Analysis | [RQ4](#rq4-router-validity-sq3), [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4) | `analysis/router_analysis.py` | ~10 min GPU |
| D-calibration: Attention-Entropy Calibration | [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4) DV-2 validation | `analysis/calibration/entropy_calibration.py` + `synth_sequences.py` | ~5 min GPU (Tier 1) + ~2-4 min GPU (Tier 2) + ~8 min CPU |
| E: Residual Diagnostics | [RQ6](#rq6-effective-dimensionality-sq1) pre-check, SQ1 | `analysis/residual_diagnostics.py` | ~5 min CPU |
| F: Effective Dimensionality | [RQ6](#rq6-effective-dimensionality-sq1) | `analysis/effective_dim.py` | ~5 min CPU |
| G: KV Compressibility | extension design, [RQ6](#rq6-effective-dimensionality-sq1) cross-validation | `analysis/kv_rank.py` | ~25 min GPU |
| H: Interpolation Validity | [RQ7](#rq7-interpolation-mechanism-validity-sq3) | `analysis/interpolation_validity.py` | ~10 min GPU |
| I: Depth-Emb Freeze | [RQ8](#rq8-depth-embedding-freeze-ablation-sq1) | `analysis/depth_emb_freeze.py` | ~21 min GPU |

All protocols share a unified analysis framework under
`analysis/common/` with modular backends. Protocols A through I
share the same forward-pass activation cache to minimise GPU time.

#### Protocol B implementation (debiased CKA)

The CKA matrix protocol implements the **debiased linear CKA** via
the unbiased HSIC U-statistic per Murphy et al. [32], not the
original biased estimator. The debiasing absorbs both the
small-N positive bias documented by Chun et al. [9] and the
spurious-similarity inflation in low-data high-dimensionality
documented by Davari et al. [31]. Empirical null distributions
are constructed by 1000-permutation shuffles of the token order
on one side of each pairwise comparison; the per-pair quantiles
of the shuffled distribution define the [RQ2](#rq2-representation-structural-similarity-sq1)
zone thresholds (99th percentile = redundant, 90th-99th =
partially distinct, below 90th = structurally distinct). The RBF
kernel cross-check uses the **median heuristic** for bandwidth
selection: `sigma = median(pairwise distances) / sqrt(2)`. The
linear estimator is reported as primary; the RBF estimator is a
robustness column. The reference for the canonical CKA
definition (linear form, orthogonal-invariance, isotropic-scaling
invariance) is Kornblith et al. [40].

#### Protocol C implementation (attention taxonomy + Gini + top-k mass)

Protocol C extends the existing 7-metric attention-geometry
analysis with two operationally independent secondary sparsity
metrics:

- **Gini coefficient** of the attention distribution
  `G(p) = sum_i sum_j |p_i - p_j| / (2 N sum_i p_i)`. Bounded in
  [0, 1] (uniform = 0; one-hot = 1). Hurley and Rickard [42]
  identify Gini as one of only two measures (alongside the
  pq-mean for `p < 1, q > 1`) that satisfies all six of the
  Robin-Hood, Scaling, Rising-Tide, Cloning, Bill-Gates, and
  Babies axioms a sparsity measure should satisfy. Shannon
  entropy fails the Cloning axiom in particular, so Gini and
  entropy are *not* monotone transforms of each other across
  duplicated mass distributions; the two metrics are
  operationally independent in the [W-18 sense](#16-reliability-discipline-and-triangulation).
- **Top-k attention mass** `M_k = sum_{j in top-k} p_j` with `k`
  selected by the ADM capacity-factor arithmetic already used in
  [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4)
  (`k = floor(c_i * n_seq)` for the per-(layer, repeat) capacity
  factor `c_i`). For non-ADM checkpoints (C1, C2, C3) `k` is set
  to `min(8, ceil(log2 n_seq))`. Top-k mass is the upper-tail
  `k`-th-order-statistic of the distribution and is also
  operationally independent of entropy per Pagliardini et al. [45].

Protocol C is run on **C1, C2, C3, C4, C5** (extended to C3 per
the [§1.5 Analysis Matrix](#15-analysis-matrix) update; the
original matrix only ran Protocol C on C1, C2, C4 because the
attention-taxonomy clustering was originally not interesting on
the trained-only-without-`ln_mid` C3, but the secondary Gini and
top-k mass metrics are informative on all five and the marginal
compute cost is ~3 min). The Gini and top-k columns are reported
alongside the entropy column in the per-RQ tables; sink
correction (mask position 0, renormalise) applies to all three
metrics, with both raw and sink-corrected variants reported.

#### Protocol D-calibration (RQ5 DV-2 attention-entropy validation)

Protocol D-calibration validates the monotone interpretation
underlying [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4)
DV-2 ("low entropy implies isolation; high entropy implies broad
integration") via an empirical mapping from attention entropy to
attention-target accuracy on synthetic sequences with known
ground-truth attention targets. The protocol is required because
the published interpretability literature (Meister et al. [43] on
sparse attention's interpretability ambiguity; Zhai et al. [44]
on entropy collapse decoupled from information content; Xiao et
al. [26] and Gu et al. [46] on attention sinks systematically
depressing entropy) does not support a bidirectional monotone
mapping a priori; it must be calibrated.

The protocol is composed of a three-tier substrate ladder, a
two-condition synthetic sequence design, and a four-gate
validation ladder leading to one of three verdicts (PASS,
AMBIGUOUS, FAIL). The verdict drives the
[RQ5 DV-2](#rq5-context-preservation-under-adaptive-depth-sq4)
operational definition.

##### D-cal three-tier substrate

Per [DEC-029](#19-decision-log) (calibration substrate
three-tier strategy), the protocol runs on three model substrates
in priority order:

| Tier | Substrate | Cost | Rule |
|---|---|---|---|
| 1 (literature-grounded primary) | **GPT-2-large** (36L, 774M, HuggingFace public weights) | ~5 min GPU forward + ~30 min one-time login-node download via the project's `cache_huggingface_model.py` script | Code validation against the published Anthropic induction-head reference point [41]; serves as the literature-grounded calibration anchor. |
| 2 (target-architecture primary) | **ADM C5** (existing 24L x 5 ADM 60k checkpoint at `/scratch/ab3u21/exps/owt2/adaptive_cotformer_.../adm_v2_lr...`) | ~2-4 min GPU forward (no training) | Target architecture; **ADM C5 governs when Tier 1 and Tier 2 disagree** because the calibration must validate the metric on the architecture where DV-2 will be applied at test time. |
| 3 (contingency only) | **24L standard transformer** (fresh training, ~40k steps on OWT2) | ~1-1.5 L4-days | Triggered only IF Tier 1 and Tier 2 disagree. Provides a third independent reading and dual-purposes as the reproduction's "CoTFormer beats standard at same depth" follow-up gap-closure (per residual flag R1). |

##### D-cal two-condition synthetic sequence design

Two conditions, 1000 sequences each, fixed length L = 256 tokens.

- **Condition A (Sparse / Single-Target)**: induction-head-style
  sequences `... q r ... q ?` with one informative target. The
  reply `r` is placed at position `p_r = p_q + 1` immediately
  after the first occurrence of the query token `q`. The query
  token recurs at position 253; positions 254-255 are filler.
  Ground-truth attention target: `T_A = {p_r}` (single
  position). Expected behaviour for a model with mature induction
  heads: low entropy at position 253, high attention mass on
  `p_r`. Methodological precedent: Olsson and Elhage et al. [41].
- **Condition B (Broad-Integration / Multi-Target)**: 10
  informative target positions `p_1, ..., p_10` drawn uniformly
  from `[1, 251]` without replacement, each carrying a token
  from a reserved subvocabulary `V_inf` of size 100. A query
  marker token at position 252; positions 253-255 are filler.
  Ground-truth attention target: `T_B = {p_1, ..., p_10}` (10
  positions, equally weighted). Expected behaviour for a model
  that integrates broadly: medium entropy near `log_2 10 ~ 3.3
  bits` with mass distributed across the 10 positions.

The two conditions are deliberately designed to produce
**equivalent attention entropies for distinct information
contents**: a model attending uniformly over 10 informative
positions and a model attending uniformly over 10 random
positions will report the same Shannon entropy but radically
different attention-target accuracies. The two-condition design
is the minimum to falsify the monotone interpretation; it is the
methodological prerequisite for any subsequent claim about DV-2.

##### D-cal four-gate validation ladder

Each gate must be passed in order; failure at any gate triggers a
fallback rather than continuing the ladder.

| Gate | Test | PASS criterion | Source |
|---|---|---|---|
| 1 | Baseline gate | Tier-1 model's condition-A attention-target accuracy >= 0.5 (the model can do the induction task) | `/research --deep` principle 10 validation gate |
| 2 | Sink-correction monotonicity | `|H_full - H_no_sink|` < 1 bit on both conditions; if exceeded, `H_no_sink` is the operative metric throughout, and a separate sink-mass-vs-survival-fraction correlation is reported | Xiao et al. [26], Gu et al. [46] |
| 3 | Triangulation gate | Tier 1 and Tier 2 substrate verdicts agree on PASS/FAIL; if they disagree, Tier 3 is triggered; if Tier 2 disagrees with Tier 1 and Tier 3 is unavailable, ADM C5 governs | DEC-029 |
| 4 | Spearman gate | `rho(entropy, attention_target_accuracy) <= -0.5` across pooled 2000 sequences (Cohen 1988 [49] "large effect"; at N = 2000 the power to detect rho = -0.5 is greater than 0.999 at alpha = 0.001) AND a linear classifier on `(entropy, attention_target_accuracy)` separates condition A from condition B with accuracy >= 0.8 | Cohen 1992 |

**Verdicts**:

- **PASS** (gates 1-4 all pass): the monotone interpretation is
  empirically defensible on the calibrated substrate. DV-2 is
  operationalised as the calibrated isolation classifier; the
  classifier-accuracy gate must hold at test time as well.
- **AMBIGUOUS** (Spearman in (-0.5, -0.3)): rerun with N = 4000
  per condition to determine whether the ambiguity is statistical
  or structural; if ambiguity persists, [DEC-030](#19-decision-log)
  scope-shedding triggers downgrade DV-2 to single-measurement-preliminary.
- **FAIL** (Spearman > -0.3, OR linear classifier on entropy
  alone >= 0.85): on FAIL, DV-2 is replaced by the fraction of attention mass at repeat 5 landing on context tokens that survived to the same repeat or deeper (no monotone interpretation needed).

##### D-cal compute and implementation

Tier 1 (GPT-2-large): ~5 min GPU forward (`L = 256`,
`batch_size = 16`, 2000 sequences = 125 batches at ~1.0s per
batch including return_attention overhead; memory ~2.4 GB
attention cache fits L4 24 GB budget). Tier 2 (ADM C5): ~2-4
min GPU forward (24L is half the depth of GPT-2-large, fewer
attention weights per sequence). Tier 3 (24L standard, fresh
training): ~1-1.5 L4-days (only if triggered). CPU analysis:
~8 min for entropy, Gini, top-k, attention-accuracy, Spearman,
linear classifier across 2000 x 256 distributions.

**Implementation files** (Phase 1 skeleton orch deliverables;
new sub-package under `analysis/`):

- `analysis/calibration/entropy_calibration.py` — main entry
  point; consumes `--ckpt`, `--n-per-condition`, `--seed`,
  `--out`; produces `metrics.csv`, `spearman.json`,
  `classifier.json`, `figure.png`.
- `analysis/calibration/synth_sequences.py` — deterministic
  helpers `generate_condition_A(n, L, vocab_size, seed)` and
  `generate_condition_B(n, L, vocab_size, inf_vocab, seed)`.
- `analysis/calibration/preregister.md` — the calibration
  pre-registration document; the four-gate verdict thresholds
  (Spearman <= -0.5, classifier accuracy >= 0.8, sink shift
  threshold 1 bit) are frozen at the Phase 1 skeleton commit and
  cannot be adjusted post-hoc.

Sink correction is mandatory throughout; both `H_full` and
`H_no_sink` are reported per sequence and `H_no_sink` is the
operative metric whenever the per-bin sink mass shift exceeds 1
bit (sink-correction monotonicity gate). The same calibration
infrastructure is reused for [Prediction P2](#14-theoretical-predictions)
"sparse attention from repetition", where the same
monotone-ambiguity applies; the calibration is computed once and
its decision boundary is reused across both DV-2 and P2.

#### Protocol E operational spec

Protocol E (Residual Diagnostics) is operationalised here as the
[RQ6](#rq6-effective-dimensionality-sq1) pre-check addressing the
pre-Phase-1 hostile review's missing-spec finding. For each
(layer, repeat) site under analysis, Protocol E computes three
per-token quantities aggregated into one summary triple:

| Quantity | Formula | Aggregation |
|---|---|---|
| Mean residual L2 norm | `mean_t ||x_{l, r, t}||_2` | per-(layer, repeat) scalar |
| Per-token L2 variance | `var_t ||x_{l, r, t}||_2` | per-(layer, repeat) scalar |
| L_inf norm | `max_t |x_{l, r, t}|_inf` | per-(layer, repeat) scalar |

The protocol produces three plots, each with 21 mid-layer curves
indexed by repeat:

- Mean L2 norm vs repeat for each of the 21 mid-layers (1 panel,
  21 curves).
- Per-token L2 variance vs repeat (1 panel, 21 curves).
- L_inf norm vs repeat (1 panel, 21 curves).

The pre-check serves
[RQ6](#rq6-effective-dimensionality-sq1) by surfacing exponential
norm growth (which would invalidate the
participation-ratio-vs-repeat trend regression, since the
Frobenius-normalised PR is not norm-invariant in the presence of
extreme outliers) and by surfacing rogue-direction emergence
(which is then handled in [RQ6](#rq6-effective-dimensionality-sq1)
via the top-2-PC-removed PR cross-check). Run on **C1, C2, C3**
in the analysis main analytic sub-stream; ~5 min CPU per checkpoint, no
GPU needed.

The counting experiment
([RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting)) is
not a protocol on existing checkpoints; it trains new models via
`iridis/counting-sweep/job.sh` (which also handles evaluation).

### 1.4 Theoretical Predictions

**Prediction 1** (weight-decay low-rank correlate; observational):
weight decay (`wd = 0.1`) is **observed in correlation with**
low-rank weight-tied QK/OV matrices in the analyses of Kobayashi
et al. [11]; the predicted rank reduction is therefore a
correlational expectation, not a causal claim about the present
checkpoints. The framing is observational rather than causal
because we do not run a `wd` ablation on our own checkpoints; we
test whether the rank profile observed at our `wd = 0.1` setting
is consistent with the rank reduction Kobayashi et al. [11]
documented at the same `wd` value on a similarly-sized model
with matched `d_head`. The 5x repetition is hypothesised to
compound the effect (the same weight matrices are exposed to the
shared optimisation pressure five times per forward pass), but
this compounding hypothesis is itself observational at frozen
checkpoints. The prediction's epistemic status is "consistent
with literature" or "inconsistent with literature", not "proven
causal".

*Test*: Protocol G; three independent measurements:
- (a) Weight-matrix pseudo-rank of `W_K^T W_Q` and `PW_V`
  (Kobayashi [11] comparison)
- (b) Activation rank of K/V caches (KV compression relevance)
- (c) KV-CoRE NER of K/V caches [8] (information-theoretic
  complement)

*Two-tier threshold*: 95 % effective rank < 48 is the
literature-confirmed correlational expectation, based on Kobayashi
et al. [11]'s direct measurements at `wd = 0.1` on a 125M model
with identical `d_head = 64`. If
rank < 40 is observed, this constitutes evidence for an additional
compounding effect from 5x weight tying not studied in the
original work; we report the result as one of three
mutually-exclusive labels emitted by `analysis.kv_rank._prediction_1_verdict`,
not as a "causal proof":

- `evidence-of-compounding` -- every mid-layer rank_95 is below 40
  (Kobayashi-bracket strictly improved upon, evidence for the
  5x-weight-tying compounding).
- `consistent-with-literature` -- no mid-layer rank_95 exceeds 48
  (Kobayashi-bracket respected, but compounding not yet evident).
- `inconsistent-with-literature` -- at least one mid-layer rank_95
  exceeds 48 (literature threshold crossed).

*Scope limitation*: Prediction 1 tests the CONSEQUENCE of weight
decay (rank of frozen checkpoints trained with `wd = 0.1`) but
cannot establish CAUSALITY (which would require training with
varied `wd` values: 0, 0.01, 0.1, 0.3). The causal claim is
deferred to future work with controlled `wd` ablation. The
[§1.6 triangulation row](#16-reliability-discipline-and-triangulation)
"Weight tying correlates with low rank" reflects the observational
framing per [DEC-022](#19-decision-log) (renaming from the
earlier "induces" framing).

**Prediction 2** (sparse attention from repetition; novel
hypothesis): We hypothesise that attention entropy decreases across
repeat indices, analogous to Zucchet et al.'s [12] finding that data
repetition accelerates sparse attention emergence. Unlike their
data-repetition setting, our weight-tied repetition provides
computational depth rather than statistical redundancy. This
prediction is novel and not directly calibrated against prior work.

- **H0**: per-head attention entropy at repeat 5 is not
  significantly lower than at repeat 1, after controlling for
  representation convergence and per-layer baseline. The original
  proposal of a 252-head Wilcoxon signed-rank test treated all 252
  (layer, head) cells as exchangeable; this is incorrect because
  heads within a single layer share their input distribution and
  are therefore non-independent. Replaced per
  [DEC-026](#19-decision-log) with a **mixed-effects model**
  fit via `statsmodels.regression.mixed_linear_model.MixedLM`:
  ```
  entropy ~ repeat + (1 | layer) + (1 | head:layer)
  ```
  with `repeat` as the fixed effect of interest and per-layer +
  per-(head, layer) random intercepts to account for the head-level
  non-independence. The p-value on the `repeat` slope is the
  primary test. Per-layer Wilcoxon signed-rank tests (one per
  mid-layer, with Holm-Bonferroni correction across the 21
  layers) are reported as a non-parametric robustness check; if
  the mixed-effects slope is significant but more than half the
  per-layer Wilcoxons are not, the claim is downgraded to
  "layer-aggregate effect, not per-layer-uniform".
- **Confounder**: convergence-induced sharpening. As
  representations converge across repeats (measured by
  [RQ2](#rq2-representation-structural-similarity-sq1) CKA),
  attention naturally sharpens because the target distribution
  becomes more peaked. Mitigation: condition the entropy
  comparison on heads whose adjacent-repeat CKA is below the
  per-pair permutation null's 90th percentile (non-converging
  heads). If the entropy decrease persists in non-converging
  heads, it is not a trivial consequence of convergence.
- **Sink-and-monotonicity confounder**: per the
  [Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation)
  monotone-ambiguity finding, low entropy at repeat 5 could be
  "repetition-induced sparsity", "attention-sink amplification",
  or "attention-collapse pathology" (Zhai et al. [44]). All three
  are observationally indistinguishable from raw entropy alone.
  Mitigation: report Gini coefficient and top-k attention mass
  alongside entropy (per
  [Protocol C](#protocol-c-implementation-attention-taxonomy--gini--top-k-mass)),
  apply sink correction (mask position 0, renormalise), and
  cross-check with the Tier-1 calibration's decision boundary
  applied to the natural OWT2 forward pass. The same
  D-calibration pre-registered classifier from
  [Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation)
  is reused for P2 — the monotone-ambiguity is the same
  underlying problem.
- **Validation gate**: repeat-1 mean attention entropy across all heads exceeds the calibration substrate's per-head H0 entropy by at least one standard deviation under the no-attention-collapse null. If already near zero (maximally sparse), no decrease is possible and P2 is vacuously unfalsifiable.

*Test*: [Protocol C](#protocol-c-implementation-attention-taxonomy--gini--top-k-mass);
mixed-effects monotonicity test plus paired Gini and top-k mass
robustness checks:

1. Mixed-effects model `entropy ~ repeat + (1|layer) + (1|head:layer)`
   on the per-(token, layer, repeat, head) panel; fixed-effect
   slope p < 0.05 with the slope's bootstrap 95 % CI excluding
   zero.
2. Per-layer Spearman `rho` < -0.5 (Cohen 1988 [49] "large
   effect") across repeats 1-5, with Holm-Bonferroni correction
   across the 21 layers (per
   [§1.6 Concern 3](#concern-3-multiple-comparisons-correction)).
3. Paired comparison: repeat-1 vs repeat-5 entropy, Gini, and
   top-k mass; the monotonicity claim requires concordant
   directionality across all three sparsity metrics; if Gini and
   top-k mass disagree with entropy, the result is reported as
   "metric-dependent" rather than supporting P2.

### 1.5 Analysis Matrix

| Protocol | C1 | C2 | C3 | C4 | C5 | C4' | C5' | GPU? |
|----------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:----:|
| A: Logit Lens | X | X | X | X | X | -- | -- | Yes |
| A-ext: Tuned Lens | X | X | X | X | X | -- | -- | CPU |
| B: CKA (debiased) | X | X | X | X | X | -- | -- | No |
| C: Attn Taxonomy | X | X | **X** | X | X | -- | -- | Yes |
| D: Router Analysis | -- | -- | -- | X | X | X | X | Yes |
| D-cal: Entropy Calibration | -- | -- | -- | -- | X | -- | -- | Yes |
| E: Residual | X | X | X | X | -- | -- | -- | No |
| F: Eff. Dim | X | X | X | X | X | -- | -- | No |
| G: KV Compress. | X | -- | X | -- | X | -- | -- | Yes |
| H: Interp. Validity | -- | -- | -- | X | X | -- | -- | Yes |
| I: Depth-Emb Freeze | -- | -- | X | -- | -- | -- | -- | Yes |

Total GPU: ~4.7 h on L4 (with 4-batch inter-batch robustness; +3
min for Protocol C extended to C3 + C5; +5 min Tier-1
calibration on GPT-2-large; +3 min Tier-2 calibration on ADM C5).
Total CPU (incl. Tuned Lens training and calibration analysis):
~5.7 h (+8 min for the calibration's Spearman + classifier fit
across 2000 sequences x 256 attention distributions).

### 1.6 Reliability Discipline and Triangulation

> **note**: Lecturer raised concerns about interpretability reliability and scope creep; pre-Phase-1 hostile review added multiple-comparisons. Each codified below as an experimental constraint.

**Concern 1 — reliability**: addressed by the triangulation rule (see [§0 Reliability Discipline](#reliability-discipline)). A row whose primary and secondary measurements share the same forward pass *and* the same derived statistic counts as a single measurement; the row's claim downgrades to "single-measurement preliminary" with the downgrade explicit in synthesis output. The [RQ1](#rq1-prediction-convergence-across-repeats-sq1) Logit Lens and Tuned Lens pair is the canonical operationally-independent example: the unweighted lens projects through `lm_head(ln_f(h_r))` while the Tuned Lens applies a learned affine translator trained on KL divergence from the final-layer distribution.

| Claim | Primary | Secondary | Tertiary |
|-------|---------|-----------|----------|
| Representations refine across repeats | RQ1 Tuned Lens (per-site identity-diagnostic-filtered) | RQ1 unweighted Logit Lens | RQ2 debiased-CKA monotonic decrease |
| KV cache is compressible | Protocol G activation rank | KV-CoRE NER per repeat | Chun PR ([RQ6](#rq6-effective-dimensionality-sq1)) |
| Weight tying correlates with low rank | Kobayashi weight-matrix rank [11] | Protocol G activation rank | Tuned Lens translator SV spectrum |
| Attention specialises by repeat | RQ3 k-means clustering (permutation-null gated) | Zheng taxonomy mapping [7] | per-head attention-distribution diversity (pairwise JS divergence across heads within a layer) |
| Contextual isolation | RQ5 DV-1 per-token CE loss vs context survival (logit-derived) | RQ5 DV-2 calibrated isolation score vs context survival (attention-derived; per [Protocol D-calibration](#protocol-d-calibration-rq5-dv-2-attention-entropy-validation)) | RQ5 DV-3 representation cosine similarity (hidden-state-derived) |
| Router learns difficulty proxy | RQ4 halting-vs-loss bootstrap CI | RQ7 update magnitude | Router-score vs state-delta correlation |
| Recurrence helps counting | RQ9 DV-1 OOD accuracy sweep | RQ9 DV-2 paired linear-and-non-linear probe R^2 per repeat | RQ9 DV-3 attention-pattern analysis at penultimate repeat AND DV-4 causal intervention via zero-ablation |

Convergence rule: "supported" if primary plus at least one
secondary agree under the operational-independence audit;
"preliminary" if only primary or if primary plus secondary fail
the operational-independence audit; "contradicted" if primary
and secondary disagree (the disagreement itself is the finding,
not a measurement failure).

**Concern 2 — scope creep**: addressed by the 2-week elapsed ceiling (see [Scope Discipline](#scope-discipline)) and the [Scope-shedding pre-registered order](#scope-shedding-pre-registered-order). Protocols that do not produce triangulated results by the ceiling are moved to "future work" regardless of completion status.

##### Concern 3: multiple comparisons correction

The pre-Phase-1 hostile review identified an unguarded multiple-
comparison hazard: across the nine RQs each producing one or more
H0 tests, the family-wise error rate at uncorrected α = 0.05 per
test exceeds 0.5 even with no underlying effect, inflating the
risk that at least one false positive is reported as a finding.
The hybrid correction adopted per [DEC-025](#19-decision-log)
(S3 resolution) is:

- **Holm-Bonferroni at family-wise α = 0.01** across the
  **9 RQ-primary tests** (RQ1 logit-lens monotonicity, RQ2 cross-
  repeat debiased CKA, RQ3 silhouette permutation, RQ4 halting
  vs loss, RQ5 joint H0 across DV-1/DV-2/DV-3, RQ6 per-layer
  log-linear slope, RQ7 clone-set entropy delta, RQ8 condition-
  vs-actual delta, RQ9 OOD accuracy McNemar; RQ9b ANOVA
  interaction is treated as part of the RQ9 family for the
  Holm-Bonferroni step). Holm-Bonferroni preserves family-wise
  error control while being uniformly more powerful than the
  plain Bonferroni adjustment.
- **Benjamini-Hochberg FDR at q = 0.05** within each RQ's
  **secondary tests** (the per-layer regressions, the
  permutation-null gates, the per-task and per-OOD-length
  contrasts in RQ9, the per-condition ANOVA contrasts in RQ8).
  FDR is appropriate for secondary tests because we tolerate a
  controlled fraction of false positives in exploratory rows but
  cannot tolerate a single false-positive primary RQ outcome.

**Exemptions**: replication-stability gates (4-batch inter-batch
CV < 5 % in §1.4 Tuned Lens training) and pre-registered
validation gates (the Pilot 1 IND > 80 % gate, the Protocol
D-calibration four-gate ladder, the
[RQ9 width sweep DEC-012a](#width-sweep-design-dec-012)
gate) are *not* part of the multiple-comparison family because
they are not hypothesis tests of the underlying scientific
claims; they are pre-condition checks whose failure aborts the
test rather than producing a finding. Cross-checkpoint
robustness (e.g., RQ1 reported on C1, C3, C5 with concordant
sign across all three) is also exempt because the claim is the
agreement across checkpoints rather than the per-checkpoint
significance.

The correction is applied **once per protocol output**, not
re-applied at synthesis time; the synthesis output records both
the raw p-value and the corrected p-value for transparency, and
the convergence rule above applies to the corrected p-value
column.

### 1.7 Code and Infrastructure Strategy

`analysis/` package file tree: see extend-technical.md §2 (canonical engineering layout). Multi-concern script ownership and the `analysis/calibration/` + `data/counting.py` module bindings are documented inline in tech §2.

##### Common loader hook for module-path generality

Per the cross-architecture finding in the pre-Phase-1 hostile review, `analysis/common/loader.py` accepts `--module-path` to switch between standard transformer (`model.transformer.h`) and CoTFormer (`model.transformer.h_mid`) module paths; default is `h_mid`. Engineering details: extend-technical.md §8.9.

#### Branch strategy (DEC-009)

The analysis develops on the `abeprobes` branch; the paper-reproduction code stays on `main`. The branch protocol is canonical in extend-technical.md §5. Rationale: the `origin/tak-ablation-code` branch is missing five DDP infrastructure fixes that `main` carries; switching would either accept DDP hang risk or require forward-porting all five (see extend-technical.md §5.5).

Tak-branch port table, SHA-pin header convention, and the Explicitly-NOT-ported list: extend-technical.md §§5.5-5.7 (engineering canonical).

### 1.8 Compute Budget and Sequencing

#### Analysis main analytic sub-stream

| Stage | Wall time |
|-------|-----------|
| Stage 1 flash collector (5 ckpts x 3 min) | ~15 min GPU |
| Stage 2 non-flash collector (3 ckpts x 5 min) | ~15 min GPU |
| Stage 3 ADM collector (2 ckpts x 5 min) | ~10 min GPU |
| logit_lens (5 ckpts x 2 min) | ~10 min GPU |
| cka debiased (5 ckpts, CPU) | ~20 min CPU |
| attention_taxonomy (5 ckpts x 3 min, extended to C3 + C5) | ~15 min GPU |
| router_analysis RQ4+RQ5 (2 ckpts x 5 min) | ~10 min GPU |
| Protocol D-calibration Tier 1 (GPT-2-large) | ~5 min GPU |
| Protocol D-calibration Tier 2 (ADM C5) | ~2-4 min GPU |
| Protocol D-calibration analysis + classifier fit | ~8 min CPU |
| residual_diagnostics extended to C3 (5 ckpts, CPU) | ~5 min CPU |
| effective_dim (5 ckpts, CPU) | ~5 min CPU |
| kv_rank (5 ckpts x 5 min) | ~25 min GPU |
| interpolation_validity / RQ7 (2 ckpts x 5 min, 10000-permutation null) | ~12 min GPU |
| depth_emb_freeze / RQ8 (1 ckpt x **8 conditions** x 3 min) | ~24 min GPU |
| tuned_lens training (5 ckpts x 105 sites x ~0.6 min) | ~5.3 h CPU |
| tuned_lens application + triangulation + identity diagnostic | ~12 min CPU |
| synthesis | ~5 min CPU |
| **Subtotal GPU (single batch)** | **~2.7 h** |
| **Subtotal GPU (4-batch robustness)** | **~4.7 h** |
| **Subtotal CPU** | **~5.7 h** |

Protocol I (depth-emb freeze) is a counterfactual intervention,
not an activation summary statistic; it is exempt from the
4-batch requirement (the intervention itself is the variable,
not the data batch). Protocol D-calibration is also exempt from
the 4-batch requirement: the calibration's 2000 synthetic
sequences are the substrate (not OWT2 batches), and the four-gate
validation ladder runs once per substrate-tier checkpoint.

Split into two HPC submissions:

- `iridis/analyze-lncot+adm/job-gpu.sh` — 1 GPU, **8 h walltime**
  (raised from the originally-planned 4 h per
  [DEC-026](#19-decision-log) T10 finding: the original
  arithmetic counted Stage 1 + Stage 2 + Stage 3 collectors at
  ~40 min and the GPU protocols at ~2 h, but did not account
  for the SLURM queueing overhead nor for the flash-vs-non-flash
  job-internal context-switch when the same job script runs both
  flash and non-flash forward passes). The 8 h walltime
  accommodates the additional Protocol D-calibration Tier 1 +
  Tier 2 forward passes plus a safety margin.
- `iridis/analyze-lncot+adm/job-cpu.sh` — 0 GPU, 8 h walltime,
  runs CPU protocols plus Tuned Lens translator training plus
  Protocol D-calibration analysis. Consumes the workspace
  activation cache from `job-gpu`.

#### Counting sub-stream

The counting sub-stream is restructured per [DEC-033](#19-decision-log)
into three arms: **Arm A** (Pilot 1 Chang and Bisk-exact at
`n_embd=1024`, 5 seeds; per [DEC-021](#19-decision-log));
**Arm B** (main sweep of four CoTFormer variants V1-V4 at widths
`n_embd in {128, 256, 512, 1024}` under the CoTFormer-native
regime, priority starts at 1024); **Arm C** (contingent C&B-adapted
regime applied to V1-V4 at the width(s) where Arm B shows
divergences). The training budget reverts to **312.5K steps at
BS=32** per cell (matching Chang and Bisk [25]'s step count exactly,
restoring [DEC-016](#19-decision-log) and superseding the
match-samples resolution of [DEC-031](#19-decision-log)).
Arm B's initial wave is **4 architectures x 3 seeds at `n_embd=1024`
= 12 cells** per [DEC-033](#19-decision-log) (Abe 2026-04-22
confirmation). The `n_embd=1024` cells dominate compute at ~24-36
L4-hours per cell; initial wave total is ~288-432 L4-hours = ~6-9
L4-days on 2x L4. Secondary-width expansion to `{128, 256, 512}`
at reduced per-cell seed count is contingent on the initial-wave
results and remaining budget; the submission order prioritises
`n_embd=1024` first for the reproduction-check alignment with
Arm A (Pilot 1 Chang and Bisk-exact).

| Component | Wall time |
|-----------|-----------|
| Dataset generation (sweep + Pilot 1) | ~10 min CPU |
| Arm A (Pilot 1): 5 seeds x 4L x 1024-dim x 312.5K steps at BS=32 (Chang and Bisk regime) | ~5-7.5 L4-hours |
| Arm B (initial wave): 4 architectures x 3 seeds x 1024-dim x 312.5K steps at BS=32 | ~288-432 L4-hours (~6-9 L4-days on 2x L4) |
| Arm B (secondary-width expansion, budget-contingent): 4 architectures x `{128, 256, 512}` x reduced per-cell seed count | ~24-96 L4-hours at 1-2 seeds per small-width cell |
| Arm C (contingent): same 4 architectures with C&B-adapted regime, triggered only on Arm B divergences | TBD at trigger time |
| OOD evaluation x 7 lengths x per-cell (DV-1) + DV-3 attention extraction + DV-4 zero-ablation (4 conditions) | ~12 h L4 at full sweep |
| RQ9b two-way ANOVA + bootstrap CI on interaction effect | ~30 min CPU |
| Analysis + plotting | ~3 h CPU |
| **Total (Arm A + Arm B initial wave)** | **~7-10 L4-days on 2x L4** |

##### Arm B 32-cell consolidated matrix (DEC-035)

| Width `n_embd` | Seeds per variant | Cells (V1 + V2 + V3 + V4) | Sub-total |
|---|---|---|---|
| 1024 | 3 | 4 × 3 | 12 |
| 512 | 2 | 4 × 2 | 8 |
| 256 | 2 | 4 × 2 | 8 |
| 128 | 1 | 4 × 1 | 4 |
| **Total** | | | **32 cells** |

Per DEC-035 (Arm B secondary-width expansion committed unconditionally; supersedes DEC-033's contingent clause).

The 1024-dim cells dominate compute at ~24-36 h each at 312.5K steps
at BS=32. The initial wave per [DEC-033](#19-decision-log)
covers all four V1-V4 variants at `n_embd=1024` with 3 seeds each;
secondary-width expansion to `{128, 256, 512}` is contingent on
initial-wave results and remaining budget.

##### Empirical step-time validation gate (pre-flight measurement)

Before submitting the main sweep, the code-wave pre-flight
measures the actual `n_embd=1024` step time on the 2x L4 target
hardware. The Pilot 1 wall time above (~5-7.5 L4-hours for 5
seeds) is the empirical anchor; if Pilot 1's per-step time
exceeds 4 s on the target hardware, the main-sweep budget is
recalculated and the wall time on the schedule above adjusted
accordingly. The pre-flight is documented as the
**empirical-step-time validation gate** in the Phase 1 skeleton
commit's pre-registration: a Pilot 1 step time more than 50 %
above the projection triggers a budget revision; a step time
within 50 % of the projection clears the main sweep for
submission.

#### Sequencing (linear, no parallelism)

1. Phase 0 cleanup (1 day): documentation and plan updates.
2. Phase 1 skeleton refactor (1 day): rename package, create
   `analysis/common/`, scaffold protocol scripts.
3. Phase 2 body implementation (2-3 days, 2 parallel orchs):
   P-A logit_lens + tuned_lens + cka + residual_diagnostics +
   effective_dim; P-B attention_taxonomy + router_analysis + kv_rank
   + interpolation_validity + depth_emb_freeze; P-C synthesis +
   integration tests.
4. Phase 3 counting sub-stream (1-2 days code):
   `iridis/counting-sweep/` + `data/counting.py`.
5. Phase 4 execution (4-5 days HPC, linear queue):
   job-gpu -> job-cpu -> counting-sweep (training then in-place eval).
6. Phase 5 synthesis and writeup (3-5 days).

**Total elapsed**: ~10-12 working days from approval, within the
2-week ceiling.

### 1.9 Decision Log

| ID | Decision | Value |
|---|---|---|
| DEC-004 | RQ9 ownership | Abe executes, Aras reviews, sequential queue after analysis main |
| DEC-007 | Counting vocab | One-to-one V=256; compositional vocabulary deferred to backlog (post-main-analysis) |
| DEC-009 | Branch strategy | Strategy C -- develop on main, port tak's algorithms as Python modules |
| DEC-012 (REVISED) | Counting model width | n_embd sweep {128, 256, 512, 1024}, n_head=4, d_ff=4 x n_embd |
| DEC-012a | Validation gate | 4L baseline >= 80 % IND accuracy before CoTFormer comparison per width |
| DEC-012b | BOS token | Include in sequence format (RoPE depends on BOS) |
| DEC-012c | Shifted PEs | Randomise starting position for OOD-position robustness |
| DEC-013 | Counting primary task | Task 1 (shifted-start sequential) primary; Tasks 2-3 secondary |
| DEC-014b | Counting start-token loss | Mask out (exclude trivial first-output from CE) |
| DEC-015 | n_repeat for RQ9 | 4 (depth-matched to 4L); repeat=5 backlogged |
| DEC-016 | Training steps | 312.5K at BS=32, matching Chang and Bisk [25] |
| DEC-017 | Tuned Lens fallback | SGD first, Muon if convergence poor |
| DEC-019 | RQ9 baseline config | Match Chang and Bisk codebase: ReLU activation, grad\_clip=0.3, `scale_attn_by_inverse_layer_idx=True`, task-specific vocabulary (V=203 for te200) |
| DEC-020 | DEC-019 reconciliation | Chang and Bisk's reference codebase diverges from paper-implied defaults on seven training-regime parameters: `max_grad_norm=1.0` (paper prose said 0.3 but the dead-code value is never read), `get_constant_schedule_with_warmup` (not cosine), AdamW defaults `wd=0.01`/`beta2=0.999` with no decay grouping, `n_head=8` at `n_embd=1024`, `fp16`, and `enable_flash=False`. [DEC-035](#19-decision-log) adds two further divergences (`--disable_decay_grouping` and `--positional_encoder none`) for a current total of nine. Pilot 1 adopts Chang and Bisk-exact; the main sweep uses CoTFormer-native defaults. Operational enumeration: [§1.2 RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting) Pilot 1 row; dead-code documentation: `docs/reprod-notes.md` §B12. | Post-hostile-review |
| DEC-021 | Twin-pilot design | Pilot 1 = Chang and Bisk-exact at `n_embd=1024` 4L with 5 seeds; main sweep = CoTFormer-native at all four widths with 5 seeds at `n_embd in {256, 1024}` and 3 seeds at `{128, 512}`. Comparison isolates the training-regime effect from the architecture effect. | Abe 2026-04-18 |
| DEC-022 | Novelty Claim 1 narrow reframing | "First mechanistic interpretability study of iterative/recurrent transformers" replaced with the CoTFormer-specific scope: cross-repeat KV attention + weight-tied mid-block with reserved begin/end layers + canonical Belrose 2023 Tuned Lens at five virtual layers. Cite Lu et al. [27] and Chen et al. [28] as closest prior art. | Abe 2026-04-18 |
| DEC-023 | Novelty Claim 2 narrow reframing | "First Tuned Lens on weight-tied architecture" reframed as the canonical Belrose 2023 [24] Tuned Lens (affine + bias, SGD + Nesterov, forward-KL, identity-init) on a CoTFormer-style weight-tied depth-recurrent transformer with cross-repeat KV attention. Closest prior art: Paulo et al. [30]. | Abe 2026-04-18 |
| DEC-024 | Novelty Claim 5 demotion | "Merrill formal gap: vertical != horizontal CoT" demoted from a novelty claim to related-work cite. RQ9 reframed as an empirical test of Xu and Sato 2025 [29] Theorem 1 on the approximate-counting expressiveness gap between latent and horizontal chain-of-thought; Merrill and Sabharwal [17] retained as an upstream complexity-class result. | Abe 2026-04-18 |
| DEC-025 | Multiple-comparison correction | Hybrid Holm-Bonferroni at family-wise alpha = 0.01 across the 9 RQ-primary tests; Benjamini-Hochberg FDR at q = 0.05 within each RQ's secondary tests. Replication-stability gates and pre-registered validation gates exempt. | Abe 2026-04-18 |
| DEC-026 | RQ5 DV-3 + calibration linkage and §1.7/§1.8 fixes | DV-3 = `cos(h_target@rep5, mean(h_context_surviving@rep5))` at final-mid-block PRE-`ln_mid`; unweighted mean; strict `halt_depth(j) >= 5`; H0 `|r| < 0.1` with 1000-permutation null. DV-2 subject to Protocol D-calibration. P2 mixed-effects replaces 252-head Wilcoxon. RQ7 permutation count raised to 10000. RQ8 zero-vector condition added (8 conditions). RQ2 debiased CKA + permutation-null zone thresholds. RQ3 silhouette permutation null. RQ4 bootstrap CI. RQ6 per-layer log-linear regression replaces Mann-Kendall. T8/T9/T10/T11 infrastructure consolidations. | Abe 2026-04-18 |
| DEC-027 | RQ9 independent DVs | Add DV-3 (attention-pattern analysis at penultimate repeat) + DV-4 (causal intervention via zero-ablation at repeat r). Both mandated per the W-18 operational-independence bar. | Abe 2026-04-18 |
| DEC-028 | RQ9b explicit pre-registration | 2x2 factorial CoT+Reserved-vs-LN-CoTFormer x `n_embd`; two-way ANOVA on the architecture x `ln_mid` interaction effect; falsification at p < 0.05 with eta-squared > 0.02. | Abe 2026-04-18 |
| DEC-029 | Calibration substrate three-tier | Tier 1 GPT-2-large (literature-grounded primary), Tier 2 ADM C5 (target-architecture primary; ADM wins when tiers disagree), Tier 3 24L standard (contingency; triggers Phase-0 conditional training). Pre-registration in `analysis/calibration/preregister.md`. | Abe 2026-04-18 |
| DEC-030 | Scope-shedding hybrid trigger | Pilot 1 fail at approximately day 3 sheds RQ9 to the extension (analytic sub-stream continues); calibration inconclusive at approximately day 5 flags DV-2 single-measurement-preliminary; day-10 progress under 60 % sheds in order RQ8 → RQ5 → RQ9. Triggers checked at day-3, day-5, day-10 review checkpoints with Abe-authorised one-way shedding inside the analysis. | Abe 2026-04-18 |
| DEC-032 | 48L standard training deferred post-analysis | Cite the paper's 48L standard baseline (24.17 PPL from Table 1 of [3]) rather than reproducing it. Train 24L standard only IF Tier 1 (GPT-2-large) and Tier 2 (ADM C5) calibration verdicts disagree; defer 48L standard to post-analysis. | Abe 2026-04-18 |
| DEC-033 | RQ9 three-arm redesign: breadth-over-depth + four CoTFormer variants + step-matched budget + contingent C&B-adapted arm | RQ9 is restructured as three arms: (Arm A) Pilot 1 Chang and Bisk-exact 4L standard at `n_embd=1024`, the ONLY strict-reproduction control; (Arm B) four CoTFormer variants V1 (`cotformer_full_depth`, no reserved), V2 (CoT+Reserved with 2+1 reserved, **7 effective layers, compared against the 4L standard with the depth asymmetry reported as a secondary finding about reserved layers plus recurrence jointly per Option (c)**), V3 (LN-CoTFormer), V4 (ADM) at widths starting from `n_embd=1024` under the CoTFormer-native regime; (Arm C) same four variants with C&B-adapted hyperparameters (ReLU, `max_grad_norm=0.3`, CoTFormer-specific `scale_attn_by_inverse_layer_idx`), triggered **contingent on Arm B divergences** only. Training-step budget reverts from match-samples 78K-at-BS-128 ([DEC-031](#19-decision-log)) to step-matched **312.5K steps at BS=32** per cell (matching Chang and Bisk [25], restoring the original [DEC-016](#19-decision-log)). Seed allocation is **breadth-over-depth**: **initial wave is 4 architectures x 3 seeds at `n_embd=1024` = 12 cells** *(secondary-width expansion now committed unconditionally per DEC-035; final Arm B = 32 cells.)* (~6-9 L4-days on 2x L4); secondary-width expansion to `{128, 256, 512}` at reduced per-cell seed count is contingent on initial-wave results and remaining budget. DEC-018 (ADM excluded) is superseded by this entry; ADM is now included with routing DISABLED (min_repeat = n_repeat = 4) for comparability with the non-adaptive variants. DEC-031 (match-samples 78K-at-BS-128 and seed-symmetry bumps) is superseded. The V2 depth-match ambiguity is resolved via **Option (c)**: run V2 at its paper-canonical 7 effective layers, compare against the 4L standard only, acknowledge the asymmetry as a secondary finding about reserved layers plus recurrence. | Abe 2026-04-22 |
| DEC-034 | Tuned Lens target-residual site + batch size alignment with canonical Belrose 2023 | The Tuned Lens target-residual is the PRE-`ln_f` residual (the input to `ln_f`), captured via the new `ActivationSite.RESIDUAL_PRE_LN_F` pre-hook; `p_final` is computed as `softmax(lm_head(ln_f(h_pre_ln_f)))` so `ln_f` is applied exactly once on both lens and target paths. This removes the earlier POST-`ln_f` approximation where `ln_f` was applied twice on the target. Translator batch size is 256 tokens per SGD step, matching Belrose 2023 [24] exactly rather than the interim 64-token default. Rationale: preserves the DEC-023 Novelty Claim 2 framing "canonical Belrose 2023 Tuned Lens" without approximation caveats; the pre-`ln_f` capture adds one extra `.npy` per workspace (~6 MB for 2048 tokens at `n_embd=768`) and negligible compute. | Abe 2026-04-22 |
| DEC-035 | RQ9 spec realignment: BLOCKER 6 closure + Arm A strict-fidelity additions + Arm B secondary-width expansion | RQ9 Phase 3 spec realignment — (a) BLOCKER 6 `attention_mask` plumbing closed across all three CoTFormer variants (ESC-001 resolution; see extend-technical.md §3.6); (b) Arm A `--disable_decay_grouping` flag added (ESC-002, eighth Chang-Bisk divergence); (c) Arm A `--positional_encoder none` flag added (ESC-003, ninth Chang-Bisk divergence); (d) Arm B secondary-width expansion committed unconditionally for a **32-cell** factorial (ESC-004 option 1; supersedes DEC-033's "contingent on initial-wave results" clause). DEC-018 and DEC-031 fully superseded by this entry. | Abe 2026-04-22 |

#### Backlog items (deferred post-main-analysis)

- **repeat=5 counting**: re-run RQ9 at n_repeat=5 to align with the original 24L CoTFormer architecture.
- **Compositional vocabulary**: multi-digit tokenisation as a follow-up to the whole-number baseline; tests OOD generalisation without a fixed vocabulary ceiling.

#### Novel contributions confirmed (narrow reframings)

The original five-claim novelty roster has been narrowed to
defensible claims following the pre-Phase-1 hostile review and
the literature-fetch sub-orch's verification of close prior art.
Claims 1, 2, and 5 are reframed (DEC-022, DEC-023,
DEC-024); Claims 3 and 4 are preserved as stated.

| Contribution | Reframed scope | Evidence and closest prior art |
|---|---|---|
| Mechanistic interpretability study **specific to weight-tied depth-recurrent transformers with cross-repeat KV attention** (per [DEC-022](#19-decision-log)) | Lu et al. [27] applied Logit Lens and Coda Lens to Huginn-3.5B [16] — depth-recurrent, but no cross-repeat KV cache; Chen et al. [28] measured cross-loop representation degradation in looped transformers — also no cross-repeat KV. Knupp et al. [37] is a contemporary depth-recurrent attention-mixture comparator with FLOP/parameter/memory matching but does not perform lens-trajectory or cross-repeat KV measurements. The CoTFormer-specific gap is the combination of cross-repeat KV cache + weight-tied mid-block + reserved begin/end layers + per-(layer, repeat) Tuned Lens at five virtual layers. Mohtashami et al. [3] Appendix F explicitly defers this analysis. |
| **Canonical Belrose 2023 [24] Tuned Lens** (affine + bias, SGD + Nesterov, forward-KL, identity-init) on a CoTFormer-style weight-tied depth-recurrent transformer with cross-repeat KV (per [DEC-023](#19-decision-log)) | Paulo et al. [30] transferred Tuned Lens to Mamba and RWKV recurrent architectures — methodological precedent but no weight-tied recurrent transformers; Belrose et al. [24] all models have distinct parameters per layer. Lu et al. [27] used Logit Lens and Coda Lens (not the canonical Tuned Lens) on Huginn-3.5B [16]. The combination of cross-repeat KV cache, five weight-shared mid-block repeats, and per-(layer, repeat) translator fitting is the specific gap. |
| First **KV-CoRE on cross-repeat cache** | Chen et al. [8]: all models are standard multi-layer; weight-tied recurrent KV cache is a structurally distinct setting. |
| First **counting experiment with CoTFormer architecture** under twin-pilot reproduction-fidelity discipline | Chang and Bisk [25] use the standard 4L baseline (no weight-tying, no recurrence); the CoTFormer [3] paper reports no synthetic-task experiments. The twin-pilot design (DEC-021) is a methodological extension that separates reproduction-fidelity (Pilot 1) from fair-comparison (main sweep) regimes — a separation not present in the prior counting literature. |
| **Empirical test of Xu and Sato 2025 [29] Theorem 1** on the approximate-counting expressiveness gap between latent and horizontal CoT (per [DEC-024](#19-decision-log) demoting the original "Merrill formal gap" claim from a novelty to a related-work cite) | Xu and Sato [29] Theorem 1 establishes the formal prior; Merrill and Sabharwal [17] is the upstream complexity-class result motivating Xu and Sato's analysis. RQ9 measures whether the predicted gap is empirically narrowed, preserved, or widened in the CoTFormer's weight-tied recurrent setting. |

---

## 2. Architectural Extensions

Each extension is motivated by specific analysis findings. Implementation details
are deliberately brief -- they will be expanded once analysis results inform design.

### 2.1 Multi-Head Latent Attention (MLA)

> **Status**: post-M2 deliverable; design pending mechanistic-analysis findings. Activated upon analysis completion.

#### Motivation

CoTFormer's cross-repeat KV cache grows as `R*T` per mid-layer (4.4x overhead
at R=5). MLA [1] compresses KV through a low-rank latent bottleneck: 224
values/token instead of 1536 (6.9x compression). Savings scale with R.

Alternatives (GQA [2], KV quantization, sparse attention, fewer repeats) are
either less expressive, orthogonal, or change the compute budget rather than
efficiency. MLA preserves full per-head diversity and the complete attention
pattern while compressing the cache.

#### Design Decisions (provisional, pending analysis)

| ID | Decision | Contingent On |
|----|----------|---------------|
| DEC-EXT-001 | Rank allocation: uniform kv_lora_rank=192 | RQ6. If d_eff varies >2x, adopt CARE [14]. |
| DEC-EXT-002 | Separate RoPE branch (d_R=32) | MHA2MLA [15] analysis |
| DEC-EXT-003 | RMSNorm on compressed latents | Standard [5] |
| DEC-EXT-004 | Mid-blocks only (prefix/suffix keep MHA) | No cache growth there |
| DEC-EXT-005 | Train from scratch, 40k steps, matched hyperparams | Fair comparison |
| DEC-EXT-006 | Cache compressed latents, decompress on access | Core MLA [1] |

Target: PPL < +0.5 vs LN-CoT, 6.9x mid-block KV compression, peak VRAM < 12 GB.

See Appendix A for projection architecture and dimensionality reference.

### 2.2 Manifold-Constrained HyperConnections (mcHC)

> **Status**: post-M2 deliverable; design pending mechanistic-analysis findings. Activated upon analysis completion.

mcHC [6] generalises residual connections via multi-stream routing (`alpha`
streams, learned `C in R^{alpha x alpha}` per layer, Stiefel-constrained).
For CoTFormer: different subspaces routed through different repeats.

**Decision gate** (from the analysis): strong RQ4 correlation = mcHC as complement;
weak = mcHC as router replacement. CKA collapse (RQ2) = strong motivation.

Design: alpha=2, mid-layers only. Memory: 2x hidden state. Params: 8/layer.

### 2.3 Causal Cross-Repeat Mask for ADM

> **Status**: post-M2 deliverable; design pending mechanistic-analysis findings. Activated upon analysis completion.

The ADM uses `is_causal=False` with an index-based mask (`k_indices <= indices`)
that enforces temporal ordering but ignores repeat-depth structure. A token at
repeat 3 attends uniformly to ALL previous tokens at ALL repeats.

**Research question** (SQ5): does adding repeat-depth-aware causal constraints
improve the perplexity-compute trade-off? Ablation: same ADM architecture,
modified mask, identical hyperparameters.

Design pending analysis Protocol C (attention taxonomy) and literature findings.

### 2.4 Residual Attention

> **Status**: post-M2 deliverable; design pending mechanistic-analysis findings. Activated upon analysis completion.

Instead of compressing the full accumulated KV cache (grows linearly with R),
attend directly to per-position residual deltas across repeats. The "residual
cache" has size `R * d_model` per position (depth dimension), not `R * T *
d_model` (sequence times depth), avoiding linear growth entirely.

**Research question**: does attending to residual deltas preserve cross-repeat
information quality while eliminating the KV growth problem?

Design pending targeted literature review on differential/residual attention.

---

## Appendix A: MLA Dimensionality Reference

### Projection Architecture

```python
class MLACausalSelfAttention(nn.Module):
    def __init__(self, config):
        d = config.n_embd          # 768
        n_h = config.n_head        # 12
        d_c = config.kv_lora_rank  # 192
        d_qc = config.q_lora_rank  # 384
        d_R = config.qk_rope_head_dim  # 32
        d_nope = config.qk_nope_head_dim  # 64
        d_v = config.v_head_dim    # 64

        # Query: down -> norm -> up + RoPE branch
        self.wq_a = nn.Linear(d, d_qc, bias=False)
        self.q_norm = RMSNorm(d_qc)
        self.wq_b = nn.Linear(d_qc, n_h * d_nope, bias=False)
        self.wq_rope = nn.Linear(d_qc, n_h * d_R, bias=False)

        # KV: joint down -> norm -> up
        self.wkv_a = nn.Linear(d, d_c + d_R, bias=False)
        self.kv_norm = RMSNorm(d_c)
        self.wkv_b = nn.Linear(d_c, n_h * (d_nope + d_v), bias=False)

        # Output
        self.c_proj = nn.Linear(n_h * d_v, d, bias=False)
```

### RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

### Dimensionality Map

```
Standard MHA:  Cache/token = 2 * 12 * 64 = 1536 floats
MLA:           Cache/token = 192 + 32 = 224 floats
Compression:   6.9x
```

---

## References

[1] DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.

[2] J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints.* arXiv:2305.13245, 2023.

[3] A. Mohtashami et al. *CoTFormer: A Chain-of-Thought Driven Architecture
    with Budget-Adaptive Computation Cost at Inference.* In ICLR, 2025.

[4] N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.*
    arXiv:1911.02150, 2019.

[5] B. Zhang and R. Sennrich. *Root Mean Square Layer Normalization.* In
    NeurIPS, 2019.

[6] DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.

[7] Z. Zheng et al. *Attention Heads of Large Language Models: A Survey.* In
    Patterns (Cell Press), 2025. arXiv:2409.03752.

[8] J. Chen et al. *KV-CoRE: Benchmarking Data-Dependent Low-Rank
    Compressibility of KV-Caches in LLMs.* arXiv:2602.05929, 2026.

[9] C. Chun et al. *Estimating Dimensionality of Neural Representations from
    Finite Samples.* arXiv:2509.26560, 2025.

[10] T. Nait Saada et al. *Mind the Gap: A Spectral Analysis of Rank Collapse
     and Signal Propagation in Attention Layers.* arXiv:2410.07799, 2025.

[11] S. Kobayashi et al. *Weight Decay Induces Low-Rank Attention Layers.*
     arXiv:2410.23819, 2024.

[12] N. Zucchet et al. *The Emergence of Sparse Attention: Impact of Data
     Distribution and Benefits of Repetition.* arXiv:2505.17863, 2025.

[13] Y. Xu et al. *Tracking the Feature Dynamics in LLM Training: A
     Mechanistic Study.* arXiv:2412.17626, 2024.

[14] Z. Zhou et al. *CARE: Covariance-Aware and Rank-Enhanced Decomposition
     for Enabling Multi-Head Latent Attention.* arXiv:2603.17946, 2026.

[15] T. Ji et al. *MHA2MLA: Towards Economical Inference -- Enabling
     DeepSeek's Multi-Head Latent Attention in Any Transformer.* In ACL, 2025.
     arXiv:2502.14837.

[16] J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R.
     Bartoldson, B. Kailkhura, A. Bhatele, T. Goldstein. *Scaling up
     Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.*
     arXiv:2502.05171, 2025. Local PDF:
     `docs/LatentReasoningScaling-Geiping2025.pdf`.

[17] W. Merrill and A. Sabharwal. *The Expressive Power of Transformers with
     Chain of Thought.* In ICLR, 2024. arXiv:2310.07923.

[18] S. Hao et al. *Training Large Language Models to Reason in a Continuous
     Latent Space (Coconut).* arXiv:2412.06769, 2024.

[19] B. Wang et al. *Grokked Transformers are Implicit Reasoners.* In
     NeurIPS, 2024. arXiv:2405.15071.

[20] H. Yao et al. *Thin Keys, Full Values: Reducing KV Cache via
     Low-Dimensional Attention Selection.* arXiv:2603.04427, 2026.

[21] J. Sok et al. *Garbage Attention in LLMs: BOS Sink Heads and Sink-aware
     Pruning.* arXiv:2601.06787, 2026.

[22] D. Lesens et al. *KQ-SVD: Compressing the KV Cache with Provable
     Guarantees on Attention Fidelity.* arXiv:2512.05916, 2025.

[23] X. Chen et al. *Reasoning Beyond Language: A Comprehensive Survey on
     Latent Chain-of-Thought Reasoning.* arXiv:2505.16782, 2025.

[24] N. Belrose et al. *Eliciting Latent Predictions from Transformers with
     the Tuned Lens.* In NeurIPS, 2023. arXiv:2303.08112.

[25] J. Chang and Y. Bisk. 2025. OpenReview `s3IBHTTDYl`. Local copy:
     `docs/InductiveBiasCheng&Bisk.pdf`.

[26] G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming
     Language Models with Attention Sinks.* In ICLR, 2024.
     arXiv:2309.17453.

[27] W. Lu, M. Yang, S. Lee, Y. Li, P. Liu. *Latent Chain-of-Thought?
     Decoding the Depth-Recurrent Transformer.* COLM 2025 Workshop on
     LLM Explainability for Reasoning and Planning. arXiv:2507.02199,
     2025. Local PDF: `docs/LatentCoT-Lu2025.pdf`.

[28] G. Chen, S. Liu, J. Shao. *Loop as a Bridge: Can Looped Transformers
     Truly Link Representation Space and Natural Language Outputs?*
     arXiv:2601.10242, 2026. Local PDF:
     `docs/LoopAsBridge-Chen2026.pdf`.

[29] L. Xu and Y. Sato. *A Formal Comparison Between Chain of Thought
     and Latent Thought.* arXiv:2509.25239, 2025. Local PDF:
     `docs/CoTvsLatent-XuSato2025.pdf`.

[30] G. Paulo, T. Marshall, N. Belrose. *Does Transformer
     Interpretability Transfer to RNNs?* AAAI 2025 (proceedings
     version). arXiv:2404.05971, 2024. Local PDF:
     `docs/InterpMethodsRNN-Paulo2025.pdf`.

[31] M. R. Davari, S. Horoi, A. Natik, G. Lajoie, G. Wolf, E.
     Belilovsky. *Reliability of CKA as a Similarity Measure in Deep
     Learning.* arXiv:2210.16156, 2022. Local PDF:
     `docs/CKAReliability-Davari2022.pdf`.

[32] A. R. Murphy, J. Zylberberg, A. Fyshe. *Correcting Biased
     Centered Kernel Alignment Measures in Biological and Artificial
     Neural Networks.* ICLR 2024 Re-Align Workshop. arXiv:2405.01012;
     OpenReview `E1NRrGtIHG`. Local PDF:
     `docs/CKADebiased-Murphy2024.pdf`.

[33] M. Chun, A. Canatar, S. Chung, K. Lee. *Estimating Neural
     Representation Alignment from Sparsely Sampled Inputs and
     Features.* arXiv:2502.15104, 2025. Local PDF:
     `docs/CKABias-Chun2025.pdf`. Note: distinct from Chun et al. [9]
     (different first author and topic).

[34] F. Wang, W. Shao, H. Yu, G. Kan, X. He, D. Zhang, M. Ren, K.
     Wang. *Re-evaluation of the Power of the Mann-Kendall Test for
     Detecting Monotonic Trends in Hydrometeorological Time Series.*
     Frontiers in Earth Science. doi:10.3389/feart.2020.00014, 2020.
     Local PDF: `docs/MannKendallPower-Wang2020.pdf`.

[35] J. Hewitt and P. Liang. *Designing and Interpreting Probes with
     Control Tasks.* EMNLP 2019. arXiv:1909.03368. Local PDF:
     `docs/ProbeControl-HewittLiang2019.pdf`.

[36] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S.
     Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R.
     Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, C.
     Olah. *Toy Models of Superposition.* transformer-circuits.pub,
     September 2022. arXiv:2209.10652. Local PDF:
     `docs/SuperpositionToyModels-Elhage2022.pdf`.

[37] J. Knupp, J. H. Metzen, V. Bohn, J. Groh, K. Kersting.
     *Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the
     Attention it Deserves.* arXiv:2601.21582, 2026. Local PDF:
     `docs/DepthRecurrentAttn-Knupp2026.pdf`.

[38] S. Takase and S. Kiyono. *Lessons on Parameter Sharing across
     Layers in Transformers.* arXiv:2104.06022, 2021. Local PDF:
     `docs/ParameterSharing-TakaseKiyono2021.pdf`.

[39] S. Tan, Y. Shen, Z. Chen, A. Courville, C. Gan. *Sparse Universal
     Transformer.* arXiv:2310.07096, 2023. Local PDF:
     `docs/SparseUniversalTransformer-Tan2023.pdf`.

[40] S. Kornblith, M. Norouzi, H. Lee, G. Hinton. *Similarity of
     Neural Network Representations Revisited.* In ICML, 2019.
     arXiv:1905.00414. Local PDF:
     `docs/CKAOriginal-Kornblith2019.pdf`.

[41] C. Olsson, N. Elhage, N. Nanda, et al. *In-context Learning and
     Induction Heads.* Transformer Circuits Thread, 2022.
     arXiv:2209.11895.

[42] N. Hurley and S. Rickard. *Comparing Measures of Sparsity.* IEEE
     Transactions on Information Theory 55(10):4723-4741, 2009.
     arXiv:0811.4706.

[43] C. Meister, S. Lazov, I. Augenstein, R. Cotterell. *Is Sparse
     Attention more Interpretable?* In ACL (short), 2021.
     arXiv:2106.01087.

[44] S. Zhai, T. Likhomanenko, E. Littwin, et al. *Stabilizing
     Transformer Training by Preventing Attention Entropy Collapse.*
     In ICML, 2023. arXiv:2303.06296.

[45] M. Pagliardini, D. Paliotta, M. Jaggi, F. Fleuret. *Faster
     Causal Attention Over Large Sequences Through Sparse Flash
     Attention.* In NeurIPS, 2023. arXiv:2306.01160.

[46] X. Gu, et al. *When Attention Sink Emerges in Language Models.*
     In ICLR, 2025.

[47] S. Bae, et al. *Mixture-of-Recursions: Learning Dynamic Recursive
     Depths for Adaptive Token-Level Computation.* arXiv:2507.10524,
     2025.

[48] D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. Humphreys, A.
     Santoro. *Mixture-of-Depths: Dynamically allocating compute in
     transformer-based language models.* arXiv:2404.02258, 2024.

[49] J. Cohen. *Statistical Power Analysis for the Behavioral
     Sciences.* 2nd ed., Lawrence Erlbaum, 1988.

[50] D. Curran-Everett. *Explorations in statistics: standard
     deviations and standard errors.* Advances in Physiology
     Education 32(3):203-208, 2008.

[51] L. Kaufman and P. J. Rousseeuw. *Finding Groups in Data: An
     Introduction to Cluster Analysis.* Wiley, 1990.

[52] S. Holm. *A Simple Sequentially Rejective Multiple Test
     Procedure.* Scandinavian Journal of Statistics 6(2):65-70, 1979.

[53] Y. Benjamini and Y. Hochberg. *Controlling the False Discovery
     Rate: A Practical and Powerful Approach to Multiple Testing.*
     Journal of the Royal Statistical Society Series B
     57(1):289-300, 1995.
