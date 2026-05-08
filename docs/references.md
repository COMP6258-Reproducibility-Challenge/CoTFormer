# External References — CoTFormer

Collected for methodology validation and novelty-claim triangulation during
the pre-Phase-1 hostile-review pass (2026-04-18). Papers are grouped by
relevance tier. Each entry gives the local filename, full citation, an
arXiv or journal identifier, and a single-sentence relevance tag that
explains the downstream use within the Unpick-and-Analyse work
programme documented in `docs/extend-notes.md`. Filenames follow the
existing `docs/*.pdf` convention `<Topic>-<FirstAuthorLastName>YYYY.pdf`;
four original working filenames were corrected to match the actual
first-author attribution (see §Attribution corrections).

## High priority — novelty-scoop evidence

- **`LatentCoT-Lu2025.pdf`** — Lu, Yang, Lee, Li, Liu. 2025. "Latent
  Chain-of-Thought? Decoding the Depth-Recurrent Transformer". COLM 2025
  Workshop on LLM Explainability for Reasoning and Planning. arXiv:2507.02199.
  *Relevance*: applies Logit Lens and Coda Lens to Huginn-3.5B and tracks
  rank trajectories of final and intermediate tokens across recurrent
  blocks — partially scoops our RQ1 lens-based latent-CoT analysis and
  independently corroborates the probing-inconsistency finding our
  triangulation design anticipates.

- **`LoopAsBridge-Chen2026.pdf`** — Chen, Liu, Shao (Shanghai AI Lab,
  Shanghai Jiao Tong University). 2026. "Loop as a Bridge: Can Looped
  Transformers Truly Link Representation Space and Natural Language
  Outputs?". arXiv:2601.10242.
  *Relevance*: empirically shows that iterating shared layers narrows the
  internal-knowledge-to-output gap partly through representation
  degradation, with representation-perception ability concentrated in the
  final loop rather than improving across loops — directly informs how
  we interpret the cascading mechanism and lens trajectories in
  LN-CoTFormer (RQ1, RQ3).

- **`CoTvsLatent-XuSato2025.pdf`** — Xu, Sato. 2025. "A Formal Comparison
  Between Chain of Thought and Latent Thought". arXiv:2509.25239.
  *Relevance*: proves latent-thought reasoning admits more efficient
  parallel computation than sequential CoT while CoT retains an
  advantage on approximate counting and stochastic sampling — bounds
  the theoretical expectations for RQ9 (counting) and delineates which
  tasks depth-recurrence should and should not improve.

- **`InterpMethodsRNN-Paulo2025.pdf`** — Paulo, Marshall, Belrose
  (EleutherAI). 2024/AAAI 2025. "Does Transformer Interpretability
  Transfer to RNNs?". arXiv:2404.05971 (AAAI 2025 proceedings version).
  *Relevance*: establishes that tuned-lens, contrastive activation
  addition, and latent-knowledge elicitation transfer from transformers
  to Mamba and RWKV recurrent architectures — methodological precedent
  for applying transformer-designed lenses to depth-recurrent
  CoTFormer variants in RQ1.

## Medium priority — methodological foundations

- **`TunedLens-Belrose2023.pdf`** — Belrose, Ostrovsky, McKinney, Furman,
  Smith, Halawi, Biderman, Steinhardt (EleutherAI, FAR AI, Toronto,
  Boston University, UC Berkeley). 2023. "Eliciting Latent Predictions
  from Transformers with the Tuned Lens". arXiv:2303.08112.
  *Relevance*: introduces the Tuned Lens — a learned affine translator
  per layer that decodes hidden states into the model's pretrained
  unembedding space, refining the brittle Logit Lens — and the
  prediction-trajectory framework on which our RQ1 lens-based
  latent-CoT analysis is built. The canonical methodological reference
  for DEC-023's Novelty Claim 2 ("canonical Belrose 2023 Tuned Lens on
  a CoTFormer-style weight-tied depth-recurrent transformer") and
  DEC-034's pre-`ln_f` target-residual + 256-token batch-size
  alignment.

- **`CKAReliability-Davari2022.pdf`** — Davari, Horoi, Natik, Lajoie,
  Wolf, Belilovsky (Concordia, Universite de Montreal, Mila). 2022.
  "Reliability of CKA as a Similarity Measure in Deep Learning".
  arXiv:2210.16156.
  *Relevance*: documents CKA's sensitivity to a small number of
  high-norm data points and its failure to reliably identify
  functionally equivalent networks — grounds the requirement in RQ6 to
  triangulate CKA with at least two other alignment protocols rather
  than treating CKA as a standalone verdict.

- **`CKADebiased-Murphy2024.pdf`** — Murphy, Zylberberg, Fyshe
  (Alberta, York). 2024. "Correcting Biased Centered Kernel Alignment
  Measures in Biological and Artificial Neural Networks". ICLR 2024
  Re-Align Workshop. arXiv:2405.01012 (OpenReview E1NRrGtIHG).
  *Relevance*: shows that biased CKA returns high similarity even on
  random matrices in the low-data high-dimensionality regime and that
  debiased CKA is required to recover stimulus-driven alignment —
  prescribes the debiased estimator we adopt for cross-block
  comparisons in RQ6.

- **`CKABias-Chun2025.pdf`** — Chun, Canatar, Chung, Lee (Weill Cornell,
  Flatiron, Cornell Tech). 2025. "Estimating Neural Representation
  Alignment from Sparsely Sampled Inputs and Features". arXiv:2502.15104.
  *Relevance*: introduces an estimator that corrects CKA for both input
  and feature sampling simultaneously — the primary alignment metric we
  pair with Kobayashi rank and KV-CoRE NER in the RQ6 three-way
  triangulation.

- **`MannKendallPower-Wang2020.pdf`** — Wang, Shao, Yu, Kan, He, Zhang,
  Ren, Wang. 2020. "Re-evaluation of the Power of the Mann-Kendall Test
  for Detecting Monotonic Trends in Hydrometeorological Time Series".
  Frontiers in Earth Science. doi:10.3389/feart.2020.00014.
  *Relevance*: Monte-Carlo assessment of Mann-Kendall power under
  serial correlation, building on Yue and Wang (2002) — grounds our
  choice of Mann-Kendall for rank-trajectory monotonicity tests in
  RQ1 and the pre-whitening or effective-sample-size correction
  required when the rank series is autocorrelated across repeats.

- **`ProbeControl-HewittLiang2019.pdf`** — Hewitt, Liang (Stanford).
  2019. "Designing and Interpreting Probes with Control Tasks".
  EMNLP 2019. arXiv:1909.03368.
  *Relevance*: introduces the selectivity metric and control tasks
  that distinguish whether a probe's accuracy reflects the
  representation encoding a property or the probe memorising the
  task — the canonical methodology our RQ4 probing protocol adopts
  for control-baseline design.

## Lower priority — good-to-have

- **`SuperpositionToyModels-Elhage2022.pdf`** — Elhage, Hume, Olsson,
  Schiefer, Henighan, Kravec, Hatfield-Dodds, Lasenby, Drain, Chen,
  Grosse, McCandlish, Kaplan, Amodei, Wattenberg, Olah
  (Anthropic, Harvard). 2022. "Toy Models of Superposition".
  transformer-circuits.pub (Sept 2022). arXiv:2209.10652.
  *Relevance*: provides the conceptual frame for interpreting
  polysemantic neuron behaviour and feature superposition in the
  mid-repeat residual stream — background for discussing why a single
  neuron or direction can encode different information at different
  repeat depths (RQ1 lens interpretation).

- **`DepthRecurrentAttn-Knupp2026.pdf`** — Knupp, Metzen, Bohn, Groh,
  Kersting. 2026. "Depth-Recurrent Attention Mixtures: Giving Latent
  Reasoning the Attention it Deserves". arXiv:2601.21582.
  *Relevance*: introduces the Dreamer framework combining sequence,
  depth, and sparse-expert attention in depth-recurrent layers with
  FLOP-, parameter-, and memory-matched baselines — contemporary
  architectural comparator that informs the framing of our
  depth-recurrence ablations and validates the importance of
  FLOP-, parameter-, and memory-matched baselines when comparing
  depth-recurrent variants.

- **`LatentReasoningScaling-Geiping2025.pdf`** — Geiping, McLeish, Jain,
  Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein
  (ELLIS Institute, Maryland, LLNL). 2025. "Scaling up Test-Time
  Compute with Latent Reasoning: A Recurrent Depth Approach".
  arXiv:2502.05171.
  *Relevance*: describes Huginn-3.5B, the 3.5B-parameter depth-recurrent
  model whose behaviour `LatentCoT-Lu2025.pdf` dissects — upstream
  context for the latent-CoT novelty-scoop assessment and the
  precedent for test-time recurrent unrolling.

- **`ParameterSharing-TakaseKiyono2021.pdf`** — Takase, Kiyono
  (LINE Corporation). 2021. "Lessons on Parameter Sharing across
  Layers in Transformers". arXiv:2104.06022.
  *Relevance*: compares SEQUENCE, CYCLE, and CYCLE (REV) parameter-
  sharing strategies as relaxations of Universal-Transformer-style
  all-shared weights — historical context for the weight-sharing
  design space surrounding the full-depth CoTFormer baseline.

- **`SparseUniversalTransformer-Tan2023.pdf`** — Tan, Shen, Chen,
  Courville, Gan (Mila, MIT-IBM Watson AI Lab). 2023. "Sparse
  Universal Transformer". arXiv:2310.07096.
  *Relevance*: combines Sparse Mixture-of-Experts with a
  stick-breaking-based dynamic halting mechanism on Universal
  Transformer, reducing compute while preserving compositional
  generalisation — related-work anchor for the Adaptive Depth Module
  (ADM) router and halting design.

- **`CKAOriginal-Kornblith2019.pdf`** — Kornblith, Norouzi, Lee, Hinton
  (Google). 2019. "Similarity of Neural Network Representations
  Revisited". ICML 2019. arXiv:1905.00414.
  *Relevance*: canonical definition of CKA as a similarity index
  invariant to orthogonal transformation and isotropic scaling —
  the reference we cite for the CKA metric itself in RQ6.

## Unavailable / paywalled

None. All fifteen papers were fetched directly as free PDFs from arXiv,
transformer-circuits.pub, or Frontiers in Earth Science.

## Attribution corrections

During author verification (first-page extraction with `pdftotext`), four
working filenames from the hostile-review shortlist were found to use
incorrect first-author attributions. The files were renamed to match the
actual first author before this index was written. The corrections:

| Working filename | Corrected filename | Reason |
|------------------|--------------------|--------|
| `LatentCoT-Bahamid2025.pdf` | `LatentCoT-Lu2025.pdf` | arXiv:2507.02199 first author is Wenquan Lu (Brown University), not Bahamid. |
| `LoopAsBridge-Yong2026.pdf` | `LoopAsBridge-Chen2026.pdf` | arXiv:2601.10242 first author is Guanxu Chen (Shanghai AI Laboratory / SJTU), not Yong. |
| `MannKendallPower-Yue2020.pdf` | `MannKendallPower-Wang2020.pdf` | The 2020 Frontiers re-evaluation (doi:10.3389/feart.2020.00014) is authored by Fan Wang et al., referencing Yue & Wang 2002 as prior work; the filename now tracks the paper's own first author. |
| `DepthRecurrentAttn-2026.pdf` | `DepthRecurrentAttn-Knupp2026.pdf` | arXiv:2601.21582 first author is Jonas Knupp (TU Munich / TU Darmstadt); the directive flagged this paper as requiring author verification on fetch. |

No PDF content was altered; only filenames changed. The arXiv and
Frontiers identifiers recorded in each bullet above are the canonical
source of truth for each paper.
