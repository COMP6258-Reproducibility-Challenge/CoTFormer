"""Counting-task synthetic dataset (Chang and Bisk 2024 reference).

Scope
-----
In-memory dataset generating the inductive-counting task at the heart
of RQ9. The task is: given ``[start_int_token, a, a, ..., a]`` as
input tokens, produce ``[ignore, start+1, start+2, ..., start+L]`` as
target tokens. The model must learn to count running sums.

Falsifiability relevance
------------------------
RQ9 main-effect H0 "recurrent weight-tied transformers offer no OOD
advantage over 4L standard transformers" is tested by training every
architectural variant on the data produced here and comparing OOD
(length-50+) accuracy (see `docs/extend-notes.md` §1.2 RQ9).
RQ9b interaction H0 is tested over the same data with a two-way
ANOVA on the architecture x `ln_mid` panel.

Ontological purpose
-------------------
Provides a ground-truth minimal task where the correct next-token
distribution is a deterministic function of the input, so per-token
cross-entropy faithfully measures the model's ability to execute a
simple algorithmic operation (addition by 1). The task is the
Xu and Sato 2025 Theorem 1 empirical venue cited in
DEC-024.

Reference
---------
Chang and Bisk 2024 ``inductive_counting_with_LMs`` public codebase
(``dataset.py``, ``config_taskspecific.py``). Per
DEC-019 and the seven-divergence reconciliation DEC-020,
Pilot 1 uses their exact hyperparameter regime; the main sweep uses
the CoTFormer-native regime at identical task design.

Implementation notes (te200 variant)
------------------------------------
- Vocab: ``[str(0), ..., str(max_out)] + ['<pad>', 'a']``; for
  ``max_out=200`` this is V = 203 tokens.
- Train-length range: [1, 50] per Chang and Bisk Table 1.
- OOD-length range: [51, 200].
- Start range: [0, max_out - L] so ``start + L <= max_out``
  guarantees target tokens stay in-vocabulary.
- Pad token ``<pad>`` fills positions [L+1, seq_length); pad_mask is
  1 for valid positions and 0 for pad positions. Loss mask is 1 only
  for the L count-prediction positions (excludes the start-int
  position at index 0, which has no predecessor to count from).
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch


TE200_MAX_OUT = 200
TE200_VOCAB_SIZE = 203  # [0..200] + '<pad>' + 'a'
TE200_TRAIN_LENGTH_RANGE = (1, 50)
TE200_OOD_LENGTH_RANGE = (51, 200)


def build_vocab(max_out: int = TE200_MAX_OUT) -> tuple[list[str], dict[str, int]]:
    """Return the ordered vocab list and its inverse mapping.

    Matches ``inductive_counting_with_LMs/scripts/causal_transformer/dataset.py:45``:
    ``vocab = [str(i) for i in range(max_out + 1)] + ['<pad>', 'a']``.
    """
    vocab = [str(i) for i in range(max_out + 1)] + ['<pad>', 'a']
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i


class CountingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset yielding ``(tokens, targets, pad_mask, loss_mask)`` 4-tuples.

    Each sample is constructed deterministically at ``__init__`` time
    using the per-instance RNG seeded from ``seed``; this allows
    disjoint train / val / ood splits via disjoint seeds.

    Parameters
    ----------
    split : {"train", "val", "ood"}
        Determines the length range used when sampling ``L``.
    seed : int
        Deterministic RNG seed. Train/val/ood use ``seed``,
        ``seed + 1000``, ``seed + 2000`` by the project's seed_list
        convention (see ``prepare_counting_dataset``).
    num_samples : int
        Number of samples to pre-generate. The full set lives in
        host memory (counting samples are tiny; a ~10 KB per sample
        upper bound means 1e5 samples fit easily).
    sequence_length : int
        Pad length. Must be at least ``max_out + 1`` so that the
        longest OOD sequence (``L = max_out``, plus the start token at
        position 0) fits. Defaults in the project's arguments table
        set this to 256 to match Chang and Bisk max_position_embeddings.
    max_out : int
        Upper bound on counting targets. For te200 this is 200;
        produces a vocab of size ``max_out + 3``.
    length_range : tuple[int, int]
        Inclusive ``(lo, hi)`` bounds on the sampled length ``L``.
    start_range : tuple[int, int] | None
        Optional explicit start range; when None the constructor
        computes ``(0, max_out - L)`` per-sample so that ``start + L
        <= max_out`` holds for every sampled ``L``.

    Output contract
    ---------------
    Each ``__getitem__(idx)`` returns:

    - ``tokens``    : LongTensor[seq_length], input token indices.
    - ``targets``   : LongTensor[seq_length], target indices (``-1``
                      for ignore per ``F.cross_entropy`` convention).
    - ``pad_mask``  : FloatTensor[seq_length], 1.0 for valid positions
                      (input has a meaningful token) and 0.0 for pad
                      positions. Consumed by BLOCKER 6 attention mask.
    - ``loss_mask`` : FloatTensor[seq_length], 1.0 for running-count
                      positions (1..L inclusive) and 0.0 elsewhere
                      (including position 0, the start-int; which has
                      no predecessor to count from). Consumed by the
                      optim/base.py counting loss branch.
    """

    def __init__(
        self,
        split: str,
        seed: int,
        num_samples: int,
        sequence_length: int,
        max_out: int = TE200_MAX_OUT,
        length_range: tuple[int, int] = TE200_TRAIN_LENGTH_RANGE,
        start_range: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if sequence_length < max_out + 1:
            raise ValueError(
                f"sequence_length={sequence_length} too small for max_out={max_out}: "
                f"OOD samples require at least {max_out + 1} positions."
            )
        self.split = split
        self.num_samples = int(num_samples)
        self.sequence_length = int(sequence_length)
        self.max_out = int(max_out)
        self.length_range = length_range
        self.start_range = start_range

        self.vocab, self.w2i = build_vocab(self.max_out)
        self.pad_id = self.w2i['<pad>']
        self.a_id = self.w2i['a']

        rng = np.random.default_rng(int(seed))
        lo_L, hi_L = self.length_range
        samples: list[tuple[int, int]] = []
        for _ in range(self.num_samples):
            L = int(rng.integers(lo_L, hi_L + 1))
            if self.start_range is None:
                max_start = self.max_out - L
            else:
                lo_s, hi_s = self.start_range
                max_start = min(hi_s, self.max_out - L)
                if max_start < lo_s:
                    raise ValueError(
                        f"start_range={self.start_range} incompatible with "
                        f"L={L} and max_out={self.max_out}"
                    )
            lo_s = 0 if self.start_range is None else self.start_range[0]
            start = int(rng.integers(lo_s, max_start + 1))
            samples.append((start, L))
        self._samples = samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, L = self._samples[int(idx)]
        seq_length = self.sequence_length

        tokens = np.full(seq_length, self.pad_id, dtype=np.int64)
        targets = np.full(seq_length, -1, dtype=np.int64)
        pad_mask = np.zeros(seq_length, dtype=np.float32)
        loss_mask = np.zeros(seq_length, dtype=np.float32)

        # Input positions 0..L: start_int at 0, 'a' at 1..L.
        tokens[0] = self.w2i[str(start)]
        if L >= 1:
            tokens[1:L + 1] = self.a_id

        # Targets at positions 1..L: predict start+1, start+2, ..., start+L.
        # Position 0's target is -1 (ignore): no predecessor to count from.
        for t in range(1, L + 1):
            targets[t] = self.w2i[str(start + t)]

        # pad_mask covers valid positions (0..L inclusive).
        pad_mask[: L + 1] = 1.0
        # loss_mask restricts CE to running-count positions (1..L inclusive).
        loss_mask[1 : L + 1] = 1.0

        return (
            torch.from_numpy(tokens),
            torch.from_numpy(targets),
            torch.from_numpy(pad_mask),
            torch.from_numpy(loss_mask),
        )


def prepare_counting_dataset(args: Namespace) -> None:
    """No-op preparation entry point.

    The counting dataset is generated in-memory at ``get_counting_data``
    time; there is no corpus to download or tokenise. The function
    exists to satisfy the ``PREPARE_GET_DATASET_MAP[...][0]`` contract
    in ``data/utils.py``.
    """
    # Assert vocab_size is configured consistently with the task.
    expected_vocab = TE200_VOCAB_SIZE
    if getattr(args, "vocab_size", expected_vocab) != expected_vocab:
        # Harmless warning; main.py does not hard-require this.
        print(
            f"[counting.prepare] warning: args.vocab_size={args.vocab_size} "
            f"differs from the te200 counting vocab size {expected_vocab}; "
            f"the CountingDataset uses its own V=203 regardless."
        )
    return None


def get_counting_data(
    args: Namespace,
    ood_length: int | None = None,
) -> dict[str, "CountingDataset"]:
    """Return ``{"train": dataset, "val": dataset}`` CountingDataset map.

    The map is consumed by ``main.py`` via ``data.utils.get_dataset`` +
    ``data.utils.get_dataloader``; ``get_dataloader`` detects the
    already-wrapped ``torch.utils.data.Dataset`` and uses it directly
    (see the defensive branch added at ``data/utils.py`` alongside the
    dispatcher extension).

    Train / val seeds are derived from ``args.data_seed`` (or
    ``args.seed`` when ``data_seed`` is None) by adding 0 / 1000
    respectively, so the splits are disjoint.

    Parameters
    ----------
    args : Namespace
        Standard project args; ``sequence_length``, ``seed`` /
        ``data_seed``, ``counting_train_samples``,
        ``counting_val_samples`` are read from here.
    ood_length : int | None
        Optional OOD-length override for the val split. When set, the
        val split is rebuilt at a fixed sequence length ``L =
        ood_length`` drawn from ``TE200_OOD_LENGTH_RANGE`` (start range
        constrained to ``[0, max_out - L]`` so target tokens stay
        in-vocab). Triggered by ``eval.py --eval-length`` for the RQ9
        DV-1 OOD accuracy sweep. The val split's seed is bumped by
        ``+2000`` (the project's "ood" seed convention) so OOD eval
        sequences are disjoint from in-distribution train and val.
        The train split is unchanged regardless of ``ood_length``;
        this override is eval-only.
    """
    base_seed = int(args.data_seed) if getattr(args, "data_seed", None) is not None else int(args.seed)
    sequence_length = int(getattr(args, "sequence_length", 256))
    max_out = TE200_MAX_OUT

    # Pilot-appropriate sample counts: enough that a 100-iteration
    # smoke test at BS=32 covers multiple gradient passes without
    # re-sampling the same sequences, but small enough that the
    # synthetic set fits comfortably in host memory.
    train_n = int(getattr(args, "counting_train_samples", 10_000))
    val_n = int(getattr(args, "counting_val_samples", 1_000))

    train = CountingDataset(
        split="train",
        seed=base_seed + 0,
        num_samples=train_n,
        sequence_length=sequence_length,
        max_out=max_out,
        length_range=TE200_TRAIN_LENGTH_RANGE,
    )
    if ood_length is None:
        val = CountingDataset(
            split="val",
            seed=base_seed + 1000,
            num_samples=val_n,
            sequence_length=sequence_length,
            max_out=max_out,
            length_range=TE200_TRAIN_LENGTH_RANGE,
        )
    else:
        ood_lo, ood_hi = TE200_OOD_LENGTH_RANGE
        if not (ood_lo <= int(ood_length) <= ood_hi):
            raise ValueError(
                f"get_counting_data: ood_length={ood_length} outside the "
                f"OOD length range {TE200_OOD_LENGTH_RANGE}; the OOD eval "
                f"split is defined for L in that closed interval."
            )
        L = int(ood_length)
        if sequence_length < L + 1:
            raise ValueError(
                f"get_counting_data: sequence_length={sequence_length} too "
                f"small for ood_length={L}; require sequence_length >= L+1."
            )
        val = CountingDataset(
            split="ood",
            seed=base_seed + 2000,
            num_samples=val_n,
            sequence_length=sequence_length,
            max_out=max_out,
            length_range=(L, L),
        )
    return {"train": train, "val": val}
