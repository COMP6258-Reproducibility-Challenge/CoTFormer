"""Deterministic Condition-A / Condition-B synthetic sequence generators.

Scope
-----
Two generators produce the sequence substrate for Protocol
D-calibration. Both take an explicit ``seed`` argument; the seed
alone determines the output, so two calls with the same seed produce
bit-identical sequences. This determinism is required for the
pre-registration audit trail -- the calibration ladder's verdict is
only reproducible if the sequences are reproducible.

- **Condition A (Sparse / Single-Target)**: induction-head-style
  ``... q r ... q ?`` sequences with one informative target. Reply
  ``r`` placed at ``p_r = p_q + 1`` immediately after the first
  occurrence of query token ``q``; query token recurs at position
  253; positions 254-255 are filler. Ground-truth attention target
  ``T_A = {p_r}``.
- **Condition B (Broad-Integration / Multi-Target)**: 10 informative
  target positions drawn uniformly from [1, 251] without replacement,
  each carrying a token from a reserved subvocabulary ``V_inf`` of
  size 100. Query marker at position 252; positions 253-255 filler.
  Ground-truth attention target ``T_B = {p_1, ..., p_10}`` with equal
  weights.

Falsifiability relevance
------------------------
The two conditions are designed to produce equivalent attention
entropies for distinct information contents: a model attending
uniformly over 10 informative positions and a model attending
uniformly over 10 random positions will report the same Shannon
entropy but radically different attention-target accuracies. Without
Condition B, the Spearman gate cannot separate "low entropy =
accurate" from "low entropy = accurate AND broad integration
impossible".

Ontological purpose
-------------------
Substrate generation: these functions are not tests themselves, they
produce the material on which the calibration test runs.
"""

from __future__ import annotations

import torch


# Reserved token IDs valid for both GPT-2-large (vocab=50257) and
# ADM C5 (vocab=50257). The query marker ID and informative
# subvocabulary lower bound are chosen to lie inside the common range
# so the same seed produces compatible sequences across substrates.
_QUERY_TOKEN_ID = 50000          # rare reserved id; valid for both substrates
_INF_VOCAB_BASE = 49000          # informative subvocabulary starts here
_FILLER_TOKEN_ID = 0             # BOS / pad / filler


def generate_condition_A(
    n: int,
    L: int,
    vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``n`` Condition A sparse-induction sequences of length ``L``.

    Layout per sequence (L = 256 in the protocol):
        positions 0 .. L-4    : random filler from [1, vocab_size) with
                                forbidden ids reserved.
        position p_q in [10, L-10) : first occurrence of query token q.
        position p_r = p_q + 1 : reply token r (the induction target).
        position L-3          : recurrence of query token q (the
                                "induction probe" position).
        positions L-2, L-1    : filler.

    The generator uses a single ``torch.Generator`` seeded by ``seed``
    so two calls with the same seed produce bit-identical output.

    Returns
    -------
    tokens : torch.Tensor
        Integer tensor of shape ``(n, L)``.
    query_positions : torch.Tensor
        Integer tensor of shape ``(n,)`` with the recurring-query
        position (always ``L - 3`` under the fixed design).
    target_positions : torch.Tensor
        Integer tensor of shape ``(n,)`` with the ground-truth reply
        position ``p_r`` for each sequence.
    """
    if L < 30:
        raise ValueError(
            f"generate_condition_A: L must be >= 30 for the induction "
            f"layout (got {L})"
        )
    if vocab_size < _QUERY_TOKEN_ID + 1:
        raise ValueError(
            f"generate_condition_A: vocab_size {vocab_size} < query "
            f"token id {_QUERY_TOKEN_ID} + 1; substrate incompatible"
        )

    g = torch.Generator()
    g.manual_seed(int(seed))

    # Random filler: draw uniformly from [1, vocab_size) (avoid 0 as
    # it is the filler / BOS token in the layout below).
    tokens = torch.randint(
        low=1, high=vocab_size, size=(n, L), generator=g, dtype=torch.long
    )

    # Sample the first-occurrence query position p_q in [10, L-10) so
    # the (q, r) pair sits well inside the sequence.
    p_q_low, p_q_high = 10, L - 10
    p_q = torch.randint(
        low=p_q_low, high=p_q_high, size=(n,), generator=g, dtype=torch.long
    )

    # Place query token at p_q and reply r immediately after.
    # r is a random non-query token to avoid trivial double-counting.
    r_tokens = torch.randint(
        low=1, high=vocab_size, size=(n,), generator=g, dtype=torch.long
    )
    # Ensure r != _QUERY_TOKEN_ID; resample any collisions.
    collisions = (r_tokens == _QUERY_TOKEN_ID)
    while bool(collisions.any()):
        n_col = int(collisions.sum().item())
        r_tokens[collisions] = torch.randint(
            low=1, high=vocab_size, size=(n_col,), generator=g, dtype=torch.long
        )
        collisions = (r_tokens == _QUERY_TOKEN_ID)

    seq_idx = torch.arange(n, dtype=torch.long)
    tokens[seq_idx, p_q] = _QUERY_TOKEN_ID
    p_r = p_q + 1
    tokens[seq_idx, p_r] = r_tokens

    # Strip stray query-token occurrences elsewhere in the sequence so
    # the recurrence at L-3 is unique. Replace with a non-query token
    # (use _FILLER_TOKEN_ID + 1 to keep distinguishable from filler).
    stray_mask = (tokens == _QUERY_TOKEN_ID)
    # Carve out the planted positions so they survive.
    stray_mask[seq_idx, p_q] = False
    if int(stray_mask.sum().item()) > 0:
        tokens[stray_mask] = 1  # any non-query, non-filler id

    # Recurrence of query at position L-3 (the induction probe).
    probe_pos = L - 3
    tokens[:, probe_pos] = _QUERY_TOKEN_ID

    # Tail filler at L-2, L-1.
    tokens[:, L - 2] = _FILLER_TOKEN_ID
    tokens[:, L - 1] = _FILLER_TOKEN_ID

    query_positions = torch.full((n,), probe_pos, dtype=torch.long)
    target_positions = p_r.clone().to(dtype=torch.long)
    return tokens, query_positions, target_positions


def generate_condition_B(
    n: int,
    L: int,
    vocab_size: int,
    inf_vocab: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``n`` Condition B broad-integration sequences of length ``L``.

    Layout per sequence (L = 256 in the protocol):
        positions 0 .. L-5    : filler from [1, vocab_size) with the
                                informative subvocabulary excluded.
        positions p_1..p_10   : 10 distinct positions drawn uniformly
                                without replacement from [1, L-5)
                                each carrying a token sampled from
                                ``V_inf = [_INF_VOCAB_BASE,
                                _INF_VOCAB_BASE + inf_vocab)``.
        position L-4          : query / broad-integration marker.
        positions L-3, L-2, L-1 : filler.

    Returns
    -------
    tokens : torch.Tensor
        Integer tensor of shape ``(n, L)``.
    query_positions : torch.Tensor
        Integer tensor of shape ``(n,)`` with the broad-query marker
        position (always ``L - 4`` under the fixed design).
    target_positions : torch.Tensor
        Integer tensor of shape ``(n, 10)`` with the 10 ground-truth
        target positions per sequence.
    """
    n_targets = 10
    if L < n_targets + 10:
        raise ValueError(
            f"generate_condition_B: L must be >= {n_targets + 10} (got {L})"
        )
    if vocab_size < _INF_VOCAB_BASE + inf_vocab:
        raise ValueError(
            f"generate_condition_B: vocab_size {vocab_size} too small "
            f"for inf_vocab {inf_vocab} based at {_INF_VOCAB_BASE}"
        )

    g = torch.Generator()
    g.manual_seed(int(seed))

    # Background filler: sample from [1, _INF_VOCAB_BASE) to keep the
    # informative subvocabulary exclusively at the planted positions.
    tokens = torch.randint(
        low=1,
        high=_INF_VOCAB_BASE,
        size=(n, L),
        generator=g,
        dtype=torch.long,
    )

    # Sample 10 distinct target positions per sequence from
    # [1, L-5) without replacement. We sample a dense permutation
    # over the candidate range and slice the first 10 to ensure
    # uniqueness without rejection sampling.
    candidate_high = L - 5  # exclusive upper bound; query at L-4
    target_positions = torch.empty((n, n_targets), dtype=torch.long)
    inf_lo, inf_hi = _INF_VOCAB_BASE, _INF_VOCAB_BASE + inf_vocab
    for i in range(n):
        # Permutation of [1, candidate_high) -> first n_targets entries.
        perm = torch.randperm(candidate_high - 1, generator=g) + 1
        positions_i = perm[:n_targets].sort().values
        target_positions[i] = positions_i
        # Plant informative tokens at those positions.
        inf_tokens = torch.randint(
            low=inf_lo,
            high=inf_hi,
            size=(n_targets,),
            generator=g,
            dtype=torch.long,
        )
        tokens[i, positions_i] = inf_tokens

    # Query marker at L-4.
    probe_pos = L - 4
    tokens[:, probe_pos] = _QUERY_TOKEN_ID
    # Tail filler.
    tokens[:, L - 3] = _FILLER_TOKEN_ID
    tokens[:, L - 2] = _FILLER_TOKEN_ID
    tokens[:, L - 1] = _FILLER_TOKEN_ID

    query_positions = torch.full((n,), probe_pos, dtype=torch.long)
    return tokens, query_positions, target_positions
