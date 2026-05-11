try:
    from bidict import bidict
except ImportError:
    class bidict(dict):
        """Minimal fallback for this generator when bidict is unavailable."""

        @property
        def inv(self):
            return {value: key for key, value in self.items()}

import numpy as np
import torch
from typing import Tuple


IGNORE_INDEX = -1


class InductionHopsFullSequenceTask:
    """Dense p-hop transduction task.

    This is useful for debugging because every sequence position receives a
    target. It is not the paper-style p-hop benchmark from the looped
    transformer paper, where the model is evaluated on one final answer.
    """

    def __init__(self, seq_len = 50, char_tokens = 5, min_hops = 0, max_hops = 3, rng: np.random.RandomState = None):
        self.seq_len = seq_len
        self.char_tokens = char_tokens
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.induction_tokens = max_hops - min_hops + 1
        self.num_tokens = char_tokens + self.induction_tokens + 2
        self.rng = rng

        self.BLANK_CHAR = '_'
        self.DOES_NOT_EXIST_CHAR = '~'

        MIN_CHAR_INT = 97
        MIN_NUM_INT = 48
        self.induction_token_map = bidict({chr(i + MIN_NUM_INT): i - min_hops for i in range(min_hops, max_hops + 1)})
        self.char_token_map = {chr(i): i - MIN_CHAR_INT + self.induction_tokens for i in range(MIN_CHAR_INT, MIN_CHAR_INT + char_tokens)}
        self.token_map = bidict({**self.induction_token_map, **self.char_token_map, **{self.BLANK_CHAR: self.num_tokens - 1, self.DOES_NOT_EXIST_CHAR: self.num_tokens - 2}})
        
    def get_batch(self, batch_size: int, metadata=False, hops=None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensors = torch.zeros((batch_size, self.seq_len-1))
        output_tensors = torch.zeros((batch_size, self.seq_len-1))
        if metadata:
            input_strings = []
            output_strings = []
            induction_hop_strings_list = []
            induction_hop_indices_list = []
        for i in range(batch_size):
            if metadata:
                input_string, output_string, induction_hop_strings, induction_hop_indices = self.get_strings(metadata=True, hops=hops)
                input_strings.append(input_string)
                output_strings.append(output_string)
                induction_hop_strings_list.append(induction_hop_strings)
                induction_hop_indices_list.append(induction_hop_indices)
            else:
                input_string, output_string = self.get_strings(hops=hops)
            input_tensors[i] = torch.tensor([self.token_map[token] for token in input_string])
            output_tensors[i] = torch.tensor([self.token_map[token] for token in output_string])
        if metadata:
            return input_tensors.long(), output_tensors.long(), input_strings, output_strings, induction_hop_strings_list, induction_hop_indices_list
        else:
            return input_tensors.long(), output_tensors.long()

    def get_strings(self, metadata=False, hops=None):
        random_char_string = ''
        for i in range(self.seq_len):
            if i == 0:
                random_char_string += self.rng.choice(list(self.char_token_map.keys()))
            else:
                random_char_string += self.rng.choice(list(self.char_token_map.keys() - {random_char_string[i-1]}))

        induction_hop_strings = [random_char_string]
        induction_hop_indices = [range(self.seq_len)]
        for i in range(self.max_hops+1):
            last_string = induction_hop_strings[-1]
            last_indices = induction_hop_indices[-1]
            new_string = ''
            new_indices = []
            for i in range(self.seq_len):
                if last_indices[i] == -1 or (last_index := random_char_string[:last_indices[i]].rfind(last_string[i])) == -1:
                    new_string += self.DOES_NOT_EXIST_CHAR
                    new_indices.append(-1)
                else:
                    new_string += random_char_string[last_index + 1]
                    new_indices.append(last_index + 1)

            induction_hop_strings.append(new_string)
            induction_hop_indices.append(new_indices)

        if hops is None:
            num_hops = self.rng.randint(self.min_hops, self.max_hops+1)
        else:
            num_hops = hops

        input_string = self.induction_token_map.inv[num_hops - self.min_hops] + random_char_string[:-2]
        output_string = self.BLANK_CHAR + induction_hop_strings[num_hops][:-2]


        if metadata:
            return input_string, output_string, induction_hop_strings, induction_hop_indices
        else:
            return input_string, output_string


class InductionHopsFinalAnswerTask:
    """Paper-style p-hop induction task with one supervised answer.

    The looped-transformer paper defines p-hop as a map from a whole sequence
    to a single symbol. We represent that in our sequence-label training setup
    by masking every label with IGNORE_INDEX except the final query position.

    The hop operation is:
      1. Start from the final/query token.
      2. Find the previous occurrence of that token.
      3. Move to the token immediately after that previous occurrence.
      4. Repeat for p hops.

    By default examples are resampled until the p-hop chain exists, matching
    the benchmark assumption that the answer is one of the alphabet symbols.
    """

    def __init__(
        self,
        seq_len: int = 256,
        char_tokens: int = 4,
        min_hops: int = 16,
        max_hops: int = 16,
        rng: np.random.RandomState = None,
        ensure_exists: bool = True,
        max_resample_attempts: int = 10000,
        include_hop_token: bool = False,
        avoid_adjacent_repeats: bool = True,
    ):
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if char_tokens < 2:
            raise ValueError("char_tokens must be at least 2.")
        if min_hops < 0 or max_hops < min_hops:
            raise ValueError("Require 0 <= min_hops <= max_hops.")
        if min_hops != max_hops and not include_hop_token:
            raise ValueError(
                "Mixed-hop training needs include_hop_token=True so the model "
                "knows which hop count to answer."
            )

        self.seq_len = seq_len
        self.char_tokens = char_tokens
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.rng = rng if rng is not None else np.random.RandomState()
        self.ensure_exists = ensure_exists
        self.max_resample_attempts = max_resample_attempts
        self.include_hop_token = include_hop_token
        self.avoid_adjacent_repeats = avoid_adjacent_repeats

        self.DOES_NOT_EXIST_CHAR = "~"
        self.IGNORE_LABEL = "-1"

        min_char_int = 97
        self.char_token_map = bidict(
            {
                chr(min_char_int + idx): idx
                for idx in range(char_tokens)
            }
        )

        next_id = char_tokens
        self.hop_token_map = bidict()
        if include_hop_token:
            for hop in range(min_hops, max_hops + 1):
                self.hop_token_map[f"<H{hop}>"] = next_id
                next_id += 1

        self.token_map = bidict({**self.char_token_map, **self.hop_token_map})
        if not ensure_exists:
            self.token_map[self.DOES_NOT_EXIST_CHAR] = next_id
        self.num_tokens = len(self.token_map)
        self.input_len = seq_len + (1 if include_hop_token else 0)

    def _sample_char_sequence(self):
        chars = list(self.char_token_map.keys())
        seq = []
        for idx in range(self.seq_len):
            choices = chars
            if self.avoid_adjacent_repeats and idx > 0:
                choices = [char for char in chars if char != seq[-1]]
            seq.append(self.rng.choice(choices))
        return seq

    @staticmethod
    def _previous_index(seq, query, before_idx):
        for idx in range(before_idx - 1, -1, -1):
            if seq[idx] == query:
                return idx
        return -1

    def _compute_final_answer(self, seq, hops):
        current_idx = len(seq) - 1
        query = seq[current_idx]
        path = [(current_idx, query)]

        for _ in range(hops):
            previous_idx = self._previous_index(seq, query, current_idx)
            if previous_idx == -1:
                return None, path
            current_idx = previous_idx + 1
            query = seq[current_idx]
            path.append((current_idx, query))

        return query, path

    def _sample_hops(self, hops=None):
        if hops is not None:
            if not (self.min_hops <= hops <= self.max_hops):
                raise ValueError(
                    f"hops={hops} outside [{self.min_hops}, {self.max_hops}]"
                )
            return hops
        return self.rng.randint(self.min_hops, self.max_hops + 1)

    def get_tokens(self, metadata=False, hops=None):
        num_hops = self._sample_hops(hops)
        for _ in range(self.max_resample_attempts):
            seq = self._sample_char_sequence()
            answer, path = self._compute_final_answer(seq, num_hops)
            if answer is not None or not self.ensure_exists:
                break
        else:
            raise RuntimeError(
                f"Could not sample a valid {num_hops}-hop example after "
                f"{self.max_resample_attempts} attempts."
            )

        if answer is None:
            answer = self.DOES_NOT_EXIST_CHAR

        input_tokens = list(seq)
        if self.include_hop_token:
            input_tokens = [self.hop_token_map.inv[num_hops]] + input_tokens

        # All positions are ignored except the final query position. This keeps
        # the task as single-answer classification while reusing LM-style logits.
        label_tokens = [self.IGNORE_LABEL] * len(input_tokens)
        label_tokens[-1] = answer

        if metadata:
            return input_tokens, label_tokens, {
                "hops": num_hops,
                "answer": answer,
                "path": path,
                "raw_sequence": seq,
            }
        return input_tokens, label_tokens

    def get_strings(self, metadata=False, hops=None):
        result = self.get_tokens(metadata=metadata, hops=hops)
        if metadata:
            input_tokens, label_tokens, meta = result
        else:
            input_tokens, label_tokens = result

        input_string = " ".join(input_tokens) if self.include_hop_token else "".join(input_tokens)
        label_string = " ".join(label_tokens)
        if metadata:
            return input_string, label_string, meta
        return input_string, label_string

    def get_batch(self, batch_size: int, metadata=False, hops=None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensors = torch.empty((batch_size, self.input_len), dtype=torch.long)
        output_tensors = torch.full((batch_size, self.input_len), IGNORE_INDEX, dtype=torch.long)

        if metadata:
            input_strings = []
            output_strings = []
            metadata_list = []

        for row_idx in range(batch_size):
            if metadata:
                input_tokens, label_tokens, meta = self.get_tokens(metadata=True, hops=hops)
                input_strings.append(
                    " ".join(input_tokens) if self.include_hop_token else "".join(input_tokens)
                )
                output_strings.append(" ".join(label_tokens))
                metadata_list.append(meta)
            else:
                input_tokens, label_tokens = self.get_tokens(hops=hops)

            input_tensors[row_idx] = torch.tensor(
                [self.token_map[token] for token in input_tokens],
                dtype=torch.long,
            )
            for col_idx, token in enumerate(label_tokens):
                if token != self.IGNORE_LABEL:
                    output_tensors[row_idx, col_idx] = self.token_map[token]

        if metadata:
            return input_tensors, output_tensors, input_strings, output_strings, metadata_list
        return input_tensors, output_tensors


def get_task(task_name, rng: np.random.RandomState = None, **task_kwargs) -> InductionHopsFullSequenceTask:
    if task_name == "InductionHopsFullSequenceTask":
        return InductionHopsFullSequenceTask(rng=rng, **task_kwargs)
    if task_name == "InductionHopsFinalAnswerTask":
        return InductionHopsFinalAnswerTask(rng=rng, **task_kwargs)
    raise ValueError(f"Unknown task: {task_name}")
