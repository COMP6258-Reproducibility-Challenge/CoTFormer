import argparse
import json
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from generate import IGNORE_INDEX, InductionHopsFinalAnswerTask
except ImportError:
    from .generate import IGNORE_INDEX, InductionHopsFinalAnswerTask


DEFAULT_PHOP_TASK = "phop_p8_seq256_a4_final"


@dataclass(frozen=True)
class PHopTaskSpec:
    name: str
    seq_len: int
    min_hops: int
    max_hops: int
    char_tokens: int
    vocab: List[str]
    include_hop_token: bool = False
    ensure_exists: bool = True
    avoid_adjacent_repeats: bool = True
    sampling_strategy: str = "rejection"
    pad_token: str = "<pad>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab.index(self.pad_token)

    @property
    def input_len(self) -> int:
        return self.seq_len + (1 if self.include_hop_token else 0)

    @property
    def minimum_sequence_length(self) -> int:
        return self.input_len


def make_phop_task_name(
    hops: int,
    seq_len: int = 256,
    char_tokens: int = 4,
    include_hop_token: bool = False,
    sampling_strategy: str = "rejection",
) -> str:
    suffixes = []
    if include_hop_token:
        suffixes.append("hoptok")
    if sampling_strategy == "constructive":
        suffixes.append("constructive")
    suffix = "".join(f"_{suffix}" for suffix in suffixes)
    return f"phop_p{hops}_seq{seq_len}_a{char_tokens}_final{suffix}"


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "p-hop"


def _char_vocab(char_tokens: int) -> List[str]:
    return [chr(97 + idx) for idx in range(char_tokens)]


def get_phop_task_spec(task: str = DEFAULT_PHOP_TASK) -> PHopTaskSpec:
    match = re.fullmatch(r"phop_p(\d+)_seq(\d+)_a(\d+)_final((?:_[a-z0-9]+)*)", task)
    if match is None:
        raise ValueError(
            "Expected p-hop task name like 'phop_p8_seq256_a4_final' or "
            "'phop_p8_seq256_a4_final_constructive'; "
            f"got {task!r}."
        )

    hops = int(match.group(1))
    seq_len = int(match.group(2))
    char_tokens = int(match.group(3))
    suffixes = set(filter(None, match.group(4).split("_")))
    allowed_suffixes = {"hoptok", "constructive", "allowadj", "noadj"}
    unknown_suffixes = suffixes - allowed_suffixes
    if unknown_suffixes:
        raise ValueError(
            f"Unknown p-hop task suffixes {sorted(unknown_suffixes)} in {task!r}."
        )
    include_hop_token = "hoptok" in suffixes
    sampling_strategy = "constructive" if "constructive" in suffixes else "rejection"
    avoid_adjacent_repeats = True
    if sampling_strategy == "constructive":
        avoid_adjacent_repeats = False
    if "allowadj" in suffixes:
        avoid_adjacent_repeats = False
    if "noadj" in suffixes:
        avoid_adjacent_repeats = True

    vocab = _char_vocab(char_tokens)
    if include_hop_token:
        vocab += [f"<H{hops}>"]
    vocab += ["<pad>"]

    return PHopTaskSpec(
        name=task,
        seq_len=seq_len,
        min_hops=hops,
        max_hops=hops,
        char_tokens=char_tokens,
        include_hop_token=include_hop_token,
        avoid_adjacent_repeats=avoid_adjacent_repeats,
        sampling_strategy=sampling_strategy,
        vocab=vocab,
    )


def make_generator(spec: PHopTaskSpec, seed: int) -> InductionHopsFinalAnswerTask:
    return InductionHopsFinalAnswerTask(
        seq_len=spec.seq_len,
        char_tokens=spec.char_tokens,
        min_hops=spec.min_hops,
        max_hops=spec.max_hops,
        rng=np.random.RandomState(seed),
        ensure_exists=spec.ensure_exists,
        include_hop_token=spec.include_hop_token,
        avoid_adjacent_repeats=spec.avoid_adjacent_repeats,
        sampling_strategy=spec.sampling_strategy,
    )


def write_phop_split(
    output_path: Path,
    spec: PHopTaskSpec,
    num_examples: int,
    seed: int,
    force: bool = False,
    progress_every: int = 10000,
) -> int:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Pass --force to overwrite it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = make_generator(spec, seed)
    with output_path.open("w", encoding="utf-8") as handle:
        for example_idx in range(num_examples):
            input_tokens, label_tokens = generator.get_tokens()
            handle.write(json.dumps([input_tokens, label_tokens]) + "\n")
            if progress_every > 0 and (example_idx + 1) % progress_every == 0:
                print(
                    f"{output_path.name}: wrote {example_idx + 1}/{num_examples} examples",
                    flush=True,
                )
    return num_examples


class JsonlOffsetDataset(Dataset):
    """A JSONL dataset that keeps only line offsets in memory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing split file: {self.path}")
        self.offsets = self._build_offsets()
        self._handle = None

    def _build_offsets(self) -> List[int]:
        offsets = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        return offsets

    def _get_handle(self):
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        handle = self._get_handle()
        handle.seek(self.offsets[idx])
        return {"text": handle.readline()}


def make_collate_fn(spec: PHopTaskSpec, sequence_length: int):
    w2i = {token: idx for idx, token in enumerate(spec.vocab)}

    def collate(rows: Iterable[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        attention_masks = []

        for row in rows:
            input_tokens, label_tokens = json.loads(row["text"])
            if len(input_tokens) > sequence_length:
                raise ValueError(
                    f"Input length {len(input_tokens)} exceeds sequence_length={sequence_length}"
                )
            if len(label_tokens) > sequence_length:
                raise ValueError(
                    f"Label length {len(label_tokens)} exceeds sequence_length={sequence_length}"
                )

            input_id = [w2i[token] for token in input_tokens]
            label = [IGNORE_INDEX if token == "-1" else w2i[token] for token in label_tokens]
            attention_mask = [0 if token == spec.pad_token else 1 for token in input_tokens]

            pad_len = sequence_length - len(input_id)
            input_id += [spec.pad_id] * pad_len
            label += [IGNORE_INDEX] * (sequence_length - len(label))
            attention_mask += [0] * pad_len

            input_ids.append(input_id)
            labels.append(label)
            attention_masks.append(attention_mask)

        return {
            "input_id": torch.LongTensor(input_ids),
            "label": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_masks),
        }

    return collate


def load_phop_split(data_root: Path, task: str, split: str) -> JsonlOffsetDataset:
    return JsonlOffsetDataset(Path(data_root) / task / f"{split}.txt")


def make_phop_dataloader(
    data_root: Path,
    task: str,
    split: str,
    spec: PHopTaskSpec,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    sampler=None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        load_phop_split(data_root, task, split),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=make_collate_fn(spec, sequence_length),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def unpack_model_outputs(outputs):
    if not isinstance(outputs, dict):
        raise TypeError("p-hop runners expect model outputs to be a dict.")
    loss = outputs.get("cross_entropy_loss", outputs.get("loss"))
    logits = outputs.get("logits")
    if loss is None:
        raise KeyError("Model output did not include 'cross_entropy_loss' or 'loss'.")
    if logits is None:
        raise KeyError("Model output did not include 'logits'. Use get_logits=True.")
    return loss, logits


def update_phop_counters(logits: torch.Tensor, labels: torch.Tensor, counters: Dict[str, float]) -> None:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    counters["correct"] += (preds[mask] == labels[mask]).float().sum().item()
    counters["demo"] += mask.float().sum().item()
    counters["examples"] += labels.shape[0]


def _safe_ratio(num: float, denom: float, empty_value: float = math.nan) -> float:
    if denom == 0:
        return empty_value
    return num / denom


@torch.no_grad()
def evaluate_phop_model(
    model,
    dataloader: DataLoader,
    device,
    max_batches: Optional[int] = None,
    ctx=None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    ctx = ctx or nullcontext()

    counters = {
        "correct": 0.0,
        "demo": 0.0,
        "examples": 0.0,
    }
    losses = []
    avg_depths = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = batch["input_id"].to(device)
        labels = batch["label"].to(device)
        with ctx:
            outputs = model(inputs, targets=labels, get_logits=True)
        loss, logits = unpack_model_outputs(outputs)
        losses.append(float(loss.detach().item()))
        update_phop_counters(logits.detach().cpu(), labels.detach().cpu(), counters)
        if outputs.get("average_depth") is not None:
            avg_depths.append(float(torch.as_tensor(outputs["average_depth"]).detach().cpu().float().item()))

    if was_training:
        model.train()

    metrics = {
        "loss": float(sum(losses) / len(losses)) if losses else math.nan,
        "acc": _safe_ratio(counters["correct"], counters["demo"]),
        "final_acc": _safe_ratio(counters["correct"], counters["demo"]),
        "num_examples": counters["examples"],
        "num_batches": float(len(losses)),
    }
    if avg_depths:
        metrics["average_depth"] = float(sum(avg_depths) / len(avg_depths))
    return metrics


def parse_split_sizes(values: List[str]) -> Dict[str, int]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Split size must look like split=num_examples, got {value!r}.")
        split, count = value.split("=", 1)
        result[split] = int(count)
    return result


def parse_auto_bool(value: str, *, name: str) -> Optional[bool]:
    normalized = value.lower()
    if normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be auto/true/false, got {value!r}.")


def main():
    parser = argparse.ArgumentParser(description="Generate fixed p-hop JSONL splits.")
    parser.add_argument("--task", default=DEFAULT_PHOP_TASK)
    parser.add_argument("--data_root", default=str(default_data_root()))
    parser.add_argument(
        "--split_sizes",
        nargs="+",
        default=["train=200000", "val=10000", "test=10000"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10000)
    parser.add_argument(
        "--sampling_strategy",
        choices=["auto", "rejection", "constructive"],
        default="auto",
        help="Override task-name sampler. 'auto' uses the task suffix/default.",
    )
    parser.add_argument(
        "--avoid_adjacent_repeats",
        default="auto",
        help="auto/true/false. Constructive paper-style data usually uses false.",
    )
    args = parser.parse_args()

    spec = get_phop_task_spec(args.task)
    avoid_adjacent_repeats = parse_auto_bool(
        args.avoid_adjacent_repeats,
        name="avoid_adjacent_repeats",
    )
    spec = replace(
        spec,
        sampling_strategy=(
            spec.sampling_strategy
            if args.sampling_strategy == "auto"
            else args.sampling_strategy
        ),
        avoid_adjacent_repeats=(
            spec.avoid_adjacent_repeats
            if avoid_adjacent_repeats is None
            else avoid_adjacent_repeats
        ),
    )
    data_root = Path(args.data_root)
    split_sizes = parse_split_sizes(args.split_sizes)
    print(
        "p-hop spec: "
        f"task={spec.name} seq_len={spec.seq_len} "
        f"hops={spec.min_hops}-{spec.max_hops} "
        f"alphabet={spec.char_tokens} "
        f"sampling={spec.sampling_strategy} "
        f"avoid_adjacent_repeats={spec.avoid_adjacent_repeats}",
        flush=True,
    )

    for split_idx, (split, num_examples) in enumerate(split_sizes.items()):
        split_seed = args.seed + split_idx
        output_path = data_root / args.task / f"{split}.txt"
        count = write_phop_split(
            output_path=output_path,
            spec=spec,
            num_examples=num_examples,
            seed=split_seed,
            force=args.force,
            progress_every=args.progress_every,
        )
        print(f"{split}: wrote {count} examples to {output_path}")


if __name__ == "__main__":
    main()
