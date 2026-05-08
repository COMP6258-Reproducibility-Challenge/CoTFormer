import json
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_SHIFTED_START_TASK = "counting_samesymbol_shiftedstart3__tr25_te200__"
IGNORE_INDEX = -1


@dataclass(frozen=True)
class CountingTaskSpec:
    name: str
    train_seq_len: int
    test_seq_len: int
    vocab: List[str]
    pad_token: str = "<pad>"
    symbol_token: str = "a"
    minimum_sequence_length: int = 256

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab.index(self.pad_token)

    @property
    def max_seen_len(self) -> int:
        return self.train_seq_len + 1


def get_shifted_start_task_spec(task: str = DEFAULT_SHIFTED_START_TASK) -> CountingTaskSpec:
    match = re.fullmatch(r"counting_samesymbol_shiftedstart3__tr(\d+)_te(\d+)__", task)
    if match is None:
        raise ValueError(
            "Only plain samesymbol shiftedstart3 tasks are supported here; "
            f"got {task!r}."
        )

    train_seq_len = int(match.group(1))
    test_seq_len = int(match.group(2))
    return CountingTaskSpec(
        name=task,
        train_seq_len=train_seq_len,
        test_seq_len=test_seq_len,
        vocab=[str(i) for i in range(test_seq_len + 1)] + ["<pad>", "a"],
        minimum_sequence_length=max(256, test_seq_len + 1),
    )


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "rasp_primitives"


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


def make_collate_fn(spec: CountingTaskSpec, sequence_length: int):
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


def load_counting_split(data_root: Path, task: str, split: str) -> JsonlOffsetDataset:
    return JsonlOffsetDataset(Path(data_root) / task / f"{split}.txt")


def make_counting_dataloader(
    data_root: Path,
    task: str,
    split: str,
    spec: CountingTaskSpec,
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
        load_counting_split(data_root, task, split),
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
        raise TypeError("CoTFormer shifted-start runners expect model outputs to be a dict.")
    loss = outputs.get("cross_entropy_loss", outputs.get("loss"))
    logits = outputs.get("logits")
    if loss is None:
        raise KeyError("Model output did not include 'cross_entropy_loss' or 'loss'.")
    if logits is None:
        raise KeyError("Model output did not include 'logits'. Use get_logits=True.")
    return loss, logits


def update_counting_counters(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_seen_len: int,
    counters: Dict[str, float],
) -> None:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX

    counters["correct"] += (preds[mask] == labels[mask]).float().sum().item()
    counters["demo"] += mask.float().sum().item()

    for pred, label in zip(preds.cpu(), labels.cpu()):
        active = label != IGNORE_INDEX
        active_pred = pred[active]
        active_label = label[active]
        if active_label.numel() == 0:
            continue

        counting_pred = active_pred[:-1]
        counting_label = active_label[:-1]
        last_pred = active_pred[-1]
        last_label = active_label[-1]

        counters["counting_correct"] += (counting_pred == counting_label).float().sum().item()
        counters["counting_demo"] += counting_label.numel()
        counters["last_correct"] += (last_pred == last_label).float().sum().item()
        counters["last_demo"] += 1

        unseen_pred = active_pred[max_seen_len:]
        unseen_label = active_label[max_seen_len:]
        counters["unseen_len_correct"] += (unseen_pred == unseen_label).float().sum().item()
        counters["unseen_len_demo"] += unseen_label.numel()


def _safe_ratio(num: float, denom: float, empty_value: float = math.nan) -> float:
    if denom == 0:
        return empty_value
    return num / denom


@torch.no_grad()
def evaluate_counting_model(
    model,
    dataloader: DataLoader,
    device,
    max_seen_len: int,
    max_batches: Optional[int] = None,
    ctx=None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    ctx = ctx or nullcontext()

    counters = {
        "correct": 0.0,
        "demo": 0.0,
        "counting_correct": 0.0,
        "counting_demo": 0.0,
        "last_correct": 0.0,
        "last_demo": 0.0,
        "unseen_len_correct": 0.0,
        "unseen_len_demo": 0.0,
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
        update_counting_counters(logits.detach().cpu(), labels.detach().cpu(), max_seen_len, counters)
        if outputs.get("average_depth") is not None:
            avg_depths.append(float(torch.as_tensor(outputs["average_depth"]).detach().cpu().float().item()))

    if was_training:
        model.train()

    metrics = {
        "loss": float(sum(losses) / len(losses)) if losses else math.nan,
        "acc": _safe_ratio(counters["correct"], counters["demo"]),
        "counting_acc": _safe_ratio(counters["counting_correct"], counters["counting_demo"]),
        "last_acc": _safe_ratio(counters["last_correct"], counters["last_demo"]),
        "unseen_len_acc": _safe_ratio(
            counters["unseen_len_correct"],
            counters["unseen_len_demo"],
            empty_value=-1.0,
        ),
        "num_examples": float(len(dataloader.dataset)),
        "num_batches": float(len(losses)),
    }
    if avg_depths:
        metrics["average_depth"] = float(sum(avg_depths) / len(avg_depths))
    return metrics
