import argparse
import json
from pathlib import Path


DEFAULT_TASK = "counting_samesymbol_shiftedstart3__tr25_te200__"


def parse_task(task):
    prefix = "counting_samesymbol_shiftedstart3__tr"
    middle = "_te"
    suffix = "__"
    if not task.startswith(prefix) or not task.endswith(suffix) or middle not in task:
        raise ValueError(f"Unsupported task name: {task}")
    body = task[len(prefix):-len(suffix)]
    train_seq_len, test_seq_len = body.split(middle)
    return int(train_seq_len), int(test_seq_len)


def default_data_root():
    return Path(__file__).resolve().parents[1] / "data" / "rasp_primitives"


def make_example(addon, count):
    input_tokens = [str(addon)] + ["a" for _ in range(count)]
    label_tokens = ["-1"] + [str(value) for value in range(addon + 1, addon + count + 1)]
    return [input_tokens, label_tokens]


def write_examples(output_path, examples, force):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Pass --force to overwrite it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")
            count += 1
    return count


def iter_train_examples(train_seq_len, test_seq_len, num_train):
    addon_range = range(test_seq_len - train_seq_len + 1)
    cycles = num_train // len(addon_range)
    for _ in range(cycles):
        for addon in addon_range:
            yield make_example(addon, train_seq_len)


def iter_val_examples(train_seq_len, test_seq_len):
    for addon in range(test_seq_len - train_seq_len + 1):
        yield make_example(addon, train_seq_len)


def iter_ood_examples(train_seq_len, test_seq_len):
    count_range = range(train_seq_len + 1)
    for _ in range(train_seq_len):
        for addon in range(test_seq_len - train_seq_len + 1):
            count = test_seq_len - addon
            if count in count_range:
                continue
            yield make_example(addon, count)


def write_split_file(output_path, split, train_seq_len, test_seq_len, num_train, force):
    if split == "train":
        examples = iter_train_examples(train_seq_len, test_seq_len, num_train)
    elif split == "val":
        examples = iter_val_examples(train_seq_len, test_seq_len)
    elif split == "ood_test":
        examples = iter_ood_examples(train_seq_len, test_seq_len)
    else:
        raise ValueError(f"Unknown split: {split}")
    return write_examples(output_path, examples, force)


def main():
    parser = argparse.ArgumentParser(
        description="Generate shiftedstart3 JSONL splits deterministically."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--data_root", default=str(default_data_root()))
    parser.add_argument("--output", default=None)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "ood_test", "all"])
    parser.add_argument("--num_train", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_seq_len, test_seq_len = parse_task(args.task)
    splits = ["train", "val", "ood_test"] if "all" in args.splits else args.splits
    if args.output is not None and len(splits) != 1:
        raise ValueError("--output can only be used when generating exactly one split.")

    outputs = {}
    for split in splits:
        output = args.output
        if output is None:
            output = Path(args.data_root) / args.task / f"{split}.txt"
        outputs[split] = {
            "output": str(output),
            "written_examples": write_split_file(
                output_path=output,
                split=split,
                train_seq_len=train_seq_len,
                test_seq_len=test_seq_len,
                num_train=args.num_train,
                force=args.force,
            ),
        }

    print(
        json.dumps(
            {
                "task": args.task,
                "requested_examples": args.num_train,
                "train_seq_len": train_seq_len,
                "test_seq_len": test_seq_len,
                "splits": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
