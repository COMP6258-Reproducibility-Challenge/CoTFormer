import argparse
import json
import random
from pathlib import Path


DEFAULT_TASK = "counting_samesymbol_shiftedstart3__tr25_te200__"
AUTHOR_SPLITS = ["train", "val", "ood_test"]
EXTENDED_SPLITS = AUTHOR_SPLITS + [
    "train_var_len",
    "val_var_len",
    "ood_test_grid",
    "ood_test_grid_ltmax",
    "ood_test_max_len",
    "ood_test_sampled",
]


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


def iter_var_len_examples(train_seq_len, test_seq_len):
    addon_range = range(test_seq_len - train_seq_len + 1)
    for count in range(1, train_seq_len + 1):
        for addon in addon_range:
            yield make_example(addon, count)


def iter_ood_grid_examples(train_seq_len, test_seq_len):
    for count in range(train_seq_len + 1, test_seq_len + 1):
        for addon in range(test_seq_len - count + 1):
            yield make_example(addon, count)


def iter_ood_grid_ltmax_examples(train_seq_len, test_seq_len):
    for count in range(train_seq_len + 1, test_seq_len):
        for addon in range(test_seq_len - count + 1):
            yield make_example(addon, count)


def iter_ood_max_len_examples(train_seq_len, test_seq_len):
    del train_seq_len
    yield make_example(0, test_seq_len)


def iter_sampled_ood_grid_examples(train_seq_len, test_seq_len, sample_size, seed):
    if sample_size < 0:
        raise ValueError("--ood_sample_size must be non-negative.")

    rng = random.Random(seed)
    pairs_by_count = {
        count: [(addon, count) for addon in range(test_seq_len - count + 1)]
        for count in range(train_seq_len + 1, test_seq_len + 1)
    }
    counts = list(pairs_by_count)
    full_grid_size = sum(len(pairs) for pairs in pairs_by_count.values())
    if sample_size >= full_grid_size:
        selected_pairs = [
            pair
            for count in counts
            for pair in pairs_by_count[count]
        ]
    else:
        remaining_by_count = {}
        for count, pairs in pairs_by_count.items():
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            remaining_by_count[count] = shuffled

        selected_pairs = []
        while len(selected_pairs) < sample_size:
            made_progress = False
            for count in counts:
                if len(selected_pairs) >= sample_size:
                    break
                if remaining_by_count[count]:
                    selected_pairs.append(remaining_by_count[count].pop())
                    made_progress = True
            if not made_progress:
                break
        selected_pairs.sort(key=lambda pair: (pair[1], pair[0]))

    for addon, count in selected_pairs:
        yield make_example(addon, count)


def write_split_file(
    output_path,
    split,
    train_seq_len,
    test_seq_len,
    num_train,
    ood_sample_size,
    seed,
    force,
):
    if split == "train":
        examples = iter_train_examples(train_seq_len, test_seq_len, num_train)
    elif split == "val":
        examples = iter_val_examples(train_seq_len, test_seq_len)
    elif split == "ood_test":
        examples = iter_ood_examples(train_seq_len, test_seq_len)
    elif split == "train_var_len":
        examples = iter_var_len_examples(train_seq_len, test_seq_len)
    elif split == "val_var_len":
        examples = iter_var_len_examples(train_seq_len, test_seq_len)
    elif split == "ood_test_grid":
        examples = iter_ood_grid_examples(train_seq_len, test_seq_len)
    elif split == "ood_test_grid_ltmax":
        examples = iter_ood_grid_ltmax_examples(train_seq_len, test_seq_len)
    elif split == "ood_test_max_len":
        examples = iter_ood_max_len_examples(train_seq_len, test_seq_len)
    elif split == "ood_test_sampled":
        examples = iter_sampled_ood_grid_examples(
            train_seq_len,
            test_seq_len,
            sample_size=ood_sample_size,
            seed=seed,
        )
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
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=EXTENDED_SPLITS + ["all", "all_extended"],
    )
    parser.add_argument("--num_train", type=int, default=1_000_000)
    parser.add_argument("--ood_sample_size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_seq_len, test_seq_len = parse_task(args.task)
    if "all_extended" in args.splits:
        splits = EXTENDED_SPLITS
    elif "all" in args.splits:
        splits = AUTHOR_SPLITS
    else:
        splits = args.splits
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
                ood_sample_size=args.ood_sample_size,
                seed=args.seed,
                force=args.force,
            ),
        }

    print(
        json.dumps(
            {
                "task": args.task,
                "requested_examples": args.num_train,
                "ood_sample_size": args.ood_sample_size,
                "seed": args.seed,
                "train_seq_len": train_seq_len,
                "test_seq_len": test_seq_len,
                "splits": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
