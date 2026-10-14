"""Build the fixed training/validation subsets and the AlpacaEval evaluation subset.

Outputs:
    data/processed/<name>/{train,val}.jsonl     full records (git-ignored, regenerated on Kaggle)
    data/manifests/<name>.json                  sources, filters, counts, selected source indices
    eval_sets/alpaca_eval_300_seed42.json       fixed AlpacaEval prompts (committed)

Usage (repository root):
    python scripts/prepare_data.py --tasks all
    python scripts/prepare_data.py --tasks hh uf --source_limit 2000   # quick local smoke run
"""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alignment.batching import encode_text  # noqa: E402
from alignment.data import (  # noqa: E402
    parse_hh_rlhf,
    parse_ultrachat,
    parse_ultrafeedback,
    sample_eval_subset,
    select_examples,
)
from alignment.prompting import format_prompt  # noqa: E402
from alignment.utils import read_json, write_json, write_jsonl  # noqa: E402

# name -> (HF dataset, train split, val split, parser, train size, val size)
TRAINING_SETS = {
    "sft": ("HuggingFaceH4/ultrachat_200k", "train_sft", "test_sft", parse_ultrachat, 30_000, 1_000),
    "hh": ("Anthropic/hh-rlhf", "train", "test", parse_hh_rlhf, 15_000, 1_000),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs", parse_ultrafeedback, 15_000, 1_000),
}
ALPACA_REPO, ALPACA_FILE = "tatsu-lab/alpaca_eval", "alpaca_eval.json"


def load_split(dataset_name: str, split: str, source_limit: int | None):
    from datasets import Dataset, load_dataset

    if source_limit:
        stream = load_dataset(dataset_name, split=split, streaming=True)
        return Dataset.from_list(list(islice(stream, source_limit)))
    return load_dataset(dataset_name, split=split)


def dataset_revision(dataset_name: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_name).sha
    except Exception:  # noqa: BLE001 - offline or rate-limited; revision is informational
        return None


def prepare_training_set(name: str, args, tokenizer) -> None:
    dataset_name, train_split, val_split, parser, train_size, val_size = TRAINING_SETS[name]
    train_size = args.train_size or train_size
    val_size = args.val_size or val_size

    keep = None
    if tokenizer is not None and args.max_prompt_tokens > 0:
        keep = lambda parsed: len(encode_text(tokenizer, format_prompt(parsed["prompt"]))) <= args.max_prompt_tokens  # noqa: E731

    out_dir = REPO_ROOT / "data" / "processed" / name
    manifest = {
        "name": name,
        "source_dataset": dataset_name,
        "source_revision": dataset_revision(dataset_name),
        "seed": args.seed,
        "selection": "seeded shuffle of source row indices; first N rows passing the parser and filters",
        "filters": {
            "parser": parser.__name__,
            "max_formatted_prompt_tokens": args.max_prompt_tokens if keep else None,
            "tokenizer": args.tokenizer if keep else None,
        },
        "source_limit": args.source_limit,
        "splits": {},
    }

    for split_name, source_split, size in (("train", train_split, train_size), ("val", val_split, val_size)):
        rows = load_split(dataset_name, source_split, args.source_limit)
        records, stats = select_examples(rows, parser, size, args.seed, f"{name}-{split_name}", keep)
        if len(records) < size:
            print(f"WARNING: {name}/{split_name}: only {len(records)} of {size} requested examples available")
        write_jsonl(out_dir / f"{split_name}.jsonl", records)
        manifest["splits"][split_name] = {
            "source_split": source_split,
            "source_rows": len(rows),
            "requested": size,
            "stats": stats,
            "source_indices": [record["source_index"] for record in records],
        }
        print(f"{name}/{split_name}: {stats}")

    write_json(REPO_ROOT / "data" / "manifests" / f"{name}.json", manifest)


def prepare_alpaca_eval(args) -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ALPACA_REPO, ALPACA_FILE, repo_type="dataset")
    rows = read_json(path)
    subset = sample_eval_subset(rows, args.alpaca_num_prompts, args.seed, f"alpaca-eval-{args.seed}", "alpaca_eval")
    out_path = REPO_ROOT / "eval_sets" / f"alpaca_eval_{args.alpaca_num_prompts}_seed{args.seed}.json"
    write_json(out_path, subset)
    print(f"alpaca_eval: sampled {len(subset)} of {len(rows)} prompts -> {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=["all", "sft", "hh", "uf", "alpaca"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="Used only for the prompt-length filter.")
    parser.add_argument("--max_prompt_tokens", type=int, default=256,
                        help="Drop rows whose templated prompt exceeds this many tokens (0 disables).")
    parser.add_argument("--train_size", type=int, default=None, help="Override the default train size.")
    parser.add_argument("--val_size", type=int, default=None, help="Override the default validation size.")
    parser.add_argument("--alpaca_num_prompts", type=int, default=300)
    parser.add_argument("--source_limit", type=int, default=None,
                        help="Stream only the first N source rows per split (local smoke tests only).")
    args = parser.parse_args()

    tasks = ["sft", "hh", "uf", "alpaca"] if "all" in args.tasks else args.tasks
    tokenizer = None
    if args.max_prompt_tokens > 0 and any(task in TRAINING_SETS for task in tasks):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for task in tasks:
        if task == "alpaca":
            prepare_alpaca_eval(args)
        else:
            prepare_training_set(task, args, tokenizer)


if __name__ == "__main__":
    main()
