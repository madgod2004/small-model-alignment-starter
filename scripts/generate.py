"""Generate responses for a fixed evaluation set and save them as JSON.

Output entries: {"id", "instruction", "output", "generator", "dataset", ...extra}.
Defaults come from ``configs/generation.yaml`` (shared by all models so comparisons
use identical decoding); explicit CLI flags override it.
A sidecar ``<output>.meta.json`` records the model path and exact decoding settings.
Progress is checkpointed to ``<output>.partial.jsonl`` after every batch, so a rerun
after a Kaggle disconnect resumes instead of starting over.

Usage (repository root):
    # sanity check on 3 prompts, printed to stdout
    python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \
        --eval_set eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/sanity.json --limit 3 --print

    # full baseline run
    python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \
        --eval_set eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/base_alpaca_eval.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from alignment.generation import DecodingConfig, generate_responses  # noqa: E402
from alignment.utils import append_jsonl, load_config, read_json, read_jsonl, set_seed, write_json  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "generation.yaml"

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
EXTRA_FIELDS = ("category", "expected_behavior", "source_subset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML with decoding defaults.")
    parser.add_argument("--model_path", required=True, help="HF model id or local checkpoint directory.")
    parser.add_argument("--tokenizer_path", default=None, help="Defaults to --model_path.")
    parser.add_argument("--generator", required=True, help="Name stored in each entry, e.g. qwen2.5-0.5b-sft.")
    parser.add_argument("--eval_set", required=True, help="JSON list with id, instruction, dataset.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt_format", default="chatml", choices=["chatml", "chat_template", "raw"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true", help="Sample instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N prompts (sanity checks).")
    parser.add_argument("--print", action="store_true", help="Print each prompt/response pair.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the output exists.")
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**load_config(config_args.config))
    args = parser.parse_args()

    output_path = Path(args.output_path)
    partial_path = output_path.with_name(output_path.name + ".partial.jsonl")
    meta_path = output_path.with_name(output_path.stem + ".meta.json")
    eval_rows = read_json(args.eval_set)[: args.limit]
    decoding = DecodingConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    run_settings = {"generator": args.generator, "prompt_format": args.prompt_format, "dtype": args.dtype,
                    "num_prompts": len(eval_rows), "decoding": decoding.to_dict()}
    # Partial checkpoints are only resumed for the same settings.
    partial_settings_path = partial_path.with_name(partial_path.name + ".settings.json")
    if partial_path.exists() and (args.overwrite or not partial_settings_path.exists()
                                  or read_json(partial_settings_path) != run_settings):
        partial_path.unlink()

    if output_path.exists() and not args.overwrite:
        previous = read_json(meta_path) if meta_path.exists() else {}
        if all(previous.get(key) == value for key, value in run_settings.items()):
            print(f"{output_path} already exists with identical settings; skipping (use --overwrite to regenerate).")
            return
        print(f"{output_path} exists but was generated with different settings; regenerating.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    write_json(partial_settings_path, run_settings)
    done = {row["id"]: row for row in read_jsonl(partial_path)}
    todo = [row for row in eval_rows if row["id"] not in done]
    print(f"{len(eval_rows)} prompts, {len(done)} already in checkpoint, {len(todo)} to generate")

    start_time = time.time()
    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=DTYPES[args.dtype])
        model.to(args.device).eval()

        def checkpoint(indices, results):
            for index, result in zip(indices, results):
                row = todo[index]
                entry = {
                    "id": row["id"],
                    "instruction": row["instruction"],
                    "output": result["output"],
                    "generator": args.generator,
                    "dataset": row.get("dataset"),
                    **{key: row[key] for key in EXTRA_FIELDS if key in row},
                    "truncated": result["truncated"],
                    "num_tokens": result["num_tokens"],
                }
                done[row["id"]] = entry
                append_jsonl(partial_path, entry)
            print(f"  generated {len(done)}/{len(eval_rows)}", flush=True)

        generate_responses(
            model, tokenizer, [row["instruction"] for row in todo], decoding,
            batch_size=args.batch_size, prompt_format=args.prompt_format, on_batch=checkpoint,
        )

    entries = [done[row["id"]] for row in eval_rows]  # original eval-set order
    write_json(output_path, entries)
    write_json(meta_path, {
        "model_path": args.model_path,
        "config": args.config,
        "generator": args.generator,
        "eval_set": args.eval_set,
        "num_prompts": len(entries),
        "prompt_format": args.prompt_format,
        "dtype": args.dtype,
        "seed": args.seed,
        "decoding": decoding.to_dict(),
        "num_truncated": sum(entry["truncated"] for entry in entries),
        "generation_seconds": round(time.time() - start_time, 1),
    })
    for path in (partial_path, partial_settings_path):
        if path.exists():
            path.unlink()
    print(f"Wrote {len(entries)} generations to {output_path} "
          f"({sum(e['truncated'] for e in entries)} hit max_new_tokens)")

    if args.print:
        for entry in entries:
            print("=" * 80, f"\n[{entry['id']}] {entry['instruction']}\n---\n{entry['output']}")


if __name__ == "__main__":
    main()
