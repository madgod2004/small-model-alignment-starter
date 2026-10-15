"""Task 5: persona DPO with TRL's DPOTrainer and a PEFT/LoRA adapter on SmolLM2-360M-Instruct.

    python scripts/train_lora_dpo.py --config configs/lora_dpo.yaml

Outputs (under output_dir): adapter/ (LoRA weights), merged/ (base + adapter, for generation),
metrics.jsonl (trainer log history), split.json (train/eval prompts), config.json, summary.json.
Any config key can be overridden with key=value (e.g. max_steps=2 for a smoke test).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

# One GPU: with two visible GPUs HF Trainer would silently wrap the model in DataParallel.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.utils import apply_overrides, load_config, read_json, set_seed, write_json, write_jsonl  # noqa: E402


def to_conversational(row: dict) -> dict:
    """TRL conversational preference format: prompt / chosen / rejected as role-message lists."""

    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def split_rows(rows: list[dict], eval_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    eval_ids = set(order[:eval_size])
    train = [row for i, row in enumerate(rows) if i not in eval_ids]
    held_out = [row for i, row in enumerate(rows) if i in eval_ids]
    return train, held_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/lora_dpo.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_json(config["data_file"])
    train_rows, eval_rows = split_rows(rows, config["eval_size"], config["seed"])
    write_json(output_dir / "split.json", {
        "seed": config["seed"],
        "train_prompts": [row["prompt"] for row in train_rows],
        "eval_prompts": [row["prompt"] for row in eval_rows],
    })
    train_dataset = Dataset.from_list([to_conversational(row) for row in train_rows])
    eval_dataset = Dataset.from_list([to_conversational(row) for row in eval_rows])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=list(config["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    use_fp16 = bool(config.get("fp16")) and torch.cuda.is_available()
    training_args = DPOConfig(
        output_dir=str(output_dir / "trainer"),
        beta=config["beta"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config.get("max_steps") or -1,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        max_length=config["max_length"],
        max_grad_norm=config["max_grad_norm"],
        weight_decay=config["weight_decay"],
        fp16=use_fp16,
        bf16=False,
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="no",
        report_to="none",
        seed=config["seed"],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: the adapter-disabled model is the frozen reference
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"LoRA trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    initial_eval = trainer.evaluate()
    start_time = time.time()
    trainer.train()
    train_seconds = time.time() - start_time
    final_eval = trainer.evaluate()

    write_jsonl(output_dir / "metrics.jsonl", [{"type": "initial_eval", **initial_eval}, *trainer.state.log_history])
    trainer.model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir / "merged", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "merged")

    summary = {
        "model": config["model_name_or_path"],
        "train_pairs": len(train_rows),
        "eval_pairs": len(eval_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100 * trainable / total,
        "optimizer_steps": trainer.state.global_step,
        "effective_batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
        "precision": "fp16 autocast" if use_fp16 else "fp32",
        "train_seconds": train_seconds,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "adapter_dir": str(output_dir / "adapter"),
        "merged_dir": str(output_dir / "merged"),
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary.json", summary)
    print(f"Done: {summary}", flush=True)


if __name__ == "__main__":
    main()
