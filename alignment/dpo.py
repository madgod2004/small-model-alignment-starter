"""From-scratch DPO training loop (Task 4), shared by the HH-RLHF and UltraFeedback runs.

The trainable policy and the frozen reference both start from the SFT best checkpoint.
The policy lives on cuda:0 and the reference on cuda:1 (falling back to one device).
Each optimizer step averages the per-pair DPO loss over ``effective_batch_size`` pairs.
The checkpoint with the best validation reward accuracy is kept, plus the final one.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator, Sequence

import torch

from alignment.batching import build_example, encode_text, get_eos_token_id, get_pad_token_id, pad_pair_examples
from alignment.losses import concatenated_logprobs, dpo_loss_from_logps
from alignment.prompting import format_prompt
from alignment.training import (
    autocast_context,
    build_optimizer,
    build_scheduler,
    make_grad_scaler,
    max_memory_gb,
    resolve_accumulation,
    save_checkpoint,
    shuffled_indices,
)
from alignment.utils import append_jsonl, read_jsonl, set_seed, write_json

Example = tuple[list[int], list[int]]
PairExample = tuple[Example, Example]

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
METRIC_KEYS = ("loss", "reward_accuracy", "reward_margin", "chosen_rewards", "rejected_rewards",
               "policy_chosen_logps", "policy_rejected_logps")


def tokenize_pairs(tokenizer, records: Sequence[dict], max_length: int, max_prompt_length: int | None) -> list[PairExample]:
    """Template + tokenise each pair once; same construction as ``build_dpo_batch``."""

    eos_token_id = get_eos_token_id(tokenizer)
    pairs = []
    for record in records:
        prompt_ids = encode_text(tokenizer, format_prompt(record["prompt"]))
        chosen = build_example(prompt_ids, encode_text(tokenizer, record["chosen"]), eos_token_id, max_length, max_prompt_length)
        rejected = build_example(prompt_ids, encode_text(tokenizer, record["rejected"]), eos_token_id, max_length, max_prompt_length)
        pairs.append((chosen, rejected))
    return pairs


def iter_pair_batches(pairs: Sequence[PairExample], indices: Sequence[int], batch_size: int, pad_token_id: int) -> Iterator[dict]:
    for start in range(0, len(indices), batch_size):
        selected = [pairs[i] for i in indices[start : start + batch_size]]
        yield pad_pair_examples([c for c, _ in selected], [r for _, r in selected], pad_token_id)


def pick_devices() -> tuple[torch.device, torch.device]:
    """Policy on cuda:0, reference on cuda:1 when two GPUs are visible."""

    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    return torch.device("cuda", 0), torch.device("cuda", 1 if torch.cuda.device_count() > 1 else 0)


def dpo_step_outputs(policy, reference, batch, beta: float, policy_device, reference_device, fp16: bool) -> dict:
    """DPO loss and diagnostics for one batch.

    The reference forward is queued first: its kernels run on cuda:1 while the policy
    forward runs on cuda:0, and the two only synchronise inside the loss.
    """

    with torch.no_grad(), autocast_context(reference_device, fp16):
        reference_chosen, reference_rejected = concatenated_logprobs(reference, batch)
    with autocast_context(policy_device, fp16):
        policy_chosen, policy_rejected = concatenated_logprobs(policy, batch)
    outputs = dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    outputs["policy_chosen_logps"] = policy_chosen.detach()
    outputs["policy_rejected_logps"] = policy_rejected.detach()
    return outputs


def summed_metrics(outputs: dict) -> dict[str, float]:
    """Per-batch sums of the per-pair metrics (averaged later over all pairs)."""

    per_pair = {"loss": outputs["losses"].detach(), **{key: outputs[key] for key in METRIC_KEYS if key != "loss"}}
    return {key: float(value.float().sum()) for key, value in per_pair.items()}


@torch.no_grad()
def evaluate_dpo(policy, reference, pairs, batch_size, pad_token_id, beta, policy_device, reference_device, fp16) -> dict:
    """Validation DPO loss, reward accuracy, reward margin, and mean rewards over all pairs."""

    was_training = policy.training
    policy.eval()
    totals = dict.fromkeys(METRIC_KEYS, 0.0)
    for batch in iter_pair_batches(pairs, range(len(pairs)), batch_size, pad_token_id):
        outputs = dpo_step_outputs(policy, reference, batch, beta, policy_device, reference_device, fp16)
        for key, value in summed_metrics(outputs).items():
            totals[key] += value
    if was_training:
        policy.train()
    if policy_device.type == "cuda":
        torch.cuda.empty_cache()
    return {key: value / max(len(pairs), 1) for key, value in totals.items()}


def load_model(path: str, dtype: torch.dtype, device: torch.device, trainable: bool):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.config.use_cache = False
    model.to(device)
    if trainable:
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def train_dpo(config: dict) -> dict:
    from transformers import AutoTokenizer

    set_seed(config["seed"])
    policy_device, reference_device = pick_devices()
    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    fp16 = bool(config.get("fp16", True))
    beta = float(config["beta"])

    # ---- data -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["sft_checkpoint"])
    pad_token_id = get_pad_token_id(tokenizer)
    train_records = read_jsonl(config["train_file"])[: config.get("train_limit")]
    val_records = read_jsonl(config["val_file"])[: config.get("val_limit")]
    max_length, max_prompt_length = config["max_length"], config.get("max_prompt_length")
    train_pairs = tokenize_pairs(tokenizer, train_records, max_length, max_prompt_length)
    val_pairs = tokenize_pairs(tokenizer, val_records, max_length, max_prompt_length)

    batch_size = config["per_device_train_batch_size"]
    accumulation = resolve_accumulation(config)
    steps_per_epoch = math.ceil(math.ceil(len(train_pairs) / batch_size) / accumulation)
    total_steps = steps_per_epoch * config["num_epochs"]
    if config.get("max_steps"):
        total_steps = min(total_steps, config["max_steps"])

    # ---- models / optimisation --------------------------------------------------------------
    policy = load_model(config["sft_checkpoint"], torch.float32, policy_device, trainable=True)
    if config.get("gradient_checkpointing"):
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = load_model(config["sft_checkpoint"], DTYPES[config.get("reference_dtype", "fp32")], reference_device, trainable=False)

    optimizer = build_optimizer(policy, config)
    scheduler = build_scheduler(optimizer, config, total_steps)
    scaler = make_grad_scaler(policy_device, fp16)

    run_info = {
        "policy_device": str(policy_device),
        "reference_device": str(reference_device),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "total_optimizer_steps": total_steps,
        "train_pairs_truncated": sum(
            1 for chosen, rejected in train_pairs if max_length in (len(chosen[0]), len(rejected[0]))
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    write_json(output_dir / "config.json", {**config, **run_info})
    print(f"DPO run: {run_info}", flush=True)

    best_accuracy = -1.0
    best_step = None
    eval_args = (config["per_device_eval_batch_size"], pad_token_id, beta, policy_device, reference_device, fp16)

    def run_eval(step: int) -> dict:
        nonlocal best_accuracy, best_step
        metrics = evaluate_dpo(policy, reference, val_pairs, *eval_args)
        # step 0 is the untrained policy (identical to the reference), never a candidate
        improved = step > 0 and metrics["reward_accuracy"] > best_accuracy
        append_jsonl(metrics_path, {"type": "eval", "step": step, **{f"val_{k}": v for k, v in metrics.items()}, "best": improved})
        print(f"[eval] step {step}: loss={metrics['loss']:.4f} acc={metrics['reward_accuracy']:.3f} "
              f"margin={metrics['reward_margin']:.4f}{' (best)' if improved else ''}", flush=True)
        if improved:
            best_accuracy, best_step = metrics["reward_accuracy"], step
            save_checkpoint(policy, tokenizer, checkpoint_dir / "best", {"step": step, **metrics})
        return metrics

    if config.get("eval_at_start", True):
        run_eval(0)

    # ---- training ---------------------------------------------------------------------------
    step = 0
    start_time = log_time = time.time()
    pairs_since_log = 0
    skipped_steps = 0
    for epoch in range(config["num_epochs"]):
        indices = shuffled_indices(len(train_pairs), config["seed"] + epoch)
        batches = list(iter_pair_batches(train_pairs, indices, batch_size, pad_token_id))
        for window_start in range(0, len(batches), accumulation):
            if step >= total_steps:
                break
            micro_batches = batches[window_start : window_start + accumulation]
            window_pairs = sum(batch["chosen"]["input_ids"].shape[0] for batch in micro_batches)
            totals = dict.fromkeys(METRIC_KEYS, 0.0)
            for batch in micro_batches:
                outputs = dpo_step_outputs(policy, reference, batch, beta, policy_device, reference_device, fp16)
                scaler.scale(outputs["losses"].sum() / window_pairs).backward()
                for key, value in summed_metrics(outputs).items():
                    totals[key] += value

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            pairs_since_log += window_pairs

            if step % config["logging_steps"] == 0 or step == total_steps:
                now = time.time()
                window = {key: value / window_pairs for key, value in totals.items()}
                record = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch + (window_start + len(micro_batches)) / len(batches),
                    **{f"train_{key}": value for key, value in window.items()},
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "loss_scale": scaler.get_scale() if scaler.is_enabled() else None,
                    "skipped_steps": skipped_steps,
                    "pairs_per_second": pairs_since_log / max(now - log_time, 1e-9),
                    "elapsed_seconds": now - start_time,
                    "max_memory_gb_policy": max_memory_gb(policy_device),
                    "max_memory_gb_reference": max_memory_gb(reference_device),
                }
                pairs_since_log, log_time = 0, now
                append_jsonl(metrics_path, record)
                memory = f"mem={record['max_memory_gb_policy']:.1f}G " if record["max_memory_gb_policy"] else ""
                print(f"step {step}/{total_steps} loss={window['loss']:.4f} acc={window['reward_accuracy']:.3f} "
                      f"margin={window['reward_margin']:.4f} gnorm={record['grad_norm']:.3f} {memory}"
                      f"elapsed={record['elapsed_seconds'] / 60:.1f}m", flush=True)

            if step % config["eval_steps"] == 0 and step != total_steps:
                run_eval(step)

    final_metrics = run_eval(step)
    save_checkpoint(policy, tokenizer, checkpoint_dir / "final", {"step": step, **final_metrics})
    summary = {
        **run_info,
        "beta": beta,
        "reference_checkpoint": config["sft_checkpoint"],
        "steps_completed": step,
        "best_step": best_step,
        "best_val_reward_accuracy": best_accuracy,
        "final_val_metrics": final_metrics,
        "skipped_steps": skipped_steps,
        "train_seconds": time.time() - start_time,
        "max_memory_gb_policy": max_memory_gb(policy_device),
        "max_memory_gb_reference": max_memory_gb(reference_device),
        "best_checkpoint": str(checkpoint_dir / "best"),
        "final_checkpoint": str(checkpoint_dir / "final"),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Done: {summary}", flush=True)
    return summary
