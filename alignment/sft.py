"""Supervised fine-tuning loop (Task 3), written without HF Trainer.

One optimizer step consumes ``gradient_accumulation_steps`` micro-batches on each
rank. The loss for a step is the mean NLL over *all* response tokens in that step
across micro-batches and GPUs, so the update matches a single batch of
``effective_batch_size`` examples regardless of how it is split.
"""

from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Iterator, Sequence

import torch

from alignment.batching import build_example, encode_text, get_eos_token_id, get_pad_token_id, pad_examples
from alignment.losses import batch_logprobs
from alignment.prompting import format_prompt
from alignment.training import (
    DistInfo,
    all_reduce_sum,
    autocast_context,
    barrier,
    build_optimizer,
    build_scheduler,
    make_grad_scaler,
    resolve_accumulation,
    save_checkpoint,
    shard_indices,
    unwrap,
)
from alignment.utils import append_jsonl, read_jsonl, set_seed, write_json

Example = tuple[list[int], list[int]]


def tokenize_sft_records(tokenizer, records: Sequence[dict], max_length: int) -> list[Example]:
    """Template + tokenise each record once; same construction as ``build_sft_batch``."""

    eos_token_id = get_eos_token_id(tokenizer)
    return [
        build_example(
            encode_text(tokenizer, format_prompt(record["prompt"])),
            encode_text(tokenizer, record["response"]),
            eos_token_id,
            max_length,
        )
        for record in records
    ]


def iter_batches(examples: Sequence[Example], indices: Sequence[int], batch_size: int, pad_token_id: int) -> Iterator[dict]:
    for start in range(0, len(indices), batch_size):
        yield pad_examples([examples[i] for i in indices[start : start + batch_size]], pad_token_id)


def count_response_tokens(batch: dict) -> int:
    """Tokens that contribute to the loss (labels shifted by one, as in ``sequence_logprobs``)."""

    return int(batch["labels"][:, 1:].ne(-100).sum())


def accumulate_window(model, micro_batches: Sequence[dict], window_tokens: int, info: DistInfo, scaler, fp16: bool) -> torch.Tensor:
    """Backward over one accumulation window; returns this rank's summed NLL (detached).

    Each micro-batch contributes ``sum_nll / window_tokens``, where ``window_tokens`` is
    the response-token total of the window across all ranks. DDP averages gradients
    over ranks, so the loss is multiplied by ``world_size`` to recover a global token mean.
    """

    nll_total = torch.zeros((), device=info.device)
    for index, batch in enumerate(micro_batches):
        is_last = index == len(micro_batches) - 1
        sync = contextlib.nullcontext() if (is_last or not info.distributed) else model.no_sync()
        with sync:
            with autocast_context(info.device, fp16):
                summed_logps, _ = batch_logprobs(model, batch)
            nll = -summed_logps.sum()
            loss = nll * info.world_size / max(window_tokens, 1)
            scaler.scale(loss).backward()
        nll_total += nll.detach()
    return nll_total


@torch.no_grad()
def evaluate_sft(model, examples: Sequence[Example], batch_size: int, pad_token_id: int, info: DistInfo, fp16: bool) -> float:
    """Token-mean validation NLL over the whole validation set (each rank scores a disjoint shard)."""

    base_model = unwrap(model)
    was_training = base_model.training
    base_model.eval()
    indices = list(range(info.rank, len(examples), info.world_size))  # disjoint, no padding duplicates
    totals = torch.zeros(2, device=info.device, dtype=torch.float64)
    for batch in iter_batches(examples, indices, batch_size, pad_token_id):
        with autocast_context(info.device, fp16):
            summed_logps, token_counts = batch_logprobs(base_model, batch)
        totals[0] += -summed_logps.double().sum()
        totals[1] += token_counts.sum()
    all_reduce_sum(totals, info)
    if was_training:
        base_model.train()
    return (totals[0] / totals[1].clamp_min(1)).item()


def train_sft(config: dict, info: DistInfo) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    fp16 = bool(config.get("fp16", True))
    log = print if info.is_main else (lambda *args, **kwargs: None)

    # ---- data -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    pad_token_id = get_pad_token_id(tokenizer)
    train_records = read_jsonl(config["train_file"])[: config.get("train_limit")]
    val_records = read_jsonl(config["val_file"])[: config.get("val_limit")]
    train_examples = tokenize_sft_records(tokenizer, train_records, config["max_length"])
    val_examples = tokenize_sft_records(tokenizer, val_records, config["max_length"])

    batch_size = config["per_device_train_batch_size"]
    accumulation = resolve_accumulation(config, info.world_size)
    rank_indices = shard_indices(len(train_examples), info, config["seed"])
    micro_batches_per_epoch = math.ceil(len(rank_indices) / batch_size)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * config["num_epochs"]
    if config.get("max_steps"):
        total_steps = min(total_steps, config["max_steps"])

    # ---- model / optimisation ---------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], torch_dtype=torch.float32)
    model.config.use_cache = False
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(info.device).train()
    if info.distributed:
        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)
    scaler = make_grad_scaler(info.device, fp16)

    run_info = {
        "world_size": info.world_size,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation * info.world_size,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "total_optimizer_steps": total_steps,
        "warmup_steps": math.ceil(config.get("warmup_ratio", 0.0) * total_steps),
        "train_examples_truncated": sum(1 for ids, _ in train_examples if len(ids) == config["max_length"]),
    }
    if info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()
        write_json(output_dir / "config.json", {**config, **run_info})
    log(f"SFT run: {run_info}")

    best_val_loss = float("inf")
    best_step = None

    def run_eval(step: int) -> float:
        nonlocal best_val_loss, best_step
        val_loss = evaluate_sft(model, val_examples, config["per_device_eval_batch_size"], pad_token_id, info, fp16)
        improved = step > 0 and val_loss < best_val_loss  # step 0 is the untrained base model
        if info.is_main:
            append_jsonl(metrics_path, {"type": "eval", "step": step, "val_loss": val_loss, "best": improved})
            log(f"[eval] step {step}: val_loss={val_loss:.4f}{' (best)' if improved else ''}")
            if improved:
                save_checkpoint(model, tokenizer, checkpoint_dir / "best", {"step": step, "val_loss": val_loss})
        if improved:
            best_val_loss, best_step = val_loss, step
        barrier(info)
        return val_loss

    if config.get("eval_at_start", True):
        run_eval(0)

    # ---- training ---------------------------------------------------------------------------
    step = 0
    start_time = time.time()
    tokens_since_log = 0
    log_time = start_time
    skipped_steps = 0
    for epoch in range(config["num_epochs"]):
        epoch_indices = shard_indices(len(train_examples), info, config["seed"] + epoch)
        batches = list(iter_batches(train_examples, epoch_indices, batch_size, pad_token_id))
        for window_start in range(0, len(batches), accumulation):
            if step >= total_steps:
                break
            micro_batches = batches[window_start : window_start + accumulation]
            token_count = torch.tensor(sum(count_response_tokens(b) for b in micro_batches), device=info.device)
            window_tokens = int(all_reduce_sum(token_count, info).item())

            nll = accumulate_window(model, micro_batches, window_tokens, info, scaler, fp16)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                skipped_steps += 1  # inf/nan gradients: GradScaler skipped this update
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            tokens_since_log += sum(int(b["attention_mask"].sum()) for b in micro_batches) * info.world_size

            if step % config["logging_steps"] == 0 or step == total_steps:
                train_loss = (all_reduce_sum(nll, info) / max(window_tokens, 1)).item()
                now = time.time()
                record = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch + (window_start + len(micro_batches)) / len(batches),
                    "train_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "loss_scale": scaler.get_scale() if scaler.is_enabled() else None,
                    "skipped_steps": skipped_steps,
                    "tokens_per_second": tokens_since_log / max(now - log_time, 1e-9),
                    "elapsed_seconds": now - start_time,
                }
                tokens_since_log, log_time = 0, now
                if info.is_main:
                    append_jsonl(metrics_path, record)
                log(f"step {step}/{total_steps} loss={train_loss:.4f} lr={record['lr']:.2e} "
                    f"gnorm={record['grad_norm']:.2f} tok/s={record['tokens_per_second']:.0f} "
                    f"elapsed={record['elapsed_seconds'] / 60:.1f}m")

            if step % config["eval_steps"] == 0 and step != total_steps:
                run_eval(step)

    final_val_loss = run_eval(step)
    if info.is_main:
        save_checkpoint(model, tokenizer, checkpoint_dir / "final", {"step": step, "val_loss": final_val_loss})
    summary = {
        **run_info,
        "steps_completed": step,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "skipped_steps": skipped_steps,
        "train_seconds": time.time() - start_time,
        "best_checkpoint": str(checkpoint_dir / "best"),
        "final_checkpoint": str(checkpoint_dir / "final"),
    }
    if info.is_main:
        write_json(output_dir / "summary.json", summary)
        log(f"Done: {summary}")
    barrier(info)
    return summary
