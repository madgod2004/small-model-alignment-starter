"""Training infrastructure shared by the SFT and DPO loops (no HF Trainer).

Covers device selection, the optimizer with decoupled weight decay, the warmup +
cosine/constant schedule, fp16 autocast with gradient scaling, and checkpoint saving.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
from pathlib import Path

import torch


def default_device(index: int = 0) -> torch.device:
    return torch.device("cuda", index) if torch.cuda.is_available() else torch.device("cpu")


def resolve_accumulation(config: dict) -> int:
    """Accumulation steps so that per_device_train_batch_size * accumulation == effective_batch_size."""

    if config.get("gradient_accumulation_steps"):
        return int(config["gradient_accumulation_steps"])
    per_step = config["per_device_train_batch_size"]
    effective = config["effective_batch_size"]
    if effective % per_step:
        raise ValueError(f"effective_batch_size={effective} is not divisible by per_device_train_batch_size={per_step}")
    return effective // per_step


def build_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """AdamW (or RMSprop); weight decay on matrices only, not on biases or norm weights."""

    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.get("weight_decay", 0.0)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = config.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=config["learning_rate"],
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=config["learning_rate"])
    raise ValueError(f"Unknown optimizer: {name}")


def lr_lambda_factory(total_steps: int, warmup_steps: int, schedule: str):
    """Linear warmup, then cosine decay to 0 or a constant learning rate."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        raise ValueError(f"Unknown lr_scheduler: {schedule}")

    return lr_lambda


def build_scheduler(optimizer, config: dict, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = math.ceil(config.get("warmup_ratio", 0.0) * total_steps)
    lr_lambda = lr_lambda_factory(total_steps, warmup_steps, config.get("lr_scheduler", "constant"))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast_context(device: torch.device, fp16: bool):
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, fp16: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")


def shuffled_indices(num_examples: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_examples, generator=generator).tolist()


def max_memory_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def save_checkpoint(model, tokenizer, path: str | os.PathLike, extra_state: dict | None = None) -> None:
    """Save weights (safetensors, current dtype) + tokenizer; written to a temp dir then renamed."""

    from alignment.utils import write_json

    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    model.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer.save_pretrained(tmp_path)
    if extra_state:
        write_json(tmp_path / "training_state.json", extra_state)
    if path.exists():
        shutil.rmtree(path)
    tmp_path.rename(path)
