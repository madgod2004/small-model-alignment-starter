"""Training infrastructure shared by the SFT and DPO loops (no HF Trainer).

Covers distributed setup (torchrun/DDP or a single device), the optimizer with
decoupled weight decay, the warmup + cosine/constant schedule, fp16 autocast with
gradient scaling, and checkpoint saving.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """Initialise DDP when launched by torchrun; otherwise run on one device."""

    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend=backend)
        return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank, device)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return DistInfo(device=device)


def cleanup_distributed(info: DistInfo) -> None:
    if info.distributed and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(tensor: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if info.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def barrier(info: DistInfo) -> None:
    if info.distributed:
        dist.barrier()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def resolve_accumulation(config: dict, world_size: int) -> int:
    """Accumulation steps so per_device * accumulation * world_size == effective batch."""

    if config.get("gradient_accumulation_steps"):
        return int(config["gradient_accumulation_steps"])
    per_step = config["per_device_train_batch_size"] * world_size
    effective = config["effective_batch_size"]
    if effective % per_step:
        raise ValueError(f"effective_batch_size={effective} is not divisible by per_device*world_size={per_step}")
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


def shard_indices(num_examples: int, info: DistInfo, seed: int | None) -> list[int]:
    """Shuffle (if seed is given), pad to a multiple of world_size by wrapping, and take this rank's shard.

    Every rank gets the same number of examples so DDP ranks stay in lock-step.
    """

    if seed is None:
        order = list(range(num_examples))
    else:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(num_examples, generator=generator).tolist()
    remainder = len(order) % info.world_size
    if remainder:
        order += order[: info.world_size - remainder]
    return order[info.rank :: info.world_size]


def save_checkpoint(model, tokenizer, path: str | os.PathLike, extra_state: dict | None = None) -> None:
    """Save weights (safetensors, current dtype) + tokenizer; written to a temp dir then renamed."""

    from alignment.utils import write_json

    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    unwrap(model).save_pretrained(tmp_path, safe_serialization=True)
    tokenizer.save_pretrained(tmp_path)
    if extra_state:
        write_json(tmp_path / "training_state.json", extra_state)
    if path.exists():
        shutil.rmtree(path)
    tmp_path.rename(path)
