"""Batched, deterministic response generation for evaluation.

Prompts use the same template as training (``alignment.prompting``) and are
tokenised the same way (no extra special tokens). Batches are left-padded, as
decoder-only generation requires. Generation stops at EOS or at ChatML turn
markers, so a base model that keeps writing a fake next turn is cut cleanly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch

from alignment.batching import encode_text, get_pad_token_id
from alignment.prompting import format_prompt

STOP_TOKENS = ("<|im_end|>", "<|im_start|>")


@dataclass
class DecodingConfig:
    max_new_tokens: int = 512
    do_sample: bool = False  # greedy by default for the main comparison
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build_prompt(tokenizer, instruction: str, prompt_format: str) -> str:
    """``chatml``: our training template. ``chat_template``: the tokenizer's own (used for SmolLM)."""

    if prompt_format == "chatml":
        return format_prompt(instruction)
    if prompt_format == "chat_template":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
    if prompt_format == "raw":
        return instruction
    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def stop_token_ids(tokenizer) -> list[int]:
    ids = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != unk_id and token_id not in ids:
            ids.append(token_id)
    return ids


def left_pad(sequences: Sequence[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in sequences)
    input_ids = [[pad_token_id] * (width - len(ids)) + ids for ids in sequences]
    attention_mask = [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences]
    return torch.tensor(input_ids), torch.tensor(attention_mask)


def cut_at_stop(token_ids: list[int], stop_ids: Sequence[int]) -> tuple[list[int], bool]:
    """Drop everything from the first stop token; returns (ids, stopped)."""

    for position, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[:position], True
    return token_ids, False


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: Sequence[str], decoding: DecodingConfig) -> list[dict]:
    """Generate for one batch of formatted prompts; returns [{"output", "truncated", "num_tokens"}]."""

    device = next(model.parameters()).device
    pad_token_id = get_pad_token_id(tokenizer)
    stop_ids = stop_token_ids(tokenizer)
    input_ids, attention_mask = left_pad([encode_text(tokenizer, p) for p in prompts], pad_token_id)

    kwargs = dict(
        max_new_tokens=decoding.max_new_tokens,
        do_sample=decoding.do_sample,
        repetition_penalty=decoding.repetition_penalty,
        eos_token_id=stop_ids,
        pad_token_id=pad_token_id,
    )
    if decoding.do_sample:
        kwargs.update(temperature=decoding.temperature, top_p=decoding.top_p)
    else:  # silence "temperature/top_p ignored" warnings from model generation configs
        kwargs.update(temperature=None, top_p=None, top_k=None)
    if decoding.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = decoding.no_repeat_ngram_size

    sequences = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), **kwargs)
    results = []
    for row in sequences[:, input_ids.shape[1] :].tolist():
        # Finished rows are padded after their stop token, so cutting at the stop removes padding too.
        new_ids, stopped = cut_at_stop(row, stop_ids)
        results.append({
            "output": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
            "truncated": not stopped,
            "num_tokens": len(new_ids),
        })
    return results


def generate_responses(
    model,
    tokenizer,
    instructions: Sequence[str],
    decoding: DecodingConfig,
    batch_size: int = 16,
    prompt_format: str = "chatml",
    on_batch: Callable[[list[int], list[dict]], None] | None = None,
) -> list[dict]:
    """Generate for all instructions, batching similar lengths together; results keep input order.

    ``on_batch(indices, results)`` is called after every batch (used for resumable checkpoints).
    """

    prompts = [build_prompt(tokenizer, instruction, prompt_format) for instruction in instructions]
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]), reverse=True)
    results: list[dict | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch_results = generate_batch(model, tokenizer, [prompts[i] for i in indices], decoding)
        for index, result in zip(indices, batch_results):
            results[index] = result
        if on_batch is not None:
            on_batch(indices, batch_results)
    return results  # type: ignore[return-value]
