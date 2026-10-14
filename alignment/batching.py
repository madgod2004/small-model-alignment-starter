"""Tokenisation, truncation, padding, and response-only label masking.

Every training example is ``prompt_tokens + response_tokens + [EOS]``. Prompt and
response are tokenised separately so the prompt/response boundary is exact. The
EOS token is appended *before* truncation, so a response cut at ``max_length``
loses its EOS. Labels copy ``input_ids`` on response positions and are ``-100``
on prompt and padding positions. Padding is on the right.

Masking is done by position, never by token id: Qwen2.5 uses the same id for
padding and EOS, so masking "all pad ids" would also hide the EOS target.
"""

from __future__ import annotations

from typing import Sequence

import torch

IGNORE_INDEX = -100


def encode_text(tokenizer, text: str) -> list[int]:
    """Tokenise without adding BOS/EOS or other special tokens."""

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # tokenizers whose encode() takes no keyword
        token_ids = tokenizer.encode(text)
    return list(token_ids)


def get_eos_token_id(tokenizer) -> int | None:
    return getattr(tokenizer, "eos_token_id", None)


def get_pad_token_id(tokenizer) -> int:
    """Pad id used for input_ids. Its value never matters for the loss (labels are -100)."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = get_eos_token_id(tokenizer)
    return 0 if pad_token_id is None else pad_token_id


def build_example(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    eos_token_id: int | None,
    max_length: int,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return unpadded ``(input_ids, labels)`` for one prompt/response pair.

    ``max_prompt_length`` (training only, e.g. 256 for DPO) keeps the *last* prompt
    tokens so the assistant header right before the response survives.
    """

    prompt_ids = list(prompt_ids)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :] if max_prompt_length > 0 else []

    target_ids = list(response_ids)
    if eos_token_id is not None:
        target_ids.append(eos_token_id)

    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + target_ids)[:max_length]
    return input_ids, labels


def pad_examples(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_token_id: int,
    width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad examples into ``input_ids``/``attention_mask``/``labels`` tensors."""

    if width is None:
        width = max((len(input_ids) for input_ids, _ in examples), default=0)
    width = max(width, 1)  # keep a valid 2-D shape even if every row is empty

    input_rows, mask_rows, label_rows = [], [], []
    for input_ids, labels in examples:
        pad = width - len(input_ids)
        input_rows.append(input_ids + [pad_token_id] * pad)
        mask_rows.append([1] * len(input_ids) + [0] * pad)
        label_rows.append(labels + [IGNORE_INDEX] * pad)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def _check_lengths(**columns: Sequence[str]) -> int:
    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Batch columns must have equal length, got {lengths}")
    return next(iter(lengths.values()))


def build_sft_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Batch of prompt+response sequences with response-only labels, in input order."""

    _check_lengths(prompts=prompts, responses=responses)
    eos_token_id = get_eos_token_id(tokenizer)
    examples = [
        build_example(
            encode_text(tokenizer, prompt),
            encode_text(tokenizer, response),
            eos_token_id,
            max_length,
            max_prompt_length,
        )
        for prompt, response in zip(prompts, responses)
    ]
    return pad_examples(examples, get_pad_token_id(tokenizer))


def build_dpo_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Chosen and rejected LM batches sharing the same prompts.

    Both sides are padded to one common width so they can be concatenated into a
    single forward pass.
    """

    _check_lengths(prompts=prompts, chosen=chosen_responses, rejected=rejected_responses)
    eos_token_id = get_eos_token_id(tokenizer)
    prompt_ids = [encode_text(tokenizer, prompt) for prompt in prompts]

    def side(responses: Sequence[str]) -> list[tuple[list[int], list[int]]]:
        return [
            build_example(ids, encode_text(tokenizer, response), eos_token_id, max_length, max_prompt_length)
            for ids, response in zip(prompt_ids, responses)
        ]

    chosen, rejected = side(chosen_responses), side(rejected_responses)
    width = max((len(input_ids) for input_ids, _ in chosen + rejected), default=0)
    pad_token_id = get_pad_token_id(tokenizer)
    return {
        "chosen": pad_examples(chosen, pad_token_id, width),
        "rejected": pad_examples(rejected, pad_token_id, width),
    }
