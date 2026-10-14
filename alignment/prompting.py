"""The single prompt template shared by SFT, DPO, and evaluation generation.

We use Qwen's ChatML layout with a fixed system message, built as a plain string
(not ``tokenizer.apply_chat_template``) so the exact text is stable across
tokenizer versions and easy to document in the report. The response that follows
the prompt is the raw assistant text; the batching code appends the tokenizer's
EOS token (``<|endoftext|>`` for Qwen2.5) to mark the end of the turn.
"""

from __future__ import annotations

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def format_prompt(instruction: str, system: str = DEFAULT_SYSTEM_MESSAGE) -> str:
    """Wrap a single-turn user instruction in the chat template, ending at the assistant turn."""

    return PROMPT_TEMPLATE.format(system=system, instruction=instruction.strip())
