"""Response-only log-probabilities, the SFT loss, and the DPO loss."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from alignment.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the causal LM; labels are deliberately not passed so the model computes no loss of its own."""

    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return outputs.logits if hasattr(outputs, "logits") else outputs[0]


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of log p(token) over unmasked label positions, per row.

    Logits at position t predict the token at t+1, so logits and labels are shifted
    by one. Returns ``(summed_logps [B], token_counts [B])`` in float32.
    """

    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:].to(logits.device)
    mask = labels.ne(IGNORE_INDEX)
    safe_labels = labels.masked_fill(~mask, 0)

    # log_softmax(x)[y] = x[y] - logsumexp(x); avoids materialising a second [B, T, V] tensor.
    token_logits = torch.gather(logits, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logits - torch.logsumexp(logits, dim=-1)
    token_logps = token_logps * mask

    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def batch_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = to_device(batch, model_device(model))
    return sequence_logprobs(forward_logits(model, batch), batch["labels"])


def sft_loss_from_batch(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean negative log-likelihood over all unmasked response tokens in the batch.

    Returns ``(loss, num_response_tokens)``; the token count lets the training loop
    weight gradient-accumulation micro-batches correctly.
    """

    summed_logps, token_counts = batch_logprobs(model, batch)
    num_tokens = token_counts.sum()
    return -summed_logps.sum() / num_tokens.clamp_min(1), num_tokens


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    batch = build_sft_batch(tokenizer, list(prompts), list(responses), max_length)
    summed_logps, _ = batch_logprobs(model, batch)
    return summed_logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    batch = build_sft_batch(tokenizer, list(prompts), list(responses), max_length)
    loss, _ = sft_loss_from_batch(model, batch)
    return loss


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Per-example DPO loss -log sigmoid(beta * (Δθ - Δref)) plus implicit-reward diagnostics.

    Reference log-probs may live on another device or dtype (e.g. fp16 on cuda:1);
    everything is moved to the policy's device in float32.
    """

    policy_chosen_logps = torch.as_tensor(policy_chosen_logps)
    policy_rejected_logps = torch.as_tensor(policy_rejected_logps)
    reference_chosen_logps = torch.as_tensor(reference_chosen_logps)
    reference_rejected_logps = torch.as_tensor(reference_rejected_logps)

    device = policy_chosen_logps.device
    policy_chosen = policy_chosen_logps.float()
    policy_rejected = policy_rejected_logps.float().to(device)
    reference_chosen = reference_chosen_logps.detach().float().to(device)
    reference_rejected = reference_rejected_logps.detach().float().to(device)

    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen - reference_rejected
    losses = -F.logsigmoid(beta * (policy_gap - reference_gap))

    chosen_rewards = (beta * (policy_chosen - reference_chosen)).detach()
    rejected_rewards = (beta * (policy_rejected - reference_rejected)).detach()
    return {
        "loss": losses.mean(),
        "losses": losses,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_accuracy": (chosen_rewards > rejected_rewards).float(),  # strict: ties count as wrong
        "reward_margin": chosen_rewards - rejected_rewards,
    }


def concatenated_logprobs(model, dpo_batch: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Score chosen and rejected rows in one forward pass; returns (chosen_logps, rejected_logps)."""

    chosen, rejected = dpo_batch["chosen"], dpo_batch["rejected"]
    num_rows = chosen["input_ids"].shape[0]
    joint = {key: torch.cat([chosen[key], rejected[key]], dim=0) for key in chosen}
    summed_logps, _ = batch_logprobs(model, joint)
    return summed_logps[:num_rows], summed_logps[num_rows:]


def dpo_forward(policy_model, reference_model, dpo_batch, beta: float) -> dict[str, torch.Tensor]:
    """DPO loss and diagnostics for a pre-built batch; the reference runs without gradients."""

    policy_chosen, policy_rejected = concatenated_logprobs(policy_model, dpo_batch)
    with torch.no_grad():
        reference_chosen, reference_rejected = concatenated_logprobs(reference_model, dpo_batch)
    return dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    beta: float,
    max_length: int,
) -> torch.Tensor:
    dpo_batch = build_dpo_batch(
        tokenizer, list(prompts), list(chosen_responses), list(rejected_responses), max_length
    )
    return dpo_forward(policy_model, reference_model, dpo_batch, beta)["loss"]
