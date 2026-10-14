"""Parsing and deterministic subsampling of the SFT, preference, and evaluation datasets.

Every record stores the *raw* user instruction as ``prompt``; the chat template from
``alignment.prompting`` is applied later at batching/generation time, so the saved
subsets do not depend on the template.

Record schemas:
    SFT:        {"id", "source_index", "prompt", "response"}
    Preference: {"id", "source_index", "prompt", "chosen", "rejected"}
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable

HH_HUMAN = "\n\nHuman:"
HH_ASSISTANT = "\n\nAssistant:"


# ---------------------------------------------------------------------------
# Row parsers: return a parsed dict, or a short string naming why the row was dropped.
# ---------------------------------------------------------------------------


def parse_ultrachat(row: dict[str, Any]) -> dict[str, str] | str:
    """First user/assistant turn of an UltraChat conversation."""

    messages = row.get("messages") or []
    if len(messages) < 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        return "bad_roles"
    prompt, response = messages[0]["content"].strip(), messages[1]["content"].strip()
    if not prompt or not response:
        return "empty"
    return {"prompt": prompt, "response": response}


def split_hh_transcript(text: str) -> tuple[str, str] | None:
    """Split a single-turn ``Human: ... Assistant: ...`` transcript into (prompt, response)."""

    if text.count(HH_HUMAN) != 1 or text.count(HH_ASSISTANT) != 1:
        return None
    human_start, assistant_start = text.find(HH_HUMAN), text.find(HH_ASSISTANT)
    if human_start > assistant_start:
        return None
    prompt = text[human_start + len(HH_HUMAN) : assistant_start].strip()
    response = text[assistant_start + len(HH_ASSISTANT) :].strip()
    return prompt, response


def parse_hh_rlhf(row: dict[str, Any]) -> dict[str, str] | str:
    """Single-turn HH-RLHF pair; multi-turn conversations are dropped."""

    chosen, rejected = split_hh_transcript(row["chosen"]), split_hh_transcript(row["rejected"])
    if chosen is None or rejected is None:
        return "multi_turn"
    if chosen[0] != rejected[0]:
        return "prompt_mismatch"
    prompt, chosen_response, rejected_response = chosen[0], chosen[1], rejected[1]
    if not prompt or not chosen_response or not rejected_response:
        return "empty"
    if chosen_response == rejected_response:
        return "identical"
    return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}


def parse_ultrafeedback(row: dict[str, Any]) -> dict[str, str] | str:
    """Single-turn UltraFeedback pair with a strict score preference."""

    chosen, rejected = row.get("chosen") or [], row.get("rejected") or []
    if len(chosen) != 2 or len(rejected) != 2:
        return "multi_turn"
    if [m.get("role") for m in chosen] != ["user", "assistant"] or [m.get("role") for m in rejected] != ["user", "assistant"]:
        return "bad_roles"
    prompt = row["prompt"].strip()
    if chosen[0]["content"].strip() != prompt or rejected[0]["content"].strip() != prompt:
        return "prompt_mismatch"
    if not row["score_chosen"] > row["score_rejected"]:
        return "score_tie"
    chosen_response, rejected_response = chosen[1]["content"].strip(), rejected[1]["content"].strip()
    if not prompt or not chosen_response or not rejected_response:
        return "empty"
    if chosen_response == rejected_response:
        return "identical"
    return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}


# ---------------------------------------------------------------------------
# Deterministic subsampling
# ---------------------------------------------------------------------------


def select_examples(
    rows: Any,
    parser: Callable[[dict[str, Any]], dict[str, str] | str],
    num_examples: int,
    seed: int,
    id_prefix: str,
    keep: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Visit rows in a seeded random order, keep the first ``num_examples`` valid ones.

    ``rows`` must support ``len()`` and integer indexing (a list or HF Dataset).
    ``keep`` is an optional extra filter (e.g. prompt token length). Returns the
    selected records and a count of drop reasons.
    """

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    selected: list[dict[str, Any]] = []
    stats: dict[str, int] = {"visited": 0}
    for source_index in order:
        if len(selected) >= num_examples:
            break
        stats["visited"] += 1
        parsed = parser(rows[source_index])
        if isinstance(parsed, str):
            stats[parsed] = stats.get(parsed, 0) + 1
            continue
        if keep is not None and not keep(parsed):
            stats["too_long_prompt"] = stats.get("too_long_prompt", 0) + 1
            continue
        selected.append({"id": f"{id_prefix}-{len(selected):05d}", "source_index": source_index, **parsed})
    stats["selected"] = len(selected)
    return selected, stats


def sample_eval_subset(
    rows: Iterable[dict[str, Any]], num_prompts: int, seed: int, id_prefix: str, dataset_name: str
) -> list[dict[str, Any]]:
    """Fixed evaluation subset: ``random.Random(seed).sample`` over row indices, kept in index order."""

    rows = list(rows)
    indices = sorted(random.Random(seed).sample(range(len(rows)), num_prompts))
    return [
        {
            "id": f"{id_prefix}-{position:04d}",
            "instruction": rows[index]["instruction"],
            "dataset": dataset_name,
            "source_index": index,
            "source_subset": rows[index].get("dataset"),
        }
        for position, index in enumerate(indices)
    ]
