"""Blind pairwise comparison files (Task 6).

Each record shows the judge only ``instruction``, ``response_a`` and ``response_b``. Which model
produced which response is stored under ``hidden`` and is used only after judging. Positions are
exactly balanced: a seeded shuffle puts the candidate in position A for half of the pairs.
"""

from __future__ import annotations

import random
from typing import Sequence

JUDGE_FIELDS = ("pair_id", "instruction", "response_a", "response_b")


def build_pairs(
    candidate_rows: Sequence[dict],
    baseline_rows: Sequence[dict],
    candidate_name: str,
    baseline_name: str,
    seed: int,
) -> list[dict]:
    """Pair two generation files row by row (same ids, same instructions) with balanced, hidden positions."""

    baseline_by_id = {row["id"]: row for row in baseline_rows}
    missing = [row["id"] for row in candidate_rows if row["id"] not in baseline_by_id]
    if missing or len(candidate_rows) != len(baseline_rows):
        raise ValueError(f"Generation files do not cover the same prompts (e.g. missing {missing[:3]})")

    count = len(candidate_rows)
    candidate_first = [True] * (count // 2) + [False] * (count - count // 2)
    random.Random(seed).shuffle(candidate_first)

    pairs = []
    for row, first in zip(candidate_rows, candidate_first):
        other = baseline_by_id[row["id"]]
        if row["instruction"] != other["instruction"]:
            raise ValueError(f"Instruction mismatch for id {row['id']}")
        a, b = (row, other) if first else (other, row)
        record = {
            "pair_id": row["id"],  # neutral: no model names anywhere in the judge-facing fields
            "id": row["id"],
            "dataset": row.get("dataset"),
            "instruction": row["instruction"],
            "response_a": a["output"],
            "response_b": b["output"],
            "hidden": {
                "candidate": candidate_name,
                "baseline": baseline_name,
                "candidate_position": "A" if first else "B",
                "model_a": candidate_name if first else baseline_name,
                "model_b": baseline_name if first else candidate_name,
                "generator_a": a.get("generator"),
                "generator_b": b.get("generator"),
            },
        }
        if "category" in row:
            record["category"] = row["category"]
        pairs.append(record)
    return pairs


def judge_view(pair: dict) -> dict:
    """The only fields a judge prompt may use (no model identities)."""

    return {key: pair[key] for key in JUDGE_FIELDS}
