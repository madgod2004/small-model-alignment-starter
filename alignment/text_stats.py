"""Simple, deterministic repetition and length statistics for generated responses."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

WORD = re.compile(r"\w+|[^\w\s]")


def repeated_line_rate(text: str) -> float:
    """Fraction of non-empty lines that exactly repeat an earlier line (0 = no repeated lines)."""

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    seen, repeats = set(), 0
    for line in lines:
        # ignore list numbering so "3. Foo" and "4. Foo" count as the same line
        key = re.sub(r"^\d+[.)]\s*", "", line)
        repeats += key in seen
        seen.add(key)
    return repeats / len(lines)


def distinct_n(text: str, n: int = 3) -> float:
    """Unique word n-grams / total word n-grams (1 = no repeated n-grams)."""

    tokens = WORD.findall(text.lower())
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def summarize(rows: Sequence[dict]) -> dict:
    """Aggregate stats over generation entries (``output``, optional ``truncated``/``num_tokens``)."""

    outputs = [row["output"] for row in rows]
    count = max(len(rows), 1)
    line_rates = [repeated_line_rate(text) for text in outputs]
    return {
        "n": len(rows),
        "truncated_rate": sum(bool(row.get("truncated")) for row in rows) / count,
        "mean_tokens": sum(row.get("num_tokens", 0) for row in rows) / count,
        "mean_repeated_line_rate": sum(line_rates) / count,
        "frac_with_repeated_lines": sum(rate > 0 for rate in line_rates) / count,
        "mean_distinct_3": sum(distinct_n(text, 3) for text in outputs) / count,
        "empty_rate": sum(not text.strip() for text in outputs) / count,
    }


def format_table(named_summaries: Iterable[tuple[str, dict]]) -> str:
    header = f"{'run':<34}{'n':>4}{'trunc':>7}{'tokens':>8}{'rep_line':>9}{'any_rep':>8}{'dist3':>7}"
    lines = [header, "-" * len(header)]
    for name, s in named_summaries:
        lines.append(
            f"{name:<34}{s['n']:>4}{s['truncated_rate']:>7.2f}{s['mean_tokens']:>8.0f}"
            f"{s['mean_repeated_line_rate']:>9.3f}{s['frac_with_repeated_lines']:>8.2f}{s['mean_distinct_3']:>7.3f}"
        )
    return "\n".join(lines)
