"""Judge blind pairwise files with a Kaggle model-proxy LLM (Task 7).

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash \
        --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

Reliability behaviour:
  - reads saved pairwise JSONL only (never regenerates outputs);
  - judges `batch_size` pairs per request, in a fresh chat each time (no history carried over);
  - appends judgments to the output JSONL after every batch;
  - waits at least `sleep_seconds` before every request attempt;
  - retries a batch up to `max_retries` times (only the still-unjudged pairs are re-sent);
  - logs unresolved batches to `<output>_failures.jsonl`;
  - with --reuse_judgments, skips pairs already in the output file;
  - writes `<output>_summary.json` with win/loss/tie/unknown/failure rates.

`--backend mock` runs the whole pipeline offline with a deterministic fake judge (for testing only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judging import (  # noqa: E402
    RUBRICS,
    SYSTEM_PROMPT,
    build_batch_prompt,
    make_judgment,
    parse_batch_response,
    rubric_for_dataset,
    summarize_judgments,
)
from alignment.utils import append_jsonl, read_jsonl, write_json  # noqa: E402


class KaggleJudge:
    """Calls a model through kaggle_benchmarks; each call runs in an isolated chat."""

    def __init__(self, model_name: str):
        import kaggle_benchmarks as kbench

        self.kbench = kbench
        if model_name in kbench.llms:
            self.llm = kbench.llms[model_name]
        else:
            available = sorted(kbench.llms)
            raise SystemExit(f"Judge model {model_name!r} is not available here. Available models: {available}")
        self.model_name = model_name

    def __call__(self, system: str, prompt: str) -> str:
        with self.kbench.chats.new(name="pairwise-judge", system_instructions=system, orphan=True):
            return self.llm.prompt(prompt, temperature=0, seed=0)


class MockJudge:
    """Offline stand-in: prefers the response with fewer repeated lines, then the shorter one."""

    model_name = "mock-judge"

    def __call__(self, system: str, prompt: str) -> str:
        import re

        from alignment.text_stats import repeated_line_rate

        blocks = re.findall(r"\[Response A\]\n(.*?)\n\n\[Response B\]\n(.*?)(?=\n\n### Comparison|\n\nReturn a JSON list)",
                            prompt, flags=re.DOTALL)
        criteria = re.search(r'"comparison": 1, (.*?), "reason"', prompt).group(1)
        names = re.findall(r'"(\w+)": "A"', criteria)
        items = []
        for number, (a, b) in enumerate(blocks, start=1):
            score_a, score_b = (repeated_line_rate(a), len(a)), (repeated_line_rate(b), len(b))
            winner = "tie" if score_a == score_b else ("A" if score_a < score_b else "B")
            items.append({"comparison": number, **{n: winner for n in names}, "reason": "mock"})
        return "```json\n" + json.dumps(items) + "\n```"


def run_judging(todo, judge, rubric, output_path, failures_path, batch_size, sleep_seconds, max_retries, total=None):
    """Judge `todo` pairs in batches with retries, appending results after every batch."""

    first_request = True
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        remaining, attempts, last_error = list(batch), 0, None
        last_parsed: dict[str, dict | None] = {}
        while remaining and attempts < max_retries:
            if not first_request:
                time.sleep(sleep_seconds)  # at least sleep_seconds between request attempts
            first_request = False
            attempts += 1
            try:
                text = judge(SYSTEM_PROMPT, build_batch_prompt(remaining, rubric))
                parsed = parse_batch_response(text, len(remaining), rubric)
            except Exception as error:  # noqa: BLE001 - network, proxy, quota, or parse errors are all retried
                last_error = f"{type(error).__name__}: {error}"[:500]
                print(f"  batch {start // batch_size + 1}: attempt {attempts} failed: {last_error}", flush=True)
                continue
            still = []
            for pair, result in zip(remaining, parsed):
                complete = result is not None and all(v is not None for v in result["verdicts"].values())
                if complete:
                    append_jsonl(output_path, make_judgment(pair, result, rubric, judge.model_name, attempts))
                else:
                    last_parsed[pair["pair_id"]] = result
                    still.append(pair)
            remaining = still

        unknown = failed = 0
        for pair in remaining:
            if pair["pair_id"] in last_parsed:
                unknown += 1
                # The judge answered but gave no valid verdict for this pair: record it as "unknown".
                append_jsonl(output_path, make_judgment(pair, last_parsed[pair["pair_id"]], rubric, judge.model_name, attempts))
            else:
                failed += 1
                append_jsonl(failures_path, {"pair_id": pair["pair_id"], "attempts": attempts, "error": last_error,
                                             "time": datetime.now(timezone.utc).isoformat()})
        judged_so_far = len(read_jsonl(output_path))
        print(f"[batch {start // batch_size + 1}/{-(-len(todo) // batch_size)}] judged {judged_so_far}/{total or len(todo)}"
              + (f" ({unknown} recorded as unknown)" if unknown else "")
              + (f" ({failed} failed after {attempts} attempts, logged to {failures_path})" if failed else ""), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3, help="Attempts per batch.")
    parser.add_argument("--reuse_judgments", action="store_true", help="Skip pairs already judged in output_path.")
    parser.add_argument("--rubric", choices=sorted(RUBRICS), default=None, help="Default: chosen from the pairs' dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N pairs (smoke test).")
    parser.add_argument("--backend", choices=["kaggle", "mock"], default="kaggle")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    stem = output_path.with_suffix("")
    failures_path = Path(f"{stem}_failures.jsonl")
    summary_path = Path(f"{stem}_summary.json")

    pairs = read_jsonl(args.pairs_path)[: args.limit]
    rubric = args.rubric or rubric_for_dataset(pairs[0].get("dataset"))
    if not args.reuse_judgments:
        for path in (output_path, failures_path):
            if path.exists():
                path.unlink()
    done = {j["pair_id"] for j in read_jsonl(output_path)}
    todo = [pair for pair in pairs if pair["pair_id"] not in done]
    judge = MockJudge() if args.backend == "mock" else KaggleJudge(args.judge_model)
    print(f"{len(pairs)} pairs, {len(done)} already judged, {len(todo)} to judge | rubric={rubric} | judge={judge.model_name}",
          flush=True)

    run_judging(todo, judge, rubric, output_path, failures_path, args.batch_size, args.sleep_seconds,
                args.max_retries, total=len(pairs))

    judgments = [j for j in read_jsonl(output_path) if j["pair_id"] in {p["pair_id"] for p in pairs}]
    summary = summarize_judgments(judgments, len(pairs), read_jsonl(failures_path))
    summary.update({"pairs_path": args.pairs_path, "rubric": rubric, "batch_size": args.batch_size,
                    "sleep_seconds": args.sleep_seconds, "max_retries": args.max_retries})
    write_json(summary_path, summary)
    for criterion, stats in summary["criteria"].items():
        print(f"{criterion}: {stats['wins']}W / {stats['losses']}L / {stats['ties']}T / {stats['unknowns']}U / "
              f"{stats['failures']} failed | win rate excl. ties {stats['win_rate_excluding_ties']} | "
              f"ties as half {stats['win_rate_ties_as_half']}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
