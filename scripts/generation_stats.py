"""Print repetition/length statistics for one or more generation JSON files.

    python scripts/generation_stats.py outputs/generations/sft_alpaca_eval.json outputs/generations/dpo_uf_alpaca_eval.json
    python scripts/generation_stats.py outputs/pilot/*.json --json_out outputs/pilot/summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.text_stats import format_table, summarize  # noqa: E402
from alignment.utils import read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    files = [f for f in args.files if not f.endswith(".meta.json") and not f.endswith("summary.json")]
    summaries = [(Path(f).stem, summarize(read_json(f))) for f in sorted(files)]
    print(format_table(summaries))
    if args.json_out:
        write_json(args.json_out, dict(summaries))


if __name__ == "__main__":
    main()
