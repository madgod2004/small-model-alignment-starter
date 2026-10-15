"""Build a blind pairwise comparison file: candidate (e.g. DPO) vs baseline (e.g. SFT).

    python scripts/build_pairs.py \
        --candidate outputs/generations/dpo_uf_alpaca_eval.json --candidate_name dpo_uf \
        --baseline outputs/generations/sft_alpaca_eval.json --baseline_name sft \
        --output outputs/eval/dpo_uf_vs_sft_alpaca_pairs.jsonl

Writes the pairs JSONL before any judging. Positions are exactly balanced and seeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.pairs import build_pairs  # noqa: E402
from alignment.utils import read_json, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="Generation JSON of the model being evaluated.")
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--baseline", required=True, help="Generation JSON of the reference model.")
    parser.add_argument("--baseline_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = build_pairs(read_json(args.candidate), read_json(args.baseline), args.candidate_name, args.baseline_name, args.seed)
    write_jsonl(args.output, pairs)
    in_a = sum(pair["hidden"]["candidate_position"] == "A" for pair in pairs)
    print(f"{args.output}: {len(pairs)} pairs, {args.candidate_name} in position A for {in_a}, B for {len(pairs) - in_a}")


if __name__ == "__main__":
    main()
