"""Build eval_sets/decoding_pilot_20.json: held-out prompts for choosing decoding settings.

None of these prompts are in the evaluation sets, so the decoding choice is not tuned on the
prompts the models are judged on:
  - 12 AlpacaEval prompts that are NOT in eval_sets/alpaca_eval_300_seed42.json
  - 8 single-turn HH-RLHF test prompts that are NOT in our DPO validation subset (data/manifests/hh.json)

    python scripts/make_decoding_pilot_set.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alignment.data import parse_hh_rlhf  # noqa: E402
from alignment.utils import read_json, write_json  # noqa: E402

SEED = 42


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    alpaca_rows = read_json(hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset"))
    used = {row["source_index"] for row in read_json(REPO_ROOT / "eval_sets" / "alpaca_eval_300_seed42.json")}
    unused = [i for i in range(len(alpaca_rows)) if i not in used]
    alpaca_pick = sorted(random.Random(SEED).sample(unused, 12))

    hh_val_indices = set(read_json(REPO_ROOT / "data" / "manifests" / "hh.json")["splits"]["val"]["source_indices"])
    hh_test = load_dataset("Anthropic/hh-rlhf", split="test")
    order = list(range(len(hh_test)))
    random.Random(SEED).shuffle(order)
    hh_pick = []
    for index in order:
        if index in hh_val_indices:
            continue
        parsed = parse_hh_rlhf(hh_test[index])
        if isinstance(parsed, dict) and len(parsed["prompt"]) < 400:
            hh_pick.append((index, parsed["prompt"]))
        if len(hh_pick) == 8:
            break

    rows = [
        {"id": f"pilot-alpaca-{i:02d}", "instruction": alpaca_rows[index]["instruction"], "dataset": "decoding_pilot",
         "source": "alpaca_eval", "source_index": index}
        for i, index in enumerate(alpaca_pick)
    ] + [
        {"id": f"pilot-hh-{i:02d}", "instruction": prompt, "dataset": "decoding_pilot",
         "source": "hh_rlhf_test", "source_index": index}
        for i, (index, prompt) in enumerate(hh_pick)
    ]
    write_json(REPO_ROOT / "eval_sets" / "decoding_pilot_20.json", rows)
    print(f"wrote {len(rows)} pilot prompts")


if __name__ == "__main__":
    main()
