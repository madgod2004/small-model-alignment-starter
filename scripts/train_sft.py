"""Train the SFT model (Task 3) on a single GPU.

    python scripts/train_sft.py --config configs/sft.yaml

Any config key can be overridden with key=value, e.g.:
    python scripts/train_sft.py --config configs/sft.yaml max_steps=20 train_limit=640 eval_steps=10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reduce CUDA memory fragmentation (set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.sft import train_sft  # noqa: E402
from alignment.utils import load_config, parse_value  # noqa: E402


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"Override must be key=value, got {pair!r}")
        overrides[key] = parse_value(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    config = load_config(args.config)
    config.update(parse_overrides(args.overrides))  # explicit overrides may also set null
    train_sft(config)


if __name__ == "__main__":
    main()
