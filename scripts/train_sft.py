"""Train the SFT model (Task 3).

Two T4s (DDP, accumulation derived as 8 so the effective batch stays 32):
    torchrun --nproc_per_node 2 scripts/train_sft.py --config configs/sft.yaml

One GPU (accumulation 16):
    python scripts/train_sft.py --config configs/sft.yaml

Any config key can be overridden with key=value, e.g.:
    python scripts/train_sft.py --config configs/sft.yaml max_steps=20 train_limit=640 eval_steps=10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.sft import train_sft  # noqa: E402
from alignment.training import cleanup_distributed, setup_distributed  # noqa: E402
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
    info = setup_distributed()
    try:
        train_sft(config, info)
    finally:
        cleanup_distributed(info)


if __name__ == "__main__":
    main()
