"""Train a from-scratch DPO model (Task 4): policy on cuda:0, frozen reference on cuda:1.

    python scripts/train_dpo.py --config configs/dpo_hh.yaml
    python scripts/train_dpo.py --config configs/dpo_uf.yaml

Any config key can be overridden with key=value, e.g.:
    python scripts/train_dpo.py --config configs/dpo_hh.yaml max_steps=10 val_limit=64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reduce CUDA memory fragmentation (set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.dpo import train_dpo  # noqa: E402
from alignment.utils import apply_overrides, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    train_dpo(apply_overrides(load_config(args.config), args.overrides))


if __name__ == "__main__":
    main()
