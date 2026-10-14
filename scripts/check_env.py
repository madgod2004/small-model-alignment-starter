"""Task 1 environment check: GPUs, package versions, and repo importability.

Usage (from the repository root, on Kaggle or locally):
    python scripts/check_env.py
    python scripts/check_env.py --expect_gpus 2
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Make the repo importable even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "tokenizers", "yaml"]


def package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 - report any import failure
        return f"NOT AVAILABLE ({type(error).__name__})"
    return getattr(module, "__version__", "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect_gpus", type=int, default=0, help="Fail if fewer CUDA GPUs are visible.")
    args = parser.parse_args()

    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    for name in PACKAGES:
        print(f"{name:<12}: {package_version(name)}")

    if shutil.which("nvidia-smi"):
        print("\n$ nvidia-smi")
        subprocess.run(["nvidia-smi"], check=False)

    gpu_count = 0
    try:
        import torch

        print(f"\nCUDA available: {torch.cuda.is_available()} (torch CUDA build: {torch.version.cuda})")
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for index in range(gpu_count):
            props = torch.cuda.get_device_properties(index)
            print(f"  cuda:{index} {props.name} {props.total_memory / 2**30:.1f} GiB")
    except ImportError:
        print("\ntorch is not installed; skipping CUDA checks.")

    try:
        import alignment  # noqa: F401
        import submission_adapter  # noqa: F401

        print("\nRepository imports OK (alignment, submission_adapter).")
    except Exception as error:  # noqa: BLE001
        print(f"\nRepository import FAILED: {error!r}")
        return 1

    if gpu_count < args.expect_gpus:
        print(f"\nERROR: expected at least {args.expect_gpus} GPUs, found {gpu_count}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
