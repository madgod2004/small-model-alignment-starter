"""Shared helpers: seeding, config loading, and durable JSON/JSONL I/O."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if available), and PyTorch (CPU and all GPUs)."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | os.PathLike, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply flat key overrides (ignoring None values)."""

    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return config


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike, data: Any) -> None:
    """Write JSON atomically so an interrupted Kaggle session never leaves a half file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def append_jsonl(path: str | os.PathLike, row: dict[str, Any]) -> None:
    """Append one record and flush, e.g. a metrics line after each evaluation step."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
