"""Utilities for deterministic patient-level train/val/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_patient_splits(
    patient_ids: list[int],
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[int]]:
    """Create deterministic patient-level splits and write JSON files.

    Files written:
      - train_ids.json
      - val_ids.json
      - test_ids.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ids = np.array(sorted(set(int(pid) for pid in patient_ids)), dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, min(n_train, max(n - 2, 1))) if n >= 3 else max(1, n)
    n_val = max(1, min(n_val, max(n - n_train - 1, 1))) if n >= 3 else max(0, n - n_train)

    train = ids[:n_train].tolist()
    val = ids[n_train : n_train + n_val].tolist()
    test = ids[n_train + n_val :].tolist()
    if not test:
        # Guarantee non-empty test split when n is small.
        test = [val.pop()] if val else [train.pop()]

    split_map = {"train": train, "val": val, "test": test}
    for split_name, split_ids in split_map.items():
        (out / f"{split_name}_ids.json").write_text(
            json.dumps(split_ids, indent=2), encoding="utf-8"
        )
    return split_map


def load_patient_ids(split_dir: str, split: str) -> set[int]:
    """Load split IDs from split_dir/<split>_ids.json."""
    path = Path(split_dir) / f"{split}_ids.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Generate cohort to create train/val/test splits."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(x) for x in data}
