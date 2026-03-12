from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def regime_from_mach(mach: np.ndarray, sub_max: float, trans_max: float) -> np.ndarray:
    labels = np.zeros(mach.shape[0], dtype=np.int64)
    labels[(mach >= sub_max) & (mach <= trans_max)] = 1
    labels[mach > trans_max] = 2
    return labels


def sample_indices(n_rows: int, max_points: int, seed: int = 42) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))
