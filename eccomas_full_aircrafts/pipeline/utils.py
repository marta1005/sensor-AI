from __future__ import annotations

import json
from pathlib import Path

import numpy as np


RAW_FILENAMES = {
    "train": ("X_train.npy", "Ytrain.npy"),
    "test": ("X_test.npy", "Ytest.npy"),
}


def raw_paths(raw_data_dir: Path, split: str) -> tuple[Path, Path]:
    x_name, y_name = RAW_FILENAMES[split]
    return raw_data_dir / x_name, raw_data_dir / y_name


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def condition_start(raw_points_per_condition: int, condition_index: int) -> int:
    return int(condition_index * raw_points_per_condition)


def condition_stop(raw_points_per_condition: int, condition_index: int) -> int:
    return int((condition_index + 1) * raw_points_per_condition)


def sample_indices(n_rows: int, max_points: int, seed: int = 42) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def regime_from_mach(mach: np.ndarray, sub_max: float, trans_max: float) -> np.ndarray:
    labels = np.zeros(mach.shape[0], dtype=np.int64)
    labels[(mach >= sub_max) & (mach <= trans_max)] = 1
    labels[mach > trans_max] = 2
    return labels
