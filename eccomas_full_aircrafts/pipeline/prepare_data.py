from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .config import FullAircraftConfig
from .utils import condition_start, condition_stop, raw_paths, save_json


def reduced_paths(cfg: FullAircraftConfig, split: str) -> tuple[Path, Path]:
    return cfg.reduced_data_dir / f"X_cut_{split}.npy", cfg.reduced_data_dir / f"Y_cut_{split}.npy"


def _condition_table(x_raw: np.memmap, cfg: FullAircraftConfig, n_conditions: int) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for idx in range(n_conditions):
        row = np.asarray(x_raw[idx * cfg.raw_points_per_condition, 6:9], dtype=np.float32)
        records.append(
            {
                "condition_index": int(idx),
                "Mach": float(row[0]),
                "AoA_deg": float(row[1]),
                "Pi": float(row[2]),
            }
        )
    return records


def prepare_reduced_data(
    cfg: FullAircraftConfig,
    surface: str | None = None,
    splits: tuple[str, ...] = ("train", "test"),
    max_conditions: int | None = None,
) -> dict:
    cfg.ensure_dirs()
    surface_name = surface or cfg.reduced_surface
    reference_path = cfg.surface_reference_path(surface_name)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference surface not found: {reference_path}. Run 'prepare-reference-surface' first.")

    ref = np.load(reference_path)
    local_idx = np.asarray(ref["local_idx"], dtype=np.int64)
    x_ref = np.asarray(ref["x"], dtype=np.float32)
    y_ref = np.asarray(ref["y"], dtype=np.float32)
    z_ref = np.asarray(ref["z"], dtype=np.float32)

    payload: dict[str, object] = {
        "surface": surface_name,
        "reference_path": str(reference_path),
        "points_per_condition_reduced": int(local_idx.size),
        "reference_ranges": {
            "x": [float(x_ref.min()), float(x_ref.max())],
            "y": [float(y_ref.min()), float(y_ref.max())],
            "z": [float(z_ref.min()), float(z_ref.max())],
        },
        "splits": {},
    }

    for split in splits:
        x_path, y_path = raw_paths(cfg.raw_data_dir, split)
        x_raw = np.load(x_path, mmap_mode="r")
        y_raw = np.load(y_path, mmap_mode="r")
        n_conditions_total = int(x_raw.shape[0] // cfg.raw_points_per_condition)
        n_conditions = n_conditions_total if max_conditions is None else min(int(max_conditions), n_conditions_total)

        x_out_path, y_out_path = reduced_paths(cfg, split)
        y_dim = int(y_raw.shape[1])
        out_rows = int(n_conditions * local_idx.size)
        x_out = open_memmap(x_out_path, mode="w+", dtype=np.float32, shape=(out_rows, cfg.input_dim_raw))
        y_out = open_memmap(y_out_path, mode="w+", dtype=np.float32, shape=(out_rows, y_dim))

        for cond_idx in range(n_conditions):
            start = condition_start(cfg.raw_points_per_condition, cond_idx)
            stop = condition_stop(cfg.raw_points_per_condition, cond_idx)
            row_start = cond_idx * local_idx.size
            row_stop = row_start + local_idx.size

            x_block = np.asarray(x_raw[start:stop, : cfg.input_dim_raw], dtype=np.float32)
            y_block = np.asarray(y_raw[start:stop], dtype=np.float32)
            x_out[row_start:row_stop] = x_block[local_idx]
            y_out[row_start:row_stop] = y_block[local_idx]

            if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
                print(f"[prepare-reduced-data] {split}: condition {cond_idx + 1}/{n_conditions}")

        split_payload = {
            "conditions_total": n_conditions_total,
            "conditions_written": n_conditions,
            "points_per_condition_raw": int(cfg.raw_points_per_condition),
            "points_per_condition_reduced": int(local_idx.size),
            "rows_written": out_rows,
            "x_shape": [out_rows, cfg.input_dim_raw],
            "y_shape": [out_rows, y_dim],
            "x_path": str(x_out_path),
            "y_path": str(y_out_path),
            "conditions": _condition_table(x_raw, cfg, n_conditions),
        }
        payload["splits"][split] = split_payload
        save_json(cfg.metadata_dir / f"reduced_{surface_name}_{split}_summary.json", split_payload)

    save_json(cfg.metadata_dir / f"reduced_{surface_name}_summary.json", payload)
    return payload
