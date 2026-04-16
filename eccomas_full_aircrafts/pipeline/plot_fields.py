from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import FullAircraftConfig
from .utils import condition_start, condition_stop, raw_paths, save_json, sample_indices

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _reference_surface_path(cfg: FullAircraftConfig, surface: str) -> Path:
    path = cfg.surfaces_dir / f"{surface}_surface_reference.npz"
    if not path.exists():
        raise FileNotFoundError(f"Reference surface not found: {path}. Run 'prepare-reference-surface' first.")
    return path


def _condition_records(cfg: FullAircraftConfig, split: str) -> list[dict[str, float]]:
    x_path, _ = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
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


def _surface_grid(ref: np.lib.npyio.NpzFile, values: np.ndarray) -> np.ma.MaskedArray:
    x_bins = int(np.asarray(ref["x_edges"]).shape[0] - 1)
    y_bins = int(np.asarray(ref["y_edges"]).shape[0] - 1)
    grid = np.full((y_bins, x_bins), np.nan, dtype=np.float32)
    x_bin = np.asarray(ref["x_bin"], dtype=np.int64)
    y_bin = np.asarray(ref["y_bin"], dtype=np.int64)
    grid[y_bin, x_bin] = values.astype(np.float32, copy=False)
    return np.ma.masked_invalid(grid)


def generate_raw_cp_field_plots(
    cfg: FullAircraftConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    surface: str = "upper",
    max_plotted_points: int = 60_000,
    mode: str = "points",
) -> dict:
    cfg.ensure_dirs()
    result_dir = cfg.results_surface_dir(surface)
    result_dir.mkdir(parents=True, exist_ok=True)
    ref = np.load(_reference_surface_path(cfg, surface))
    local_idx = np.asarray(ref["local_idx"], dtype=np.int64)
    x_ref = np.asarray(ref["x"], dtype=np.float32)
    y_ref = np.asarray(ref["y"], dtype=np.float32)
    x_edges = np.asarray(ref["x_edges"], dtype=np.float32)
    y_edges = np.asarray(ref["y_edges"], dtype=np.float32)

    x_path, y_path = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    y_raw = np.load(y_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
    condition_indices = list(range(n_conditions)) if condition_indices is None else [int(idx) for idx in condition_indices]
    records = _condition_records(cfg, split)

    plot_idx = sample_indices(local_idx.size, min(max_plotted_points, local_idx.size), seed=17)
    selected_local = local_idx[plot_idx]
    selected_x = x_ref[plot_idx]
    selected_y = y_ref[plot_idx]

    summary: list[dict[str, float | str]] = []
    for cond_idx in condition_indices:
        start = condition_start(cfg.raw_points_per_condition, cond_idx)
        stop = condition_stop(cfg.raw_points_per_condition, cond_idx)
        cp_block = np.asarray(y_raw[start:stop, cfg.cp_column], dtype=np.float32)
        cp_surface = cp_block[selected_local]
        cp_full_surface = cp_block[local_idx]

        rec = records[cond_idx]
        fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
        if mode == "surface":
            cp_grid = _surface_grid(ref, cp_full_surface)
            sc = ax.pcolormesh(x_edges, y_edges, cp_grid, cmap="jet", shading="flat")
            plotted_count = int(cp_full_surface.size)
        elif mode == "points":
            sc = ax.scatter(selected_x, selected_y, c=cp_surface, s=2.6, cmap="jet", linewidths=0)
            plotted_count = int(selected_local.size)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Full-aircraft {surface} {mode} Cp | cond {cond_idx} | "
            f"Mach={rec['Mach']:.2f}, AoA={rec['AoA_deg']:.1f}, Pi={rec['Pi']:.1f}"
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.9)
        cb.set_label("Cp")
        out_path = result_dir / f"cp_full_aircraft_{surface}_{mode}_cond{cond_idx}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        summary.append(
            {
                "condition_index": int(cond_idx),
                "surface": surface,
                "mode": mode,
                "Mach": float(rec["Mach"]),
                "AoA_deg": float(rec["AoA_deg"]),
                "Pi": float(rec["Pi"]),
                "points_plotted": plotted_count,
                "cp_min": float(cp_full_surface.min()),
                "cp_max": float(cp_full_surface.max()),
                "path": str(out_path),
            }
        )

    payload = {
        "split": split,
        "surface": surface,
        "mode": mode,
        "n_conditions": len(summary),
        "results": summary,
    }
    save_json(result_dir / f"cp_full_aircraft_{surface}_{mode}_{split}_summary.json", payload)
    return payload
