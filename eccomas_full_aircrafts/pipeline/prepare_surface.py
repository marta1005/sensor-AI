from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import FullAircraftConfig
from .utils import condition_start, condition_stop, raw_paths, save_json

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def inspect_raw_dataset(cfg: FullAircraftConfig) -> dict:
    cfg.ensure_dirs()
    payload: dict[str, dict] = {}
    for split in ["train", "test"]:
        x_path, y_path = raw_paths(cfg.raw_data_dir, split)
        x_raw = np.load(x_path, mmap_mode="r")
        y_raw = np.load(y_path, mmap_mode="r")
        n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
        first_block = np.asarray(x_raw[: cfg.raw_points_per_condition, : cfg.input_dim_raw], dtype=np.float32)
        payload[split] = {
            "x_shape": list(x_raw.shape),
            "y_shape": list(y_raw.shape),
            "x_dtype": str(x_raw.dtype),
            "y_dtype": str(y_raw.dtype),
            "conditions": n_conditions,
            "points_per_condition": cfg.raw_points_per_condition,
            "x_min": np.min(first_block[:, 0:3], axis=0).astype(float).tolist(),
            "x_max": np.max(first_block[:, 0:3], axis=0).astype(float).tolist(),
            "mach_values": np.unique(np.asarray(x_raw[:: cfg.raw_points_per_condition, 6], dtype=np.float32)).astype(float).tolist(),
            "aoa_values": np.unique(np.asarray(x_raw[:: cfg.raw_points_per_condition, 7], dtype=np.float32)).astype(float).tolist(),
            "pi_values": np.unique(np.asarray(x_raw[:: cfg.raw_points_per_condition, 8], dtype=np.float32)).astype(float).tolist(),
        }
    save_json(cfg.metadata_dir / "raw_dataset_summary.json", payload)
    return payload


def _bin_ids(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, x_edges.size - 2)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, y_edges.size - 2)
    cell_id = x_bin.astype(np.int64) * (y_edges.size - 1) + y_bin.astype(np.int64)
    return x_bin, y_bin, cell_id


def _surface_indices_by_envelope(xyz: np.ndarray, x_bins: int, y_bins: int) -> dict[str, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_edges = np.linspace(float(x.min()), float(x.max()), x_bins + 1, dtype=np.float32)
    y_edges = np.linspace(float(y.min()), float(y.max()), y_bins + 1, dtype=np.float32)
    x_bin, y_bin, cell_id = _bin_ids(x, y, x_edges, y_edges)

    upper: dict[int, int] = {}
    lower: dict[int, int] = {}
    for local_idx, cid in enumerate(cell_id.tolist()):
        prev_upper = upper.get(cid)
        prev_lower = lower.get(cid)
        if prev_upper is None or z[local_idx] > z[prev_upper]:
            upper[cid] = local_idx
        if prev_lower is None or z[local_idx] < z[prev_lower]:
            lower[cid] = local_idx

    def _pack(selected: dict[int, int]) -> np.ndarray:
        local_idx = np.array([selected[cid] for cid in sorted(selected)], dtype=np.int64)
        packed = np.column_stack([local_idx, x_bin[local_idx], y_bin[local_idx], cell_id[local_idx]]).astype(np.int64)
        return packed

    return {
        "upper": _pack(upper),
        "lower": _pack(lower),
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def _save_surface_reference(
    cfg: FullAircraftConfig,
    surface_name: str,
    packed_indices: np.ndarray,
    xyz: np.ndarray,
    normals: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> dict:
    local_idx = packed_indices[:, 0]
    x_bin = packed_indices[:, 1]
    y_bin = packed_indices[:, 2]
    cell_id = packed_indices[:, 3]
    coords = xyz[local_idx]
    norms = normals[local_idx]

    out_path = cfg.surfaces_dir / f"{surface_name}_surface_reference.npz"
    np.savez_compressed(
        out_path,
        local_idx=local_idx.astype(np.int64),
        x=coords[:, 0].astype(np.float32),
        y=coords[:, 1].astype(np.float32),
        z=coords[:, 2].astype(np.float32),
        nx=norms[:, 0].astype(np.float32),
        ny=norms[:, 1].astype(np.float32),
        nz=norms[:, 2].astype(np.float32),
        x_bin=x_bin.astype(np.int64),
        y_bin=y_bin.astype(np.int64),
        cell_id=cell_id.astype(np.int64),
        x_edges=x_edges.astype(np.float32),
        y_edges=y_edges.astype(np.float32),
    )
    return {
        "surface": surface_name,
        "rows": int(local_idx.size),
        "path": str(out_path),
        "x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
        "y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
        "z_range": [float(coords[:, 2].min()), float(coords[:, 2].max())],
    }


def _plot_surface_reference_preview(cfg: FullAircraftConfig, upper_path: Path, lower_path: Path) -> Path:
    upper = np.load(upper_path)
    lower = np.load(lower_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    scat0 = axes[0].scatter(upper["x"], upper["y"], c=upper["z"], s=2.0, cmap="viridis", linewidths=0)
    axes[0].set_title(f"Upper surface reference ({upper['x'].shape[0]:,} pts)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(scat0, ax=axes[0], shrink=0.85, label="z")

    scat1 = axes[1].scatter(lower["x"], lower["y"], c=lower["z"], s=2.0, cmap="viridis", linewidths=0)
    axes[1].set_title(f"Lower surface reference ({lower['x'].shape[0]:,} pts)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(scat1, ax=axes[1], shrink=0.85, label="z")

    out_path = cfg.shared_results_dir / "reference_surface_preview.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def prepare_reference_surface(
    cfg: FullAircraftConfig,
    reference_split: str = "train",
    reference_condition_index: int = 0,
    x_bins: int | None = None,
    y_bins: int | None = None,
) -> dict:
    cfg.ensure_dirs()
    x_bins = int(x_bins or cfg.x_bins)
    y_bins = int(y_bins or cfg.y_bins)

    x_path, _ = raw_paths(cfg.raw_data_dir, reference_split)
    x_raw = np.load(x_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
    if reference_condition_index < 0 or reference_condition_index >= n_conditions:
        raise IndexError(f"Condition index {reference_condition_index} out of range for split {reference_split} ({n_conditions} conditions).")

    start = condition_start(cfg.raw_points_per_condition, reference_condition_index)
    stop = condition_stop(cfg.raw_points_per_condition, reference_condition_index)
    ref_block = np.asarray(x_raw[start:stop, : cfg.input_dim_raw], dtype=np.float32)
    xyz = ref_block[:, 0:3]
    normals = ref_block[:, 3:6]
    packed = _surface_indices_by_envelope(xyz, x_bins=x_bins, y_bins=y_bins)

    upper_meta = _save_surface_reference(
        cfg,
        surface_name="upper",
        packed_indices=packed["upper"],
        xyz=xyz,
        normals=normals,
        x_edges=packed["x_edges"],
        y_edges=packed["y_edges"],
    )
    lower_meta = _save_surface_reference(
        cfg,
        surface_name="lower",
        packed_indices=packed["lower"],
        xyz=xyz,
        normals=normals,
        x_edges=packed["x_edges"],
        y_edges=packed["y_edges"],
    )
    preview_path = _plot_surface_reference_preview(
        cfg,
        cfg.surfaces_dir / "upper_surface_reference.npz",
        cfg.surfaces_dir / "lower_surface_reference.npz",
    )

    payload = {
        "reference_split": reference_split,
        "reference_condition_index": int(reference_condition_index),
        "x_bins": int(x_bins),
        "y_bins": int(y_bins),
        "raw_points_per_condition": int(cfg.raw_points_per_condition),
        "upper": upper_meta,
        "lower": lower_meta,
        "preview_path": str(preview_path),
    }
    save_json(cfg.metadata_dir / "reference_surface_summary.json", payload)
    return payload
