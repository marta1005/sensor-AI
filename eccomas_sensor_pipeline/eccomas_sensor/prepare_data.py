from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .config import PipelineConfig
from .utils import sample_indices, save_json

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


RAW_FILENAMES = {
    "train": ("X_train.npy", "Ytrain.npy"),
    "test": ("X_test.npy", "Ytest.npy"),
}

CUT_FILENAMES = {
    "train": ("X_cut_train.npy", "Y_cut_train.npy", "idx_keep_train.npy"),
    "test": ("X_cut_test.npy", "Y_cut_test.npy", "idx_keep_test.npy"),
}


def _raw_paths(cfg: PipelineConfig, split: str) -> tuple[Path, Path]:
    x_name, y_name = RAW_FILENAMES[split]
    return cfg.raw_data_dir / x_name, cfg.raw_data_dir / y_name


def _cut_paths(cfg: PipelineConfig, split: str) -> tuple[Path, Path, Path]:
    return tuple(cfg.cut_data_dir / name for name in CUT_FILENAMES[split])


def _build_keep_indices(cfg: PipelineConfig) -> np.ndarray:
    train_x_path, _ = _raw_paths(cfg, "train")
    train_x = np.load(train_x_path, mmap_mode="r")
    first_condition = np.asarray(train_x[: cfg.raw_points_per_condition, : cfg.input_dim_raw], dtype=np.float32)
    y = first_condition[:, 1]
    if cfg.keep_ge_threshold:
        mask = y >= cfg.cut_threshold_y
    else:
        mask = y <= cfg.cut_threshold_y
    idx_keep = np.flatnonzero(mask).astype(np.int64)
    if idx_keep.size == 0:
        raise ValueError("The cut produced zero points. Check threshold and side.")
    return idx_keep


def _validate_constant_mask(cfg: PipelineConfig, idx_keep: np.ndarray) -> None:
    for split in ["train", "test"]:
        x_path, _ = _raw_paths(cfg, split)
        x_raw = np.load(x_path, mmap_mode="r")
        if x_raw.shape[0] < 2 * cfg.raw_points_per_condition:
            continue
        first_y = np.asarray(x_raw[: cfg.raw_points_per_condition, 1], dtype=np.float32)
        second_y = np.asarray(x_raw[cfg.raw_points_per_condition : 2 * cfg.raw_points_per_condition, 1], dtype=np.float32)
        first_mask = np.flatnonzero(first_y >= cfg.cut_threshold_y if cfg.keep_ge_threshold else first_y <= cfg.cut_threshold_y)
        second_mask = np.flatnonzero(second_y >= cfg.cut_threshold_y if cfg.keep_ge_threshold else second_y <= cfg.cut_threshold_y)
        if not np.array_equal(first_mask, idx_keep) or not np.array_equal(second_mask, idx_keep):
            raise ValueError(f"The cut mask is not constant across the {split} split.")


def _write_cut_arrays(cfg: PipelineConfig, split: str, idx_keep: np.ndarray) -> dict:
    x_path, y_path = _raw_paths(cfg, split)
    out_x_path, out_y_path, out_idx_path = _cut_paths(cfg, split)

    x_raw = np.load(x_path, mmap_mode="r")
    y_raw = np.load(y_path, mmap_mode="r")

    if x_raw.shape[0] % cfg.raw_points_per_condition != 0:
        raise ValueError(f"{x_path.name} rows are not divisible by raw_points_per_condition")
    n_conditions = x_raw.shape[0] // cfg.raw_points_per_condition
    kept_per_condition = idx_keep.size
    total_kept = n_conditions * kept_per_condition

    x_cut = open_memmap(out_x_path, mode="w+", dtype=np.float32, shape=(total_kept, cfg.input_dim_raw))
    y_cut = open_memmap(out_y_path, mode="w+", dtype=np.float32, shape=(total_kept, y_raw.shape[1]))

    for condition_idx in range(n_conditions):
        src_start = condition_idx * cfg.raw_points_per_condition
        src_end = src_start + cfg.raw_points_per_condition
        dst_start = condition_idx * kept_per_condition
        dst_end = dst_start + kept_per_condition

        x_condition = np.asarray(x_raw[src_start:src_end, : cfg.input_dim_raw], dtype=np.float32)
        y_condition = np.asarray(y_raw[src_start:src_end], dtype=np.float32)

        x_cut[dst_start:dst_end] = x_condition[idx_keep]
        y_cut[dst_start:dst_end] = y_condition[idx_keep]

        if condition_idx % 25 == 0 or condition_idx == n_conditions - 1:
            print(f"[prepare-data] {split}: condition {condition_idx + 1}/{n_conditions}")

    np.save(out_idx_path, idx_keep)
    np.save(cfg.cut_data_dir / "idx_keep.npy", idx_keep)

    return {
        "split": split,
        "n_conditions": int(n_conditions),
        "raw_rows": int(x_raw.shape[0]),
        "kept_rows": int(total_kept),
        "kept_per_condition": int(kept_per_condition),
        "x_path": str(out_x_path),
        "y_path": str(out_y_path),
    }


def _set_equal_projection_limits(ax, x_vals: np.ndarray, y_vals: np.ndarray, pad_ratio: float = 0.03) -> None:
    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min)
    radius *= 1.0 + pad_ratio

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_aspect("equal", adjustable="box")


def _plot_cut_validation(cfg: PipelineConfig, idx_keep: np.ndarray) -> None:
    train_x_path, _ = _raw_paths(cfg, "train")
    x_raw = np.load(train_x_path, mmap_mode="r")
    first_condition = np.asarray(x_raw[: cfg.raw_points_per_condition, : cfg.input_dim_raw], dtype=np.float32)
    cut_condition = first_condition[idx_keep]

    sample_full = sample_indices(first_condition.shape[0], cfg.plot_sample_size, seed=7)
    sample_cut = sample_indices(cut_condition.shape[0], min(cfg.plot_sample_size, cut_condition.shape[0]), seed=7)

    raw_s = first_condition[sample_full]
    cut_s = cut_condition[sample_cut]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].scatter(raw_s[:, 0], raw_s[:, 1], s=1.2, alpha=0.18, color="0.55", label="raw")
    axes[0].scatter(cut_s[:, 0], cut_s[:, 1], s=1.2, alpha=0.55, color="#d95f02", label="kept")
    axes[0].axhline(cfg.cut_threshold_y, color="#1b9e77", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("x-y projection")
    axes[0].legend(markerscale=4)
    _set_equal_projection_limits(axes[0], first_condition[:, 0], first_condition[:, 1])

    axes[1].scatter(raw_s[:, 1], raw_s[:, 2], s=1.2, alpha=0.18, color="0.55")
    axes[1].scatter(cut_s[:, 1], cut_s[:, 2], s=1.2, alpha=0.55, color="#d95f02")
    axes[1].axvline(cfg.cut_threshold_y, color="#1b9e77", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("z")
    axes[1].set_title("y-z projection")
    _set_equal_projection_limits(axes[1], first_condition[:, 1], first_condition[:, 2])

    axes[2].scatter(raw_s[:, 0], raw_s[:, 2], s=1.2, alpha=0.18, color="0.55")
    axes[2].scatter(cut_s[:, 0], cut_s[:, 2], s=1.2, alpha=0.55, color="#d95f02")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("z")
    axes[2].set_title("x-z projection")
    _set_equal_projection_limits(axes[2], first_condition[:, 0], first_condition[:, 2])

    fig.savefig(cfg.cut_data_dir / "cut_geometry_projections.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(first_condition[:, 1], bins=120, color="#4c78a8", alpha=0.85)
    ax.axvline(cfg.cut_threshold_y, color="#e45756", linestyle="--", linewidth=1.5)
    ax.set_xlabel("y")
    ax.set_ylabel("count")
    ax.set_title("Distribution of y in one condition")
    fig.savefig(cfg.cut_data_dir / "cut_y_histogram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _design_space_records_from_csv(csv_path: Path) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "Pi": float(row["Pi *1e-5"]),
                    "Mach": float(row["Mach"]),
                    "AoA": float(row["AoA"]),
                    "split": "train" if row["Train"].strip().lower() == "true" else "test",
                }
            )
    return records


def _design_space_records_from_arrays(cfg: PipelineConfig) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for split in ["train", "test"]:
        x_path, _ = _raw_paths(cfg, split)
        x_raw = np.load(x_path, mmap_mode="r")
        if x_raw.shape[0] % cfg.raw_points_per_condition != 0:
            raise ValueError(f"{x_path.name} rows are not divisible by raw_points_per_condition")
        n_conditions = x_raw.shape[0] // cfg.raw_points_per_condition
        for idx in range(n_conditions):
            row = np.asarray(x_raw[idx * cfg.raw_points_per_condition, 6:9], dtype=np.float32)
            records.append(
                {
                    "Mach": float(row[0]),
                    "AoA": float(row[1]),
                    "Pi": float(row[2]),
                    "split": split,
                }
            )
    return records


def _load_design_space_records(cfg: PipelineConfig) -> list[dict[str, float | str]]:
    csv_path = cfg.raw_data_dir / "dataset.csv"
    if csv_path.exists():
        return _design_space_records_from_csv(csv_path)
    return _design_space_records_from_arrays(cfg)


def plot_design_space(cfg: PipelineConfig) -> Path:
    cfg.ensure_dirs()
    records = _load_design_space_records(cfg)
    if not records:
        raise ValueError("No design-space records found.")

    pi_values = sorted({float(record["Pi"]) for record in records})
    mach_values = [float(record["Mach"]) for record in records]
    aoa_values = [float(record["AoA"]) for record in records]

    ncols = min(3, max(1, len(pi_values)))
    nrows = math.ceil(len(pi_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), constrained_layout=True, squeeze=False)

    legend_handles = None
    mach_max = max(mach_values)
    aoa_min = min(aoa_values)
    aoa_max = max(aoa_values)

    for ax, pi_val in zip(axes.flat, pi_values):
        pi_records = [record for record in records if float(record["Pi"]) == pi_val]
        train_records = [record for record in pi_records if record["split"] == "train"]
        test_records = [record for record in pi_records if record["split"] == "test"]

        ax.axvspan(min(mach_values), cfg.mach_sub_max, color="#4c78a8", alpha=0.06)
        ax.axvspan(cfg.mach_sub_max, cfg.mach_trans_max, color="#f2cf5b", alpha=0.08)
        ax.axvspan(cfg.mach_trans_max, mach_max, color="#e45756", alpha=0.05)
        ax.axvline(cfg.mach_sub_max, color="#4c78a8", linestyle="--", linewidth=1.1)
        ax.axvline(cfg.mach_trans_max, color="#e45756", linestyle="--", linewidth=1.1)

        h1 = ax.scatter(
            [record["Mach"] for record in train_records],
            [record["AoA"] for record in train_records],
            s=42,
            alpha=0.82,
            color="#1f77b4",
            edgecolors="white",
            linewidths=0.3,
            label="train",
        )
        h2 = ax.scatter(
            [record["Mach"] for record in test_records],
            [record["AoA"] for record in test_records],
            s=42,
            alpha=0.9,
            color="#d95f02",
            marker="x",
            linewidths=1.2,
            label="test",
        )
        if legend_handles is None:
            legend_handles = [h1, h2]

        ax.set_title(f"Pi = {pi_val:g}")
        ax.set_xlabel("Mach")
        ax.set_ylabel("AoA")
        ax.set_xlim(min(mach_values) - 0.02, mach_max + 0.02)
        ax.set_ylim(aoa_min - 1.0, aoa_max + 1.0)
        ax.grid(True, alpha=0.25)

    for ax in axes.flat[len(pi_values) :]:
        ax.axis("off")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["train", "test"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=2,
            frameon=False,
        )

    fig.suptitle("Design space by Pi: AoA vs Mach", fontsize=14, y=1.11)
    out_path = cfg.cut_data_dir / "design_space_by_pi.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-design-space] Saved {out_path}")
    return out_path


def prepare_cut_data(cfg: PipelineConfig) -> None:
    cfg.ensure_dirs()
    idx_keep = _build_keep_indices(cfg)
    _validate_constant_mask(cfg, idx_keep)

    train_meta = _write_cut_arrays(cfg, "train", idx_keep)
    test_meta = _write_cut_arrays(cfg, "test", idx_keep)
    _plot_cut_validation(cfg, idx_keep)
    plot_design_space(cfg)

    save_json(
        cfg.cut_data_dir / "metadata.json",
        {
            "cut_threshold_y": cfg.cut_threshold_y,
            "keep_side": ">=" if cfg.keep_ge_threshold else "<=",
            "idx_keep_size": int(idx_keep.size),
            "train": train_meta,
            "test": test_meta,
        },
    )
    print(f"[prepare-data] Finished. Cut data stored in {cfg.cut_data_dir}")
