from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .cluster_partition import expert_names
from .config import FullAircraftConfig
from .utils import raw_paths, save_json

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _prediction_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"expert_pred_{split}.npy"


def _condition_count(cfg: FullAircraftConfig, split: str) -> int:
    x_path, _ = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    return int(x_raw.shape[0] // cfg.raw_points_per_condition)


def _points_per_condition(cfg: FullAircraftConfig, split: str) -> int:
    x_red = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    n_conditions = _condition_count(cfg, split)
    points = int(x_red.shape[0] // max(1, n_conditions))
    if points * n_conditions != x_red.shape[0]:
        raise ValueError(
            f"Reduced rows ({x_red.shape[0]}) are not divisible by the number of conditions ({n_conditions}) "
            f"for split={split}."
        )
    return points


def _load_cp_scaler(cfg: FullAircraftConfig) -> tuple[float, float]:
    payload = np.load(cfg.scalers_dir / "cp_scaler.npz")
    return float(payload["mean"][0]), float(payload["scale"][0])


def _unscale(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values.astype(np.float32) * np.float32(scale) + np.float32(mean)


def _partition_names(cfg: FullAircraftConfig) -> list[str]:
    return expert_names(cfg)


def _heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
    fmt: str = ".4f",
) -> None:
    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(col_labels), 1.8 + 0.9 * len(row_labels)))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(float(matrix[i, j]), fmt), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _condition_level_summary(
    abs_err: np.ndarray,
    partition_label: np.ndarray,
) -> dict[str, np.ndarray]:
    n_conditions, n_experts = abs_err.shape
    n_partitions = int(partition_label.max()) + 1
    mae_sum = np.zeros((n_partitions, n_experts), dtype=np.float64)
    best_counts = np.zeros((n_partitions, n_experts), dtype=np.int64)
    assigned_gap_sum = np.zeros((n_partitions,), dtype=np.float64)
    second_gap_sum = np.zeros((n_partitions,), dtype=np.float64)
    counts = np.zeros((n_partitions,), dtype=np.int64)

    best_expert = np.argmin(abs_err, axis=1)
    assigned_err = abs_err[np.arange(n_conditions), partition_label]
    oracle_err = abs_err[np.arange(n_conditions), best_expert]
    sorted_err = np.sort(abs_err, axis=1)
    best_margin = sorted_err[:, 1] - sorted_err[:, 0] if n_experts > 1 else np.zeros((n_conditions,), dtype=np.float64)

    for part in range(n_partitions):
        mask = partition_label == part
        if not np.any(mask):
            continue
        mae_sum[part] = abs_err[mask].mean(axis=0)
        local_best, local_counts = np.unique(best_expert[mask], return_counts=True)
        best_counts[part, local_best.astype(np.int64)] = local_counts.astype(np.int64)
        assigned_gap_sum[part] = float((assigned_err[mask] - oracle_err[mask]).mean())
        second_gap_sum[part] = float(best_margin[mask].mean())
        counts[part] = int(mask.sum())

    return {
        "mae": mae_sum,
        "best_counts": best_counts,
        "assigned_gap": assigned_gap_sum,
        "best_margin": second_gap_sum,
        "counts": counts,
    }


def diagnose_experts(cfg: FullAircraftConfig, splits: tuple[str, ...] = ("train", "test")) -> None:
    cfg.ensure_dirs()
    names = _partition_names(cfg)
    cp_mean, cp_scale = _load_cp_scaler(cfg)

    for split in splits:
        pred_path = _prediction_path(cfg, split)
        cp_path = cfg.features_dir / f"cp_{split}.npy"
        expert_id_path = cfg.features_dir / f"expert_id_{split}.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Expert predictions not found: {pred_path}. Run 'train-experts' first.")
        pred = np.load(pred_path, mmap_mode="r")
        cp = np.load(cp_path, mmap_mode="r")
        expert_id = np.load(expert_id_path, mmap_mode="r").astype(np.int64)

        y_true = _unscale(cp[:, 0], cp_mean, cp_scale)
        pred_cp = _unscale(pred, cp_mean, cp_scale)
        abs_err = np.abs(pred_cp - y_true[:, None]).astype(np.float32)

        n_partitions = len(names)
        row_counts = np.zeros((n_partitions,), dtype=np.int64)
        row_mae = np.zeros((n_partitions, cfg.n_experts), dtype=np.float64)
        row_rmse = np.zeros((n_partitions, cfg.n_experts), dtype=np.float64)
        row_best_counts = np.zeros((n_partitions, cfg.n_experts), dtype=np.int64)
        row_assigned_gap = np.zeros((n_partitions,), dtype=np.float64)

        best_expert_rows = np.argmin(abs_err, axis=1)
        oracle_err_rows = abs_err[np.arange(abs_err.shape[0]), best_expert_rows]
        assigned_err_rows = abs_err[np.arange(abs_err.shape[0]), expert_id]

        for part in range(n_partitions):
            mask = expert_id == part
            if not np.any(mask):
                continue
            row_counts[part] = int(mask.sum())
            row_mae[part] = abs_err[mask].mean(axis=0)
            row_rmse[part] = np.sqrt((abs_err[mask] ** 2).mean(axis=0))
            local_best, local_counts = np.unique(best_expert_rows[mask], return_counts=True)
            row_best_counts[part, local_best.astype(np.int64)] = local_counts.astype(np.int64)
            row_assigned_gap[part] = float((assigned_err_rows[mask] - oracle_err_rows[mask]).mean())

        points_per_condition = _points_per_condition(cfg, split)
        n_conditions = int(abs_err.shape[0] // points_per_condition)
        cond_abs_err = abs_err.reshape(n_conditions, points_per_condition, cfg.n_experts).mean(axis=1)
        cond_partition = expert_id.reshape(n_conditions, points_per_condition)[:, 0]
        cond_summary = _condition_level_summary(cond_abs_err, cond_partition)

        payload = {
            "split": split,
            "surface": cfg.reduced_surface,
            "partition_mode": cfg.expert_partition_mode,
            "expert_names": names,
            "rows_per_condition": int(points_per_condition),
            "condition_counts": {name: int(count) for name, count in zip(names, cond_summary["counts"])},
            "row_counts": {name: int(count) for name, count in zip(names, row_counts)},
            "row_level": {
                "mae_matrix": row_mae.tolist(),
                "rmse_matrix": row_rmse.tolist(),
                "best_expert_counts": row_best_counts.tolist(),
                "self_best_fraction": [
                    float(row_best_counts[i, i] / max(1, row_counts[i])) for i in range(n_partitions)
                ],
                "assigned_minus_oracle_mae": row_assigned_gap.tolist(),
            },
            "condition_level": {
                "mae_matrix": cond_summary["mae"].tolist(),
                "best_expert_counts": cond_summary["best_counts"].tolist(),
                "self_best_fraction": [
                    float(cond_summary["best_counts"][i, i] / max(1, cond_summary["counts"][i]))
                    for i in range(n_partitions)
                ],
                "assigned_minus_oracle_mae": cond_summary["assigned_gap"].tolist(),
                "best_vs_second_margin_mae": cond_summary["best_margin"].tolist(),
            },
        }
        out_json = cfg.metrics_dir / f"expert_diagnostics_{split}.json"
        save_json(out_json, payload)

        _heatmap(
            row_mae,
            row_labels=names,
            col_labels=names,
            title=f"Row-level MAE by partition vs expert ({split})",
            out_path=cfg.plots_dir / f"expert_partition_row_mae_{split}.png",
        )
        _heatmap(
            cond_summary["mae"],
            row_labels=names,
            col_labels=names,
            title=f"Condition-level MAE by partition vs expert ({split})",
            out_path=cfg.plots_dir / f"expert_partition_condition_mae_{split}.png",
        )
        _heatmap(
            row_best_counts.astype(np.float64),
            row_labels=names,
            col_labels=names,
            title=f"Row-level best-expert counts by partition ({split})",
            out_path=cfg.plots_dir / f"expert_partition_row_best_counts_{split}.png",
            fmt=".0f",
        )
        _heatmap(
            cond_summary["best_counts"].astype(np.float64),
            row_labels=names,
            col_labels=names,
            title=f"Condition-level best-expert counts by partition ({split})",
            out_path=cfg.plots_dir / f"expert_partition_condition_best_counts_{split}.png",
            fmt=".0f",
        )

        print(f"[diagnose-experts] {split} diagnostics written to {out_json}")
        for idx, name in enumerate(names):
            row_self = payload["row_level"]["self_best_fraction"][idx]
            cond_self = payload["condition_level"]["self_best_fraction"][idx]
            gap = payload["condition_level"]["assigned_minus_oracle_mae"][idx]
            print(
                f"[diagnose-experts] {split} {name}: "
                f"row_self_best={row_self:.3f}, cond_self_best={cond_self:.3f}, "
                f"assigned-oracle MAE={gap:.5f}"
            )
