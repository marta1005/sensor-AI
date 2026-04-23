from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import FullAircraftConfig
from .prepare_data import reduced_paths
from .utils import raw_paths, regime_from_mach, save_json

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _exploration_root(cfg: FullAircraftConfig) -> Path:
    return cfg.project_root / "exploration_data"


def _extract_conditions(cfg: FullAircraftConfig, split: str) -> dict[str, np.ndarray]:
    x_path, _ = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    step = cfg.raw_points_per_condition
    rows = np.asarray(x_raw[::step, 6:9], dtype=np.float32)
    return {
        "Mach": rows[:, 0],
        "AoA_deg": rows[:, 1],
        "Pi": rows[:, 2],
        "regime": regime_from_mach(rows[:, 0], cfg.mach_sub_max, cfg.mach_trans_max),
    }


def _condition_summary(cond: dict[str, np.ndarray]) -> dict[str, object]:
    mach = cond["Mach"]
    aoa = cond["AoA_deg"]
    pi = cond["Pi"]
    regime = cond["regime"]
    return {
        "n_conditions": int(mach.shape[0]),
        "mach_values": np.unique(mach).astype(float).tolist(),
        "aoa_values": np.unique(aoa).astype(float).tolist(),
        "pi_values": np.unique(pi).astype(float).tolist(),
        "mach_range": [float(mach.min()), float(mach.max())],
        "aoa_range": [float(aoa.min()), float(aoa.max())],
        "regime_counts": {
            "subsonic": int(np.sum(regime == 0)),
            "transonic": int(np.sum(regime == 1)),
            "supersonic": int(np.sum(regime == 2)),
        },
    }


def _plot_design_space(plots_dir: Path, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, title, cond in zip(axes, ["Train", "Test"], [train, test]):
        sc = ax.scatter(cond["Mach"], cond["AoA_deg"], c=cond["Pi"], s=38, cmap="viridis", edgecolors="black", linewidths=0.25)
        ax.set_title(f"{title} design space")
        ax.set_xlabel("Mach")
        ax.set_ylabel("AoA [deg]")
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, shrink=0.85, label="Pi")
    fig.savefig(plots_dir / "design_space_train_test.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_design_space_by_pi(plots_dir: Path, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> None:
    pi_values = sorted({float(v) for v in train["Pi"].tolist() + test["Pi"].tolist()})
    fig, axes = plt.subplots(1, len(pi_values), figsize=(5.5 * len(pi_values), 5.2), constrained_layout=True)
    if len(pi_values) == 1:
        axes = [axes]
    for ax, pi in zip(axes, pi_values):
        train_mask = np.isclose(train["Pi"], pi)
        test_mask = np.isclose(test["Pi"], pi)
        ax.scatter(train["Mach"][train_mask], train["AoA_deg"][train_mask], s=38, marker="o", c="#1f77b4", label="train", alpha=0.8)
        ax.scatter(test["Mach"][test_mask], test["AoA_deg"][test_mask], s=44, marker="^", c="#d62728", label="test", alpha=0.8)
        ax.set_title(f"Pi = {pi:g}")
        ax.set_xlabel("Mach")
        ax.set_ylabel("AoA [deg]")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(plots_dir / "design_space_by_pi.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_counts(plots_dir: Path, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> None:
    names = ["subsonic", "transonic", "supersonic"]
    x = np.arange(len(names))
    train_counts = [int(np.sum(train["regime"] == idx)) for idx in range(3)]
    test_counts = [int(np.sum(test["regime"] == idx)) for idx in range(3)]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.36
    ax.bar(x - width / 2, train_counts, width=width, label="train", color="#1f77b4")
    ax.bar(x + width / 2, test_counts, width=width, label="test", color="#ff7f0e")
    ax.set_xticks(x, names)
    ax.set_ylabel("Number of conditions")
    ax.set_title("Condition counts by Mach regime")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(plots_dir / "regime_condition_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reference_surface(plots_dir: Path, cfg: FullAircraftConfig) -> dict[str, object]:
    upper = np.load(cfg.surfaces_dir / "upper_surface_reference.npz")
    lower = np.load(cfg.surfaces_dir / "lower_surface_reference.npz")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    sc0 = axes[0].scatter(upper["x"], upper["y"], c=upper["z"], s=2.0, cmap="viridis", linewidths=0)
    axes[0].set_title("Upper reference surface")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(sc0, ax=axes[0], shrink=0.85, label="z")

    sc1 = axes[1].scatter(lower["x"], lower["y"], c=lower["z"], s=2.0, cmap="viridis", linewidths=0)
    axes[1].set_title("Lower reference surface")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(sc1, ax=axes[1], shrink=0.85, label="z")
    fig.savefig(plots_dir / "reference_surface_xy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "upper_points": int(upper["x"].shape[0]),
        "lower_points": int(lower["x"].shape[0]),
        "upper_grid_shape": [int(upper["x_edges"].shape[0] - 1), int(upper["y_edges"].shape[0] - 1)],
        "lower_grid_shape": [int(lower["x_edges"].shape[0] - 1), int(lower["y_edges"].shape[0] - 1)],
    }


def _plot_cp_ranges(plots_dir: Path, cfg: FullAircraftConfig, split: str = "test", surface: str = "upper") -> dict[str, object]:
    x_path, y_path = reduced_paths(FullAircraftConfig(
        project_root=cfg.project_root,
        raw_data_dir=cfg.raw_data_dir,
        pipeline_root=cfg.pipeline_root,
        reduced_surface=surface,
    ), split)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    ref = np.load(cfg.surfaces_dir / f"{surface}_surface_reference.npz")
    points = int(ref["local_idx"].shape[0])
    n_conditions = int(x.shape[0] // points)

    cp = np.asarray(y[:, 0], dtype=np.float32).reshape(n_conditions, points)
    mach = np.asarray(x[::points, 6], dtype=np.float32)[:n_conditions]
    aoa = np.asarray(x[::points, 7], dtype=np.float32)[:n_conditions]

    cp_min = cp.min(axis=1)
    cp_max = cp.max(axis=1)
    cp_mean = cp.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    idx = np.arange(n_conditions)
    axes[0].plot(idx, cp_min, label="Cp min", color="#1f77b4")
    axes[0].plot(idx, cp_mean, label="Cp mean", color="#2ca02c")
    axes[0].plot(idx, cp_max, label="Cp max", color="#d62728")
    axes[0].set_title(f"{surface} {split}: Cp range by condition")
    axes[0].set_xlabel("condition index")
    axes[0].set_ylabel("Cp")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    sc = axes[1].scatter(mach, aoa, c=cp_min, s=42, cmap="coolwarm", edgecolors="black", linewidths=0.25)
    axes[1].set_title(f"{surface} {split}: condition space colored by Cp min")
    axes[1].set_xlabel("Mach")
    axes[1].set_ylabel("AoA [deg]")
    axes[1].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[1], shrink=0.85, label="Cp min")

    out_name = f"cp_ranges_{surface}_{split}.png"
    fig.savefig(plots_dir / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)

    worst_idx = np.argsort(cp_min)[:10]
    return {
        "surface": surface,
        "split": split,
        "n_conditions": int(n_conditions),
        "cp_min_range": [float(cp_min.min()), float(cp_min.max())],
        "cp_max_range": [float(cp_max.min()), float(cp_max.max())],
        "worst_cp_min_conditions": [
            {
                "condition_index": int(i),
                "Mach": float(mach[i]),
                "AoA_deg": float(aoa[i]),
                "cp_min": float(cp_min[i]),
                "cp_max": float(cp_max[i]),
                "cp_mean": float(cp_mean[i]),
            }
            for i in worst_idx.tolist()
        ],
    }


def characterize_dataset(cfg: FullAircraftConfig) -> dict[str, object]:
    root = _exploration_root(cfg)
    plots_dir = root / "plots"
    summary_dir = root / "summary"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    train = _extract_conditions(cfg, "train")
    test = _extract_conditions(cfg, "test")

    _plot_design_space(plots_dir, train, test)
    _plot_design_space_by_pi(plots_dir, train, test)
    _plot_regime_counts(plots_dir, train, test)
    surface_summary = _plot_reference_surface(plots_dir, cfg)
    cp_upper_test = _plot_cp_ranges(plots_dir, cfg, split="test", surface="upper")
    cp_upper_train = _plot_cp_ranges(plots_dir, cfg, split="train", surface="upper")

    payload = {
        "train": _condition_summary(train),
        "test": _condition_summary(test),
        "surface_reference": surface_summary,
        "cp_upper_test": cp_upper_test,
        "cp_upper_train": cp_upper_train,
        "plots": {
            "design_space_train_test": str(plots_dir / "design_space_train_test.png"),
            "design_space_by_pi": str(plots_dir / "design_space_by_pi.png"),
            "regime_condition_counts": str(plots_dir / "regime_condition_counts.png"),
            "reference_surface_xy": str(plots_dir / "reference_surface_xy.png"),
            "cp_ranges_upper_test": str(plots_dir / "cp_ranges_upper_test.png"),
            "cp_ranges_upper_train": str(plots_dir / "cp_ranges_upper_train.png"),
        },
    }
    save_json(summary_dir / "dataset_characterization.json", payload)
    print(f"[explore-dataset] Characterization written to {root}")
    return payload
