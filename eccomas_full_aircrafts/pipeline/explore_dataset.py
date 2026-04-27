from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import FullAircraftConfig
from .prepare_data import reduced_paths
from .utils import raw_paths, regime_from_mach, save_json

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


CLUSTER_FEATURE_NAMES = [
    "Mach",
    "AoA_deg",
    "Pi",
    "cp_min",
    "cp_mean",
    "cp_std",
    "negative_cp_fraction",
    "strong_suction_fraction",
    "grad_mean",
    "grad_p95",
    "grad_max",
    "shock_x_weighted",
    "shock_y_weighted",
    "shock_span_std",
    "front_cp_mean",
    "rear_cp_mean",
]


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
        sc = ax.scatter(
            cond["Mach"],
            cond["AoA_deg"],
            c=cond["Pi"],
            s=38,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.25,
        )
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
    x_path, y_path = reduced_paths(
        FullAircraftConfig(
            project_root=cfg.project_root,
            raw_data_dir=cfg.raw_data_dir,
            pipeline_root=cfg.pipeline_root,
            reduced_surface=surface,
        ),
        split,
    )
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


def _load_surface_reference(cfg: FullAircraftConfig, surface: str) -> np.lib.npyio.NpzFile:
    path = cfg.surfaces_dir / f"{surface}_surface_reference.npz"
    if not path.exists():
        raise FileNotFoundError(f"Surface reference not found: {path}. Run 'prepare-reference-surface' first.")
    return np.load(path)


def _compact_surface_geometry(ref: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    x_bin = np.asarray(ref["x_bin"], dtype=np.int64)
    y_bin = np.asarray(ref["y_bin"], dtype=np.int64)
    x_edges = np.asarray(ref["x_edges"], dtype=np.float32)
    y_edges = np.asarray(ref["y_edges"], dtype=np.float32)

    unique_x = np.unique(x_bin)
    unique_y = np.unique(y_bin)
    x_lookup = np.full(x_edges.shape[0] - 1, -1, dtype=np.int64)
    y_lookup = np.full(y_edges.shape[0] - 1, -1, dtype=np.int64)
    x_lookup[unique_x] = np.arange(unique_x.shape[0], dtype=np.int64)
    y_lookup[unique_y] = np.arange(unique_y.shape[0], dtype=np.int64)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    return {
        "row_idx": y_lookup[y_bin],
        "col_idx": x_lookup[x_bin],
        "height": np.array(unique_y.shape[0], dtype=np.int64),
        "width": np.array(unique_x.shape[0], dtype=np.int64),
        "x_coords": x_centers[unique_x].astype(np.float32),
        "y_coords": y_centers[unique_y].astype(np.float32),
        "x_points": np.asarray(ref["x"], dtype=np.float32),
    }


def _scatter_flat_to_compact_grid(values: np.ndarray, geom: dict[str, np.ndarray]) -> np.ndarray:
    grid = np.full((int(geom["height"]), int(geom["width"])), np.nan, dtype=np.float32)
    grid[geom["row_idx"], geom["col_idx"]] = values.astype(np.float32, copy=False)
    return grid


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0 or float(np.sum(weights)) <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0 or float(np.sum(weights)) <= 0.0:
        return float("nan")
    mean = np.sum(values * weights) / np.sum(weights)
    var = np.sum(weights * np.square(values - mean)) / np.sum(weights)
    return float(np.sqrt(max(var, 0.0)))


def _flow_descriptors_for_split(cfg: FullAircraftConfig, split: str, surface: str) -> list[dict[str, Any]]:
    surface_cfg = FullAircraftConfig(
        project_root=cfg.project_root,
        raw_data_dir=cfg.raw_data_dir,
        pipeline_root=cfg.pipeline_root,
        reduced_surface=surface,
    )
    x_path, y_path = reduced_paths(surface_cfg, split)
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Reduced data not found for surface={surface}, split={split}. Run 'prepare-reduced-data' first.")

    ref = _load_surface_reference(cfg, surface)
    geom = _compact_surface_geometry(ref)
    points = int(np.asarray(ref["local_idx"]).shape[0])
    x_red = np.load(x_path, mmap_mode="r")
    y_red = np.load(y_path, mmap_mode="r")
    n_conditions = int(x_red.shape[0] // points)

    x_points = geom["x_points"]
    x_q20 = float(np.quantile(x_points, 0.20))
    x_q80 = float(np.quantile(x_points, 0.80))
    front_mask = x_points <= x_q20
    rear_mask = x_points >= x_q80

    x_mesh, y_mesh = np.meshgrid(geom["x_coords"], geom["y_coords"])

    descriptors: list[dict[str, Any]] = []
    for cond_idx in range(n_conditions):
        row_start = cond_idx * points
        row_stop = row_start + points
        cp_flat = np.asarray(y_red[row_start:row_stop, cfg.cp_column], dtype=np.float32)
        row = np.asarray(x_red[row_start, 6:9], dtype=np.float32)

        cp_grid = _scatter_flat_to_compact_grid(cp_flat, geom)
        dcp_dy, dcp_dx = np.gradient(cp_grid, geom["y_coords"], geom["x_coords"], edge_order=1)
        grad_mag = np.sqrt(np.square(dcp_dx) + np.square(dcp_dy)).astype(np.float32)
        grad_valid = grad_mag[np.isfinite(grad_mag)]

        if grad_valid.size > 0:
            grad_mean = float(np.mean(grad_valid))
            grad_std = float(np.std(grad_valid))
            grad_p90 = float(np.quantile(grad_valid, 0.90))
            grad_p95 = float(np.quantile(grad_valid, 0.95))
            grad_max = float(np.max(grad_valid))
            high_grad_fraction = float(np.mean(grad_valid >= 0.5 * grad_max)) if grad_max > 0.0 else 0.0
            top_threshold = float(np.quantile(grad_valid, 0.99))
            top_mask = np.isfinite(grad_mag) & (grad_mag >= top_threshold)
            top_weights = grad_mag[top_mask]
            shock_x = _weighted_mean(x_mesh[top_mask], top_weights)
            shock_y = _weighted_mean(y_mesh[top_mask], top_weights)
            shock_span_std = _weighted_std(y_mesh[top_mask], top_weights)
        else:
            grad_mean = grad_std = grad_p90 = grad_p95 = grad_max = 0.0
            high_grad_fraction = 0.0
            shock_x = float("nan")
            shock_y = float("nan")
            shock_span_std = float("nan")

        descriptors.append(
            {
                "split": split,
                "surface": surface,
                "condition_index": int(cond_idx),
                "Mach": float(row[0]),
                "AoA_deg": float(row[1]),
                "Pi": float(row[2]),
                "cp_min": float(np.min(cp_flat)),
                "cp_max": float(np.max(cp_flat)),
                "cp_mean": float(np.mean(cp_flat)),
                "cp_std": float(np.std(cp_flat)),
                "cp_p05": float(np.quantile(cp_flat, 0.05)),
                "cp_p95": float(np.quantile(cp_flat, 0.95)),
                "negative_cp_fraction": float(np.mean(cp_flat < 0.0)),
                "strong_suction_fraction": float(np.mean(cp_flat < -1.0)),
                "front_cp_mean": float(np.mean(cp_flat[front_mask])),
                "rear_cp_mean": float(np.mean(cp_flat[rear_mask])),
                "grad_mean": grad_mean,
                "grad_std": grad_std,
                "grad_p90": grad_p90,
                "grad_p95": grad_p95,
                "grad_max": grad_max,
                "high_grad_fraction": high_grad_fraction,
                "shock_x_weighted": shock_x,
                "shock_y_weighted": shock_y,
                "shock_span_std": shock_span_std,
            }
        )
    return descriptors


def _write_descriptor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _descriptor_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def _plot_descriptor_correlation(
    plots_dir: Path,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    surface: str,
) -> Path:
    data = _descriptor_matrix(rows, feature_names)
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(12.2, 10.2), constrained_layout=True)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(feature_names)), feature_names, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(feature_names)), feature_names)
    ax.set_title(f"{surface} flow descriptors correlation")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson correlation")
    out_path = plots_dir / f"{surface}_flow_descriptor_correlations.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fit_cluster_model(algorithm: str, n_clusters: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    if algorithm == "gmm":
        return GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42, n_init=5)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _predict_cluster_labels(model, algorithm: str, x_scaled: np.ndarray) -> np.ndarray:
    if algorithm == "kmeans":
        return model.fit_predict(x_scaled)
    model.fit(x_scaled)
    return model.predict(x_scaled)


def _cluster_evaluation(train_scaled: np.ndarray, algorithm: str, k_values: list[int]) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for k in k_values:
        model = _fit_cluster_model(algorithm, k)
        labels = _predict_cluster_labels(model, algorithm, train_scaled)
        unique, counts = np.unique(labels, return_counts=True)
        if unique.shape[0] < 2:
            continue
        evaluations.append(
            {
                "algorithm": algorithm,
                "n_clusters": int(k),
                "silhouette": float(silhouette_score(train_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(train_scaled, labels)),
                "cluster_sizes": {str(int(label)): int(count) for label, count in zip(unique, counts)},
                "min_cluster_size": int(np.min(counts)),
            }
        )
    return evaluations


def _select_best_evaluation(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        raise RuntimeError("No valid clustering evaluation could be computed.")
    return max(
        evaluations,
        key=lambda item: (
            item["silhouette"],
            -item["davies_bouldin"],
            item["min_cluster_size"],
        ),
    )


def _plot_cluster_selection(plots_dir: Path, surface: str, algorithm: str, evaluations: list[dict[str, Any]]) -> Path:
    ks = [item["n_clusters"] for item in evaluations]
    silhouettes = [item["silhouette"] for item in evaluations]
    dbi = [item["davies_bouldin"] for item in evaluations]

    fig, ax1 = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(ks, silhouettes, marker="o", color="#1f77b4", label="silhouette")
    ax2.plot(ks, dbi, marker="s", color="#d62728", label="davies-bouldin")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Silhouette", color="#1f77b4")
    ax2.set_ylabel("Davies-Bouldin", color="#d62728")
    ax1.set_title(f"{surface} {algorithm.upper()} cluster selection")
    ax1.grid(True, alpha=0.25)
    out_path = plots_dir / f"{surface}_{algorithm}_cluster_selection.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fit_best_cluster_model(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    algorithm: str,
    feature_names: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    train_x = _descriptor_matrix(train_rows, feature_names)
    test_x = _descriptor_matrix(test_rows, feature_names)

    scaler = StandardScaler()
    train_scaled = np.asarray(scaler.fit_transform(train_x), dtype=np.float64)
    test_scaled = np.asarray(scaler.transform(test_x), dtype=np.float64)
    evaluations = _cluster_evaluation(train_scaled, algorithm, k_values)
    best = _select_best_evaluation(evaluations)

    model = _fit_cluster_model(algorithm, int(best["n_clusters"]))
    if algorithm == "kmeans":
        train_labels = model.fit_predict(train_scaled)
        test_labels = model.predict(test_scaled)
    else:
        model.fit(train_scaled)
        train_labels = model.predict(train_scaled)
        test_labels = model.predict(test_scaled)

    pca = PCA(n_components=2, svd_solver="full")
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    return {
        "algorithm": algorithm,
        "feature_names": feature_names,
        "scaler": scaler,
        "model": model,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "train_labels": train_labels.astype(int),
        "test_labels": test_labels.astype(int),
        "train_pca": train_pca.astype(np.float32),
        "test_pca": test_pca.astype(np.float32),
        "evaluations": evaluations,
        "best": best,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
    }


def _plot_cluster_design_space(
    plots_dir: Path,
    surface: str,
    algorithm: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for ax, title, rows, labels in [
        (axes[0], "Train", train_rows, train_labels),
        (axes[1], "Test", test_rows, test_labels),
    ]:
        mach = np.asarray([row["Mach"] for row in rows], dtype=np.float32)
        aoa = np.asarray([row["AoA_deg"] for row in rows], dtype=np.float32)
        sc = ax.scatter(mach, aoa, c=labels, s=44, cmap="tab10", edgecolors="black", linewidths=0.25)
        ax.set_title(f"{title} design space")
        ax.set_xlabel("Mach")
        ax.set_ylabel("AoA [deg]")
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, shrink=0.82, label="Cluster")
    out_path = plots_dir / f"{surface}_{algorithm}_cluster_design_space.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cluster_pca(
    plots_dir: Path,
    surface: str,
    algorithm: str,
    clustering: dict[str, Any],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for ax, title, coords, labels in [
        (axes[0], "Train", clustering["train_pca"], clustering["train_labels"]),
        (axes[1], "Test", clustering["test_pca"], clustering["test_labels"]),
    ]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=44, cmap="tab10", edgecolors="black", linewidths=0.25)
        ax.set_title(f"{title} PCA projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, shrink=0.82, label="Cluster")
    out_path = plots_dir / f"{surface}_{algorithm}_cluster_pca.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cluster_flow_map(
    plots_dir: Path,
    surface: str,
    algorithm: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    split: str,
) -> Path:
    x = np.asarray([row["shock_x_weighted"] for row in rows], dtype=np.float32)
    y = np.asarray([row["grad_p95"] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    sc = ax.scatter(x, y, c=labels, s=48, cmap="tab10", edgecolors="black", linewidths=0.25)
    ax.set_xlabel("Shock x (weighted)")
    ax.set_ylabel("Gradient p95")
    ax.set_title(f"{surface} {split} flow descriptors colored by {algorithm.upper()} cluster")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, shrink=0.84, label="Cluster")
    out_path = plots_dir / f"{surface}_{algorithm}_cluster_flow_map_{split}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _attach_cluster_labels(rows: list[dict[str, Any]], algorithm: str, labels: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, label in zip(rows, labels.tolist()):
        row_copy = dict(row)
        row_copy[f"{algorithm}_cluster"] = int(label)
        out.append(row_copy)
    return out


def _cluster_payload(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    clustering: dict[str, Any],
    surface: str,
    plots: dict[str, str],
) -> dict[str, Any]:
    algorithm = clustering["algorithm"]
    train_labels = clustering["train_labels"]
    test_labels = clustering["test_labels"]
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    test_unique, test_counts = np.unique(test_labels, return_counts=True)
    return {
        "surface": surface,
        "algorithm": algorithm,
        "feature_names": clustering["feature_names"],
        "evaluations": clustering["evaluations"],
        "best": clustering["best"],
        "pca_explained_variance_ratio": clustering["explained_variance_ratio"],
        "train_cluster_counts": {str(int(k)): int(v) for k, v in zip(train_unique, train_counts)},
        "test_cluster_counts": {str(int(k)): int(v) for k, v in zip(test_unique, test_counts)},
        "plots": plots,
        "train_assignments_csv": f"{surface}_{algorithm}_train_condition_descriptors.csv",
        "test_assignments_csv": f"{surface}_{algorithm}_test_condition_descriptors.csv",
        "n_train_conditions": len(train_rows),
        "n_test_conditions": len(test_rows),
    }


def characterize_dataset(cfg: FullAircraftConfig) -> dict[str, object]:
    root = _exploration_root(cfg)
    plots_dir = root / "plots"
    summary_dir = root / "summary"
    tables_dir = root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train = _extract_conditions(cfg, "train")
    test = _extract_conditions(cfg, "test")

    _plot_design_space(plots_dir, train, test)
    _plot_design_space_by_pi(plots_dir, train, test)
    _plot_regime_counts(plots_dir, train, test)
    surface_summary = _plot_reference_surface(plots_dir, cfg)
    cp_upper_test = _plot_cp_ranges(plots_dir, cfg, split="test", surface="upper")
    cp_upper_train = _plot_cp_ranges(plots_dir, cfg, split="train", surface="upper")

    surface = cfg.reduced_surface
    train_descriptors = _flow_descriptors_for_split(cfg, "train", surface)
    test_descriptors = _flow_descriptors_for_split(cfg, "test", surface)
    all_descriptors = train_descriptors + test_descriptors

    descriptor_csv_train = tables_dir / f"{surface}_train_condition_descriptors.csv"
    descriptor_csv_test = tables_dir / f"{surface}_test_condition_descriptors.csv"
    descriptor_csv_all = tables_dir / f"{surface}_all_condition_descriptors.csv"
    _write_descriptor_csv(descriptor_csv_train, train_descriptors)
    _write_descriptor_csv(descriptor_csv_test, test_descriptors)
    _write_descriptor_csv(descriptor_csv_all, all_descriptors)

    corr_path = _plot_descriptor_correlation(
        plots_dir,
        all_descriptors,
        [
            "Mach",
            "AoA_deg",
            "Pi",
            "cp_min",
            "cp_std",
            "negative_cp_fraction",
            "grad_mean",
            "grad_p95",
            "grad_max",
            "shock_x_weighted",
            "shock_span_std",
            "front_cp_mean",
            "rear_cp_mean",
        ],
        surface,
    )

    clustering_payload: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    for algorithm in ("kmeans", "gmm"):
        clustering = _fit_best_cluster_model(
            train_rows=train_descriptors,
            test_rows=test_descriptors,
            algorithm=algorithm,
            feature_names=CLUSTER_FEATURE_NAMES,
            k_values=[2, 3, 4, 5, 6],
        )
        train_labeled = _attach_cluster_labels(train_descriptors, algorithm, clustering["train_labels"])
        test_labeled = _attach_cluster_labels(test_descriptors, algorithm, clustering["test_labels"])
        _write_descriptor_csv(tables_dir / f"{surface}_{algorithm}_train_condition_descriptors.csv", train_labeled)
        _write_descriptor_csv(tables_dir / f"{surface}_{algorithm}_test_condition_descriptors.csv", test_labeled)

        selection_path = _plot_cluster_selection(plots_dir, surface, algorithm, clustering["evaluations"])
        design_path = _plot_cluster_design_space(
            plots_dir,
            surface,
            algorithm,
            train_descriptors,
            test_descriptors,
            clustering["train_labels"],
            clustering["test_labels"],
        )
        pca_path = _plot_cluster_pca(plots_dir, surface, algorithm, clustering)
        flow_train_path = _plot_cluster_flow_map(plots_dir, surface, algorithm, train_descriptors, clustering["train_labels"], "train")
        flow_test_path = _plot_cluster_flow_map(plots_dir, surface, algorithm, test_descriptors, clustering["test_labels"], "test")

        plots = {
            "selection": str(selection_path),
            "design_space": str(design_path),
            "pca": str(pca_path),
            "flow_map_train": str(flow_train_path),
            "flow_map_test": str(flow_test_path),
        }
        clustering_payload[algorithm] = _cluster_payload(
            train_rows=train_descriptors,
            test_rows=test_descriptors,
            clustering=clustering,
            surface=surface,
            plots=plots,
        )
        comparison.append(
            {
                "algorithm": algorithm,
                "best_n_clusters": clustering["best"]["n_clusters"],
                "best_silhouette": clustering["best"]["silhouette"],
                "best_davies_bouldin": clustering["best"]["davies_bouldin"],
                "min_cluster_size": clustering["best"]["min_cluster_size"],
            }
        )

    recommended = max(comparison, key=lambda item: (item["best_silhouette"], -item["best_davies_bouldin"], item["min_cluster_size"]))

    payload = {
        "train": _condition_summary(train),
        "test": _condition_summary(test),
        "surface_reference": surface_summary,
        "cp_upper_test": cp_upper_test,
        "cp_upper_train": cp_upper_train,
        "flow_descriptors": {
            "surface": surface,
            "feature_names": CLUSTER_FEATURE_NAMES,
            "train_csv": str(descriptor_csv_train),
            "test_csv": str(descriptor_csv_test),
            "all_csv": str(descriptor_csv_all),
            "correlation_plot": str(corr_path),
            "n_train_conditions": len(train_descriptors),
            "n_test_conditions": len(test_descriptors),
        },
        "clustering": {
            "recommended_algorithm": recommended["algorithm"],
            "comparison": comparison,
            "kmeans": clustering_payload["kmeans"],
            "gmm": clustering_payload["gmm"],
        },
        "plots": {
            "design_space_train_test": str(plots_dir / "design_space_train_test.png"),
            "design_space_by_pi": str(plots_dir / "design_space_by_pi.png"),
            "regime_condition_counts": str(plots_dir / "regime_condition_counts.png"),
            "reference_surface_xy": str(plots_dir / "reference_surface_xy.png"),
            "cp_ranges_upper_test": str(plots_dir / "cp_ranges_upper_test.png"),
            "cp_ranges_upper_train": str(plots_dir / "cp_ranges_upper_train.png"),
            "flow_descriptor_correlations": str(corr_path),
        },
    }
    save_json(summary_dir / "dataset_characterization.json", payload)
    print(f"[explore-dataset] Characterization written to {root}")
    return payload
