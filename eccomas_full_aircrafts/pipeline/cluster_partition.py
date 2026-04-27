from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import FullAircraftConfig
from .utils import raw_paths, regime_from_mach, save_json


MACH_EXPERT_NAMES = ["subsonic", "transonic", "supersonic"]
HYBRID_EXPERT_NAMES = ["negative_branch", "mild_branch", "positive_branch"]
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


def expert_names(cfg: FullAircraftConfig) -> list[str]:
    if cfg.expert_partition_mode == "mach":
        return MACH_EXPERT_NAMES[: cfg.n_experts]
    if cfg.expert_partition_mode == "hybrid":
        return HYBRID_EXPERT_NAMES[: cfg.n_experts]
    return [f"cluster{idx}" for idx in range(cfg.n_experts)]


def _descriptor_csv_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.project_root / "exploration_data" / "tables" / f"{cfg.reduced_surface}_{split}_condition_descriptors.csv"


def _cluster_cache_prefix(cfg: FullAircraftConfig) -> str:
    return f"{cfg.reduced_surface}_{cfg.cluster_algorithm}{cfg.cluster_count}"


def _cluster_cache_paths(cfg: FullAircraftConfig) -> tuple[Path, Path]:
    prefix = _cluster_cache_prefix(cfg)
    return (
        cfg.metadata_dir / f"{prefix}_condition_clusters.npz",
        cfg.metadata_dir / f"{prefix}_condition_clusters.json",
    )


def _hybrid_cache_paths(cfg: FullAircraftConfig) -> tuple[Path, Path]:
    prefix = f"{cfg.reduced_surface}_hybrid_kmeans{cfg.hybrid_source_clusters}"
    return (
        cfg.metadata_dir / f"{prefix}_condition_clusters.npz",
        cfg.metadata_dir / f"{prefix}_condition_clusters.json",
    )


def _load_descriptor_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Descriptor table not found: {path}. Run 'explore-dataset' first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _descriptor_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in CLUSTER_FEATURE_NAMES] for row in rows], dtype=np.float64)


def _fit_cluster_model(cfg: FullAircraftConfig):
    if cfg.cluster_algorithm == "kmeans":
        return KMeans(n_clusters=cfg.cluster_count, random_state=42, n_init=20)
    if cfg.cluster_algorithm == "gmm":
        return GaussianMixture(n_components=cfg.cluster_count, covariance_type="full", random_state=42, n_init=5)
    raise ValueError(f"Unsupported cluster algorithm: {cfg.cluster_algorithm}")


def _fit_condition_clusters(cfg: FullAircraftConfig) -> dict[str, Any]:
    if cfg.cluster_count != cfg.n_experts:
        raise ValueError(
            f"cluster_count ({cfg.cluster_count}) must match n_experts ({cfg.n_experts}) "
            "for cluster-based expert training."
        )

    train_rows = _load_descriptor_rows(_descriptor_csv_path(cfg, "train"))
    test_rows = _load_descriptor_rows(_descriptor_csv_path(cfg, "test"))
    train_x = _descriptor_matrix(train_rows)
    test_x = _descriptor_matrix(test_rows)

    scaler = StandardScaler()
    train_scaled = np.asarray(scaler.fit_transform(train_x), dtype=np.float64)
    test_scaled = np.asarray(scaler.transform(test_x), dtype=np.float64)

    model = _fit_cluster_model(cfg)
    if cfg.cluster_algorithm == "kmeans":
        train_labels = model.fit_predict(train_scaled)
        test_labels = model.predict(test_scaled)
    else:
        model.fit(train_scaled)
        train_labels = model.predict(train_scaled)
        test_labels = model.predict(test_scaled)

    sil = float(silhouette_score(train_scaled, train_labels))
    dbi = float(davies_bouldin_score(train_scaled, train_labels))
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_test, counts_test = np.unique(test_labels, return_counts=True)

    npz_path, json_path = _cluster_cache_paths(cfg)
    np.savez_compressed(
        npz_path,
        train_labels=train_labels.astype(np.int64),
        test_labels=test_labels.astype(np.int64),
    )
    save_json(
        json_path,
        {
            "partition_mode": cfg.expert_partition_mode,
            "cluster_algorithm": cfg.cluster_algorithm,
            "cluster_count": int(cfg.cluster_count),
            "feature_names": CLUSTER_FEATURE_NAMES,
            "surface": cfg.reduced_surface,
            "train_cluster_counts": {str(int(k)): int(v) for k, v in zip(unique_train, counts_train)},
            "test_cluster_counts": {str(int(k)): int(v) for k, v in zip(unique_test, counts_test)},
            "silhouette_train": sil,
            "davies_bouldin_train": dbi,
        },
    )
    return {
        "train_labels": train_labels.astype(np.int64),
        "test_labels": test_labels.astype(np.int64),
    }


def _fit_hybrid_condition_clusters(cfg: FullAircraftConfig) -> dict[str, Any]:
    if cfg.n_experts != 3:
        raise ValueError(f"Hybrid partition expects n_experts=3, got {cfg.n_experts}.")
    if cfg.hybrid_source_clusters < 5:
        raise ValueError(
            f"hybrid_source_clusters should be at least 5 for the negative/mild/positive merge, got {cfg.hybrid_source_clusters}."
        )

    train_rows = _load_descriptor_rows(_descriptor_csv_path(cfg, "train"))
    test_rows = _load_descriptor_rows(_descriptor_csv_path(cfg, "test"))
    train_x = _descriptor_matrix(train_rows)
    test_x = _descriptor_matrix(test_rows)

    scaler = StandardScaler()
    train_scaled = np.asarray(scaler.fit_transform(train_x), dtype=np.float64)
    test_scaled = np.asarray(scaler.transform(test_x), dtype=np.float64)

    source_model = KMeans(n_clusters=cfg.hybrid_source_clusters, random_state=42, n_init=20)
    source_train = source_model.fit_predict(train_scaled)
    source_test = source_model.predict(test_scaled)

    aoa_idx = CLUSTER_FEATURE_NAMES.index("AoA_deg")
    source_ids = np.arange(cfg.hybrid_source_clusters, dtype=np.int64)
    aoa_means = {
        int(cluster_id): float(train_x[source_train == cluster_id, aoa_idx].mean())
        for cluster_id in source_ids
    }
    ordered = sorted(source_ids.tolist(), key=lambda cluster_id: aoa_means[int(cluster_id)])
    if len(ordered) < 5:
        raise ValueError("Hybrid source clustering produced fewer than 5 populated clusters.")

    merge_map = {
        int(ordered[0]): 0,
        int(ordered[1]): 0,
        int(ordered[2]): 1,
        int(ordered[3]): 2,
        int(ordered[4]): 2,
    }
    for cluster_id in ordered[5:]:
        aoa_value = aoa_means[int(cluster_id)]
        merge_map[int(cluster_id)] = 0 if aoa_value < 0.0 else 2

    hybrid_train = np.asarray([merge_map[int(cluster_id)] for cluster_id in source_train], dtype=np.int64)
    hybrid_test = np.asarray([merge_map[int(cluster_id)] for cluster_id in source_test], dtype=np.int64)

    unique_train, counts_train = np.unique(hybrid_train, return_counts=True)
    unique_test, counts_test = np.unique(hybrid_test, return_counts=True)
    source_unique_train, source_counts_train = np.unique(source_train, return_counts=True)
    source_unique_test, source_counts_test = np.unique(source_test, return_counts=True)

    npz_path, json_path = _hybrid_cache_paths(cfg)
    np.savez_compressed(
        npz_path,
        train_labels=hybrid_train.astype(np.int64),
        test_labels=hybrid_test.astype(np.int64),
        source_train_labels=source_train.astype(np.int64),
        source_test_labels=source_test.astype(np.int64),
    )
    save_json(
        json_path,
        {
            "partition_mode": cfg.expert_partition_mode,
            "surface": cfg.reduced_surface,
            "hybrid_source_algorithm": "kmeans",
            "hybrid_source_clusters": int(cfg.hybrid_source_clusters),
            "expert_names": HYBRID_EXPERT_NAMES,
            "feature_names": CLUSTER_FEATURE_NAMES,
            "source_cluster_aoa_means": {str(int(k)): float(v) for k, v in aoa_means.items()},
            "source_cluster_merge_map": {str(int(k)): int(v) for k, v in merge_map.items()},
            "source_train_cluster_counts": {str(int(k)): int(v) for k, v in zip(source_unique_train, source_counts_train)},
            "source_test_cluster_counts": {str(int(k)): int(v) for k, v in zip(source_unique_test, source_counts_test)},
            "train_cluster_counts": {str(int(k)): int(v) for k, v in zip(unique_train, counts_train)},
            "test_cluster_counts": {str(int(k)): int(v) for k, v in zip(unique_test, counts_test)},
        },
    )
    return {
        "train_labels": hybrid_train.astype(np.int64),
        "test_labels": hybrid_test.astype(np.int64),
    }


def load_condition_partition_labels(cfg: FullAircraftConfig, split: str) -> np.ndarray:
    if cfg.expert_partition_mode == "mach":
        x_path, _ = raw_paths(cfg.raw_data_dir, split)
        x_raw = np.load(x_path, mmap_mode="r")
        mach = np.asarray(x_raw[:: cfg.raw_points_per_condition, 6], dtype=np.float32)
        return regime_from_mach(mach, cfg.mach_sub_max, cfg.mach_trans_max)

    if cfg.expert_partition_mode == "hybrid":
        npz_path, _json_path = _hybrid_cache_paths(cfg)
        if not npz_path.exists():
            _fit_hybrid_condition_clusters(cfg)
        payload = np.load(npz_path)
        key = f"{split}_labels"
        if key not in payload:
            raise KeyError(f"Missing {key} in {npz_path}")
        return np.asarray(payload[key], dtype=np.int64)

    npz_path, _json_path = _cluster_cache_paths(cfg)
    if not npz_path.exists():
        _fit_condition_clusters(cfg)
    payload = np.load(npz_path)
    key = f"{split}_labels"
    if key not in payload:
        raise KeyError(f"Missing {key} in {npz_path}")
    return np.asarray(payload[key], dtype=np.int64)
