from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .config import PipelineConfig
from .features import ENCODER_FEATURE_NAMES, EXPERT_FEATURE_NAMES, build_encoder_features, build_expert_features
from .utils import regime_from_mach, save_json


def _cut_paths(cfg: PipelineConfig, split: str) -> tuple[Path, Path]:
    return cfg.cut_data_dir / f"X_cut_{split}.npy", cfg.cut_data_dir / f"Y_cut_{split}.npy"


def _feature_paths(cfg: PipelineConfig, split: str) -> dict[str, Path]:
    return {
        "expert": cfg.features_dir / f"expert_features_{split}.npy",
        "gate": cfg.features_dir / f"gate_features_{split}.npy",
        "cp": cfg.features_dir / f"cp_{split}.npy",
        "expert_id": cfg.features_dir / f"expert_id_{split}.npy",
    }


def _stats_from_builder(x_path: Path, builder, cfg: PipelineConfig, batch_rows: int = 400_000) -> tuple[np.ndarray, np.ndarray]:
    x_raw = np.load(x_path, mmap_mode="r")
    total_count = 0
    total_sum = None
    total_sum_sq = None
    for start in range(0, x_raw.shape[0], batch_rows):
        end = min(x_raw.shape[0], start + batch_rows)
        feats = builder(np.asarray(x_raw[start:end, : cfg.input_dim_raw], dtype=np.float32), cfg.cut_threshold_y)
        feats64 = feats.astype(np.float64)
        chunk_sum = feats64.sum(axis=0)
        chunk_sum_sq = np.square(feats64).sum(axis=0)
        if total_sum is None:
            total_sum = chunk_sum
            total_sum_sq = chunk_sum_sq
        else:
            total_sum += chunk_sum
            total_sum_sq += chunk_sum_sq
        total_count += feats.shape[0]
        if start == 0 or end == x_raw.shape[0] or (start // batch_rows) % 20 == 0:
            print(f"[prepare-features] accumulate stats rows {end}/{x_raw.shape[0]}")
    mean = total_sum / total_count
    var = np.maximum(total_sum_sq / total_count - mean * mean, 1e-12)
    scale = np.sqrt(var)
    return mean.astype(np.float32), scale.astype(np.float32)


def _stats_from_cp(y_path: Path, batch_rows: int = 800_000) -> tuple[np.ndarray, np.ndarray]:
    y_raw = np.load(y_path, mmap_mode="r")
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    for start in range(0, y_raw.shape[0], batch_rows):
        end = min(y_raw.shape[0], start + batch_rows)
        cp = np.asarray(y_raw[start:end, 0], dtype=np.float64)
        total_sum += cp.sum()
        total_sum_sq += np.square(cp).sum()
        total_count += cp.shape[0]
    mean = np.array([total_sum / total_count], dtype=np.float32)
    var = max(total_sum_sq / total_count - float(mean[0] ** 2), 1e-12)
    scale = np.array([np.sqrt(var)], dtype=np.float32)
    return mean, scale


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def _transform_split(
    cfg: PipelineConfig,
    split: str,
    expert_mean: np.ndarray,
    expert_scale: np.ndarray,
    gate_mean: np.ndarray,
    gate_scale: np.ndarray,
    cp_mean: np.ndarray,
    cp_scale: np.ndarray,
    batch_rows: int = 400_000,
) -> dict[str, int]:
    x_path, y_path = _cut_paths(cfg, split)
    x_raw = np.load(x_path, mmap_mode="r")
    y_raw = np.load(y_path, mmap_mode="r")
    paths = _feature_paths(cfg, split)

    n_rows = x_raw.shape[0]
    expert_example = build_expert_features(np.asarray(x_raw[:1, : cfg.input_dim_raw], dtype=np.float32), cfg.cut_threshold_y)
    gate_example = build_encoder_features(np.asarray(x_raw[:1, : cfg.input_dim_raw], dtype=np.float32), cfg.cut_threshold_y)

    expert_out = open_memmap(paths["expert"], mode="w+", dtype=np.float32, shape=(n_rows, expert_example.shape[1]))
    gate_out = open_memmap(paths["gate"], mode="w+", dtype=np.float32, shape=(n_rows, gate_example.shape[1]))
    cp_out = open_memmap(paths["cp"], mode="w+", dtype=np.float32, shape=(n_rows, 1))
    expert_id_out = open_memmap(paths["expert_id"], mode="w+", dtype=np.int64, shape=(n_rows,))

    for start in range(0, n_rows, batch_rows):
        end = min(n_rows, start + batch_rows)
        x_chunk = np.asarray(x_raw[start:end, : cfg.input_dim_raw], dtype=np.float32)
        y_chunk = np.asarray(y_raw[start:end, [0]], dtype=np.float32)

        expert_chunk = build_expert_features(x_chunk, cfg.cut_threshold_y)
        gate_chunk = build_encoder_features(x_chunk, cfg.cut_threshold_y)
        mach = x_chunk[:, 6]
        expert_ids = regime_from_mach(mach, cfg.mach_sub_max, cfg.mach_trans_max)

        expert_out[start:end] = _standardize(expert_chunk, expert_mean, expert_scale)
        gate_out[start:end] = _standardize(gate_chunk, gate_mean, gate_scale)
        cp_out[start:end] = _standardize(y_chunk, cp_mean, cp_scale)
        expert_id_out[start:end] = expert_ids

        if start == 0 or end == n_rows or (start // batch_rows) % 20 == 0:
            print(f"[prepare-features] transform {split}: {end}/{n_rows}")

    return {
        "rows": int(n_rows),
        "expert_dim": int(expert_example.shape[1]),
        "gate_dim": int(gate_example.shape[1]),
    }


def prepare_feature_store(cfg: PipelineConfig) -> None:
    cfg.ensure_dirs()

    train_x_path, train_y_path = _cut_paths(cfg, "train")
    if not train_x_path.exists() or not train_y_path.exists():
        raise FileNotFoundError("Cut data not found. Run 'prepare-data' first.")

    expert_mean, expert_scale = _stats_from_builder(train_x_path, build_expert_features, cfg)
    gate_mean, gate_scale = _stats_from_builder(train_x_path, build_encoder_features, cfg)
    cp_mean, cp_scale = _stats_from_cp(train_y_path)

    np.savez(cfg.scalers_dir / "expert_scaler.npz", mean=expert_mean, scale=expert_scale)
    np.savez(cfg.scalers_dir / "gate_scaler.npz", mean=gate_mean, scale=gate_scale)
    np.savez(cfg.scalers_dir / "cp_scaler.npz", mean=cp_mean, scale=cp_scale)

    train_meta = _transform_split(cfg, "train", expert_mean, expert_scale, gate_mean, gate_scale, cp_mean, cp_scale)
    test_meta = _transform_split(cfg, "test", expert_mean, expert_scale, gate_mean, gate_scale, cp_mean, cp_scale)

    save_json(
        cfg.features_dir / "metadata.json",
        {
            "expert_feature_names": EXPERT_FEATURE_NAMES,
            "encoder_feature_names": ENCODER_FEATURE_NAMES,
            "train": train_meta,
            "test": test_meta,
        },
    )
    print(f"[prepare-features] Finished. Features stored in {cfg.features_dir}")
