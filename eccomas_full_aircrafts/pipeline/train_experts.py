from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from eccomas_sensor_pipeline.eccomas_sensor.datasets import IndexedCpDataset, WeightedIndexedCpDataset
from eccomas_sensor_pipeline.eccomas_sensor.models import CpExpertNet
from eccomas_sensor_pipeline.eccomas_sensor.train_experts import (
    REGIME_NAMES,
    _evaluate,
    _regime_sample_weights,
    _weighted_smooth_l1,
)

from .config import FullAircraftConfig

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _feature_paths(cfg: FullAircraftConfig, split: str) -> tuple[Path, Path, Path]:
    return (
        cfg.features_dir / f"expert_features_{split}.npy",
        cfg.features_dir / f"cp_{split}.npy",
        cfg.features_dir / f"expert_id_{split}.npy",
    )


def _cache_path(cfg: FullAircraftConfig) -> Path:
    return cfg.features_dir / "expert_regime_cache.npz"


def _build_regime_cache(cfg: FullAircraftConfig) -> dict[str, np.ndarray]:
    train_raw_path = cfg.cut_data_dir / "X_cut_train.npy"
    test_id_path = _feature_paths(cfg, "test")[2]

    train_raw = np.load(train_raw_path, mmap_mode="r")
    test_ids = np.load(test_id_path, mmap_mode="r")

    print("[train-experts] building regime cache from reduced full-aircraft arrays")
    train_mach = np.asarray(train_raw[:, 6], dtype=np.float32)

    cache: dict[str, np.ndarray] = {}
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        train_weights_full = _regime_sample_weights(train_mach, cfg, regime_id)
        train_idx = np.flatnonzero(train_weights_full > 0.0)
        train_weights = train_weights_full[train_idx]
        test_idx = np.flatnonzero(np.asarray(test_ids) == regime_id)

        cache[f"{regime_name}_train_idx"] = train_idx.astype(np.int64)
        cache[f"{regime_name}_train_weights"] = train_weights.astype(np.float32)
        cache[f"{regime_name}_test_idx"] = test_idx.astype(np.int64)

        print(
            f"[train-experts] cache {regime_name}: "
            f"train_rows={train_idx.size:,}, test_rows={test_idx.size:,}"
        )

    np.savez_compressed(_cache_path(cfg), **cache)
    return cache


def _load_or_build_regime_cache(cfg: FullAircraftConfig) -> dict[str, np.ndarray]:
    path = _cache_path(cfg)
    if path.exists():
        payload = np.load(path)
        print(f"[train-experts] using cached regime split metadata from {path}")
        return {key: payload[key] for key in payload.files}
    return _build_regime_cache(cfg)


def _train_single_regime(
    cfg: FullAircraftConfig,
    regime_id: int,
    regime_name: str,
    train_x_path: Path,
    train_cp_path: Path,
    test_x_path: Path,
    test_cp_path: Path,
    input_dim: int,
    regime_cache: dict[str, np.ndarray],
) -> None:
    train_idx = np.asarray(regime_cache[f"{regime_name}_train_idx"], dtype=np.int64)
    train_weights = np.asarray(regime_cache[f"{regime_name}_train_weights"], dtype=np.float32)
    test_idx = np.asarray(regime_cache[f"{regime_name}_test_idx"], dtype=np.int64)

    if train_idx.size == 0:
        raise ValueError(f"No train samples for regime {regime_name}")

    train_set = WeightedIndexedCpDataset(str(train_x_path), str(train_cp_path), train_idx, train_weights)
    test_set = IndexedCpDataset(str(test_x_path), str(test_cp_path), test_idx if test_idx.size > 0 else None)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.expert_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.expert_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CpExpertNet(input_dim=input_dim).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.expert_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.expert_epochs)

    history_train: list[float] = []
    history_test: list[float] = []

    print(
        f"[train-experts] start {regime_name}: "
        f"train_rows={train_idx.size:,}, test_rows={test_idx.size:,}, device={cfg.device}"
    )

    for epoch in range(1, cfg.expert_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[expert:{regime_name}] epoch {epoch}/{cfg.expert_epochs}")
        for bx, by, bw in pbar:
            bx = bx.to(cfg.device, non_blocking=True)
            by = by.to(cfg.device, non_blocking=True)
            bw = bw.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = _weighted_smooth_l1(pred, by, bw)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bw.sum().item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        scheduler.step()

        train_loss = epoch_loss / max(1.0, float(train_weights.sum()))
        test_metrics = _evaluate(model, test_loader, cfg.device)
        history_train.append(train_loss)
        history_test.append(test_metrics["smooth_l1"])
        print(
            f"[expert:{regime_name}] epoch {epoch:03d} | train={train_loss:.6f} | "
            f"test={test_metrics['smooth_l1']:.6f} | rmse_norm={test_metrics['rmse_norm']:.6f}"
        )

    model_path = cfg.models_dir / f"expert_{regime_name}.pth"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "regime": regime_name,
        "train_rows": int(train_idx.size),
        "train_weight_sum": float(train_weights.sum()),
        "test_rows": int(test_idx.size),
        "final_train_smooth_l1": float(history_train[-1]),
        "final_test_smooth_l1": float(history_test[-1]),
        "overlap_margin": float(cfg.expert_overlap_margin),
        "overlap_min_weight": float(cfg.expert_overlap_min_weight),
    }
    with (cfg.metrics_dir / f"expert_{regime_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plt.figure(figsize=(8, 4.8))
    plt.plot(history_train, label="train")
    plt.plot(history_test, label="test")
    plt.xlabel("epoch")
    plt.ylabel("SmoothL1")
    plt.title(f"Expert {regime_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / f"expert_loss_{regime_name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def train_all_experts(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()

    train_x_path, train_cp_path, _ = _feature_paths(cfg, "train")
    test_x_path, test_cp_path, _ = _feature_paths(cfg, "test")
    input_dim = int(np.load(train_x_path, mmap_mode="r").shape[1])

    regime_cache = _load_or_build_regime_cache(cfg)

    for regime_id, regime_name in enumerate(REGIME_NAMES):
        _train_single_regime(
            cfg,
            regime_id=regime_id,
            regime_name=regime_name,
            train_x_path=train_x_path,
            train_cp_path=train_cp_path,
            test_x_path=test_x_path,
            test_cp_path=test_cp_path,
            input_dim=input_dim,
            regime_cache=regime_cache,
        )

    print(f"[train-experts] Finished. Models stored in {cfg.models_dir}")
