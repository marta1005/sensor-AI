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

from .config import PipelineConfig
from .datasets import IndexedCpDataset
from .models import CpExpertNet

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


REGIME_NAMES = ["subsonic", "transonic", "supersonic"]


def _feature_paths(cfg: PipelineConfig, split: str) -> tuple[Path, Path, Path]:
    return (
        cfg.features_dir / f"expert_features_{split}.npy",
        cfg.features_dir / f"cp_{split}.npy",
        cfg.features_dir / f"expert_id_{split}.npy",
    )


def _evaluate(model: CpExpertNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    criterion = nn.SmoothL1Loss(beta=1.0)
    mse = nn.MSELoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            pred = model(bx)
            total_loss += criterion(pred, by).item() * bx.shape[0]
            total_mse += mse(pred, by).item()
            n += bx.shape[0]
    return {
        "smooth_l1": total_loss / max(1, n),
        "rmse_norm": float(np.sqrt(total_mse / max(1, n))),
    }


def _train_single_regime(cfg: PipelineConfig, regime_id: int, regime_name: str) -> None:
    train_x_path, train_cp_path, train_id_path = _feature_paths(cfg, "train")
    test_x_path, test_cp_path, test_id_path = _feature_paths(cfg, "test")

    train_ids = np.load(train_id_path, mmap_mode="r")
    test_ids = np.load(test_id_path, mmap_mode="r")
    train_idx = np.flatnonzero(np.asarray(train_ids) == regime_id)
    test_idx = np.flatnonzero(np.asarray(test_ids) == regime_id)

    if train_idx.size == 0:
        raise ValueError(f"No train samples for regime {regime_name}")

    train_set = IndexedCpDataset(str(train_x_path), str(train_cp_path), train_idx)
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

    input_dim = np.load(train_x_path, mmap_mode="r").shape[1]
    model = CpExpertNet(input_dim=input_dim).to(cfg.device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.expert_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.expert_epochs)

    history_train: list[float] = []
    history_test: list[float] = []

    for epoch in range(1, cfg.expert_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_rows = 0
        pbar = tqdm(train_loader, desc=f"[expert:{regime_name}] epoch {epoch}/{cfg.expert_epochs}")
        for bx, by in pbar:
            bx = bx.to(cfg.device, non_blocking=True)
            by = by.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.shape[0]
            n_rows += bx.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        scheduler.step()
        train_loss = epoch_loss / max(1, n_rows)
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
        "test_rows": int(test_idx.size),
        "final_train_smooth_l1": float(history_train[-1]),
        "final_test_smooth_l1": float(history_test[-1]),
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



def train_all_experts(cfg: PipelineConfig) -> None:
    cfg.ensure_dirs()
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        _train_single_regime(cfg, regime_id, regime_name)
    print(f"[train-experts] Finished. Models stored in {cfg.models_dir}")
