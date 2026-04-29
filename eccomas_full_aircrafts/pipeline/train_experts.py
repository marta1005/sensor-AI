from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.lib.format import open_memmap
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .cluster_partition import expert_names, load_condition_partition_labels
from .config import FullAircraftConfig
from .models import FullAircraftExpertUNet
from .surface_grid import CompactSurfaceGrid
from .utils import regime_from_mach

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _regime_sample_weights(mach: np.ndarray, cfg: FullAircraftConfig, regime_id: int) -> np.ndarray:
    mach = np.asarray(mach, dtype=np.float32)
    weights = np.zeros_like(mach, dtype=np.float32)
    margin = float(cfg.expert_overlap_margin)
    floor = float(cfg.expert_overlap_min_weight)

    if regime_id == 0:
        core = mach <= cfg.mach_sub_max
        weights[core] = 1.0
        if margin > 0.0:
            overlap = (mach > cfg.mach_sub_max) & (mach <= cfg.mach_sub_max + margin)
            frac = (mach[overlap] - cfg.mach_sub_max) / margin
            weights[overlap] = 1.0 - (1.0 - floor) * frac
    elif regime_id == 1:
        core = (mach >= cfg.mach_sub_max) & (mach <= cfg.mach_trans_max)
        weights[core] = 1.0
        if margin > 0.0:
            lower = (mach >= cfg.mach_sub_max - margin) & (mach < cfg.mach_sub_max)
            lower_frac = (mach[lower] - (cfg.mach_sub_max - margin)) / margin
            weights[lower] = floor + (1.0 - floor) * lower_frac

            upper = (mach > cfg.mach_trans_max) & (mach <= cfg.mach_trans_max + margin)
            upper_frac = 1.0 - (mach[upper] - cfg.mach_trans_max) / margin
            weights[upper] = floor + (1.0 - floor) * upper_frac
    else:
        core = mach >= cfg.mach_trans_max
        weights[core] = 1.0
        if margin > 0.0:
            overlap = (mach >= cfg.mach_trans_max - margin) & (mach < cfg.mach_trans_max)
            frac = (cfg.mach_trans_max - mach[overlap]) / margin
            weights[overlap] = 1.0 - (1.0 - floor) * frac

    return weights.astype(np.float32)


def _feature_paths(cfg: FullAircraftConfig, split: str) -> tuple[Path, Path, Path]:
    return (
        cfg.features_dir / f"expert_features_{split}.npy",
        cfg.features_dir / f"cp_{split}.npy",
        cfg.features_dir / f"expert_id_{split}.npy",
    )


def _prediction_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"expert_pred_{split}.npy"


def _expert_model_config_path(cfg: FullAircraftConfig) -> Path:
    return cfg.models_dir / "expert_model_config.json"


@dataclass(frozen=True)
class _ConditionTable:
    mach: np.ndarray
    aoa_deg: np.ndarray
    partition_label: np.ndarray
    n_conditions: int
    points_per_condition: int


def _condition_table(cfg: FullAircraftConfig, split: str, grid: CompactSurfaceGrid) -> _ConditionTable:
    x_path = cfg.cut_data_dir / f"X_cut_{split}.npy"
    x_raw = np.load(x_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // grid.n_points)
    mach = np.asarray(x_raw[:: grid.n_points, 6], dtype=np.float32)[:n_conditions]
    aoa_deg = np.asarray(x_raw[:: grid.n_points, 7], dtype=np.float32)[:n_conditions]
    if cfg.expert_partition_mode == "mach":
        partition_label = regime_from_mach(mach, cfg.mach_sub_max, cfg.mach_trans_max)
    else:
        partition_label = load_condition_partition_labels(cfg, split)
        if partition_label.shape[0] != n_conditions:
            raise ValueError(
                f"Condition labels ({partition_label.shape[0]}) do not match reduced conditions ({n_conditions}) "
                f"for split={split}."
            )
    return _ConditionTable(
        mach=mach,
        aoa_deg=aoa_deg,
        partition_label=np.asarray(partition_label, dtype=np.int64),
        n_conditions=n_conditions,
        points_per_condition=grid.n_points,
    )


def _hybrid_positive_branch_weights(
    cfg: FullAircraftConfig,
    regime_name: str,
    train_table: _ConditionTable,
    condition_indices: np.ndarray,
) -> np.ndarray:
    weights = np.ones(condition_indices.shape[0], dtype=np.float32)
    if cfg.expert_partition_mode != "hybrid" or regime_name != "positive_branch" or condition_indices.size == 0:
        return weights

    mach = train_table.mach[condition_indices]
    aoa_deg = train_table.aoa_deg[condition_indices]

    focus_mask = (mach <= cfg.positive_branch_focus_mach_max) & (aoa_deg >= cfg.positive_branch_focus_aoa_deg)
    extreme_mask = (mach <= cfg.positive_branch_extreme_mach_max) & (aoa_deg >= cfg.positive_branch_extreme_aoa_deg)

    weights[focus_mask] = np.maximum(weights[focus_mask], np.float32(cfg.positive_branch_focus_weight))
    weights[extreme_mask] = np.maximum(weights[extreme_mask], np.float32(cfg.positive_branch_extreme_weight))
    return weights


class _FieldConditionDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        cp_path: Path,
        grid: CompactSurfaceGrid,
        condition_indices: np.ndarray,
        condition_weights: np.ndarray | None = None,
    ):
        self.features = np.load(feature_path, mmap_mode="r")
        self.cp = np.load(cp_path, mmap_mode="r")
        self.grid = grid
        self.condition_indices = np.asarray(condition_indices, dtype=np.int64)
        self.condition_weights = (
            np.asarray(condition_weights, dtype=np.float32)
            if condition_weights is not None
            else np.ones(self.condition_indices.shape[0], dtype=np.float32)
        )
        self.points_per_condition = grid.n_points
        self.mask = grid.valid_mask[None, ...].astype(np.float32)

    def __len__(self) -> int:
        return int(self.condition_indices.shape[0])

    def __getitem__(self, idx: int):
        cond_idx = int(self.condition_indices[idx])
        row_start = cond_idx * self.points_per_condition
        row_stop = row_start + self.points_per_condition

        feat_flat = np.asarray(self.features[row_start:row_stop], dtype=np.float32)
        cp_flat = np.asarray(self.cp[row_start:row_stop], dtype=np.float32)

        feat_grid = self.grid.scatter_numpy(feat_flat)
        cp_grid = self.grid.scatter_numpy(cp_flat)
        feat_grid = np.concatenate([feat_grid, self.mask], axis=0)
        weight = np.array(self.condition_weights[idx], dtype=np.float32)

        return (
            torch.from_numpy(feat_grid),
            torch.from_numpy(cp_grid),
            torch.from_numpy(self.mask),
            torch.from_numpy(weight),
        )


def _masked_weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    per_sample = loss.flatten(1).sum(dim=1) / denom
    weights = sample_weight.reshape(-1).clamp_min(1e-8)
    return (per_sample * weights).sum() / weights.sum()


def _evaluate(model: FullAircraftExpertUNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_s1 = 0.0
    total_rmse = 0.0
    count = 0
    with torch.no_grad():
        for feat, cp, mask, _weight in loader:
            feat = feat.to(device, non_blocking=True)
            cp = cp.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            pred = model(feat)
            diff = (pred - cp) * mask
            denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
            s1 = F.smooth_l1_loss(pred, cp, reduction="none")
            s1 = (s1 * mask).flatten(1).sum(dim=1) / denom
            rmse = torch.sqrt((diff.square().flatten(1).sum(dim=1) / denom).clamp_min(1e-12))

            total_s1 += s1.sum().item()
            total_rmse += rmse.sum().item()
            count += pred.shape[0]

    return {
        "smooth_l1": total_s1 / max(1, count),
        "rmse_norm": total_rmse / max(1, count),
    }


def _predict_split(
    cfg: FullAircraftConfig,
    model: FullAircraftExpertUNet,
    split: str,
    grid: CompactSurfaceGrid,
) -> np.ndarray:
    x_path, _, _ = _feature_paths(cfg, split)
    features = np.load(x_path, mmap_mode="r")
    n_conditions = int(features.shape[0] // grid.n_points)
    preds = np.zeros((features.shape[0], 1), dtype=np.float32)
    mask = grid.valid_mask[None, ...].astype(np.float32)

    model.eval()
    with torch.no_grad():
        for cond_idx in range(n_conditions):
            row_start = cond_idx * grid.n_points
            row_stop = row_start + grid.n_points
            feat_flat = np.asarray(features[row_start:row_stop], dtype=np.float32)
            feat_grid = grid.scatter_numpy(feat_flat)
            feat_grid = np.concatenate([feat_grid, mask], axis=0)[None, ...]

            pred_grid = model(torch.from_numpy(feat_grid).to(cfg.device, non_blocking=True)).detach().cpu().numpy()
            pred_flat = grid.gather_numpy(pred_grid)[0]
            preds[row_start:row_stop] = pred_flat

            if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
                print(f"[train-experts] export {split}: condition {cond_idx + 1}/{n_conditions}")

    return preds


def _save_predictions(
    cfg: FullAircraftConfig,
    regime_id: int,
    split: str,
    preds: np.ndarray,
) -> None:
    out_path = _prediction_path(cfg, split)
    if out_path.exists():
        out = np.load(out_path, mmap_mode="r+")
        out[:, regime_id] = preds[:, 0]
        return

    out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(preds.shape[0], cfg.n_experts))
    out[:] = 0.0
    out[:, regime_id] = preds[:, 0]


def _save_expert_model_config(cfg: FullAircraftConfig, grid: CompactSurfaceGrid, input_channels: int) -> None:
    payload = {
        "expert_model_architecture": cfg.expert_model_architecture,
        "expert_partition_mode": cfg.expert_partition_mode,
        "cluster_algorithm": cfg.cluster_algorithm,
        "cluster_count": int(cfg.cluster_count),
        "expert_names": expert_names(cfg),
        "input_channels": int(input_channels),
        "base_channels": int(cfg.expert_unet_base_channels),
        "grid_height": int(grid.height),
        "grid_width": int(grid.width),
        "points_per_condition": int(grid.n_points),
    }
    with _expert_model_config_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _train_single_regime(
    cfg: FullAircraftConfig,
    regime_id: int,
    regime_name: str,
    grid: CompactSurfaceGrid,
    train_table: _ConditionTable,
    test_table: _ConditionTable,
    feature_dim: int,
) -> None:
    if cfg.expert_partition_mode == "mach":
        train_weights_full = _regime_sample_weights(train_table.mach, cfg, regime_id).astype(np.float32)
        train_condition_idx = np.flatnonzero(train_weights_full > 0.0).astype(np.int64)
        train_condition_weights = train_weights_full[train_condition_idx]
    else:
        train_condition_idx = np.flatnonzero(train_table.partition_label == regime_id).astype(np.int64)
        train_condition_weights = _hybrid_positive_branch_weights(cfg, regime_name, train_table, train_condition_idx)
    test_condition_idx = np.flatnonzero(test_table.partition_label == regime_id).astype(np.int64)

    positive_focus_count = 0
    positive_extreme_count = 0
    if cfg.expert_partition_mode == "hybrid" and regime_name == "positive_branch" and train_condition_idx.size > 0:
        mach = train_table.mach[train_condition_idx]
        aoa_deg = train_table.aoa_deg[train_condition_idx]
        focus_mask = (mach <= cfg.positive_branch_focus_mach_max) & (aoa_deg >= cfg.positive_branch_focus_aoa_deg)
        extreme_mask = (mach <= cfg.positive_branch_extreme_mach_max) & (aoa_deg >= cfg.positive_branch_extreme_aoa_deg)
        positive_focus_count = int(focus_mask.sum())
        positive_extreme_count = int(extreme_mask.sum())

    train_x_path, train_cp_path, _ = _feature_paths(cfg, "train")
    test_x_path, test_cp_path, _ = _feature_paths(cfg, "test")

    train_set = _FieldConditionDataset(train_x_path, train_cp_path, grid, train_condition_idx, train_condition_weights)
    test_set = _FieldConditionDataset(test_x_path, test_cp_path, grid, test_condition_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.expert_field_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(1, cfg.expert_field_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FullAircraftExpertUNet(input_channels=feature_dim + 1, base_channels=cfg.expert_unet_base_channels).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.expert_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.expert_epochs)

    history_train: list[float] = []
    history_test: list[float] = []

    print(
        f"[train-experts] start {regime_name}: "
        f"train_conditions={train_condition_idx.size:,}, test_conditions={test_condition_idx.size:,}, "
        f"grid={grid.height}x{grid.width}, device={cfg.device}, "
        f"train_weight_sum={train_condition_weights.sum():.1f}"
    )
    if cfg.expert_partition_mode == "hybrid" and regime_name == "positive_branch":
        print(
            f"[train-experts] positive_branch focus cases={positive_focus_count:,} "
            f"(w={cfg.positive_branch_focus_weight:.2f}), "
            f"extreme cases={positive_extreme_count:,} "
            f"(w={cfg.positive_branch_extreme_weight:.2f})"
        )

    for epoch in range(1, cfg.expert_epochs + 1):
        model.train()
        total_weighted = 0.0
        total_weight = 0.0
        pbar = tqdm(train_loader, desc=f"[expert-unet:{regime_name}] epoch {epoch}/{cfg.expert_epochs}")
        for feat, cp, mask, weight in pbar:
            feat = feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)
            weight = weight.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(feat)
            loss = _masked_weighted_smooth_l1(pred, cp, mask, weight)
            loss.backward()
            optimizer.step()

            total_weighted += loss.item() * float(weight.sum().item())
            total_weight += float(weight.sum().item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        train_loss = total_weighted / max(1e-8, total_weight)
        test_metrics = _evaluate(model, test_loader, cfg.device)
        history_train.append(train_loss)
        history_test.append(test_metrics["smooth_l1"])
        print(
            f"[expert-unet:{regime_name}] epoch {epoch:03d} | train={train_loss:.6f} | "
            f"test={test_metrics['smooth_l1']:.6f} | rmse_norm={test_metrics['rmse_norm']:.6f}"
        )

    model_path = cfg.models_dir / f"expert_{regime_name}.pth"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "regime": regime_name,
        "partition_mode": cfg.expert_partition_mode,
        "model_architecture": cfg.expert_model_architecture,
        "train_conditions": int(train_condition_idx.size),
        "train_rows": int(train_condition_idx.size * grid.n_points),
        "train_weight_sum": float(train_condition_weights.sum()),
        "test_conditions": int(test_condition_idx.size),
        "test_rows": int(test_condition_idx.size * grid.n_points),
        "final_train_smooth_l1": float(history_train[-1]),
        "final_test_smooth_l1": float(history_test[-1]),
    }
    if cfg.expert_partition_mode == "mach":
        metrics["overlap_margin"] = float(cfg.expert_overlap_margin)
        metrics["overlap_min_weight"] = float(cfg.expert_overlap_min_weight)
    if cfg.expert_partition_mode == "hybrid" and regime_name == "positive_branch":
        metrics["positive_branch_focus_mach_max"] = float(cfg.positive_branch_focus_mach_max)
        metrics["positive_branch_focus_aoa_deg"] = float(cfg.positive_branch_focus_aoa_deg)
        metrics["positive_branch_focus_weight"] = float(cfg.positive_branch_focus_weight)
        metrics["positive_branch_extreme_mach_max"] = float(cfg.positive_branch_extreme_mach_max)
        metrics["positive_branch_extreme_aoa_deg"] = float(cfg.positive_branch_extreme_aoa_deg)
        metrics["positive_branch_extreme_weight"] = float(cfg.positive_branch_extreme_weight)
        metrics["positive_branch_focus_case_count"] = int(positive_focus_count)
        metrics["positive_branch_extreme_case_count"] = int(positive_extreme_count)
    with (cfg.metrics_dir / f"expert_{regime_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plt.figure(figsize=(8, 4.8))
    plt.plot(history_train, label="train")
    plt.plot(history_test, label="test")
    plt.xlabel("epoch")
    plt.ylabel("SmoothL1")
    plt.title(f"Expert {regime_name} ({cfg.expert_model_architecture})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / f"expert_loss_{regime_name}.png", dpi=220, bbox_inches="tight")
    plt.close()

    for split in ("train", "test"):
        preds = _predict_split(cfg, model, split, grid)
        _save_predictions(cfg, regime_id, split, preds)


def train_all_experts(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    grid = CompactSurfaceGrid.from_reference(cfg)
    regime_names = expert_names(cfg)
    if len(regime_names) != cfg.n_experts:
        raise ValueError(f"Expert name count ({len(regime_names)}) does not match n_experts ({cfg.n_experts}).")

    train_x_path, _, _ = _feature_paths(cfg, "train")
    feature_dim = int(np.load(train_x_path, mmap_mode="r").shape[1])
    train_table = _condition_table(cfg, "train", grid)
    test_table = _condition_table(cfg, "test", grid)

    # Reset exported predictions so the current run owns them.
    for split in ("train", "test"):
        path = _prediction_path(cfg, split)
        if path.exists():
            path.unlink()

    _save_expert_model_config(cfg, grid, feature_dim + 1)

    for regime_id, regime_name in enumerate(regime_names):
        _train_single_regime(
            cfg=cfg,
            regime_id=regime_id,
            regime_name=regime_name,
            grid=grid,
            train_table=train_table,
            test_table=test_table,
            feature_dim=feature_dim,
        )

    print(f"[train-experts] Finished. U-Net experts stored in {cfg.models_dir}")
