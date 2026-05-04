from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import FullAircraftConfig
from .features import ENCODER_FEATURE_NAMES, SYMBOLIC_GATE_ENCODER_FEATURE_NAMES, SYMBOLIC_GATE_ENCODER_INDICES, build_encoder_features
from .models import FullAircraftShockSplitUNet
from .surface_grid import CompactSurfaceGrid
from .utils import sample_indices, save_json

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


SHOCK_SENSOR_EXTRA_FEATURE_NAMES = [
    "x_sq",
    "y_sq",
    "z_sq",
    "AoA_sq",
    "radius_yz_sq",
    "Mach_AoA",
]
SHOCK_SENSOR_FEATURE_NAMES = SYMBOLIC_GATE_ENCODER_FEATURE_NAMES + SHOCK_SENSOR_EXTRA_FEATURE_NAMES


def _model_path(cfg: FullAircraftConfig) -> Path:
    return cfg.models_dir / "shock_split_unet.pth"


def _model_config_path(cfg: FullAircraftConfig) -> Path:
    return cfg.models_dir / "shock_split_config.json"


def _training_metrics_path(cfg: FullAircraftConfig) -> Path:
    return cfg.metrics_dir / "shock_split_training.json"


def _diagnostics_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.metrics_dir / f"shock_split_diagnostics_{split}.json"


def _sensor_json_path(cfg: FullAircraftConfig) -> Path:
    return cfg.sensor_dir / "shock_symbolic_sensor.json"


def _sensor_txt_path(cfg: FullAircraftConfig) -> Path:
    return cfg.sensor_dir / "shock_symbolic_sensor.txt"


def _shock_target_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"shock_target_{split}.npy"


def _teacher_alpha_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"shock_teacher_alpha_{split}.npy"


def _default_inference_output_path(cfg: FullAircraftConfig, input_path: Path) -> Path:
    return cfg.inference_dir / f"{input_path.stem}_shock_symbolic.npz"


def _load_cp_scaler(cfg: FullAircraftConfig) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(cfg.scalers_dir / "cp_scaler.npz")
    return payload["mean"].astype(np.float32), payload["scale"].astype(np.float32)


def _load_expert_scaler(cfg: FullAircraftConfig) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(cfg.scalers_dir / "expert_scaler.npz")
    return payload["mean"].astype(np.float32), payload["scale"].astype(np.float32)


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def _destandardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values * scale + mean).astype(np.float32)


def _finite_diff_x(field: torch.Tensor) -> torch.Tensor:
    return F.pad(field[..., 1:] - field[..., :-1], (0, 1, 0, 0))


def _finite_diff_y(field: torch.Tensor) -> torch.Tensor:
    return F.pad(field[..., 1:, :] - field[..., :-1, :], (0, 0, 0, 1))


def _masked_weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weight_map
    denom = weight_map.flatten(1).sum(dim=1).clamp_min(1.0)
    return (loss.flatten(1).sum(dim=1) / denom).mean()


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    loss = torch.abs(pred - target) * weight_map
    denom = weight_map.flatten(1).sum(dim=1).clamp_min(1.0)
    return (loss.flatten(1).sum(dim=1) / denom).mean()


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dx = _finite_diff_x(pred)
    pred_dy = _finite_diff_y(pred)
    target_dx = _finite_diff_x(target)
    target_dy = _finite_diff_y(target)
    valid_dx = F.pad(mask[..., 1:] * mask[..., :-1], (0, 1, 0, 0))
    valid_dy = F.pad(mask[..., 1:, :] * mask[..., :-1, :], (0, 0, 0, 1))
    weight_dx = weight_map * valid_dx
    weight_dy = weight_map * valid_dy
    loss_dx = F.smooth_l1_loss(pred_dx, target_dx, reduction="none") * weight_dx
    loss_dy = F.smooth_l1_loss(pred_dy, target_dy, reduction="none") * weight_dy
    denom_dx = weight_dx.flatten(1).sum(dim=1).clamp_min(1.0)
    denom_dy = weight_dy.flatten(1).sum(dim=1).clamp_min(1.0)
    return 0.5 * (
        (loss_dx.flatten(1).sum(dim=1) / denom_dx).mean()
        + (loss_dy.flatten(1).sum(dim=1) / denom_dy).mean()
    )


def _shock_target_map_from_cp(cp_grid: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    grad_x = _finite_diff_x(cp_grid)
    grad_y = _finite_diff_y(cp_grid)
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12) * mask
    flat = grad_mag.flatten(1)
    flat_mask = mask.flatten(1) > 0.5
    scales: list[torch.Tensor] = []
    for idx in range(cp_grid.shape[0]):
        valid = flat[idx][flat_mask[idx]]
        if valid.numel() == 0:
            scales.append(torch.ones((), device=cp_grid.device, dtype=cp_grid.dtype))
            continue
        scales.append(torch.quantile(valid, q=float(quantile)).clamp_min(1e-6))
    scale = torch.stack(scales).view(-1, 1, 1, 1)
    return torch.clamp(grad_mag / scale, 0.0, 1.0) * mask


def _shock_target_map_numpy(cp_grid: np.ndarray, mask: np.ndarray, quantile: float) -> np.ndarray:
    grad_y, grad_x = np.gradient(cp_grid.astype(np.float32), edge_order=1)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y) * mask
    valid = grad_mag[mask > 0.5]
    if valid.size == 0:
        return np.zeros_like(cp_grid, dtype=np.float32)
    scale = float(np.quantile(valid, quantile))
    if scale <= 1e-6:
        return np.zeros_like(cp_grid, dtype=np.float32)
    return np.clip(grad_mag / scale, 0.0, 1.0).astype(np.float32) * mask


class _ShockConditionDataset(Dataset):
    def __init__(self, cfg: FullAircraftConfig, split: str, grid: CompactSurfaceGrid):
        self.features = np.load(cfg.features_dir / f"expert_features_{split}.npy", mmap_mode="r")
        self.cp = np.load(cfg.features_dir / f"cp_{split}.npy", mmap_mode="r")
        self.grid = grid
        self.points_per_condition = grid.n_points
        self.n_conditions = int(self.features.shape[0] // self.points_per_condition)
        self.mask = grid.valid_mask[None, ...].astype(np.float32)

    def __len__(self) -> int:
        return self.n_conditions

    def __getitem__(self, idx: int):
        row_start = idx * self.points_per_condition
        row_stop = row_start + self.points_per_condition
        feat_flat = np.asarray(self.features[row_start:row_stop], dtype=np.float32)
        cp_flat = np.asarray(self.cp[row_start:row_stop], dtype=np.float32)
        feat_grid = self.grid.scatter_numpy(feat_flat)
        cp_grid = self.grid.scatter_numpy(cp_flat)
        feat_grid = np.concatenate([feat_grid, self.mask], axis=0)
        return (
            torch.from_numpy(feat_grid),
            torch.from_numpy(cp_grid),
            torch.from_numpy(self.mask),
        )


def _save_model_config(cfg: FullAircraftConfig, grid: CompactSurfaceGrid, input_channels: int) -> None:
    payload = {
        "architecture": "shock_split_latent_unet_v2",
        "input_channels": int(input_channels),
        "base_channels": int(cfg.shock_local_base_channels),
        "latent_dim": int(cfg.latent_dim),
        "grid_height": int(grid.height),
        "grid_width": int(grid.width),
        "points_per_condition": int(grid.n_points),
        "target_quantile": float(cfg.shock_local_target_quantile),
    }
    with _model_config_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _split_predictions(
    model: FullAircraftShockSplitUNet,
    feat: torch.Tensor,
    cp: torch.Tensor,
    mask: torch.Tensor,
    cfg: FullAircraftConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    smooth_pred, shock_pred, alpha_logits, latent_map = model(feat)
    alpha_teacher = torch.sigmoid(alpha_logits) * mask
    shock_target = _shock_target_map_from_cp(cp, mask, cfg.shock_local_target_quantile)
    mixed_pred = (1.0 - alpha_teacher) * smooth_pred + alpha_teacher * shock_pred
    return smooth_pred, shock_pred, alpha_teacher, mixed_pred, shock_target, latent_map


def _train_objective(
    smooth_pred: torch.Tensor,
    shock_pred: torch.Tensor,
    alpha_teacher: torch.Tensor,
    mixed_pred: torch.Tensor,
    cp: torch.Tensor,
    mask: torch.Tensor,
    shock_target: torch.Tensor,
    cfg: FullAircraftConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    mixed_loss = _masked_weighted_smooth_l1(mixed_pred, cp, mask)
    smooth_loss = _masked_weighted_smooth_l1(smooth_pred, cp, (1.0 - shock_target) * mask)
    shock_loss = _masked_weighted_smooth_l1(shock_pred, cp, shock_target * mask)
    alpha_loss = _masked_weighted_smooth_l1(alpha_teacher, shock_target, mask)
    grad_weight = (1.0 + float(cfg.expert_shock_weight) * shock_target) * mask
    grad_loss = _gradient_loss(mixed_pred, cp, grad_weight, mask)
    total = (
        float(cfg.shock_local_mixed_loss_weight) * mixed_loss
        + float(cfg.shock_local_smooth_head_weight) * smooth_loss
        + float(cfg.shock_local_shock_head_weight) * shock_loss
        + float(cfg.shock_local_alpha_loss_weight) * alpha_loss
        + float(cfg.shock_local_grad_loss_weight) * grad_loss
    )
    return total, {
        "mixed": float(mixed_loss.item()),
        "smooth": float(smooth_loss.item()),
        "shock": float(shock_loss.item()),
        "alpha": float(alpha_loss.item()),
        "grad": float(grad_loss.item()),
    }


def _evaluate_model(
    cfg: FullAircraftConfig,
    model: FullAircraftShockSplitUNet,
    loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    total_mixed = 0.0
    total_rmse = 0.0
    total_alpha_mae = 0.0
    total_smooth_zone_smooth = 0.0
    total_smooth_zone_shock = 0.0
    total_shock_zone_smooth = 0.0
    total_shock_zone_shock = 0.0
    total_smooth_best = 0.0
    total_shock_best = 0.0
    n_batches = 0
    with torch.no_grad():
        for feat, cp, mask in loader:
            feat = feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)
            smooth_pred, shock_pred, alpha_teacher, mixed_pred, shock_target, _ = _split_predictions(model, feat, cp, mask, cfg)

            mixed_mae = _masked_mae(mixed_pred, cp, mask)
            rmse = torch.sqrt((((mixed_pred - cp) * mask).square().flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)).mean())
            alpha_mae = _masked_mae(alpha_teacher, shock_target, mask)

            smooth_zone = ((shock_target < cfg.shock_local_binary_threshold).float()) * mask
            shock_zone = ((shock_target >= cfg.shock_local_binary_threshold).float()) * mask
            smooth_zone_smooth = _masked_mae(smooth_pred, cp, smooth_zone)
            smooth_zone_shock = _masked_mae(shock_pred, cp, smooth_zone)
            shock_zone_smooth = _masked_mae(smooth_pred, cp, shock_zone)
            shock_zone_shock = _masked_mae(shock_pred, cp, shock_zone)

            smooth_abs = torch.abs(smooth_pred - cp)
            shock_abs = torch.abs(shock_pred - cp)
            smooth_best = (((smooth_abs <= shock_abs).float() * smooth_zone).flatten(1).sum(dim=1) / smooth_zone.flatten(1).sum(dim=1).clamp_min(1.0)).mean()
            shock_best = (((shock_abs <= smooth_abs).float() * shock_zone).flatten(1).sum(dim=1) / shock_zone.flatten(1).sum(dim=1).clamp_min(1.0)).mean()

            total_mixed += float(mixed_mae.item())
            total_rmse += float(rmse.item())
            total_alpha_mae += float(alpha_mae.item())
            total_smooth_zone_smooth += float(smooth_zone_smooth.item())
            total_smooth_zone_shock += float(smooth_zone_shock.item())
            total_shock_zone_smooth += float(shock_zone_smooth.item())
            total_shock_zone_shock += float(shock_zone_shock.item())
            total_smooth_best += float(smooth_best.item())
            total_shock_best += float(shock_best.item())
            n_batches += 1

    denom = max(1, n_batches)
    return {
        "mixed_mae": total_mixed / denom,
        "mixed_rmse": total_rmse / denom,
        "teacher_alpha_mae": total_alpha_mae / denom,
        "smooth_zone_smooth_mae": total_smooth_zone_smooth / denom,
        "smooth_zone_shock_mae": total_smooth_zone_shock / denom,
        "shock_zone_smooth_mae": total_shock_zone_smooth / denom,
        "shock_zone_shock_mae": total_shock_zone_shock / denom,
        "smooth_zone_smooth_best_fraction": total_smooth_best / denom,
        "shock_zone_shock_best_fraction": total_shock_best / denom,
    }


def _diagnose_split(cfg: FullAircraftConfig, model: FullAircraftShockSplitUNet, split: str, grid: CompactSurfaceGrid) -> dict[str, float]:
    dataset = _ShockConditionDataset(cfg, split, grid)
    loader = DataLoader(
        dataset,
        batch_size=max(1, cfg.expert_field_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    metrics = _evaluate_model(cfg, model, loader)
    payload = {
        "split": split,
        "surface": cfg.reduced_surface,
        **{key: float(value) for key, value in metrics.items()},
    }
    save_json(_diagnostics_path(cfg, split), payload)
    return payload


def _plot_training_curves(cfg: FullAircraftConfig, train_hist: list[float], test_hist: list[float]) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(train_hist, label="train objective")
    plt.plot(test_hist, label="test mixed MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Shock-split experts training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / "shock_split_training.png", dpi=220, bbox_inches="tight")
    plt.close()


def _condition_count_from_reduced(cfg: FullAircraftConfig, split: str) -> int:
    x = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    return int(x.shape[0] // CompactSurfaceGrid.from_reference(cfg).n_points)


def _build_shock_target_array(cfg: FullAircraftConfig, split: str, grid: CompactSurfaceGrid) -> np.ndarray:
    target_path = _shock_target_path(cfg, split)
    if target_path.exists():
        return np.load(target_path, mmap_mode="r")

    cp = np.load(cfg.features_dir / f"cp_{split}.npy", mmap_mode="r")
    n_conditions = int(cp.shape[0] // grid.n_points)
    alpha = np.zeros((cp.shape[0], 1), dtype=np.float32)
    for cond_idx in range(n_conditions):
        row_start = cond_idx * grid.n_points
        row_stop = row_start + grid.n_points
        cp_grid = grid.scatter_numpy(np.asarray(cp[row_start:row_stop], dtype=np.float32))[0]
        alpha_grid = _shock_target_map_numpy(cp_grid, grid.valid_mask, cfg.shock_local_target_quantile)
        alpha[row_start:row_stop, 0] = grid.gather_numpy(alpha_grid[None, ...])[:, 0]
        if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
            print(f"[shock-target] {split}: condition {cond_idx + 1}/{n_conditions}")
    np.save(target_path, alpha.astype(np.float32))
    return np.load(target_path, mmap_mode="r")


def _build_teacher_alpha_array(
    cfg: FullAircraftConfig,
    split: str,
    grid: CompactSurfaceGrid,
    model: FullAircraftShockSplitUNet,
) -> np.ndarray:
    target_path = _teacher_alpha_path(cfg, split)
    dataset = _ShockConditionDataset(cfg, split, grid)
    alpha = np.zeros((dataset.n_conditions * grid.n_points, 1), dtype=np.float32)
    with torch.no_grad():
        for cond_idx in range(dataset.n_conditions):
            feat, _, mask = dataset[cond_idx]
            feat_t = feat.unsqueeze(0).to(cfg.device, non_blocking=True)
            mask_t = mask.unsqueeze(0).to(cfg.device, non_blocking=True)
            _, _, alpha_logits, _ = model(feat_t)
            alpha_grid = (torch.sigmoid(alpha_logits) * mask_t).detach().cpu().numpy()
            row_start = cond_idx * grid.n_points
            row_stop = row_start + grid.n_points
            alpha[row_start:row_stop, 0] = grid.gather_numpy(alpha_grid)[:, 0]
            if cond_idx == 0 or cond_idx + 1 == dataset.n_conditions or cond_idx % 25 == 0:
                print(f"[teacher-alpha] {split}: condition {cond_idx + 1}/{dataset.n_conditions}")
    np.save(target_path, alpha.astype(np.float32))
    return np.load(target_path, mmap_mode="r")


def _build_sensor_basis(x_raw: np.ndarray) -> np.ndarray:
    encoder = build_encoder_features(x_raw)
    base = encoder[:, SYMBOLIC_GATE_ENCODER_INDICES].astype(np.float64)
    x = encoder[:, 0:1].astype(np.float64)
    y = encoder[:, 1:2].astype(np.float64)
    z = encoder[:, 2:3].astype(np.float64)
    mach = encoder[:, 6:7].astype(np.float64)
    aoa = encoder[:, 8:9].astype(np.float64)
    radius = encoder[:, 12:13].astype(np.float64)
    extra = np.concatenate([x**2, y**2, z**2, aoa**2, radius**2, mach * aoa], axis=1)
    return np.concatenate([base, extra], axis=1).astype(np.float64)


def _render_linear_equation(intercept: float, coefficients: np.ndarray, basis_names: list[str], top_k: int = 10) -> str:
    coef = np.asarray(coefficients, dtype=np.float64)
    order = np.argsort(-np.abs(coef))
    parts = [f"{intercept:.4f}"]
    used = 0
    for idx in order:
        if abs(float(coef[idx])) < 1e-6:
            continue
        sign = "+" if float(coef[idx]) >= 0.0 else "-"
        parts.append(f" {sign} {abs(float(coef[idx])):.4f}*{basis_names[idx]}")
        used += 1
        if used >= top_k:
            break
    return "clip(" + "".join(parts) + ", 0, 1)"


def _solve_linear_sensor(
    basis_train: np.ndarray,
    target_train: np.ndarray,
    cfg: FullAircraftConfig,
) -> dict[str, object]:
    mean = basis_train.mean(axis=0)
    scale = basis_train.std(axis=0) + 1e-6
    standardized = np.clip((basis_train - mean) / scale, -cfg.sensor_feature_clip, cfg.sensor_feature_clip)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    gram = design.T @ design
    ridge = np.eye(gram.shape[0], dtype=np.float64)
    ridge[0, 0] = 0.0
    system = gram + float(cfg.shock_symbolic_ridge_alpha) * ridge
    rhs = design.T @ target_train.astype(np.float64)
    coef_std = np.linalg.solve(system, rhs)
    coef_raw = coef_std[1:] / scale
    intercept_raw = float(coef_std[0] - np.sum(coef_std[1:] * mean / scale))
    equation = _render_linear_equation(intercept_raw, coef_raw, SHOCK_SENSOR_FEATURE_NAMES)
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients_std": coef_std.tolist(),
        "coefficients_raw": coef_raw.tolist(),
        "intercept_raw": intercept_raw,
        "equation": equation,
    }


def _apply_linear_sensor_basis(basis: np.ndarray, artifact: dict[str, object], cfg: FullAircraftConfig) -> np.ndarray:
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
    coef_std = np.asarray(artifact["coefficients_std"], dtype=np.float64)
    standardized = np.clip((basis - mean) / scale, -cfg.sensor_feature_clip, cfg.sensor_feature_clip)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    return np.clip(design @ coef_std, 0.0, 1.0).astype(np.float32)


def _binary_metrics(target: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    target_bin = target >= threshold
    pred_bin = pred >= threshold
    tp = float(np.logical_and(target_bin, pred_bin).sum())
    fp = float(np.logical_and(~target_bin, pred_bin).sum())
    fn = float(np.logical_and(target_bin, ~pred_bin).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    iou = tp / max(1.0, tp + fp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def train_shock_experts(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    grid = CompactSurfaceGrid.from_reference(cfg)
    train_set = _ShockConditionDataset(cfg, "train", grid)
    test_set = _ShockConditionDataset(cfg, "test", grid)

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

    feature_dim = int(np.load(cfg.features_dir / "expert_features_train.npy", mmap_mode="r").shape[1]) + 1
    model = FullAircraftShockSplitUNet(
        input_channels=feature_dim,
        base_channels=cfg.shock_local_base_channels,
        latent_dim=cfg.latent_dim,
    ).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.shock_local_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.shock_local_epochs)

    train_objective_hist: list[float] = []
    test_mae_hist: list[float] = []

    print(
        f"[train-shock-experts] train_conditions={len(train_set):,}, test_conditions={len(test_set):,}, "
        f"grid={grid.height}x{grid.width}, device={cfg.device}"
    )

    for epoch in range(1, cfg.shock_local_epochs + 1):
        model.train()
        total_objective = 0.0
        total_batches = 0
        pbar = tqdm(train_loader, desc=f"[shock-experts] epoch {epoch}/{cfg.shock_local_epochs}")
        for feat, cp, mask in pbar:
            feat = feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            smooth_pred, shock_pred, alpha_teacher, mixed_pred, shock_target, _ = _split_predictions(model, feat, cp, mask, cfg)
            loss, terms = _train_objective(smooth_pred, shock_pred, alpha_teacher, mixed_pred, cp, mask, shock_target, cfg)
            loss.backward()
            optimizer.step()

            total_objective += float(loss.item())
            total_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}", mix=f"{terms['mixed']:.5f}", alpha=f"{terms['alpha']:.5f}", grad=f"{terms['grad']:.5f}")

        scheduler.step()
        train_objective = total_objective / max(1, total_batches)
        test_metrics = _evaluate_model(cfg, model, test_loader)
        train_objective_hist.append(train_objective)
        test_mae_hist.append(test_metrics["mixed_mae"])
        print(
            f"[train-shock-experts] epoch {epoch:03d} | train_obj={train_objective:.6f} | "
            f"test_mixed_mae={test_metrics['mixed_mae']:.6f} | "
            f"smooth_best={test_metrics['smooth_zone_smooth_best_fraction']:.3f} | "
            f"shock_best={test_metrics['shock_zone_shock_best_fraction']:.3f}"
        )

    torch.save(model.state_dict(), _model_path(cfg))
    _save_model_config(cfg, grid, feature_dim)
    train_diag = _diagnose_split(cfg, model, "train", grid)
    test_diag = _diagnose_split(cfg, model, "test", grid)
    save_json(
        _training_metrics_path(cfg),
        {
            "architecture": "shock_split_latent_unet_v2",
            "surface": cfg.reduced_surface,
            "final_train_objective": float(train_objective_hist[-1]),
            "final_test_mixed_mae": float(test_mae_hist[-1]),
            "train_diagnostics": train_diag,
            "test_diagnostics": test_diag,
            "shock_local_base_channels": int(cfg.shock_local_base_channels),
            "latent_dim": int(cfg.latent_dim),
            "shock_local_target_quantile": float(cfg.shock_local_target_quantile),
            "shock_local_binary_threshold": float(cfg.shock_local_binary_threshold),
            "shock_local_alpha_loss_weight": float(cfg.shock_local_alpha_loss_weight),
            "shock_local_grad_loss_weight": float(cfg.shock_local_grad_loss_weight),
        },
    )
    _plot_training_curves(cfg, train_objective_hist, test_mae_hist)
    print(f"[train-shock-experts] Finished. Model stored in {_model_path(cfg)}")


def distill_shock_sensor(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    grid = CompactSurfaceGrid.from_reference(cfg)
    model = _load_shock_model(cfg)
    alpha_train = np.asarray(_build_teacher_alpha_array(cfg, "train", grid, model)[:, 0], dtype=np.float32)
    alpha_test = np.asarray(_build_teacher_alpha_array(cfg, "test", grid, model)[:, 0], dtype=np.float32)
    x_train = np.load(cfg.cut_data_dir / "X_cut_train.npy", mmap_mode="r")
    x_test = np.load(cfg.cut_data_dir / "X_cut_test.npy", mmap_mode="r")

    train_idx = sample_indices(x_train.shape[0], min(cfg.shock_symbolic_max_samples, int(x_train.shape[0])), seed=42)
    test_idx = sample_indices(x_test.shape[0], min(cfg.shock_symbolic_max_samples, int(x_test.shape[0])), seed=123)
    basis_train = _build_sensor_basis(np.asarray(x_train[train_idx, : cfg.input_dim_raw], dtype=np.float32))
    basis_test = _build_sensor_basis(np.asarray(x_test[test_idx, : cfg.input_dim_raw], dtype=np.float32))

    artifact = {
        "type": "local_shock_linear_map",
        "description": "Symbolic soft shock sensor distilled from the latent local shock teacher.",
        "surface": cfg.reduced_surface,
        "feature_names": SHOCK_SENSOR_FEATURE_NAMES,
        "binary_threshold": float(cfg.shock_local_binary_threshold),
        "ridge_alpha": float(cfg.shock_symbolic_ridge_alpha),
        "feature_clip": float(cfg.sensor_feature_clip),
        "teacher_source": "teacher_alpha",
        "latent_dim": int(cfg.latent_dim),
        **_solve_linear_sensor(basis_train, alpha_train[train_idx], cfg),
    }

    pred_train = _apply_linear_sensor_basis(basis_train, artifact, cfg)
    pred_test = _apply_linear_sensor_basis(basis_test, artifact, cfg)
    train_mae = float(np.mean(np.abs(pred_train - alpha_train[train_idx])))
    test_mae = float(np.mean(np.abs(pred_test - alpha_test[test_idx])))
    artifact["train_mae"] = train_mae
    artifact["test_mae"] = test_mae
    artifact["train_binary_metrics"] = _binary_metrics(alpha_train[train_idx], pred_train, cfg.shock_local_binary_threshold)
    artifact["test_binary_metrics"] = _binary_metrics(alpha_test[test_idx], pred_test, cfg.shock_local_binary_threshold)

    with _sensor_json_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    with _sensor_txt_path(cfg).open("w", encoding="utf-8") as handle:
        handle.write(artifact["equation"] + "\n")
        handle.write(json.dumps({"train_mae": train_mae, "test_mae": test_mae}, indent=2) + "\n")

    print(
        f"[distill-shock-sensor] train_mae={train_mae:.5f}, test_mae={test_mae:.5f}, "
        f"test_iou={artifact['test_binary_metrics']['iou']:.3f}"
    )
    print(f"[distill-shock-sensor] Sensor stored in {_sensor_json_path(cfg)}")


def _load_shock_model(cfg: FullAircraftConfig) -> FullAircraftShockSplitUNet:
    config = json.loads(_model_config_path(cfg).read_text())
    model = FullAircraftShockSplitUNet(
        input_channels=int(config["input_channels"]),
        base_channels=int(config["base_channels"]),
        latent_dim=int(config.get("latent_dim", cfg.latent_dim)),
    )
    state = torch.load(_model_path(cfg), map_location="cpu")
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()
    return model


def _load_shock_sensor(cfg: FullAircraftConfig) -> dict[str, object]:
    with _sensor_json_path(cfg).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_symbolic_shock_sensor(cfg: FullAircraftConfig, x_raw: np.ndarray, artifact: dict[str, object]) -> np.ndarray:
    basis = _build_sensor_basis(x_raw[:, : cfg.input_dim_raw].astype(np.float32))
    return _apply_linear_sensor_basis(basis, artifact, cfg)


def infer_shock_symbolic(
    cfg: FullAircraftConfig,
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_rows: int | None = None,
) -> Path:
    cfg.ensure_dirs()
    input_path = Path(input_path).expanduser().resolve() if input_path is not None else (cfg.cut_data_dir / "X_cut_test.npy")
    output_path = Path(output_path).expanduser().resolve() if output_path is not None else _default_inference_output_path(cfg, input_path)
    x_raw = np.load(input_path, mmap_mode="r")
    if x_raw.shape[0] % CompactSurfaceGrid.from_reference(cfg).n_points != 0:
        raise ValueError("Input rows are not divisible by the reduced grid point count.")
    if max_rows is not None:
        x_raw = np.asarray(x_raw[:max_rows], dtype=np.float32)

    grid = CompactSurfaceGrid.from_reference(cfg)
    expert_mean, expert_scale = _load_expert_scaler(cfg)
    cp_mean, cp_scale = _load_cp_scaler(cfg)
    model = _load_shock_model(cfg)
    artifact = _load_shock_sensor(cfg)

    n_rows = int(x_raw.shape[0])
    n_conditions = int(n_rows // grid.n_points)
    cp_pred = np.zeros((n_rows, 1), dtype=np.float32)
    smooth_pred = np.zeros((n_rows, 1), dtype=np.float32)
    shock_pred = np.zeros((n_rows, 1), dtype=np.float32)
    shock_alpha = np.zeros((n_rows, 1), dtype=np.float32)
    teacher_alpha = np.zeros((n_rows, 1), dtype=np.float32)
    mask = grid.valid_mask[None, ...].astype(np.float32)

    with torch.no_grad():
        for cond_idx in range(n_conditions):
            row_start = cond_idx * grid.n_points
            row_stop = row_start + grid.n_points
            x_chunk = np.asarray(x_raw[row_start:row_stop, : cfg.input_dim_raw], dtype=np.float32)
            feat_chunk = _standardize(build_encoder_input_from_expert(x_chunk), expert_mean, expert_scale)
            feat_grid = grid.scatter_numpy(feat_chunk)
            feat_grid = np.concatenate([feat_grid, mask], axis=0)[None, ...]
            feat_tensor = torch.from_numpy(feat_grid).to(cfg.device, non_blocking=True)
            smooth_grid_t, shock_grid_t, alpha_logits_t, _ = model(feat_tensor)
            smooth_flat = grid.gather_numpy(smooth_grid_t.detach().cpu().numpy())[0]
            shock_flat = grid.gather_numpy(shock_grid_t.detach().cpu().numpy())[0]
            teacher_alpha_flat = grid.gather_numpy((torch.sigmoid(alpha_logits_t) * feat_tensor[:, -1:, :, :]).detach().cpu().numpy())[0]
            alpha_flat = apply_symbolic_shock_sensor(cfg, x_chunk, artifact).reshape(-1, 1)
            mixed_flat = (1.0 - alpha_flat) * smooth_flat + alpha_flat * shock_flat

            smooth_pred[row_start:row_stop] = _destandardize(smooth_flat, cp_mean, cp_scale)
            shock_pred[row_start:row_stop] = _destandardize(shock_flat, cp_mean, cp_scale)
            cp_pred[row_start:row_stop] = _destandardize(mixed_flat, cp_mean, cp_scale)
            shock_alpha[row_start:row_stop] = alpha_flat.astype(np.float32)
            teacher_alpha[row_start:row_stop] = teacher_alpha_flat.astype(np.float32)

            if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
                print(f"[infer-shock-symbolic] condition {cond_idx + 1}/{n_conditions}")

    np.savez_compressed(
        output_path,
        cp_pred=cp_pred.astype(np.float32),
        smooth_pred=smooth_pred.astype(np.float32),
        shock_pred=shock_pred.astype(np.float32),
        shock_alpha=shock_alpha.astype(np.float32),
        teacher_alpha=teacher_alpha.astype(np.float32),
        sensor_type=np.array([artifact["type"]], dtype=object),
    )
    print(f"[infer-shock-symbolic] Saved predictions to {output_path}")
    return output_path


def build_encoder_input_from_expert(x_raw: np.ndarray) -> np.ndarray:
    from .features import build_expert_features

    return build_expert_features(x_raw)
