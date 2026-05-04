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
from .features import SYMBOLIC_GATE_ENCODER_FEATURE_NAMES, SYMBOLIC_GATE_ENCODER_INDICES, build_encoder_features, build_expert_features
from .models import FullAircraftMeshTeacher
from .surface_graph import CompactSurfaceGraph
from .utils import sample_indices, save_json

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


MESH_SENSOR_EXTRA_FEATURE_NAMES = [
    "x_sq",
    "y_sq",
    "z_sq",
    "AoA_sq",
    "radius_yz_sq",
    "Mach_AoA",
]
MESH_SENSOR_FEATURE_NAMES = SYMBOLIC_GATE_ENCODER_FEATURE_NAMES + MESH_SENSOR_EXTRA_FEATURE_NAMES


def _model_path(cfg: FullAircraftConfig) -> Path:
    return cfg.models_dir / "mesh_teacher.pth"


def _model_config_path(cfg: FullAircraftConfig) -> Path:
    return cfg.models_dir / "mesh_teacher_config.json"


def _training_metrics_path(cfg: FullAircraftConfig) -> Path:
    return cfg.metrics_dir / "mesh_teacher_training.json"


def _diagnostics_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.metrics_dir / f"mesh_teacher_diagnostics_{split}.json"


def _sensor_json_path(cfg: FullAircraftConfig) -> Path:
    return cfg.sensor_dir / "mesh_symbolic_sensor.json"


def _sensor_txt_path(cfg: FullAircraftConfig) -> Path:
    return cfg.sensor_dir / "mesh_symbolic_sensor.txt"


def _teacher_alpha_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"mesh_teacher_alpha_{split}.npy"


def _default_inference_output_path(cfg: FullAircraftConfig, input_path: Path) -> Path:
    return cfg.inference_dir / f"{input_path.stem}_mesh_symbolic.npz"


def _load_cp_scaler(cfg: FullAircraftConfig) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(cfg.scalers_dir / "cp_scaler.npz")
    return payload["mean"].astype(np.float32), payload["scale"].astype(np.float32)


def _load_expert_scaler(cfg: FullAircraftConfig) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(cfg.scalers_dir / "expert_scaler.npz")
    return payload["mean"].astype(np.float32), payload["scale"].astype(np.float32)


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


def _gradient_loss(pred_grid: torch.Tensor, target_grid: torch.Tensor, weight_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dx = _finite_diff_x(pred_grid)
    pred_dy = _finite_diff_y(pred_grid)
    target_dx = _finite_diff_x(target_grid)
    target_dy = _finite_diff_y(target_grid)
    valid_dx = F.pad(mask[..., 1:] * mask[..., :-1], (0, 1, 0, 0))
    valid_dy = F.pad(mask[..., 1:, :] * mask[..., :-1, :], (0, 0, 0, 1))
    weight_dx = weight_map * valid_dx
    weight_dy = weight_map * valid_dy
    loss_dx = F.smooth_l1_loss(pred_dx, target_dx, reduction="none") * weight_dx
    loss_dy = F.smooth_l1_loss(pred_dy, target_dy, reduction="none") * weight_dy
    denom_dx = weight_dx.flatten(1).sum(dim=1).clamp_min(1.0)
    denom_dy = weight_dy.flatten(1).sum(dim=1).clamp_min(1.0)
    return 0.5 * ((loss_dx.flatten(1).sum(dim=1) / denom_dx).mean() + (loss_dy.flatten(1).sum(dim=1) / denom_dy).mean())


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


def _build_shock_target_array(cfg: FullAircraftConfig, split: str, graph: CompactSurfaceGraph) -> np.ndarray:
    target_path = cfg.features_dir / f"mesh_shock_target_{split}.npy"
    if target_path.exists():
        return np.load(target_path, mmap_mode="r")

    cp = np.load(cfg.features_dir / f"cp_{split}.npy", mmap_mode="r")
    n_conditions = int(cp.shape[0] // graph.n_points)
    alpha = np.zeros((cp.shape[0], 1), dtype=np.float32)
    for cond_idx in range(n_conditions):
        row_start = cond_idx * graph.n_points
        row_stop = row_start + graph.n_points
        cp_grid = graph.scatter_numpy(np.asarray(cp[row_start:row_stop], dtype=np.float32))[0]
        alpha_grid = _shock_target_map_numpy(cp_grid, graph.valid_mask, cfg.mesh_teacher_shock_quantile)
        alpha[row_start:row_stop, 0] = graph.gather_numpy(alpha_grid[None, ...])[:, 0]
        if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
            print(f"[mesh-shock-target] {split}: condition {cond_idx + 1}/{n_conditions}")
    np.save(target_path, alpha.astype(np.float32))
    return np.load(target_path, mmap_mode="r")


class _MeshConditionDataset(Dataset):
    def __init__(self, cfg: FullAircraftConfig, split: str, graph: CompactSurfaceGraph):
        self.features = np.load(cfg.features_dir / f"expert_features_{split}.npy", mmap_mode="r")
        self.cp = np.load(cfg.features_dir / f"cp_{split}.npy", mmap_mode="r")
        self.shock_target = _build_shock_target_array(cfg, split, graph)
        self.graph = graph
        self.points_per_condition = graph.n_points
        self.n_conditions = int(self.features.shape[0] // self.points_per_condition)

    def __len__(self) -> int:
        return self.n_conditions

    def __getitem__(self, idx: int):
        row_start = idx * self.points_per_condition
        row_stop = row_start + self.points_per_condition
        feat_flat = np.array(self.features[row_start:row_stop], dtype=np.float32, copy=True)
        cp_flat = np.array(self.cp[row_start:row_stop], dtype=np.float32, copy=True)
        shock_flat = np.array(self.shock_target[row_start:row_stop], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(feat_flat),
            torch.from_numpy(cp_flat),
            torch.from_numpy(shock_flat),
        )


def _save_model_config(cfg: FullAircraftConfig, graph: CompactSurfaceGraph, input_dim: int) -> None:
    payload = {
        "architecture": "mesh_teacher_v1",
        "input_dim": int(input_dim),
        "hidden_dim": int(cfg.mesh_teacher_hidden_dim),
        "latent_dim": int(cfg.latent_dim),
        "message_passing_steps": int(cfg.mesh_teacher_message_passing_steps),
        "dropout": float(cfg.mesh_teacher_dropout),
        "points_per_condition": int(graph.n_points),
        "height": int(graph.height),
        "width": int(graph.width),
        "shock_quantile": float(cfg.mesh_teacher_shock_quantile),
    }
    with _model_config_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def _solve_linear_sensor(basis_train: np.ndarray, target_train: np.ndarray, cfg: FullAircraftConfig) -> dict[str, object]:
    mean = basis_train.mean(axis=0)
    scale = basis_train.std(axis=0) + 1e-6
    standardized = np.clip((basis_train - mean) / scale, -cfg.sensor_feature_clip, cfg.sensor_feature_clip)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    gram = design.T @ design
    ridge = np.eye(gram.shape[0], dtype=np.float64)
    ridge[0, 0] = 0.0
    system = gram + float(cfg.mesh_symbolic_ridge_alpha) * ridge
    rhs = design.T @ target_train.astype(np.float64)
    coef_std = np.linalg.solve(system, rhs)
    coef_raw = coef_std[1:] / scale
    intercept_raw = float(coef_std[0] - np.sum(coef_std[1:] * mean / scale))
    equation = _render_linear_equation(intercept_raw, coef_raw, MESH_SENSOR_FEATURE_NAMES)
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
    return {"precision": precision, "recall": recall, "iou": iou, "f1": f1}


def _forward_teacher(
    model: FullAircraftMeshTeacher,
    feat: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_attr: torch.Tensor,
    mask_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    smooth, shock, alpha_logits, latent = model(feat, edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_attr)
    alpha_teacher = torch.sigmoid(alpha_logits) * mask_flat
    mixed = (1.0 - alpha_teacher) * smooth + alpha_teacher * shock
    return smooth, shock, alpha_teacher, mixed, latent


def _teacher_loss(
    cfg: FullAircraftConfig,
    graph: CompactSurfaceGraph,
    smooth: torch.Tensor,
    shock: torch.Tensor,
    alpha_teacher: torch.Tensor,
    mixed: torch.Tensor,
    cp: torch.Tensor,
    shock_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_flat = torch.ones_like(cp)
    mixed_loss = _masked_weighted_smooth_l1(mixed, cp, mask_flat)
    smooth_loss = _masked_weighted_smooth_l1(smooth, cp, (1.0 - shock_target) * mask_flat)
    shock_loss = _masked_weighted_smooth_l1(shock, cp, shock_target * mask_flat)
    alpha_loss = _masked_weighted_smooth_l1(alpha_teacher, shock_target, mask_flat)

    mixed_grid = graph.scatter_tensor(mixed)
    cp_grid = graph.scatter_tensor(cp)
    shock_grid = graph.scatter_tensor(shock_target)
    mask_grid = graph.mask_tensor(device=cp.device).expand(cp.shape[0], -1, -1, -1)
    grad_weight = (1.0 + float(cfg.expert_shock_weight) * shock_grid) * mask_grid
    grad_loss = _gradient_loss(mixed_grid, cp_grid, grad_weight, mask_grid)

    total = (
        float(cfg.mesh_teacher_mixed_loss_weight) * mixed_loss
        + float(cfg.mesh_teacher_smooth_head_weight) * smooth_loss
        + float(cfg.mesh_teacher_shock_head_weight) * shock_loss
        + float(cfg.mesh_teacher_alpha_loss_weight) * alpha_loss
        + float(cfg.mesh_teacher_grad_loss_weight) * grad_loss
    )
    return total, {
        "mixed": float(mixed_loss.item()),
        "smooth": float(smooth_loss.item()),
        "shock": float(shock_loss.item()),
        "alpha": float(alpha_loss.item()),
        "grad": float(grad_loss.item()),
    }


def _evaluate_teacher(cfg: FullAircraftConfig, graph: CompactSurfaceGraph, model: FullAircraftMeshTeacher, loader: DataLoader) -> dict[str, float]:
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)
    total_mixed = 0.0
    total_rmse = 0.0
    total_alpha = 0.0
    total_smooth_best = 0.0
    total_shock_best = 0.0
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for feat, cp, shock_target in loader:
            feat = feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            shock_target = shock_target.to(cfg.device, non_blocking=True)
            mask_flat = torch.ones_like(cp)
            smooth, shock, alpha_teacher, mixed, _ = _forward_teacher(model, feat, edge_src, edge_dst, edge_attr, mask_flat)
            mixed_mae = _masked_mae(mixed, cp, mask_flat)
            rmse = torch.sqrt(torch.mean((mixed - cp).square()))
            alpha_mae = _masked_mae(alpha_teacher, shock_target, mask_flat)
            smooth_abs = torch.abs(smooth - cp)
            shock_abs = torch.abs(shock - cp)
            smooth_zone = (shock_target < cfg.mesh_teacher_binary_threshold).float()
            shock_zone = (shock_target >= cfg.mesh_teacher_binary_threshold).float()
            smooth_best = (((smooth_abs <= shock_abs).float() * smooth_zone).sum(dim=1) / smooth_zone.sum(dim=1).clamp_min(1.0)).mean()
            shock_best = (((shock_abs <= smooth_abs).float() * shock_zone).sum(dim=1) / shock_zone.sum(dim=1).clamp_min(1.0)).mean()
            total_mixed += float(mixed_mae.item())
            total_rmse += float(rmse.item())
            total_alpha += float(alpha_mae.item())
            total_smooth_best += float(smooth_best.item())
            total_shock_best += float(shock_best.item())
            n_batches += 1
    denom = max(1, n_batches)
    return {
        "mixed_mae": total_mixed / denom,
        "mixed_rmse": total_rmse / denom,
        "teacher_alpha_mae": total_alpha / denom,
        "smooth_zone_smooth_best_fraction": total_smooth_best / denom,
        "shock_zone_shock_best_fraction": total_shock_best / denom,
    }


def _plot_training_curves(cfg: FullAircraftConfig, train_hist: list[float], test_hist: list[float]) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(train_hist, label="train objective")
    plt.plot(test_hist, label="test mixed MAE")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mesh teacher training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / "mesh_teacher_training.png", dpi=220, bbox_inches="tight")
    plt.close()


def train_mesh_teacher(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    graph = CompactSurfaceGraph.from_reference(cfg)
    train_set = _MeshConditionDataset(cfg, "train", graph)
    test_set = _MeshConditionDataset(cfg, "test", graph)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.mesh_teacher_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(1, cfg.mesh_teacher_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_dim = int(np.load(cfg.features_dir / "expert_features_train.npy", mmap_mode="r").shape[1])
    model = FullAircraftMeshTeacher(
        input_dim=feature_dim,
        hidden_dim=cfg.mesh_teacher_hidden_dim,
        latent_dim=cfg.latent_dim,
        message_passing_steps=cfg.mesh_teacher_message_passing_steps,
        dropout=cfg.mesh_teacher_dropout,
    ).to(cfg.device)
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.mesh_teacher_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.mesh_teacher_epochs)

    train_objective_hist: list[float] = []
    test_mae_hist: list[float] = []

    print(
        f"[train-mesh-teacher] train_conditions={len(train_set):,}, test_conditions={len(test_set):,}, "
        f"nodes={graph.n_points:,}, edges={graph.edge_src.shape[0]:,}, device={cfg.device}"
    )

    for epoch in range(1, cfg.mesh_teacher_epochs + 1):
        model.train()
        total_objective = 0.0
        total_batches = 0
        pbar = tqdm(train_loader, desc=f"[mesh-teacher] epoch {epoch}/{cfg.mesh_teacher_epochs}")
        for feat, cp, shock_target in pbar:
            feat = feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            shock_target = shock_target.to(cfg.device, non_blocking=True)
            mask_flat = torch.ones_like(cp)
            optimizer.zero_grad(set_to_none=True)
            smooth, shock, alpha_teacher, mixed, _ = _forward_teacher(model, feat, edge_src, edge_dst, edge_attr, mask_flat)
            loss, terms = _teacher_loss(cfg, graph, smooth, shock, alpha_teacher, mixed, cp, shock_target)
            loss.backward()
            optimizer.step()
            total_objective += float(loss.item())
            total_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}", mix=f"{terms['mixed']:.5f}", alpha=f"{terms['alpha']:.5f}")
        scheduler.step()
        train_objective = total_objective / max(1, total_batches)
        test_metrics = _evaluate_teacher(cfg, graph, model, test_loader)
        train_objective_hist.append(train_objective)
        test_mae_hist.append(test_metrics["mixed_mae"])
        print(
            f"[train-mesh-teacher] epoch {epoch:03d} | train_obj={train_objective:.6f} | "
            f"test_mixed_mae={test_metrics['mixed_mae']:.6f} | alpha_mae={test_metrics['teacher_alpha_mae']:.6f}"
        )

    torch.save(model.state_dict(), _model_path(cfg))
    _save_model_config(cfg, graph, feature_dim)
    train_diag = _evaluate_teacher(cfg, graph, model, train_loader)
    test_diag = _evaluate_teacher(cfg, graph, model, test_loader)
    save_json(
        _training_metrics_path(cfg),
        {
            "architecture": "mesh_teacher_v1",
            "surface": cfg.reduced_surface,
            "final_train_objective": float(train_objective_hist[-1]),
            "final_test_mixed_mae": float(test_mae_hist[-1]),
            "train_diagnostics": train_diag,
            "test_diagnostics": test_diag,
            "hidden_dim": int(cfg.mesh_teacher_hidden_dim),
            "latent_dim": int(cfg.latent_dim),
            "message_passing_steps": int(cfg.mesh_teacher_message_passing_steps),
        },
    )
    save_json(_diagnostics_path(cfg, "train"), {"split": "train", **{k: float(v) for k, v in train_diag.items()}})
    save_json(_diagnostics_path(cfg, "test"), {"split": "test", **{k: float(v) for k, v in test_diag.items()}})
    _plot_training_curves(cfg, train_objective_hist, test_mae_hist)
    print(f"[train-mesh-teacher] Finished. Model stored in {_model_path(cfg)}")


def _load_mesh_teacher(cfg: FullAircraftConfig) -> FullAircraftMeshTeacher:
    config = json.loads(_model_config_path(cfg).read_text())
    model = FullAircraftMeshTeacher(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config.get("latent_dim", cfg.latent_dim)),
        message_passing_steps=int(config["message_passing_steps"]),
        dropout=float(config.get("dropout", cfg.mesh_teacher_dropout)),
    )
    state = torch.load(_model_path(cfg), map_location="cpu")
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()
    return model


def _build_teacher_alpha_array(cfg: FullAircraftConfig, split: str, graph: CompactSurfaceGraph, model: FullAircraftMeshTeacher) -> np.ndarray:
    target_path = _teacher_alpha_path(cfg, split)
    dataset = _MeshConditionDataset(cfg, split, graph)
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)
    alpha = np.zeros((dataset.n_conditions * graph.n_points, 1), dtype=np.float32)
    with torch.no_grad():
        for cond_idx in range(dataset.n_conditions):
            feat, _, _ = dataset[cond_idx]
            feat_t = feat.unsqueeze(0).to(cfg.device, non_blocking=True)
            mask_flat = torch.ones((1, graph.n_points, 1), device=cfg.device, dtype=torch.float32)
            _, _, alpha_teacher, _, _ = _forward_teacher(model, feat_t, edge_src, edge_dst, edge_attr, mask_flat)
            row_start = cond_idx * graph.n_points
            row_stop = row_start + graph.n_points
            alpha[row_start:row_stop, 0] = alpha_teacher[0, :, 0].detach().cpu().numpy().astype(np.float32)
            if cond_idx == 0 or cond_idx + 1 == dataset.n_conditions or cond_idx % 25 == 0:
                print(f"[mesh-teacher-alpha] {split}: condition {cond_idx + 1}/{dataset.n_conditions}")
    np.save(target_path, alpha.astype(np.float32))
    return np.load(target_path, mmap_mode="r")


def distill_mesh_sensor(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    graph = CompactSurfaceGraph.from_reference(cfg)
    model = _load_mesh_teacher(cfg)
    alpha_train = np.asarray(_build_teacher_alpha_array(cfg, "train", graph, model)[:, 0], dtype=np.float32)
    alpha_test = np.asarray(_build_teacher_alpha_array(cfg, "test", graph, model)[:, 0], dtype=np.float32)
    x_train = np.load(cfg.cut_data_dir / "X_cut_train.npy", mmap_mode="r")
    x_test = np.load(cfg.cut_data_dir / "X_cut_test.npy", mmap_mode="r")

    train_idx = sample_indices(x_train.shape[0], min(cfg.mesh_symbolic_max_samples, int(x_train.shape[0])), seed=42)
    test_idx = sample_indices(x_test.shape[0], min(cfg.mesh_symbolic_max_samples, int(x_test.shape[0])), seed=123)
    basis_train = _build_sensor_basis(np.asarray(x_train[train_idx, : cfg.input_dim_raw], dtype=np.float32))
    basis_test = _build_sensor_basis(np.asarray(x_test[test_idx, : cfg.input_dim_raw], dtype=np.float32))

    artifact = {
        "type": "mesh_local_shock_linear_map",
        "description": "Symbolic local shock sensor distilled from the MeshGraphNet teacher alpha.",
        "surface": cfg.reduced_surface,
        "feature_names": MESH_SENSOR_FEATURE_NAMES,
        "binary_threshold": float(cfg.mesh_teacher_binary_threshold),
        "ridge_alpha": float(cfg.mesh_symbolic_ridge_alpha),
        "feature_clip": float(cfg.sensor_feature_clip),
        "teacher_source": "mesh_teacher_alpha",
        "latent_dim": int(cfg.latent_dim),
        **_solve_linear_sensor(basis_train, alpha_train[train_idx], cfg),
    }
    pred_train = _apply_linear_sensor_basis(basis_train, artifact, cfg)
    pred_test = _apply_linear_sensor_basis(basis_test, artifact, cfg)
    train_mae = float(np.mean(np.abs(pred_train - alpha_train[train_idx])))
    test_mae = float(np.mean(np.abs(pred_test - alpha_test[test_idx])))
    artifact["train_mae"] = train_mae
    artifact["test_mae"] = test_mae
    artifact["train_binary_metrics"] = _binary_metrics(alpha_train[train_idx], pred_train, cfg.mesh_teacher_binary_threshold)
    artifact["test_binary_metrics"] = _binary_metrics(alpha_test[test_idx], pred_test, cfg.mesh_teacher_binary_threshold)

    with _sensor_json_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    with _sensor_txt_path(cfg).open("w", encoding="utf-8") as handle:
        handle.write(artifact["equation"] + "\n")
        handle.write(json.dumps({"train_mae": train_mae, "test_mae": test_mae}, indent=2) + "\n")
    print(
        f"[distill-mesh-sensor] train_mae={train_mae:.5f}, test_mae={test_mae:.5f}, "
        f"test_iou={artifact['test_binary_metrics']['iou']:.3f}"
    )
    print(f"[distill-mesh-sensor] Sensor stored in {_sensor_json_path(cfg)}")


def _load_mesh_sensor(cfg: FullAircraftConfig) -> dict[str, object]:
    with _sensor_json_path(cfg).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_symbolic_mesh_sensor(cfg: FullAircraftConfig, x_raw: np.ndarray, artifact: dict[str, object]) -> np.ndarray:
    basis = _build_sensor_basis(x_raw[:, : cfg.input_dim_raw].astype(np.float32))
    return _apply_linear_sensor_basis(basis, artifact, cfg)


def infer_mesh_symbolic(
    cfg: FullAircraftConfig,
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_rows: int | None = None,
) -> Path:
    cfg.ensure_dirs()
    input_path = Path(input_path).expanduser().resolve() if input_path is not None else (cfg.cut_data_dir / "X_cut_test.npy")
    output_path = Path(output_path).expanduser().resolve() if output_path is not None else _default_inference_output_path(cfg, input_path)
    x_raw = np.load(input_path, mmap_mode="r")
    graph = CompactSurfaceGraph.from_reference(cfg)
    if x_raw.shape[0] % graph.n_points != 0:
        raise ValueError("Input rows are not divisible by the reduced graph point count.")
    if max_rows is not None:
        x_raw = np.asarray(x_raw[:max_rows], dtype=np.float32)

    expert_mean, expert_scale = _load_expert_scaler(cfg)
    cp_mean, cp_scale = _load_cp_scaler(cfg)
    model = _load_mesh_teacher(cfg)
    artifact = _load_mesh_sensor(cfg)
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)

    n_rows = int(x_raw.shape[0])
    n_conditions = int(n_rows // graph.n_points)
    cp_pred = np.zeros((n_rows, 1), dtype=np.float32)
    smooth_pred = np.zeros((n_rows, 1), dtype=np.float32)
    shock_pred = np.zeros((n_rows, 1), dtype=np.float32)
    shock_alpha = np.zeros((n_rows, 1), dtype=np.float32)
    teacher_alpha = np.zeros((n_rows, 1), dtype=np.float32)

    with torch.no_grad():
        for cond_idx in range(n_conditions):
            row_start = cond_idx * graph.n_points
            row_stop = row_start + graph.n_points
            x_chunk = np.asarray(x_raw[row_start:row_stop, : cfg.input_dim_raw], dtype=np.float32)
            feat_flat = build_expert_features(x_chunk)
            feat_flat = ((feat_flat - expert_mean) / expert_scale).astype(np.float32)
            feat_tensor = torch.from_numpy(feat_flat[None, ...]).to(cfg.device, non_blocking=True)
            mask_flat = torch.ones((1, graph.n_points, 1), device=cfg.device, dtype=torch.float32)
            smooth_t, shock_t, alpha_teacher_t, _, _ = _forward_teacher(model, feat_tensor, edge_src, edge_dst, edge_attr, mask_flat)
            smooth_flat = smooth_t[0].detach().cpu().numpy().astype(np.float32)
            shock_flat = shock_t[0].detach().cpu().numpy().astype(np.float32)
            teacher_alpha_flat = alpha_teacher_t[0].detach().cpu().numpy().astype(np.float32)
            alpha_flat = apply_symbolic_mesh_sensor(cfg, x_chunk, artifact).reshape(-1, 1)
            mixed_flat = (1.0 - alpha_flat) * smooth_flat + alpha_flat * shock_flat

            smooth_pred[row_start:row_stop] = _destandardize(smooth_flat, cp_mean, cp_scale)
            shock_pred[row_start:row_stop] = _destandardize(shock_flat, cp_mean, cp_scale)
            cp_pred[row_start:row_stop] = _destandardize(mixed_flat, cp_mean, cp_scale)
            shock_alpha[row_start:row_stop] = alpha_flat.astype(np.float32)
            teacher_alpha[row_start:row_stop] = teacher_alpha_flat.astype(np.float32)

            if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
                print(f"[infer-mesh-symbolic] condition {cond_idx + 1}/{n_conditions}")

    np.savez_compressed(
        output_path,
        cp_pred=cp_pred.astype(np.float32),
        smooth_pred=smooth_pred.astype(np.float32),
        shock_pred=shock_pred.astype(np.float32),
        shock_alpha=shock_alpha.astype(np.float32),
        teacher_alpha=teacher_alpha.astype(np.float32),
        sensor_type=np.array([artifact["type"]], dtype=object),
    )
    print(f"[infer-mesh-symbolic] Saved predictions to {output_path}")
    return output_path
