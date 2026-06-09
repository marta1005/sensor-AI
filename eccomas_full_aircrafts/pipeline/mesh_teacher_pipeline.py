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
from .utils import save_json

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
MESH_SHOCK_LINE_FEATURE_NAMES = [
    "row_y",
    "row_z",
    "Mach",
    "Pi",
    "AoA_deg",
    "row_y_sq",
    "row_z_sq",
    "Mach_sq",
    "AoA_sq",
    "Mach_AoA",
    "Mach_row_y",
    "AoA_row_y",
]


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


def _teacher_shock_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"mesh_teacher_shock_{split}.npy"


def _shock_target_meta_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"mesh_shock_target_{split}_meta.json"


def _default_inference_output_path(cfg: FullAircraftConfig, input_path: Path) -> Path:
    return cfg.inference_dir / f"{input_path.stem}_mesh_symbolic.npz"


def _default_teacher_inference_output_path(cfg: FullAircraftConfig, input_path: Path) -> Path:
    return cfg.inference_dir / f"{input_path.stem}_mesh_teacher.npz"


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


def _masked_focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = target * prob + (1.0 - target) * (1.0 - prob)
    focal = ((1.0 - pt).clamp_min(1e-6) ** float(gamma)) * bce * weight_map
    denom = weight_map.flatten(1).sum(dim=1).clamp_min(1.0)
    return (focal.flatten(1).sum(dim=1) / denom).mean()


def _masked_soft_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    numer = 2.0 * (prob * target * weight_map).flatten(1).sum(dim=1) + 1e-6
    denom = ((prob + target) * weight_map).flatten(1).sum(dim=1) + 1e-6
    return (1.0 - numer / denom).mean()


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


def _normalized_gradient_map(
    field_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mask: np.ndarray,
    quantile: float,
) -> np.ndarray:
    field = field_grid.astype(np.float32, copy=False)
    valid_mask = mask.astype(bool, copy=False)
    grad_x = np.zeros_like(field, dtype=np.float32)
    grad_y = np.zeros_like(field, dtype=np.float32)

    valid_x = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid_y = valid_mask[1:, :] & valid_mask[:-1, :]
    dx = field[:, 1:] - field[:, :-1]
    dy = field[1:, :] - field[:-1, :]
    dist_x = np.sqrt(
        (x_grid[:, 1:] - x_grid[:, :-1]) ** 2
        + (y_grid[:, 1:] - y_grid[:, :-1]) ** 2
    ).astype(np.float32)
    dist_y = np.sqrt(
        (x_grid[1:, :] - x_grid[:-1, :]) ** 2
        + (y_grid[1:, :] - y_grid[:-1, :]) ** 2
    ).astype(np.float32)
    dx[~valid_x] = 0.0
    dy[~valid_y] = 0.0
    dist_x[~valid_x] = 1.0
    dist_y[~valid_y] = 1.0
    dx = dx / np.maximum(dist_x, 1e-3)
    dy = dy / np.maximum(dist_y, 1e-3)
    grad_x[:, :-1] = dx
    grad_y[:-1, :] = dy

    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y) * mask
    valid = grad_mag[mask > 0.5]
    if valid.size == 0:
        return np.zeros_like(field_grid, dtype=np.float32)
    scale = float(np.quantile(valid, quantile))
    if scale <= 1e-6:
        return np.zeros_like(field_grid, dtype=np.float32)
    return np.clip(grad_mag / scale, 0.0, 1.0).astype(np.float32) * mask


def _shock_line_target_from_score_map(
    score_map: np.ndarray,
    x_grid: np.ndarray,
    mask: np.ndarray,
    activation_threshold: float,
    weight_power: float,
    band_width: float,
) -> np.ndarray:
    height, width = score_map.shape
    target = np.zeros((height, width), dtype=np.float32)
    min_width = max(float(band_width), 1e-3)
    for row in range(height):
        valid = mask[row] > 0.5
        if not np.any(valid):
            continue
        row_score = score_map[row, valid].astype(np.float32)
        if row_score.size == 0:
            continue
        peak = float(np.max(row_score))
        if peak < float(activation_threshold):
            continue
        row_x = x_grid[row, valid].astype(np.float32)
        weights = np.maximum(row_score - float(activation_threshold), 0.0) ** float(weight_power)
        if float(np.sum(weights)) <= 1e-8:
            weights = np.maximum(row_score, 0.0) ** float(weight_power)
        if float(np.sum(weights)) <= 1e-8:
            continue
        center = float(np.sum(weights * row_x) / np.sum(weights))
        row_target = peak * np.exp(-0.5 * ((row_x - center) / min_width) ** 2)
        target[row, valid] = np.clip(row_target, 0.0, 1.0)
    return target.astype(np.float32) * mask


def _shock_target_map_numpy(
    cp_grid: np.ndarray,
    cfx_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mask: np.ndarray,
    quantile: float,
    cfx_weight: float,
) -> np.ndarray:
    cp_map = _normalized_gradient_map(cp_grid, x_grid, y_grid, mask, quantile)
    cfx_map = _normalized_gradient_map(cfx_grid, x_grid, y_grid, mask, quantile)
    return np.clip(cp_map + float(cfx_weight) * cfx_map, 0.0, 1.0).astype(np.float32) * mask


def _build_shock_target_array(cfg: FullAircraftConfig, split: str, graph: CompactSurfaceGraph) -> np.ndarray:
    target_path = cfg.features_dir / f"mesh_shock_target_{split}.npy"
    meta_path = _shock_target_meta_path(cfg, split)
    y_red = np.load(cfg.reduced_data_dir / f"Y_cut_{split}.npy", mmap_mode="r")
    expected_rows = int(y_red.shape[0])
    expected_config = {
        "rows": expected_rows,
        "points_per_condition": int(graph.n_points),
        "graph_connectivity": "projected_xy_offsets_plus_row_col_successors_v3",
        "gradient_mode": "masked_projected_xy_distance_v4",
        "shock_quantile": float(cfg.mesh_teacher_shock_quantile),
        "cfx_weight": float(cfg.mesh_teacher_cfx_weight),
        "band_width": float(cfg.mesh_teacher_shock_band_width),
        "presence_threshold": float(cfg.mesh_teacher_shock_presence_threshold),
        "weight_power": float(cfg.mesh_teacher_shock_weight_power),
    }
    if target_path.exists():
        cached = np.load(target_path, mmap_mode="r")
        cached_meta = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                cached_meta = json.load(handle)
        cached_config = cached_meta.get("config") if isinstance(cached_meta, dict) else None
        if int(cached.shape[0]) == expected_rows and cached_config == expected_config:
            return cached
        target_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    n_conditions = int(y_red.shape[0] // graph.n_points)
    alpha = np.zeros((y_red.shape[0], 1), dtype=np.float32)
    x_grid = graph.scatter_numpy(graph.coords[:, [0]])[0]
    y_grid = graph.scatter_numpy(graph.coords[:, [1]])[0]
    for cond_idx in range(n_conditions):
        row_start = cond_idx * graph.n_points
        row_stop = row_start + graph.n_points
        cp_grid = graph.scatter_numpy(np.asarray(y_red[row_start:row_stop, [cfg.cp_column]], dtype=np.float32))[0]
        cfx_grid = graph.scatter_numpy(np.asarray(y_red[row_start:row_stop, [cfg.cfx_column]], dtype=np.float32))[0]
        score_grid = _shock_target_map_numpy(
            cp_grid,
            cfx_grid,
            x_grid,
            y_grid,
            graph.valid_mask,
            cfg.mesh_teacher_shock_quantile,
            cfg.mesh_teacher_cfx_weight,
        )
        alpha_grid = _shock_line_target_from_score_map(
            score_grid,
            x_grid,
            graph.valid_mask,
            cfg.mesh_teacher_shock_presence_threshold,
            cfg.mesh_teacher_shock_weight_power,
            cfg.mesh_teacher_shock_band_width,
        )
        alpha[row_start:row_stop, 0] = graph.gather_numpy(alpha_grid[None, ...])[:, 0]
        if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
            print(f"[mesh-shock-target] {split}: condition {cond_idx + 1}/{n_conditions}")
    np.save(target_path, alpha.astype(np.float32))
    valid = alpha[:, 0]
    summary = {
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
        "fraction_ge_025": float(np.mean(valid >= 0.25)),
        "fraction_ge_050": float(np.mean(valid >= 0.50)),
        "fraction_ge_075": float(np.mean(valid >= 0.75)),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump({"config": expected_config, "summary": summary}, handle, indent=2)
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
    architecture = "mesh_teacher_cp_shock_residual_v4" if cfg.mesh_teacher_use_shock_residual else "mesh_teacher_cp_shock_v3"
    payload = {
        "architecture": architecture,
        "input_dim": int(input_dim),
        "hidden_dim": int(cfg.mesh_teacher_hidden_dim),
        "latent_dim": int(cfg.latent_dim),
        "message_passing_steps": int(cfg.mesh_teacher_message_passing_steps),
        "dropout": float(cfg.mesh_teacher_dropout),
        "use_shock_residual": bool(cfg.mesh_teacher_use_shock_residual),
        "points_per_condition": int(graph.n_points),
        "height": int(graph.height),
        "width": int(graph.width),
        "shock_quantile": float(cfg.mesh_teacher_shock_quantile),
        "cfx_weight": float(cfg.mesh_teacher_cfx_weight),
        "shock_band_width": float(cfg.mesh_teacher_shock_band_width),
        "shock_presence_threshold": float(cfg.mesh_teacher_shock_presence_threshold),
        "shock_weight_power": float(cfg.mesh_teacher_shock_weight_power),
        "cp_shock_weight": float(cfg.expert_shock_weight),
        "shock_focal_gamma": float(cfg.mesh_teacher_shock_focal_gamma),
        "shock_bce_weight": float(cfg.mesh_teacher_shock_bce_weight),
        "shock_dice_weight": float(cfg.mesh_teacher_shock_dice_weight),
        "x_dilations": [int(v) for v in cfg.mesh_graph_x_dilations],
        "include_diagonals": bool(cfg.mesh_graph_include_diagonals),
        "graph_connectivity": "projected_xy_offsets_plus_row_col_successors_v3",
        "edge_projection": "xy",
        "cp_loss_weight": float(cfg.mesh_teacher_cp_loss_weight),
        "shock_loss_weight": float(cfg.mesh_teacher_shock_loss_weight),
        "grad_loss_weight": float(cfg.mesh_teacher_grad_loss_weight),
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


def _render_linear_expression(intercept: float, coefficients: np.ndarray, basis_names: list[str], top_k: int = 10) -> str:
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
    return "".join(parts)


def _render_linear_equation(intercept: float, coefficients: np.ndarray, basis_names: list[str], top_k: int = 10) -> str:
    return "clip(" + _render_linear_expression(intercept, coefficients, basis_names, top_k=top_k) + ", 0, 1)"


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


def _row_geometry(graph: CompactSurfaceGraph) -> dict[str, np.ndarray]:
    height = int(graph.height)
    row_y = np.zeros(height, dtype=np.float32)
    row_z = np.zeros(height, dtype=np.float32)
    row_x_min = np.zeros(height, dtype=np.float32)
    row_x_max = np.zeros(height, dtype=np.float32)
    row_valid = np.zeros(height, dtype=bool)
    for row in range(height):
        idx = graph.row_idx == row
        if not np.any(idx):
            continue
        row_valid[row] = True
        row_xyz = graph.coords[idx]
        row_y[row] = float(np.mean(row_xyz[:, 1]))
        row_z[row] = float(np.mean(row_xyz[:, 2]))
        row_x_min[row] = float(np.min(row_xyz[:, 0]))
        row_x_max[row] = float(np.max(row_xyz[:, 0]))
    row_x_mid = 0.5 * (row_x_min + row_x_max)
    row_x_span = np.maximum(row_x_max - row_x_min, 1e-3)
    return {
        "row_y": row_y,
        "row_z": row_z,
        "row_x_min": row_x_min,
        "row_x_max": row_x_max,
        "row_x_mid": row_x_mid.astype(np.float32),
        "row_x_span": row_x_span.astype(np.float32),
        "row_valid": row_valid,
    }


def _build_shock_line_basis(condition_raw: np.ndarray, graph: CompactSurfaceGraph) -> np.ndarray:
    if condition_raw.ndim != 2 or condition_raw.shape[1] < 9:
        raise ValueError(f"Expected [n_conditions, >=9] raw condition array, got {condition_raw.shape}")
    geometry = _row_geometry(graph)
    row_y = geometry["row_y"].astype(np.float64)
    row_z = geometry["row_z"].astype(np.float64)
    n_conditions = int(condition_raw.shape[0])
    n_rows = int(graph.height)

    mach = np.repeat(condition_raw[:, 6:7].astype(np.float64), n_rows, axis=0)
    pi = np.repeat(condition_raw[:, 8:9].astype(np.float64), n_rows, axis=0)
    aoa = np.repeat(condition_raw[:, 7:8].astype(np.float64), n_rows, axis=0)
    row_y_rep = np.tile(row_y, n_conditions)[:, None]
    row_z_rep = np.tile(row_z, n_conditions)[:, None]

    basis = np.concatenate(
        [
            row_y_rep,
            row_z_rep,
            mach,
            pi,
            aoa,
            row_y_rep * row_y_rep,
            row_z_rep * row_z_rep,
            mach * mach,
            aoa * aoa,
            mach * aoa,
            mach * row_y_rep,
            aoa * row_y_rep,
        ],
        axis=1,
    )
    return basis.astype(np.float64)


def _solve_weighted_linear_model(
    basis_train: np.ndarray,
    target_train: np.ndarray,
    feature_names: list[str],
    ridge_alpha: float,
    clip_value: float,
    weights: np.ndarray | None = None,
    output_kind: str = "identity",
    clip_min: float | None = None,
    clip_max: float | None = None,
    top_k: int = 12,
) -> dict[str, object]:
    mean = basis_train.mean(axis=0)
    scale = basis_train.std(axis=0) + 1e-6
    standardized = np.clip((basis_train - mean) / scale, -clip_value, clip_value)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    weight_vec = np.ones((design.shape[0],), dtype=np.float64) if weights is None else np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    sqrt_w = np.sqrt(weight_vec)[:, None]
    design_w = design * sqrt_w
    target_w = target_train.astype(np.float64) * sqrt_w[:, 0]
    gram = design_w.T @ design_w
    ridge = np.eye(gram.shape[0], dtype=np.float64)
    ridge[0, 0] = 0.0
    system = gram + float(ridge_alpha) * ridge
    rhs = design_w.T @ target_w
    coef_std = np.linalg.solve(system, rhs)
    coef_raw = coef_std[1:] / scale
    intercept_raw = float(coef_std[0] - np.sum(coef_std[1:] * mean / scale))
    expression = _render_linear_expression(intercept_raw, coef_raw, feature_names, top_k=top_k)
    if output_kind == "clip01":
        equation = f"clip({expression}, 0, 1)"
    elif output_kind == "exp_clip":
        floor = float(clip_min if clip_min is not None else 0.0)
        ceiling = float(clip_max if clip_max is not None else np.inf)
        equation = f"clip(exp({expression}), {floor:.4f}, {ceiling:.4f})"
    else:
        equation = expression
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients_std": coef_std.tolist(),
        "coefficients_raw": coef_raw.tolist(),
        "intercept_raw": intercept_raw,
        "equation": equation,
        "output_kind": output_kind,
        "clip_min": None if clip_min is None else float(clip_min),
        "clip_max": None if clip_max is None else float(clip_max),
    }


def _apply_weighted_linear_model(basis: np.ndarray, artifact: dict[str, object], clip_value: float) -> np.ndarray:
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
    coef_std = np.asarray(artifact["coefficients_std"], dtype=np.float64)
    standardized = np.clip((basis - mean) / scale, -clip_value, clip_value)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    raw = design @ coef_std
    output_kind = str(artifact.get("output_kind", "identity"))
    if output_kind == "clip01":
        return np.clip(raw, 0.0, 1.0).astype(np.float32)
    if output_kind == "exp_clip":
        floor = float(artifact.get("clip_min", 0.0))
        ceiling = float(artifact.get("clip_max", np.inf))
        return np.clip(np.exp(raw), floor, ceiling).astype(np.float32)
    return raw.astype(np.float32)


def _derive_shock_line_targets(alpha_flat: np.ndarray, graph: CompactSurfaceGraph, cfg: FullAircraftConfig) -> dict[str, np.ndarray]:
    n_conditions = int(alpha_flat.shape[0] // graph.n_points)
    geometry = _row_geometry(graph)
    row_valid = geometry["row_valid"]
    row_x_mid = geometry["row_x_mid"]
    x_grid = graph.scatter_numpy(graph.coords[:, [0]])[0]
    n_rows = int(graph.height)
    presence = np.zeros((n_conditions, n_rows), dtype=np.float32)
    center = np.tile(row_x_mid[None, :], (n_conditions, 1)).astype(np.float32)
    width = np.full((n_conditions, n_rows), float(cfg.mesh_shock_line_width_floor), dtype=np.float32)

    for cond_idx in range(n_conditions):
        row_start = cond_idx * graph.n_points
        row_stop = row_start + graph.n_points
        alpha_values = np.asarray(alpha_flat[row_start:row_stop], dtype=np.float32).reshape(graph.n_points, 1)
        alpha_grid = graph.scatter_numpy(alpha_values)[0]
        for row in range(n_rows):
            if not row_valid[row]:
                continue
            valid = graph.valid_mask[row] > 0.5
            row_alpha = alpha_grid[row, valid]
            row_x = x_grid[row, valid]
            if row_alpha.size == 0:
                continue
            peak = float(np.max(row_alpha))
            presence[cond_idx, row] = peak
            weights = np.maximum(row_alpha - float(cfg.mesh_shock_line_activation_threshold), 0.0) ** float(cfg.mesh_shock_line_weight_power)
            if float(np.sum(weights)) <= 1e-8:
                weights = np.maximum(row_alpha, 0.0) ** float(cfg.mesh_shock_line_weight_power)
            if float(np.sum(weights)) <= 1e-8:
                continue
            center_val = float(np.sum(weights * row_x) / np.sum(weights))
            variance = float(np.sum(weights * (row_x - center_val) ** 2) / np.sum(weights))
            width_val = float(np.clip(2.0 * np.sqrt(max(variance, 1e-8)), cfg.mesh_shock_line_width_floor, cfg.mesh_shock_line_width_ceiling))
            center[cond_idx, row] = center_val
            width[cond_idx, row] = width_val
    return {
        "presence": presence,
        "center": center,
        "width": width,
    }


def _predict_shock_line_rows(
    cfg: FullAircraftConfig,
    condition_raw: np.ndarray,
    graph: CompactSurfaceGraph,
    artifact: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = _row_geometry(graph)
    basis = _build_shock_line_basis(condition_raw[None, : cfg.input_dim_raw].astype(np.float32), graph)
    presence = _apply_weighted_linear_model(basis, artifact["presence_model"], cfg.sensor_feature_clip)
    center = _apply_weighted_linear_model(basis, artifact["center_model"], cfg.sensor_feature_clip)
    width = _apply_weighted_linear_model(basis, artifact["width_model"], cfg.sensor_feature_clip)
    presence = np.clip(presence.reshape(-1), 0.0, 1.0)
    center = np.clip(center.reshape(-1), geometry["row_x_min"], geometry["row_x_max"])
    width = np.clip(width.reshape(-1), cfg.mesh_shock_line_width_floor, cfg.mesh_shock_line_width_ceiling)
    return presence.astype(np.float32), center.astype(np.float32), width.astype(np.float32)


def _apply_shock_line_sensor_condition(
    cfg: FullAircraftConfig,
    x_chunk: np.ndarray,
    graph: CompactSurfaceGraph,
    artifact: dict[str, object],
) -> np.ndarray:
    presence, center, width = _predict_shock_line_rows(cfg, x_chunk[0], graph, artifact)
    point_rows = graph.row_idx
    point_x = x_chunk[:, 0].astype(np.float32)
    row_presence = presence[point_rows]
    row_center = center[point_rows]
    row_width = width[point_rows]
    alpha = row_presence * np.exp(-0.5 * ((point_x - row_center) / np.maximum(row_width, 1e-6)) ** 2)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _evaluate_shock_line_sensor(
    cfg: FullAircraftConfig,
    x_raw: np.ndarray,
    alpha_true: np.ndarray,
    graph: CompactSurfaceGraph,
    artifact: dict[str, object],
) -> dict[str, float]:
    n_conditions = int(x_raw.shape[0] // graph.n_points)
    abs_err = 0.0
    count = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    threshold = float(cfg.mesh_teacher_binary_threshold)
    for cond_idx in range(n_conditions):
        row_start = cond_idx * graph.n_points
        row_stop = row_start + graph.n_points
        x_chunk = np.asarray(x_raw[row_start:row_stop, : cfg.input_dim_raw], dtype=np.float32)
        pred = _apply_shock_line_sensor_condition(cfg, x_chunk, graph, artifact)
        target = np.asarray(alpha_true[row_start:row_stop], dtype=np.float32).reshape(-1)
        abs_err += float(np.abs(pred - target).sum())
        count += float(target.shape[0])
        target_bin = target >= threshold
        pred_bin = pred >= threshold
        tp += float(np.logical_and(target_bin, pred_bin).sum())
        fp += float(np.logical_and(~target_bin, pred_bin).sum())
        fn += float(np.logical_and(target_bin, ~pred_bin).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    iou = tp / max(1.0, tp + fp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "mae": abs_err / max(1.0, count),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


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
    cp_pred, shock_logits, latent = model(feat, edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_attr)
    shock_pred = torch.sigmoid(shock_logits) * mask_flat
    return cp_pred, shock_logits, shock_pred, latent


def _teacher_loss(
    cfg: FullAircraftConfig,
    graph: CompactSurfaceGraph,
    cp_pred: torch.Tensor,
    shock_logits: torch.Tensor,
    shock_pred: torch.Tensor,
    cp: torch.Tensor,
    shock_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_flat = torch.ones_like(cp)
    cp_weight = (1.0 + float(cfg.expert_shock_weight) * shock_target) * mask_flat
    cp_loss = _masked_weighted_smooth_l1(cp_pred, cp, cp_weight)
    focal_loss = _masked_focal_bce_with_logits(
        shock_logits,
        shock_target,
        mask_flat,
        cfg.mesh_teacher_shock_focal_gamma,
    )
    dice_loss = _masked_soft_dice_loss(shock_pred, shock_target, mask_flat)
    shock_loss = (
        float(cfg.mesh_teacher_shock_bce_weight) * focal_loss
        + float(cfg.mesh_teacher_shock_dice_weight) * dice_loss
    )

    pred_grid = graph.scatter_tensor(cp_pred)
    cp_grid = graph.scatter_tensor(cp)
    shock_grid = graph.scatter_tensor(shock_target)
    mask_grid = graph.mask_tensor(device=cp.device).expand(cp.shape[0], -1, -1, -1)
    grad_weight = (1.0 + float(cfg.expert_shock_weight) * shock_grid) * mask_grid
    grad_loss = _gradient_loss(pred_grid, cp_grid, grad_weight, mask_grid)

    total = (
        float(cfg.mesh_teacher_cp_loss_weight) * cp_loss
        + float(cfg.mesh_teacher_shock_loss_weight) * shock_loss
        + float(cfg.mesh_teacher_grad_loss_weight) * grad_loss
    )
    return total, {
        "cp": float(cp_loss.item()),
        "shock": float(shock_loss.item()),
        "shock_focal": float(focal_loss.item()),
        "shock_dice": float(dice_loss.item()),
        "grad": float(grad_loss.item()),
    }


def _evaluate_teacher(cfg: FullAircraftConfig, graph: CompactSurfaceGraph, model: FullAircraftMeshTeacher, loader: DataLoader) -> dict[str, float]:
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)
    total_cp = 0.0
    total_rmse = 0.0
    total_shock = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_shock_zone_cp = 0.0
    total_smooth_zone_cp = 0.0
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for feat, cp, shock_target in loader:
            feat = feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            shock_target = shock_target.to(cfg.device, non_blocking=True)
            mask_flat = torch.ones_like(cp)
            cp_pred, _, shock_pred, _ = _forward_teacher(model, feat, edge_src, edge_dst, edge_attr, mask_flat)
            cp_mae = _masked_mae(cp_pred, cp, mask_flat)
            rmse = torch.sqrt(torch.mean((cp_pred - cp).square()))
            shock_mae = _masked_mae(shock_pred, shock_target, mask_flat)
            shock_zone = (shock_target >= 0.50).to(dtype=cp.dtype)
            smooth_zone = (shock_target < 0.25).to(dtype=cp.dtype)
            shock_zone_cp = _masked_mae(cp_pred, cp, shock_zone) if torch.any(shock_zone > 0.5) else cp.new_tensor(0.0)
            smooth_zone_cp = _masked_mae(cp_pred, cp, smooth_zone) if torch.any(smooth_zone > 0.5) else cp.new_tensor(0.0)
            binary = _binary_metrics(
                shock_target.detach().cpu().numpy().reshape(-1),
                shock_pred.detach().cpu().numpy().reshape(-1),
                float(cfg.mesh_teacher_binary_threshold),
            )
            total_cp += float(cp_mae.item())
            total_rmse += float(rmse.item())
            total_shock += float(shock_mae.item())
            total_precision += float(binary["precision"])
            total_recall += float(binary["recall"])
            total_iou += float(binary["iou"])
            total_f1 += float(binary["f1"])
            total_shock_zone_cp += float(shock_zone_cp.item())
            total_smooth_zone_cp += float(smooth_zone_cp.item())
            n_batches += 1
    denom = max(1, n_batches)
    return {
        "cp_mae": total_cp / denom,
        "cp_rmse": total_rmse / denom,
        "shock_mae": total_shock / denom,
        "shock_precision": total_precision / denom,
        "shock_recall": total_recall / denom,
        "shock_iou": total_iou / denom,
        "shock_f1": total_f1 / denom,
        "cp_mae_shock_zone": total_shock_zone_cp / denom,
        "cp_mae_smooth_zone": total_smooth_zone_cp / denom,
    }


def _plot_training_curves(cfg: FullAircraftConfig, train_hist: list[float], test_hist: list[float]) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(train_hist, label="train objective")
    plt.plot(test_hist, label="test Cp MAE")
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
        use_shock_residual=cfg.mesh_teacher_use_shock_residual,
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
            cp_pred, shock_logits, shock_pred, _ = _forward_teacher(model, feat, edge_src, edge_dst, edge_attr, mask_flat)
            loss, terms = _teacher_loss(cfg, graph, cp_pred, shock_logits, shock_pred, cp, shock_target)
            loss.backward()
            optimizer.step()
            total_objective += float(loss.item())
            total_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                cp=f"{terms['cp']:.5f}",
                shock=f"{terms['shock']:.5f}",
                focal=f"{terms['shock_focal']:.5f}",
            )
        scheduler.step()
        train_objective = total_objective / max(1, total_batches)
        test_metrics = _evaluate_teacher(cfg, graph, model, test_loader)
        train_objective_hist.append(train_objective)
        test_mae_hist.append(test_metrics["cp_mae"])
        print(
            f"[train-mesh-teacher] epoch {epoch:03d} | train_obj={train_objective:.6f} | "
            f"test_cp_mae={test_metrics['cp_mae']:.6f} | shock_mae={test_metrics['shock_mae']:.6f} | "
            f"shock_iou={test_metrics['shock_iou']:.3f}"
        )

    torch.save(model.state_dict(), _model_path(cfg))
    _save_model_config(cfg, graph, feature_dim)
    train_diag = _evaluate_teacher(cfg, graph, model, train_loader)
    test_diag = _evaluate_teacher(cfg, graph, model, test_loader)
    save_json(
        _training_metrics_path(cfg),
        {
            "architecture": "mesh_teacher_cp_shock_residual_v4" if cfg.mesh_teacher_use_shock_residual else "mesh_teacher_cp_shock_v3",
            "use_shock_residual": bool(cfg.mesh_teacher_use_shock_residual),
            "surface": cfg.reduced_surface,
            "final_train_objective": float(train_objective_hist[-1]),
            "final_test_cp_mae": float(test_mae_hist[-1]),
            "train_diagnostics": train_diag,
            "test_diagnostics": test_diag,
            "hidden_dim": int(cfg.mesh_teacher_hidden_dim),
            "latent_dim": int(cfg.latent_dim),
            "message_passing_steps": int(cfg.mesh_teacher_message_passing_steps),
            "shock_quantile": float(cfg.mesh_teacher_shock_quantile),
            "cfx_weight": float(cfg.mesh_teacher_cfx_weight),
        },
    )
    save_json(_diagnostics_path(cfg, "train"), {"split": "train", **{k: float(v) for k, v in train_diag.items()}})
    save_json(_diagnostics_path(cfg, "test"), {"split": "test", **{k: float(v) for k, v in test_diag.items()}})
    _plot_training_curves(cfg, train_objective_hist, test_mae_hist)
    print(f"[train-mesh-teacher] Finished. Model stored in {_model_path(cfg)}")


def _load_mesh_teacher(cfg: FullAircraftConfig) -> FullAircraftMeshTeacher:
    config = json.loads(_model_config_path(cfg).read_text())
    architecture = str(config.get("architecture", "mesh_teacher_v1"))
    supported = {"mesh_teacher_cp_shock_v3", "mesh_teacher_cp_shock_residual_v4"}
    if architecture not in supported:
        raise ValueError(
            f"Stored mesh teacher architecture is {architecture!r}, but the current pipeline expects "
            f"one of {sorted(supported)}. Re-run 'train-mesh-teacher' before distilling or inferring."
        )
    use_shock_residual = bool(config.get("use_shock_residual", architecture == "mesh_teacher_cp_shock_residual_v4"))
    model = FullAircraftMeshTeacher(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config.get("latent_dim", cfg.latent_dim)),
        message_passing_steps=int(config["message_passing_steps"]),
        dropout=float(config.get("dropout", cfg.mesh_teacher_dropout)),
        use_shock_residual=use_shock_residual,
    )
    state = torch.load(_model_path(cfg), map_location="cpu")
    model.load_state_dict(state)
    try:
        model.to(cfg.device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and str(cfg.device).startswith("cuda"):
            raise RuntimeError(
                "CUDA ran out of memory while loading the mesh teacher. "
                "Free GPU memory, select another GPU with CUDA_VISIBLE_DEVICES, or rerun with '--device cpu'."
            ) from exc
        raise
    model.eval()
    return model


def _build_teacher_shock_array(cfg: FullAircraftConfig, split: str, graph: CompactSurfaceGraph, model: FullAircraftMeshTeacher) -> np.ndarray:
    target_path = _teacher_shock_path(cfg, split)
    dataset = _MeshConditionDataset(cfg, split, graph)
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)
    shock = np.zeros((dataset.n_conditions * graph.n_points, 1), dtype=np.float32)
    with torch.no_grad():
        for cond_idx in range(dataset.n_conditions):
            feat, _, _ = dataset[cond_idx]
            feat_t = feat.unsqueeze(0).to(cfg.device, non_blocking=True)
            mask_flat = torch.ones((1, graph.n_points, 1), device=cfg.device, dtype=torch.float32)
            _, _, shock_teacher, _ = _forward_teacher(model, feat_t, edge_src, edge_dst, edge_attr, mask_flat)
            row_start = cond_idx * graph.n_points
            row_stop = row_start + graph.n_points
            shock[row_start:row_stop, 0] = shock_teacher[0, :, 0].detach().cpu().numpy().astype(np.float32)
            if cond_idx == 0 or cond_idx + 1 == dataset.n_conditions or cond_idx % 25 == 0:
                print(f"[mesh-teacher-shock] {split}: condition {cond_idx + 1}/{dataset.n_conditions}")
    np.save(target_path, shock.astype(np.float32))
    return np.load(target_path, mmap_mode="r")


def distill_mesh_sensor(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    graph = CompactSurfaceGraph.from_reference(cfg)
    model = _load_mesh_teacher(cfg)
    shock_train = np.asarray(_build_teacher_shock_array(cfg, "train", graph, model)[:, 0], dtype=np.float32)
    shock_test = np.asarray(_build_teacher_shock_array(cfg, "test", graph, model)[:, 0], dtype=np.float32)
    x_train = np.load(cfg.cut_data_dir / "X_cut_train.npy", mmap_mode="r")
    x_test = np.load(cfg.cut_data_dir / "X_cut_test.npy", mmap_mode="r")
    train_cond = np.asarray(x_train[:: graph.n_points, : cfg.input_dim_raw], dtype=np.float32)
    test_cond = np.asarray(x_test[:: graph.n_points, : cfg.input_dim_raw], dtype=np.float32)
    row_basis_train = _build_shock_line_basis(train_cond, graph)
    row_basis_test = _build_shock_line_basis(test_cond, graph)
    train_targets = _derive_shock_line_targets(shock_train, graph, cfg)
    test_targets = _derive_shock_line_targets(shock_test, graph, cfg)

    presence_train = train_targets["presence"].reshape(-1)
    center_train = train_targets["center"].reshape(-1)
    width_train = train_targets["width"].reshape(-1)
    presence_test = test_targets["presence"].reshape(-1)
    center_test = test_targets["center"].reshape(-1)
    width_test = test_targets["width"].reshape(-1)
    active_train = presence_train >= float(cfg.mesh_shock_line_activation_threshold)
    active_test = presence_test >= float(cfg.mesh_shock_line_activation_threshold)

    artifact = {
        "type": "mesh_shock_line_symbolic",
        "description": "Symbolic shock-line sensor distilled from the MeshGraphNet auxiliary shock head.",
        "surface": cfg.reduced_surface,
        "feature_names": MESH_SHOCK_LINE_FEATURE_NAMES,
        "binary_threshold": float(cfg.mesh_teacher_binary_threshold),
        "ridge_alpha": float(cfg.mesh_symbolic_ridge_alpha),
        "feature_clip": float(cfg.sensor_feature_clip),
        "teacher_source": "mesh_teacher_shock",
        "latent_dim": int(cfg.latent_dim),
        "activation_threshold": float(cfg.mesh_shock_line_activation_threshold),
        "weight_power": float(cfg.mesh_shock_line_weight_power),
        "width_floor": float(cfg.mesh_shock_line_width_floor),
        "width_ceiling": float(cfg.mesh_shock_line_width_ceiling),
        "alpha_formula": "alpha(x,y)=clip(presence(y)*exp(-0.5*((x-x_shock(y))/width(y))^2), 0, 1)",
        "presence_model": _solve_weighted_linear_model(
            row_basis_train,
            presence_train,
            MESH_SHOCK_LINE_FEATURE_NAMES,
            cfg.mesh_symbolic_ridge_alpha,
            cfg.sensor_feature_clip,
            output_kind="clip01",
        ),
        "center_model": _solve_weighted_linear_model(
            row_basis_train,
            center_train,
            MESH_SHOCK_LINE_FEATURE_NAMES,
            cfg.mesh_symbolic_ridge_alpha,
            cfg.sensor_feature_clip,
            weights=np.maximum(presence_train, 1e-3),
            output_kind="identity",
        ),
        "width_model": _solve_weighted_linear_model(
            row_basis_train,
            np.log(np.clip(width_train, cfg.mesh_shock_line_width_floor, None)),
            MESH_SHOCK_LINE_FEATURE_NAMES,
            cfg.mesh_symbolic_ridge_alpha,
            cfg.sensor_feature_clip,
            weights=np.maximum(presence_train, 1e-3),
            output_kind="exp_clip",
            clip_min=cfg.mesh_shock_line_width_floor,
            clip_max=cfg.mesh_shock_line_width_ceiling,
        ),
    }
    train_eval = _evaluate_shock_line_sensor(cfg, x_train, shock_train, graph, artifact)
    test_eval = _evaluate_shock_line_sensor(cfg, x_test, shock_test, graph, artifact)
    pred_presence_train = _apply_weighted_linear_model(row_basis_train, artifact["presence_model"], cfg.sensor_feature_clip).reshape(-1)
    pred_center_train = _apply_weighted_linear_model(row_basis_train, artifact["center_model"], cfg.sensor_feature_clip).reshape(-1)
    pred_width_train = _apply_weighted_linear_model(row_basis_train, artifact["width_model"], cfg.sensor_feature_clip).reshape(-1)
    pred_presence_test = _apply_weighted_linear_model(row_basis_test, artifact["presence_model"], cfg.sensor_feature_clip).reshape(-1)
    pred_center_test = _apply_weighted_linear_model(row_basis_test, artifact["center_model"], cfg.sensor_feature_clip).reshape(-1)
    pred_width_test = _apply_weighted_linear_model(row_basis_test, artifact["width_model"], cfg.sensor_feature_clip).reshape(-1)
    artifact["train_mae"] = float(train_eval["mae"])
    artifact["test_mae"] = float(test_eval["mae"])
    artifact["train_binary_metrics"] = {k: float(v) for k, v in train_eval.items() if k != "mae"}
    artifact["test_binary_metrics"] = {k: float(v) for k, v in test_eval.items() if k != "mae"}
    artifact["row_metrics"] = {
        "train_presence_mae": float(np.mean(np.abs(pred_presence_train - presence_train))),
        "test_presence_mae": float(np.mean(np.abs(pred_presence_test - presence_test))),
        "train_center_mae_active": float(np.mean(np.abs(pred_center_train[active_train] - center_train[active_train]))) if np.any(active_train) else 0.0,
        "test_center_mae_active": float(np.mean(np.abs(pred_center_test[active_test] - center_test[active_test]))) if np.any(active_test) else 0.0,
        "train_width_mae_active": float(np.mean(np.abs(pred_width_train[active_train] - width_train[active_train]))) if np.any(active_train) else 0.0,
        "test_width_mae_active": float(np.mean(np.abs(pred_width_test[active_test] - width_test[active_test]))) if np.any(active_test) else 0.0,
    }

    with _sensor_json_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    with _sensor_txt_path(cfg).open("w", encoding="utf-8") as handle:
        handle.write("presence(y, z, Mach, Pi, AoA) = " + artifact["presence_model"]["equation"] + "\n")
        handle.write("x_shock(y, z, Mach, Pi, AoA) = " + artifact["center_model"]["equation"] + "\n")
        handle.write("width(y, z, Mach, Pi, AoA) = " + artifact["width_model"]["equation"] + "\n")
        handle.write(artifact["alpha_formula"] + "\n")
        handle.write(json.dumps({"train_mae": artifact["train_mae"], "test_mae": artifact["test_mae"]}, indent=2) + "\n")
    print(
        f"[distill-mesh-sensor] train_mae={artifact['train_mae']:.5f}, test_mae={artifact['test_mae']:.5f}, "
        f"test_iou={artifact['test_binary_metrics']['iou']:.3f}"
    )
    print(f"[distill-mesh-sensor] Sensor stored in {_sensor_json_path(cfg)}")


def _load_mesh_sensor(cfg: FullAircraftConfig) -> dict[str, object]:
    with _sensor_json_path(cfg).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_symbolic_mesh_sensor(cfg: FullAircraftConfig, x_raw: np.ndarray, artifact: dict[str, object]) -> np.ndarray:
    sensor_type = str(artifact.get("type", "mesh_local_shock_linear_map"))
    if sensor_type == "mesh_shock_line_symbolic":
        graph = CompactSurfaceGraph.from_reference(cfg)
        return _apply_shock_line_sensor_condition(cfg, x_raw[:, : cfg.input_dim_raw].astype(np.float32), graph, artifact)
    basis = _build_sensor_basis(x_raw[:, : cfg.input_dim_raw].astype(np.float32))
    return _apply_linear_sensor_basis(basis, artifact, cfg)


def infer_mesh_teacher(
    cfg: FullAircraftConfig,
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_rows: int | None = None,
) -> Path:
    cfg.ensure_dirs()
    input_path = Path(input_path).expanduser().resolve() if input_path is not None else (cfg.cut_data_dir / "X_cut_test.npy")
    output_path = Path(output_path).expanduser().resolve() if output_path is not None else _default_teacher_inference_output_path(cfg, input_path)
    x_raw = np.load(input_path, mmap_mode="r")
    graph = CompactSurfaceGraph.from_reference(cfg)
    if x_raw.shape[0] % graph.n_points != 0:
        raise ValueError("Input rows are not divisible by the reduced graph point count.")
    if max_rows is not None:
        x_raw = np.asarray(x_raw[:max_rows], dtype=np.float32)

    expert_mean, expert_scale = _load_expert_scaler(cfg)
    cp_mean, cp_scale = _load_cp_scaler(cfg)
    model = _load_mesh_teacher(cfg)
    edge_src, edge_dst, edge_attr = graph.edge_tensors(cfg.device)

    n_rows = int(x_raw.shape[0])
    n_conditions = int(n_rows // graph.n_points)
    cp_pred = np.zeros((n_rows, 1), dtype=np.float32)
    shock_pred = np.zeros((n_rows, 1), dtype=np.float32)

    with torch.no_grad():
        for cond_idx in range(n_conditions):
            row_start = cond_idx * graph.n_points
            row_stop = row_start + graph.n_points
            x_chunk = np.asarray(x_raw[row_start:row_stop, : cfg.input_dim_raw], dtype=np.float32)
            feat_flat = build_expert_features(x_chunk)
            feat_flat = ((feat_flat - expert_mean) / expert_scale).astype(np.float32)
            feat_tensor = torch.from_numpy(feat_flat[None, ...]).to(cfg.device, non_blocking=True)
            mask_flat = torch.ones((1, graph.n_points, 1), device=cfg.device, dtype=torch.float32)
            cp_t, _, shock_t, _ = _forward_teacher(model, feat_tensor, edge_src, edge_dst, edge_attr, mask_flat)
            cp_flat = cp_t[0].detach().cpu().numpy().astype(np.float32)
            shock_flat = shock_t[0].detach().cpu().numpy().astype(np.float32)

            cp_pred[row_start:row_stop] = _destandardize(cp_flat, cp_mean, cp_scale)
            shock_pred[row_start:row_stop] = shock_flat

            if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
                print(f"[infer-mesh-teacher] condition {cond_idx + 1}/{n_conditions}")

    np.savez_compressed(
        output_path,
        cp_pred=cp_pred.astype(np.float32),
        shock_pred=shock_pred.astype(np.float32),
        shock_alpha=shock_pred.astype(np.float32),
        teacher_alpha=shock_pred.astype(np.float32),
        sensor_type=np.array(["mesh_teacher_shock"], dtype=object),
    )
    print(f"[infer-mesh-teacher] Saved predictions to {output_path}")
    return output_path


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
            cp_t, _, shock_t, _ = _forward_teacher(model, feat_tensor, edge_src, edge_dst, edge_attr, mask_flat)
            cp_flat = cp_t[0].detach().cpu().numpy().astype(np.float32)
            teacher_shock_flat = shock_t[0].detach().cpu().numpy().astype(np.float32)
            if str(artifact.get("type", "")) == "mesh_shock_line_symbolic":
                alpha_flat = _apply_shock_line_sensor_condition(cfg, x_chunk, graph, artifact).reshape(-1, 1)
            else:
                alpha_flat = apply_symbolic_mesh_sensor(cfg, x_chunk, artifact).reshape(-1, 1)

            cp_pred[row_start:row_stop] = _destandardize(cp_flat, cp_mean, cp_scale)
            shock_pred[row_start:row_stop] = alpha_flat.astype(np.float32)
            shock_alpha[row_start:row_stop] = alpha_flat.astype(np.float32)
            teacher_alpha[row_start:row_stop] = teacher_shock_flat.astype(np.float32)

            if cond_idx == 0 or cond_idx + 1 == n_conditions or cond_idx % 25 == 0:
                print(f"[infer-mesh-symbolic] condition {cond_idx + 1}/{n_conditions}")

    np.savez_compressed(
        output_path,
        cp_pred=cp_pred.astype(np.float32),
        shock_pred=shock_pred.astype(np.float32),
        shock_alpha=shock_alpha.astype(np.float32),
        teacher_alpha=teacher_alpha.astype(np.float32),
        sensor_type=np.array([artifact["type"]], dtype=object),
    )
    print(f"[infer-mesh-symbolic] Saved predictions to {output_path}")
    return output_path


def _mach_regime(mach: float, cfg: FullAircraftConfig) -> str:
    if mach < float(cfg.mach_sub_max):
        return "subsonic"
    if mach <= float(cfg.mach_trans_max):
        return "transonic"
    return "supersonic"


def _condition_error_records(
    cfg: FullAircraftConfig,
    split: str,
    graph: CompactSurfaceGraph,
    prediction_path: Path,
    shock_target: np.ndarray,
) -> list[dict[str, float | int | str]]:
    x_red = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    y_red = np.load(cfg.cut_data_dir / f"Y_cut_{split}.npy", mmap_mode="r")
    pred_payload = np.load(prediction_path)
    if "cp_pred" not in pred_payload:
        raise KeyError(f"Prediction file does not contain 'cp_pred': {prediction_path}")
    cp_pred = np.asarray(pred_payload["cp_pred"], dtype=np.float32)
    if cp_pred.shape[0] != y_red.shape[0]:
        raise ValueError(f"Prediction rows ({cp_pred.shape[0]}) do not match {split} target rows ({y_red.shape[0]}).")

    n_conditions = int(y_red.shape[0] // graph.n_points)
    records: list[dict[str, float | int | str]] = []
    for cond_idx in range(n_conditions):
        row_start = cond_idx * graph.n_points
        row_stop = row_start + graph.n_points
        truth = np.asarray(y_red[row_start:row_stop, [cfg.cp_column]], dtype=np.float32)
        pred = np.asarray(cp_pred[row_start:row_stop], dtype=np.float32)
        err = pred - truth
        shock = np.asarray(shock_target[row_start:row_stop], dtype=np.float32).reshape(-1)
        abs_err = np.abs(err).reshape(-1)
        shock_zone = shock >= 0.50
        smooth_zone = shock < 0.25
        cond = np.asarray(x_red[row_start, : cfg.input_dim_raw], dtype=np.float32)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err * err)))
        p95 = float(np.quantile(abs_err, 0.95))
        records.append(
            {
                "condition_index": int(cond_idx),
                "Mach": float(cond[6]),
                "AoA_deg": float(cond[7]),
                "Pi": float(cond[8]),
                "regime": _mach_regime(float(cond[6]), cfg),
                "mae": mae,
                "rmse": rmse,
                "abs_error_p95": p95,
                "shock_fraction": float(np.mean(shock_zone)),
                "shock_zone_mae": float(np.mean(abs_err[shock_zone])) if np.any(shock_zone) else 0.0,
                "smooth_zone_mae": float(np.mean(abs_err[smooth_zone])) if np.any(smooth_zone) else 0.0,
            }
        )
    return records


def _aggregate_condition_records(records: list[dict[str, float | int | str]]) -> dict[str, object]:
    if not records:
        return {"conditions": 0}
    maes = np.asarray([float(row["mae"]) for row in records], dtype=np.float64)
    rmses = np.asarray([float(row["rmse"]) for row in records], dtype=np.float64)
    shock_maes = np.asarray([float(row["shock_zone_mae"]) for row in records if float(row["shock_fraction"]) > 0.0], dtype=np.float64)
    smooth_maes = np.asarray([float(row["smooth_zone_mae"]) for row in records], dtype=np.float64)
    payload: dict[str, object] = {
        "conditions": int(len(records)),
        "mae_mean": float(np.mean(maes)),
        "mae_median": float(np.median(maes)),
        "mae_max": float(np.max(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "shock_zone_mae_mean": float(np.mean(shock_maes)) if shock_maes.size else 0.0,
        "smooth_zone_mae_mean": float(np.mean(smooth_maes)) if smooth_maes.size else 0.0,
        "worst_conditions": sorted(records, key=lambda row: float(row["mae"]), reverse=True)[:10],
        "by_regime": {},
    }
    by_regime: dict[str, list[dict[str, float | int | str]]] = {}
    for row in records:
        by_regime.setdefault(str(row["regime"]), []).append(row)
    payload["by_regime"] = {
        regime: {
            "conditions": int(len(rows)),
            "mae_mean": float(np.mean([float(row["mae"]) for row in rows])),
            "mae_max": float(np.max([float(row["mae"]) for row in rows])),
        }
        for regime, rows in sorted(by_regime.items())
    }
    return payload


def _graph_summary(graph: CompactSurfaceGraph) -> dict[str, object]:
    degree = np.bincount(graph.edge_dst, minlength=graph.n_points).astype(np.float32)
    valid_fraction = float(np.mean(graph.valid_mask > 0.5))
    edge_distance = graph.edge_attr[:, 3] if graph.edge_attr.size else np.asarray([], dtype=np.float32)
    return {
        "height": int(graph.height),
        "width": int(graph.width),
        "points": int(graph.n_points),
        "valid_fraction": valid_fraction,
        "edges": int(graph.edge_src.shape[0]),
        "degree_mean": float(np.mean(degree)) if degree.size else 0.0,
        "degree_min": float(np.min(degree)) if degree.size else 0.0,
        "degree_max": float(np.max(degree)) if degree.size else 0.0,
        "edge_distance_mean": float(np.mean(edge_distance)) if edge_distance.size else 0.0,
        "edge_distance_p95": float(np.quantile(edge_distance, 0.95)) if edge_distance.size else 0.0,
    }


def _plot_mesh_diagnostic_grid(
    cfg: FullAircraftConfig,
    split: str,
    graph: CompactSurfaceGraph,
    prediction_path: Path,
    records: list[dict[str, float | int | str]],
    max_conditions: int = 4,
) -> Path:
    y_red = np.load(cfg.cut_data_dir / f"Y_cut_{split}.npy", mmap_mode="r")
    pred_payload = np.load(prediction_path)
    cp_pred = np.asarray(pred_payload["cp_pred"], dtype=np.float32)
    shock_target = np.asarray(_build_shock_target_array(cfg, split, graph), dtype=np.float32)
    selected = sorted(records, key=lambda row: float(row["mae"]), reverse=True)[:max_conditions]
    if not selected:
        raise ValueError("No condition records available for mesh diagnostic plot.")

    fig, axes = plt.subplots(len(selected), 4, figsize=(15.5, 3.2 * len(selected)), constrained_layout=True)
    if len(selected) == 1:
        axes = axes[None, :]
    mask = graph.valid_mask <= 0.5
    columns = ["truth Cp", "pred Cp", "|error|", "shock target"]
    for row_idx_plot, record in enumerate(selected):
        cond_idx = int(record["condition_index"])
        row_start = cond_idx * graph.n_points
        row_stop = row_start + graph.n_points
        truth_grid = graph.scatter_numpy(np.asarray(y_red[row_start:row_stop, [cfg.cp_column]], dtype=np.float32))[0]
        pred_grid = graph.scatter_numpy(np.asarray(cp_pred[row_start:row_stop], dtype=np.float32))[0]
        err_grid = np.abs(pred_grid - truth_grid)
        shock_grid = graph.scatter_numpy(np.asarray(shock_target[row_start:row_stop], dtype=np.float32))[0]
        grids = [truth_grid, pred_grid, err_grid, shock_grid]
        cmaps = ["jet", "jet", "magma", "viridis"]
        for col_idx_plot, (grid, cmap) in enumerate(zip(grids, cmaps)):
            ax = axes[row_idx_plot, col_idx_plot]
            data = np.ma.masked_where(mask, grid)
            image = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
            if row_idx_plot == 0:
                ax.set_title(columns[col_idx_plot])
            if col_idx_plot == 0:
                ax.set_ylabel(
                    f"cond {cond_idx}\nM={float(record['Mach']):.2f}, "
                    f"AoA={float(record['AoA_deg']):.1f}\nMAE={float(record['mae']):.3f}"
                )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, shrink=0.82)

    out_path = cfg.results_surface_dir(cfg.reduced_surface) / f"mesh_diagnostics_{split}_worst.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def diagnose_mesh_pipeline(
    cfg: FullAircraftConfig,
    split: str = "test",
    prediction_path: Path | None = None,
) -> dict[str, object]:
    cfg.ensure_dirs()
    graph = CompactSurfaceGraph.from_reference(cfg)
    prediction_path = (
        Path(prediction_path).expanduser().resolve()
        if prediction_path is not None
        else cfg.inference_dir / f"X_cut_{split}_mesh_teacher.npz"
    )
    if not prediction_path.exists():
        fallback = cfg.inference_dir / f"X_cut_{split}_mesh_symbolic.npz"
        if fallback.exists():
            prediction_path = fallback
        else:
            raise FileNotFoundError(f"No mesh prediction file found: {prediction_path} or {fallback}. Run infer-mesh-teacher first.")

    target = np.asarray(_build_shock_target_array(cfg, split, graph), dtype=np.float32).reshape(-1)
    records = _condition_error_records(cfg, split, graph, prediction_path, target)
    plot_path = _plot_mesh_diagnostic_grid(cfg, split, graph, prediction_path, records)
    payload: dict[str, object] = {
        "surface": cfg.reduced_surface,
        "split": split,
        "prediction_path": str(prediction_path),
        "diagnostic_plot": str(plot_path),
        "graph": _graph_summary(graph),
        "shock_target": {
            "mean": float(np.mean(target)),
            "max": float(np.max(target)),
            "fraction_ge_025": float(np.mean(target >= 0.25)),
            "fraction_ge_050": float(np.mean(target >= 0.50)),
            "fraction_ge_075": float(np.mean(target >= 0.75)),
        },
        "error_summary": _aggregate_condition_records(records),
        "conditions": records,
    }
    out_path = cfg.metrics_dir / f"mesh_pipeline_diagnostics_{split}.json"
    save_json(out_path, payload)
    print(f"[diagnose-mesh] Saved diagnostics to {out_path}")
    return payload
