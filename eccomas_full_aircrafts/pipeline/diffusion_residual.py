from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import FullAircraftConfig
from .features import SYMBOLIC_GATE_ENCODER_INDICES
from .inference import run_inference
from .models import ResidualDiffusionUNet
from .surface_grid import CompactSurfaceGrid
from .utils import save_json

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _diffusion_model_path(cfg: FullAircraftConfig, baseline_mode: str) -> Path:
    return cfg.models_dir / f"diffusion_residual_{baseline_mode}.pth"


def _diffusion_config_path(cfg: FullAircraftConfig, baseline_mode: str) -> Path:
    return cfg.diffusion_dir / f"diffusion_residual_{baseline_mode}.json"


def _default_baseline_prediction_path(cfg: FullAircraftConfig, split: str, baseline_mode: str) -> Path:
    return cfg.inference_dir / f"X_cut_{split}_{baseline_mode}.npz"


def _load_cp_scaler(cfg: FullAircraftConfig) -> tuple[float, float]:
    payload = np.load(cfg.scalers_dir / "cp_scaler.npz")
    mean = float(payload["mean"][0])
    scale = float(payload["scale"][0])
    return mean, scale


def _selected_gate_indices(cfg: FullAircraftConfig) -> list[int]:
    n = min(int(cfg.diffusion_cond_feature_dim), len(SYMBOLIC_GATE_ENCODER_INDICES))
    return [int(idx) for idx in SYMBOLIC_GATE_ENCODER_INDICES[:n]]


def _ensure_baseline_prediction(
    cfg: FullAircraftConfig,
    split: str,
    baseline_mode: str,
    prediction_path: Path | None = None,
) -> Path:
    path = Path(prediction_path) if prediction_path is not None else _default_baseline_prediction_path(cfg, split, baseline_mode)
    path = path.expanduser().resolve()
    if path.exists():
        return path
    input_path = cfg.cut_data_dir / f"X_cut_{split}.npy"
    print(f"[diffusion] baseline predictions not found for {split}; generating {baseline_mode} baseline at {path}")
    run_inference(cfg, input_path=input_path, mode=baseline_mode, output_path=path)
    return path


def _gradient_weight_map(cp_grid: np.ndarray, mask: np.ndarray, shock_weight: float) -> np.ndarray:
    weights = np.ones_like(cp_grid, dtype=np.float32)
    if shock_weight <= 0.0:
        return weights * mask
    grad_y, grad_x = np.gradient(cp_grid.astype(np.float32), edge_order=1)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y) * mask
    valid = grad_mag[mask > 0.5]
    if valid.size == 0:
        return weights * mask
    scale = float(np.quantile(valid, 0.95))
    if scale <= 1e-6:
        return weights * mask
    normalized = np.clip(grad_mag / scale, 0.0, 1.0)
    return (1.0 + float(shock_weight) * normalized).astype(np.float32) * mask


class _ResidualDiffusionConditionDataset(Dataset):
    def __init__(
        self,
        cfg: FullAircraftConfig,
        split: str,
        baseline_prediction_path: Path,
        grid: CompactSurfaceGrid,
        gate_feature_indices: list[int],
    ):
        self.cfg = cfg
        self.split = split
        self.grid = grid
        self.gate_feature_indices = gate_feature_indices
        self.gate = np.load(cfg.features_dir / f"gate_features_{split}.npy", mmap_mode="r")
        self.y = np.load(cfg.reduced_data_dir / f"Y_cut_{split}.npy", mmap_mode="r")
        self.baseline = np.load(baseline_prediction_path, mmap_mode="r")["cp_pred"]
        self.points_per_condition = grid.n_points
        self.n_conditions = int(self.gate.shape[0] // self.points_per_condition)
        self.cp_mean, self.cp_scale = _load_cp_scaler(cfg)
        self.mask = grid.valid_mask[None, ...].astype(np.float32)

    def __len__(self) -> int:
        return self.n_conditions

    def __getitem__(self, idx: int):
        row_start = idx * self.points_per_condition
        row_stop = row_start + self.points_per_condition

        gate_flat = np.asarray(self.gate[row_start:row_stop, self.gate_feature_indices], dtype=np.float32)
        cp_true_flat = np.asarray(self.y[row_start:row_stop, self.cfg.cp_column], dtype=np.float32).reshape(-1, 1)
        cp_base_flat = np.asarray(self.baseline[row_start:row_stop], dtype=np.float32).reshape(-1, 1)

        cp_base_norm = ((cp_base_flat - self.cp_mean) / self.cp_scale).astype(np.float32)
        residual_norm = ((cp_true_flat - cp_base_flat) / self.cp_scale).astype(np.float32)
        cp_true_grid = self.grid.scatter_numpy(cp_true_flat)
        weight_map = _gradient_weight_map(cp_true_grid[0], self.grid.valid_mask, self.cfg.diffusion_shock_weight)[None, ...]

        gate_grid = self.grid.scatter_numpy(gate_flat)
        base_grid = self.grid.scatter_numpy(cp_base_norm)
        residual_grid = self.grid.scatter_numpy(residual_norm)
        cond = np.concatenate([base_grid, gate_grid, self.mask], axis=0).astype(np.float32)

        return (
            torch.from_numpy(residual_grid),
            torch.from_numpy(cond),
            torch.from_numpy(self.mask),
            torch.from_numpy(weight_map.astype(np.float32)),
        )


def _beta_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 2e-2, num_steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def _masked_weighted_mse(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    loss = (pred_noise - target_noise).square() * mask * weight
    denom = (mask * weight).sum().clamp_min(1.0)
    return loss.sum() / denom


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _build_diffusion_model(cfg: FullAircraftConfig, cond_channels: int) -> nn.Module:
    model: nn.Module = ResidualDiffusionUNet(
        cond_channels=cond_channels,
        base_channels=cfg.diffusion_base_channels,
    )
    if cfg.device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(cfg.device)


def _evaluate_diffusion(
    cfg: FullAircraftConfig,
    model: nn.Module,
    loader: DataLoader,
    alpha_bars: torch.Tensor,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for residual, cond, mask, weight in loader:
            residual = residual.to(cfg.device, non_blocking=True)
            cond = cond.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)
            weight = weight.to(cfg.device, non_blocking=True)
            timesteps = torch.randint(0, cfg.diffusion_timesteps, (residual.shape[0],), device=cfg.device)
            noise = torch.randn_like(residual) * mask
            alpha_bar = alpha_bars[timesteps].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar) * residual + torch.sqrt(1.0 - alpha_bar) * noise
            pred_noise = model(x_t, cond, timesteps)
            loss = _masked_weighted_mse(pred_noise, noise, mask, weight)
            total += float(loss.item())
            n_batches += 1
    return total / max(1, n_batches)


def train_diffusion_residual(
    cfg: FullAircraftConfig,
    baseline_mode: str = "symbolic",
    train_prediction_path: Path | None = None,
    test_prediction_path: Path | None = None,
) -> None:
    cfg.ensure_dirs()
    torch.backends.cudnn.benchmark = cfg.device.type == "cuda"

    train_prediction_path = _ensure_baseline_prediction(cfg, "train", baseline_mode, train_prediction_path)
    test_prediction_path = _ensure_baseline_prediction(cfg, "test", baseline_mode, test_prediction_path)

    grid = CompactSurfaceGrid.from_reference(cfg)
    gate_feature_indices = _selected_gate_indices(cfg)
    train_set = _ResidualDiffusionConditionDataset(cfg, "train", train_prediction_path, grid, gate_feature_indices)
    test_set = _ResidualDiffusionConditionDataset(cfg, "test", test_prediction_path, grid, gate_feature_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.diffusion_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(1, cfg.diffusion_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cond_channels = 1 + len(gate_feature_indices) + 1
    model = _build_diffusion_model(cfg, cond_channels=cond_channels)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.diffusion_lr, weight_decay=cfg.diffusion_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.diffusion_epochs)
    scaler = GradScaler(enabled=cfg.device.type == "cuda")
    _betas, _alphas, alpha_bars = _beta_schedule(cfg.diffusion_timesteps, cfg.device)

    history_train: list[float] = []
    history_test: list[float] = []

    print(
        f"[train-diffusion] baseline={baseline_mode} | train_conditions={len(train_set):,} | "
        f"test_conditions={len(test_set):,} | cond_channels={cond_channels} | device={cfg.device}"
    )

    for epoch in range(1, cfg.diffusion_epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[diffusion] epoch {epoch}/{cfg.diffusion_epochs}")
        for residual, cond, mask, weight in pbar:
            residual = residual.to(cfg.device, non_blocking=True)
            cond = cond.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)
            weight = weight.to(cfg.device, non_blocking=True)

            timesteps = torch.randint(0, cfg.diffusion_timesteps, (residual.shape[0],), device=cfg.device)
            noise = torch.randn_like(residual) * mask
            alpha_bar = alpha_bars[timesteps].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar) * residual + torch.sqrt(1.0 - alpha_bar) * noise

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.device.type == "cuda"):
                pred_noise = model(x_t, cond, timesteps)
                loss = _masked_weighted_mse(pred_noise, noise, mask, weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        train_loss = total / max(1, n_batches)
        test_loss = _evaluate_diffusion(cfg, model, test_loader, alpha_bars)
        history_train.append(train_loss)
        history_test.append(test_loss)
        print(f"[train-diffusion] epoch {epoch:03d} | train={train_loss:.6f} | test={test_loss:.6f}")

    state = _unwrap_model(model).state_dict()
    model_path = _diffusion_model_path(cfg, baseline_mode)
    torch.save(state, model_path)

    payload = {
        "baseline_mode": baseline_mode,
        "baseline_train_prediction_path": str(train_prediction_path),
        "baseline_test_prediction_path": str(test_prediction_path),
        "model_architecture": "residual_diffusion_unet_v1",
        "grid_height": int(grid.height),
        "grid_width": int(grid.width),
        "points_per_condition": int(grid.n_points),
        "cond_gate_feature_indices": gate_feature_indices,
        "cond_channels": int(cond_channels),
        "base_channels": int(cfg.diffusion_base_channels),
        "timesteps": int(cfg.diffusion_timesteps),
        "sample_steps": int(cfg.diffusion_sample_steps),
        "shock_weight": float(cfg.diffusion_shock_weight),
        "diffusion_batch_size": int(cfg.diffusion_batch_size),
        "diffusion_epochs": int(cfg.diffusion_epochs),
        "final_train_loss": float(history_train[-1]),
        "final_test_loss": float(history_test[-1]),
    }
    save_json(_diffusion_config_path(cfg, baseline_mode), payload)

    plt.figure(figsize=(8, 4.8))
    plt.plot(history_train, label="train")
    plt.plot(history_test, label="test")
    plt.xlabel("epoch")
    plt.ylabel("noise MSE")
    plt.title(f"Residual diffusion training ({baseline_mode})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / f"diffusion_training_loss_{baseline_mode}.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[train-diffusion] Finished. Model stored in {model_path}")


def _load_diffusion_model(cfg: FullAircraftConfig, baseline_mode: str) -> tuple[nn.Module, dict[str, object]]:
    config_path = _diffusion_config_path(cfg, baseline_mode)
    model_path = _diffusion_model_path(cfg, baseline_mode)
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Diffusion artifacts not found for baseline={baseline_mode}. "
            f"Expected {config_path} and {model_path}. Run 'train-diffusion' first."
        )
    payload = json.loads(config_path.read_text())
    cond_channels = int(payload["cond_channels"])
    model = _build_diffusion_model(cfg, cond_channels=cond_channels)
    state = torch.load(model_path, map_location="cpu")
    _unwrap_model(model).load_state_dict(state)
    model.eval()
    return model, payload


def _sample_residual_ddim(
    cfg: FullAircraftConfig,
    model: nn.Module,
    cond: torch.Tensor,
    mask: torch.Tensor,
    timesteps: int,
    sample_steps: int,
) -> torch.Tensor:
    _betas, _alphas, alpha_bars = _beta_schedule(timesteps, cfg.device)
    sample_steps = max(2, min(sample_steps, timesteps))
    schedule = np.linspace(timesteps - 1, 0, sample_steps, dtype=np.int64)
    x = torch.randn((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=cfg.device) * mask

    with torch.no_grad():
        for step_idx, step in enumerate(schedule):
            t = torch.full((cond.shape[0],), int(step), device=cfg.device, dtype=torch.long)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            pred_noise = model(x, cond, t)
            x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t.clamp_min(1e-8))
            x0 = x0 * mask
            if step_idx + 1 < len(schedule):
                next_step = int(schedule[step_idx + 1])
                alpha_bar_next = alpha_bars[next_step].view(1, 1, 1, 1)
                x = torch.sqrt(alpha_bar_next) * x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
                x = x * mask
            else:
                x = x0
    return x


def infer_diffusion_residual(
    cfg: FullAircraftConfig,
    split: str = "test",
    baseline_mode: str = "symbolic",
    baseline_prediction_path: Path | None = None,
    output_path: Path | None = None,
    sample_steps: int | None = None,
) -> Path:
    cfg.ensure_dirs()
    torch.backends.cudnn.benchmark = cfg.device.type == "cuda"

    baseline_prediction_path = _ensure_baseline_prediction(cfg, split, baseline_mode, baseline_prediction_path)
    model, payload = _load_diffusion_model(cfg, baseline_mode)
    grid = CompactSurfaceGrid.from_reference(cfg)
    gate_feature_indices = [int(idx) for idx in payload["cond_gate_feature_indices"]]
    cp_mean, cp_scale = _load_cp_scaler(cfg)

    gate = np.load(cfg.features_dir / f"gate_features_{split}.npy", mmap_mode="r")
    baseline = np.load(baseline_prediction_path, mmap_mode="r")["cp_pred"]
    n_rows = int(gate.shape[0])
    n_conditions = int(n_rows // grid.n_points)
    cond_batch = max(1, cfg.diffusion_batch_size)
    mask = torch.from_numpy(grid.valid_mask[None, None, ...].astype(np.float32)).to(cfg.device)

    cp_pred = np.zeros((n_rows, 1), dtype=np.float32)
    residual_norm_flat = np.zeros((n_rows, 1), dtype=np.float32)

    for start_cond in range(0, n_conditions, cond_batch):
        end_cond = min(n_conditions, start_cond + cond_batch)
        cond_tensors: list[np.ndarray] = []
        base_grids: list[np.ndarray] = []
        for cond_idx in range(start_cond, end_cond):
            row_start = cond_idx * grid.n_points
            row_stop = row_start + grid.n_points
            gate_flat = np.asarray(gate[row_start:row_stop, gate_feature_indices], dtype=np.float32)
            cp_base_flat = np.asarray(baseline[row_start:row_stop], dtype=np.float32).reshape(-1, 1)
            cp_base_norm = ((cp_base_flat - cp_mean) / cp_scale).astype(np.float32)
            gate_grid = grid.scatter_numpy(gate_flat)
            base_grid = grid.scatter_numpy(cp_base_norm)
            cond_grid = np.concatenate([base_grid, gate_grid, grid.valid_mask[None, ...].astype(np.float32)], axis=0)
            cond_tensors.append(cond_grid)
            base_grids.append(base_grid)

        cond = torch.from_numpy(np.stack(cond_tensors, axis=0)).to(cfg.device, non_blocking=True)
        mask_batch = mask.expand(cond.shape[0], -1, -1, -1)
        residual_norm = _sample_residual_ddim(
            cfg,
            model,
            cond,
            mask_batch,
            timesteps=int(payload["timesteps"]),
            sample_steps=int(sample_steps or payload.get("sample_steps", cfg.diffusion_sample_steps)),
        )
        residual_norm_np = residual_norm.detach().cpu().numpy().astype(np.float32)

        for local_idx, cond_idx in enumerate(range(start_cond, end_cond)):
            row_start = cond_idx * grid.n_points
            row_stop = row_start + grid.n_points
            residual_flat = grid.gather_numpy(residual_norm_np[local_idx])[..., 0:1]
            cp_base_flat = np.asarray(baseline[row_start:row_stop], dtype=np.float32).reshape(-1, 1)
            cp_pred[row_start:row_stop] = cp_base_flat + residual_flat * cp_scale
            residual_norm_flat[row_start:row_stop] = residual_flat

        if start_cond == 0 or end_cond == n_conditions or start_cond % 16 == 0:
            print(f"[infer-diffusion] {split}: condition {end_cond}/{n_conditions}")

    if output_path is None:
        output_path = cfg.inference_dir / f"X_cut_{split}_diffusion_{baseline_mode}.npz"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        cp_pred=cp_pred.astype(np.float32),
        cp_base=np.asarray(baseline, dtype=np.float32).reshape(-1, 1),
        residual_norm=residual_norm_flat.astype(np.float32),
        mode=np.array("diffusion_residual"),
        baseline_mode=np.array(baseline_mode),
    )
    print(f"[infer-diffusion] Saved refined predictions to {output_path}")
    return output_path
