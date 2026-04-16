from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PipelineConfig
from .datasets import LatentMoEDataset
from .latent_viz import plot_latent_summary
from .models import LatentSensorMoE

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _expert_model_paths(cfg: PipelineConfig) -> list[Path]:
    return [
        cfg.models_dir / "expert_subsonic.pth",
        cfg.models_dir / "expert_transonic.pth",
        cfg.models_dir / "expert_supersonic.pth",
    ]


def _latent_dataset(cfg: PipelineConfig, split: str, indices: np.ndarray | None = None) -> LatentMoEDataset:
    return LatentMoEDataset(
        expert_path=str(cfg.features_dir / f"expert_features_{split}.npy"),
        gate_path=str(cfg.features_dir / f"gate_features_{split}.npy"),
        cp_path=str(cfg.features_dir / f"cp_{split}.npy"),
        expert_id_path=str(cfg.features_dir / f"expert_id_{split}.npy"),
        indices=indices,
    )


def _latent_sample_indices(cfg: PipelineConfig, split: str, max_points: int, seed: int) -> np.ndarray:
    expert_id = np.load(cfg.features_dir / f"expert_id_{split}.npy", mmap_mode="r")
    n_rows = int(expert_id.shape[0])
    if max_points <= 0 or n_rows <= max_points:
        return np.arange(n_rows, dtype=np.int64)

    x_raw = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    mach = np.asarray(x_raw[:, 6], dtype=np.float32)
    aoa = np.asarray(x_raw[:, 7], dtype=np.float32)
    aoa_abs = np.abs(aoa)

    weights = np.ones(n_rows, dtype=np.float64)
    transition_band = (mach >= cfg.mach_sub_max - 0.05) & (mach <= cfg.mach_trans_max + 0.05)
    sub_extreme = (mach < cfg.mach_sub_max) & (aoa_abs >= 10.0)
    trans_extreme = (mach >= cfg.mach_sub_max) & (mach <= cfg.mach_trans_max) & (aoa_abs >= 8.0)
    high_aoa = aoa_abs >= 12.5

    weights[transition_band] += 1.0
    weights[sub_extreme] += 2.5
    weights[trans_extreme] += 1.5
    weights[high_aoa] += 1.0

    rng = np.random.default_rng(seed)
    probs = weights / weights.sum()
    sampled = rng.choice(n_rows, size=max_points, replace=False, p=probs)
    return np.sort(sampled.astype(np.int64))


def _soft_routing_targets(expert_stack: torch.Tensor, cp: torch.Tensor, temperature: float) -> torch.Tensor:
    errors = torch.abs(expert_stack - cp.unsqueeze(2)).squeeze(1)
    return F.softmax(-errors / max(temperature, 1e-6), dim=1)


def _oracle_expert_targets(expert_stack: torch.Tensor, cp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    errors = torch.abs(expert_stack - cp.unsqueeze(2)).squeeze(1)
    oracle_expert_id = torch.argmin(errors, dim=1)
    best_two = torch.topk(errors, k=2, largest=False, dim=1).values
    oracle_margin = best_two[:, 1] - best_two[:, 0]
    return oracle_expert_id, oracle_margin


def _soft_gate_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=1), soft_targets, reduction="batchmean")


def _masked_hard_gate_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[mask], targets[mask])


def _evaluate(
    model: LatentSensorMoE,
    loader: DataLoader,
    device: torch.device,
    soft_gate_weight: float,
    hard_gate_weight: float,
    routing_temperature: float,
    oracle_margin_threshold: float,
) -> dict[str, float]:
    reg_criterion = nn.SmoothL1Loss(beta=1.0)
    model.eval()
    total = 0.0
    total_reg = 0.0
    total_soft_gate = 0.0
    total_hard_gate = 0.0
    total_hard_acc = 0.0
    total_soft_agreement = 0.0
    n = 0
    with torch.no_grad():
        for expert_feat, gate_feat, cp, expert_id in loader:
            expert_feat = expert_feat.to(device, non_blocking=True)
            gate_feat = gate_feat.to(device, non_blocking=True)
            cp = cp.to(device, non_blocking=True)
            expert_id = expert_id.to(device, non_blocking=True)
            pred, _, logits, gates, expert_stack = model(expert_feat, gate_feat, return_expert_stack=True)
            soft_targets = _soft_routing_targets(expert_stack, cp, routing_temperature)
            reg_loss = reg_criterion(pred, cp)
            soft_gate_loss = _soft_gate_loss(logits, soft_targets)
            oracle_expert_id, oracle_margin = _oracle_expert_targets(expert_stack, cp)
            ambiguous_mask = oracle_margin <= oracle_margin_threshold
            hard_gate_loss = _masked_hard_gate_loss(logits, oracle_expert_id, ambiguous_mask)
            loss = reg_loss + soft_gate_weight * soft_gate_loss + hard_gate_weight * hard_gate_loss
            total += loss.item() * expert_feat.shape[0]
            total_reg += reg_loss.item() * expert_feat.shape[0]
            total_soft_gate += soft_gate_loss.item() * expert_feat.shape[0]
            total_hard_gate += hard_gate_loss.item() * expert_feat.shape[0]
            total_hard_acc += (torch.argmax(gates, dim=1) == oracle_expert_id).float().sum().item()
            total_soft_agreement += (torch.argmax(gates, dim=1) == torch.argmax(soft_targets, dim=1)).float().sum().item()
            n += expert_feat.shape[0]
    return {
        "loss": total / max(1, n),
        "reg": total_reg / max(1, n),
        "soft_gate": total_soft_gate / max(1, n),
        "hard_gate": total_hard_gate / max(1, n),
        "hard_acc": total_hard_acc / max(1, n),
        "soft_agreement": total_soft_agreement / max(1, n),
    }


def _export_latent_artifacts(cfg: PipelineConfig, model: LatentSensorMoE, split: str) -> None:
    dataset = _latent_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=cfg.latent_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cut_x = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    cp_scaler = None
    z_all = np.zeros((len(dataset), cfg.latent_dim), dtype=np.float32)
    gates_all = np.zeros((len(dataset), cfg.n_experts), dtype=np.float32)
    cp_pred_all = np.zeros((len(dataset), 1), dtype=np.float32)
    expert_id_all = np.load(cfg.features_dir / f"expert_id_{split}.npy", mmap_mode="r")

    cursor = 0
    model.eval()
    with torch.no_grad():
        for expert_feat, gate_feat, _, _ in loader:
            batch = expert_feat.shape[0]
            expert_feat = expert_feat.to(cfg.device, non_blocking=True)
            gate_feat = gate_feat.to(cfg.device, non_blocking=True)
            cp_pred, z, _, gates = model(expert_feat, gate_feat)
            z_all[cursor : cursor + batch] = z.detach().cpu().numpy().astype(np.float32)
            gates_all[cursor : cursor + batch] = gates.detach().cpu().numpy().astype(np.float32)
            cp_pred_all[cursor : cursor + batch] = cp_pred.detach().cpu().numpy().astype(np.float32)
            cursor += batch

    np.savez_compressed(
        cfg.latent_dir / f"latent_{split}.npz",
        z=z_all,
        gates=gates_all,
        expert_id=np.asarray(expert_id_all, dtype=np.int64),
        mach=np.asarray(cut_x[:, 6], dtype=np.float32),
        aoa=np.asarray(cut_x[:, 7], dtype=np.float32),
        cp_pred=cp_pred_all,
    )
    plot_latent_summary(cfg, split)


def train_latent_pipeline(cfg: PipelineConfig) -> None:
    cfg.ensure_dirs()
    expert_paths = _expert_model_paths(cfg)
    for path in expert_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expert model not found: {path}. Run 'train-experts' first.")

    train_indices = _latent_sample_indices(cfg, "train", cfg.latent_train_max_samples, seed=31)
    test_indices = _latent_sample_indices(cfg, "test", cfg.latent_test_max_samples, seed=32)
    train_dataset = _latent_dataset(cfg, "train", indices=train_indices)
    test_dataset = _latent_dataset(cfg, "test", indices=test_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.latent_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.latent_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    expert_input_dim = np.load(cfg.features_dir / "expert_features_train.npy", mmap_mode="r").shape[1]
    gate_input_dim = np.load(cfg.features_dir / "gate_features_train.npy", mmap_mode="r").shape[1]

    model = LatentSensorMoE(
        gate_input_dim=gate_input_dim,
        expert_input_dim=expert_input_dim,
        latent_dim=cfg.latent_dim,
        expert_paths=expert_paths,
    ).to(cfg.device)

    optimizer = optim.AdamW(model.gate_net.parameters(), lr=cfg.latent_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.latent_epochs)
    reg_criterion = nn.SmoothL1Loss(beta=1.0)
    history_train: list[float] = []
    history_test: list[float] = []

    print(
        f"[train-latent] sampled rows | train={len(train_dataset):,}/{np.load(cfg.features_dir / 'expert_id_train.npy', mmap_mode='r').shape[0]:,} "
        f"| test={len(test_dataset):,}/{np.load(cfg.features_dir / 'expert_id_test.npy', mmap_mode='r').shape[0]:,}"
    )

    for epoch in range(1, cfg.latent_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_rows = 0
        pbar = tqdm(train_loader, desc=f"[latent] epoch {epoch}/{cfg.latent_epochs}")
        for expert_feat, gate_feat, cp, expert_id in pbar:
            expert_feat = expert_feat.to(cfg.device, non_blocking=True)
            gate_feat = gate_feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            expert_id = expert_id.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred, _, logits, gates, expert_stack = model(expert_feat, gate_feat, return_expert_stack=True)
            soft_targets = _soft_routing_targets(expert_stack, cp, cfg.routing_soft_temperature)
            reg_loss = reg_criterion(pred, cp)
            soft_gate_loss = _soft_gate_loss(logits, soft_targets)
            oracle_expert_id, oracle_margin = _oracle_expert_targets(expert_stack, cp)
            ambiguous_mask = oracle_margin <= cfg.routing_oracle_margin
            hard_gate_loss = _masked_hard_gate_loss(logits, oracle_expert_id, ambiguous_mask)
            loss = reg_loss + cfg.latent_gate_weight * soft_gate_loss + cfg.latent_hard_gate_weight * hard_gate_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * expert_feat.shape[0]
            n_rows += expert_feat.shape[0]
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                reg=f"{reg_loss.item():.5f}",
                soft=f"{soft_gate_loss.item():.5f}",
                hard=f"{hard_gate_loss.item():.5f}",
                acc=f"{(torch.argmax(gates, dim=1) == oracle_expert_id).float().mean().item():.3f}",
            )

        scheduler.step()
        train_loss = epoch_loss / max(1, n_rows)
        test_metrics = _evaluate(
            model,
            test_loader,
            cfg.device,
            cfg.latent_gate_weight,
            cfg.latent_hard_gate_weight,
            cfg.routing_soft_temperature,
            cfg.routing_oracle_margin,
        )
        history_train.append(train_loss)
        history_test.append(test_metrics["loss"])
        print(
            f"[latent] epoch {epoch:03d} | train={train_loss:.6f} | "
            f"test={test_metrics['loss']:.6f} | reg={test_metrics['reg']:.6f} | "
            f"soft_gate={test_metrics['soft_gate']:.6f} | hard_gate={test_metrics['hard_gate']:.6f} | "
            f"hard_acc={test_metrics['hard_acc']:.4f} | soft_agree={test_metrics['soft_agreement']:.4f}"
        )

    torch.save(model.state_dict(), cfg.models_dir / "latent_sensor_moe.pth")

    with (cfg.metrics_dir / "latent_training.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "final_train_loss": float(history_train[-1]),
                "final_test_loss": float(history_test[-1]),
                "latent_dim": cfg.latent_dim,
                "soft_gate_weight": float(cfg.latent_gate_weight),
                "hard_gate_weight": float(cfg.latent_hard_gate_weight),
                "routing_soft_temperature": float(cfg.routing_soft_temperature),
                "routing_oracle_margin": float(cfg.routing_oracle_margin),
                "train_rows": int(len(train_dataset)),
                "test_rows": int(len(test_dataset)),
            },
            handle,
            indent=2,
        )

    plt.figure(figsize=(8, 4.8))
    plt.plot(history_train, label="train")
    plt.plot(history_test, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Latent encoder + gating")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / "latent_training_loss.png", dpi=220, bbox_inches="tight")
    plt.close()

    _export_latent_artifacts(cfg, model, "train")
    _export_latent_artifacts(cfg, model, "test")
    print(f"[train-latent] Finished. Model stored in {cfg.models_dir / 'latent_sensor_moe.pth'}")
