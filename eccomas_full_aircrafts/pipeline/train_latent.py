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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eccomas_sensor_pipeline.eccomas_sensor.latent_viz import plot_latent_summary

from .config import FullAircraftConfig
from .features import SYMBOLIC_GATE_ENCODER_FEATURE_NAMES, SYMBOLIC_GATE_ENCODER_INDICES
from .utils import raw_paths

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt

from .models import FullAircraftLatentMixer


def _expert_prediction_path(cfg: FullAircraftConfig, split: str) -> Path:
    return cfg.features_dir / f"expert_pred_{split}.npy"


def _latent_gate_config_path(cfg: FullAircraftConfig) -> Path:
    return cfg.models_dir / "latent_gate_config.json"


def _save_latent_gate_config(cfg: FullAircraftConfig) -> None:
    payload = {
        "gate_architecture": cfg.latent_gate_architecture,
        "latent_dim": int(cfg.latent_dim),
        "gate_feature_mode": "symbolic_subset",
        "gate_feature_indices": [int(idx) for idx in SYMBOLIC_GATE_ENCODER_INDICES],
        "gate_feature_names": list(SYMBOLIC_GATE_ENCODER_FEATURE_NAMES),
    }
    with _latent_gate_config_path(cfg).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _n_conditions(cfg: FullAircraftConfig, split: str) -> int:
    x_path, _ = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    return int(x_raw.shape[0] // cfg.raw_points_per_condition)


def _reduced_points_per_condition(cfg: FullAircraftConfig, split: str) -> int:
    x_red = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    return int(x_red.shape[0] // max(_n_conditions(cfg, split), 1))


def _condition_sampling_weights(mach: np.ndarray, aoa: np.ndarray, cfg: FullAircraftConfig) -> np.ndarray:
    aoa_abs = np.abs(aoa)
    weights = np.ones(mach.shape[0], dtype=np.float64)
    transition_band = (mach >= cfg.mach_sub_max - 0.05) & (mach <= cfg.mach_trans_max + 0.05)
    sub_extreme = (mach < cfg.mach_sub_max) & (aoa_abs >= 10.0)
    trans_extreme = (mach >= cfg.mach_sub_max) & (mach <= cfg.mach_trans_max) & (aoa_abs >= 8.0)
    high_aoa = aoa_abs >= 12.5

    weights[transition_band] += 1.0
    weights[sub_extreme] += 2.5
    weights[trans_extreme] += 1.5
    weights[high_aoa] += 1.0
    return weights


def _allocate_condition_counts(
    weights: np.ndarray,
    total_points: int,
    capacity_per_condition: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_conditions = int(weights.shape[0])
    if total_points >= n_conditions * capacity_per_condition:
        return np.full(n_conditions, capacity_per_condition, dtype=np.int64)

    base = total_points // n_conditions
    counts = np.full(n_conditions, min(base, capacity_per_condition), dtype=np.int64)
    remaining = int(total_points - counts.sum())
    if remaining <= 0:
        return counts

    probs = weights / weights.sum()
    while remaining > 0:
        available = counts < capacity_per_condition
        if not np.any(available):
            break
        probs_available = probs[available]
        probs_available = probs_available / probs_available.sum()
        add = rng.multinomial(remaining, probs_available)
        counts_available = counts[available]
        capacity_left = capacity_per_condition - counts_available
        accepted = np.minimum(add, capacity_left)
        counts[available] = counts_available + accepted
        used = int(accepted.sum())
        if used == 0:
            break
        remaining -= used
    return counts


def _latent_sample_indices(cfg: FullAircraftConfig, split: str, max_points: int, seed: int) -> np.ndarray:
    x_red = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    n_rows = int(x_red.shape[0])
    if max_points <= 0 or n_rows <= max_points:
        return np.arange(n_rows, dtype=np.int64)

    n_conditions = _n_conditions(cfg, split)
    points_per_condition = int(n_rows // max(n_conditions, 1))
    mach = np.asarray(x_red[::points_per_condition, 6], dtype=np.float32)[:n_conditions]
    aoa = np.asarray(x_red[::points_per_condition, 7], dtype=np.float32)[:n_conditions]
    weights = _condition_sampling_weights(mach, aoa, cfg)

    rng = np.random.default_rng(seed)
    counts = _allocate_condition_counts(weights, max_points, points_per_condition, rng)
    sampled: list[np.ndarray] = []
    for cond_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        start = cond_idx * points_per_condition
        local = rng.choice(points_per_condition, size=count, replace=False)
        sampled.append(np.sort(start + local.astype(np.int64)))

    if not sampled:
        return np.arange(min(max_points, n_rows), dtype=np.int64)
    return np.concatenate(sampled, axis=0).astype(np.int64)


def _load_sampled_tensors(
    cfg: FullAircraftConfig,
    split: str,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_pred_path = _expert_prediction_path(cfg, split)
    if not expert_pred_path.exists():
        raise FileNotFoundError(f"Expert predictions not found: {expert_pred_path}. Run 'train-experts' first.")
    expert = np.asarray(np.load(expert_pred_path, mmap_mode="r")[indices], dtype=np.float32)
    gate = np.asarray(np.load(cfg.features_dir / f"gate_features_{split}.npy", mmap_mode="r")[indices], dtype=np.float32)
    cp = np.asarray(np.load(cfg.features_dir / f"cp_{split}.npy", mmap_mode="r")[indices], dtype=np.float32)
    expert_id = np.asarray(np.load(cfg.features_dir / f"expert_id_{split}.npy", mmap_mode="r")[indices], dtype=np.int64)

    gate = gate[:, SYMBOLIC_GATE_ENCODER_INDICES]
    return (
        torch.from_numpy(expert[:, None, :]),
        torch.from_numpy(gate),
        torch.from_numpy(cp),
        torch.from_numpy(expert_id),
    )


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


def _gate_entropy(gates: torch.Tensor) -> torch.Tensor:
    return -(gates * torch.log(gates.clamp_min(1e-8))).sum(dim=1).mean()


def _evaluate(
    model: FullAircraftLatentMixer,
    loader: DataLoader,
    device: torch.device,
    soft_gate_weight: float,
    hard_gate_weight: float,
    entropy_weight: float,
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
    total_entropy = 0.0
    n = 0
    with torch.no_grad():
        for expert_stack, gate_feat, cp, expert_id in loader:
            expert_stack = expert_stack.to(device, non_blocking=True)
            gate_feat = gate_feat.to(device, non_blocking=True)
            cp = cp.to(device, non_blocking=True)
            expert_id = expert_id.to(device, non_blocking=True)
            pred, _, logits, gates, expert_stack = model(expert_stack, gate_feat, return_expert_stack=True)
            soft_targets = _soft_routing_targets(expert_stack, cp, routing_temperature)
            reg_loss = reg_criterion(pred, cp)
            soft_gate_loss = _soft_gate_loss(logits, soft_targets)
            oracle_expert_id, oracle_margin = _oracle_expert_targets(expert_stack, cp)
            confident_mask = oracle_margin >= oracle_margin_threshold
            hard_gate_loss = _masked_hard_gate_loss(logits, oracle_expert_id, confident_mask)
            entropy = _gate_entropy(gates)
            loss = reg_loss + soft_gate_weight * soft_gate_loss + hard_gate_weight * hard_gate_loss - entropy_weight * entropy
            total += loss.item() * expert_stack.shape[0]
            total_reg += reg_loss.item() * expert_stack.shape[0]
            total_soft_gate += soft_gate_loss.item() * expert_stack.shape[0]
            total_hard_gate += hard_gate_loss.item() * expert_stack.shape[0]
            total_entropy += entropy.item() * expert_stack.shape[0]
            total_hard_acc += (torch.argmax(gates, dim=1) == oracle_expert_id).float().sum().item()
            total_soft_agreement += (torch.argmax(gates, dim=1) == torch.argmax(soft_targets, dim=1)).float().sum().item()
            n += expert_stack.shape[0]
    return {
        "loss": total / max(1, n),
        "reg": total_reg / max(1, n),
        "soft_gate": total_soft_gate / max(1, n),
        "hard_gate": total_hard_gate / max(1, n),
        "entropy": total_entropy / max(1, n),
        "hard_acc": total_hard_acc / max(1, n),
        "soft_agreement": total_soft_agreement / max(1, n),
    }


def _export_latent_artifacts(
    cfg: FullAircraftConfig,
    model: FullAircraftLatentMixer,
    split: str,
    gate_feature_indices: list[int],
) -> None:
    expert_stack = np.load(_expert_prediction_path(cfg, split), mmap_mode="r")
    gate_features = np.load(cfg.features_dir / f"gate_features_{split}.npy", mmap_mode="r")
    cut_x = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    expert_id_all = np.load(cfg.features_dir / f"expert_id_{split}.npy", mmap_mode="r")

    z_all = np.zeros((expert_stack.shape[0], cfg.latent_dim), dtype=np.float32)
    gates_all = np.zeros((expert_stack.shape[0], cfg.n_experts), dtype=np.float32)
    cp_pred_all = np.zeros((expert_stack.shape[0], 1), dtype=np.float32)

    cursor = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, expert_stack.shape[0], cfg.latent_batch_size):
            end = min(expert_stack.shape[0], start + cfg.latent_batch_size)
            expert_batch = torch.from_numpy(np.asarray(expert_stack[start:end], dtype=np.float32)[:, None, :]).to(cfg.device, non_blocking=True)
            gate_batch = np.asarray(gate_features[start:end], dtype=np.float32)[:, gate_feature_indices]
            gate_batch = torch.from_numpy(gate_batch).to(cfg.device, non_blocking=True)
            cp_pred, z, _, gates = model(expert_batch, gate_batch)
            batch = end - start
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


def train_latent_pipeline(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()
    for split in ("train", "test"):
        pred_path = _expert_prediction_path(cfg, split)
        if not pred_path.exists():
            raise FileNotFoundError(f"Expert predictions not found: {pred_path}. Run 'train-experts' first.")

    train_indices = _latent_sample_indices(cfg, "train", cfg.latent_train_max_samples, seed=31)
    test_indices = _latent_sample_indices(cfg, "test", cfg.latent_test_max_samples, seed=32)

    train_tensors = _load_sampled_tensors(cfg, "train", train_indices)
    test_tensors = _load_sampled_tensors(cfg, "test", test_indices)

    train_dataset = TensorDataset(*train_tensors)
    test_dataset = TensorDataset(*test_tensors)
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

    gate_input_dim = int(train_tensors[1].shape[1])

    model = FullAircraftLatentMixer(
        gate_input_dim=gate_input_dim,
        latent_dim=cfg.latent_dim,
        n_experts=cfg.n_experts,
    ).to(cfg.device)

    optimizer = optim.AdamW(model.gate_net.parameters(), lr=cfg.latent_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.latent_epochs)
    reg_criterion = nn.SmoothL1Loss(beta=1.0)
    history_train: list[float] = []
    history_test: list[float] = []

    print(
        f"[train-latent] sampled rows | train={len(train_dataset):,} | test={len(test_dataset):,} | "
        f"gate_input={gate_input_dim} symbolic-aligned features"
    )

    for epoch in range(1, cfg.latent_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_rows = 0
        pbar = tqdm(train_loader, desc=f"[latent-full] epoch {epoch}/{cfg.latent_epochs}")
        for expert_stack, gate_feat, cp, expert_id in pbar:
            expert_stack = expert_stack.to(cfg.device, non_blocking=True)
            gate_feat = gate_feat.to(cfg.device, non_blocking=True)
            cp = cp.to(cfg.device, non_blocking=True)
            expert_id = expert_id.to(cfg.device, non_blocking=True)

            if cfg.latent_gate_noise_std > 0.0:
                gate_feat = gate_feat + cfg.latent_gate_noise_std * torch.randn_like(gate_feat)

            optimizer.zero_grad(set_to_none=True)
            pred, _, logits, gates, expert_stack = model(expert_stack, gate_feat, return_expert_stack=True)
            soft_targets = _soft_routing_targets(expert_stack, cp, cfg.routing_soft_temperature)
            reg_loss = reg_criterion(pred, cp)
            soft_gate_loss = _soft_gate_loss(logits, soft_targets)
            oracle_expert_id, oracle_margin = _oracle_expert_targets(expert_stack, cp)
            confident_mask = oracle_margin >= cfg.routing_oracle_margin
            hard_gate_loss = _masked_hard_gate_loss(logits, oracle_expert_id, confident_mask)
            entropy = _gate_entropy(gates)
            loss = (
                reg_loss
                + cfg.latent_gate_weight * soft_gate_loss
                + cfg.latent_hard_gate_weight * hard_gate_loss
                - cfg.latent_entropy_weight * entropy
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * expert_stack.shape[0]
            n_rows += expert_stack.shape[0]
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                reg=f"{reg_loss.item():.5f}",
                soft=f"{soft_gate_loss.item():.5f}",
                hard=f"{hard_gate_loss.item():.5f}",
                ent=f"{entropy.item():.5f}",
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
            cfg.latent_entropy_weight,
            cfg.routing_soft_temperature,
            cfg.routing_oracle_margin,
        )
        history_train.append(train_loss)
        history_test.append(test_metrics["loss"])
        print(
            f"[latent-full] epoch {epoch:03d} | train={train_loss:.6f} | "
            f"test={test_metrics['loss']:.6f} | reg={test_metrics['reg']:.6f} | "
            f"soft_gate={test_metrics['soft_gate']:.6f} | hard_gate={test_metrics['hard_gate']:.6f} | "
            f"entropy={test_metrics['entropy']:.6f} | "
            f"hard_acc={test_metrics['hard_acc']:.4f} | soft_agree={test_metrics['soft_agreement']:.4f}"
        )

    torch.save(model.state_dict(), cfg.models_dir / "latent_sensor_moe.pth")
    _save_latent_gate_config(cfg)

    with (cfg.metrics_dir / "latent_training.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "final_train_loss": float(history_train[-1]),
                "final_test_loss": float(history_test[-1]),
                "latent_dim": cfg.latent_dim,
                "gate_architecture": cfg.latent_gate_architecture,
                "soft_gate_weight": float(cfg.latent_gate_weight),
                "hard_gate_weight": float(cfg.latent_hard_gate_weight),
                "entropy_weight": float(cfg.latent_entropy_weight),
                "gate_noise_std": float(cfg.latent_gate_noise_std),
                "routing_soft_temperature": float(cfg.routing_soft_temperature),
                "routing_oracle_margin": float(cfg.routing_oracle_margin),
                "train_rows": int(len(train_dataset)),
                "test_rows": int(len(test_dataset)),
                "gate_feature_mode": "symbolic_subset",
                "gate_feature_indices": [int(idx) for idx in SYMBOLIC_GATE_ENCODER_INDICES],
                "gate_feature_names": list(SYMBOLIC_GATE_ENCODER_FEATURE_NAMES),
            },
            handle,
            indent=2,
        )

    plt.figure(figsize=(8, 4.8))
    plt.plot(history_train, label="train")
    plt.plot(history_test, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Latent encoder + gating (symbolic-aligned)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(cfg.plots_dir / "latent_training_loss.png", dpi=220, bbox_inches="tight")
    plt.close()

    _export_latent_artifacts(cfg, model, "train", SYMBOLIC_GATE_ENCODER_INDICES)
    _export_latent_artifacts(cfg, model, "test", SYMBOLIC_GATE_ENCODER_INDICES)
