from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import FullAircraftConfig
from .features import ENCODER_FEATURE_NAMES, SYMBOLIC_GATE_ENCODER_FEATURE_NAMES, SYMBOLIC_GATE_ENCODER_INDICES, build_encoder_features
from .models import FullAircraftLatentGateNet, LegacyLatentGateNet
from .utils import sample_indices

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt

from .cluster_partition import expert_names

HYBRID_FEATURE_INDICES = SYMBOLIC_GATE_ENCODER_INDICES
HYBRID_FEATURE_NAMES = SYMBOLIC_GATE_ENCODER_FEATURE_NAMES
BAND_EXTRA_FEATURE_NAMES = [
    "Mach_ctr",
    "Mach_ctr_sq",
    "Mach_ctr_cu",
    "AoA_deg_sq",
    "y_sq",
    "z_sq",
    "radius_yz_sq",
    "Pi_sq",
    "Mach_ctr_AoA_deg",
    "Mach_ctr_Pi",
    "Mach_ctr_y",
    "Mach_ctr_z",
    "Mach_ctr_nz",
    "AoA_deg_y",
    "AoA_deg_z",
    "y_Pi",
    "z_Pi",
]
SYMBOLIC_BASIS_NAMES = HYBRID_FEATURE_NAMES + BAND_EXTRA_FEATURE_NAMES


def _normalize_positive_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.maximum(scores, 0.0).astype(np.float32)
    denom = scores.sum(axis=1, keepdims=True)
    denom = np.where(denom <= eps, 1.0, denom)
    gates = scores / denom
    zero_rows = np.flatnonzero(scores.sum(axis=1) <= eps)
    if zero_rows.size > 0:
        gates[zero_rows] = 1.0 / scores.shape[1]
    return gates.astype(np.float32)


def _band_limits(cfg: FullAircraftConfig) -> tuple[float, float]:
    return float(cfg.mach_sub_max - cfg.expert_overlap_margin), float(cfg.mach_trans_max + cfg.expert_overlap_margin)


def _solve_linear_symbolic_scores(
    basis_train: np.ndarray,
    teacher_gates_train: np.ndarray,
    cfg: FullAircraftConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], list[str]]:
    mean = basis_train.mean(axis=0)
    scale = basis_train.std(axis=0) + 1e-6
    standardized = np.clip((basis_train - mean) / scale, -cfg.sensor_feature_clip, cfg.sensor_feature_clip)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    gram = design.T @ design
    ridge = np.eye(gram.shape[0], dtype=np.float64)
    ridge[0, 0] = 0.0
    system = gram + float(cfg.sensor_ridge_alpha) * ridge

    coefficients_std: list[np.ndarray] = []
    coefficients_raw: list[np.ndarray] = []
    intercepts_raw: list[float] = []
    equations: list[str] = []

    for dim in range(cfg.n_experts):
        target = teacher_gates_train[:, dim].astype(np.float64)
        rhs = design.T @ target
        coef_std = np.linalg.solve(system, rhs)
        coef_raw = coef_std[1:] / scale
        intercept_raw = float(coef_std[0] - np.sum(coef_std[1:] * mean / scale))

        coefficients_std.append(coef_std.astype(np.float64))
        coefficients_raw.append(coef_raw.astype(np.float64))
        intercepts_raw.append(intercept_raw)
        equations.append(_render_linear_equation(intercept_raw, coef_raw, SYMBOLIC_BASIS_NAMES))

    return (
        mean.astype(np.float64),
        scale.astype(np.float64),
        np.stack(coefficients_std, axis=1).astype(np.float64),
        np.stack(coefficients_raw, axis=1).astype(np.float64),
        intercepts_raw,
        equations,
    )


def _build_symbolic_basis(x_raw: np.ndarray, cfg: FullAircraftConfig) -> np.ndarray:
    encoder = build_encoder_features(x_raw)
    base = encoder[:, HYBRID_FEATURE_INDICES].astype(np.float64)
    y = encoder[:, 1:2].astype(np.float64)
    z = encoder[:, 2:3].astype(np.float64)
    nz = encoder[:, 5:6].astype(np.float64)
    mach = encoder[:, 6:7].astype(np.float64)
    pi = encoder[:, 7:8].astype(np.float64)
    aoa_deg = encoder[:, 8:9].astype(np.float64)
    radius_yz = encoder[:, 12:13].astype(np.float64)

    band_center = 0.5 * (_band_limits(cfg)[0] + _band_limits(cfg)[1])
    mach_ctr = mach - band_center
    extra = np.concatenate(
        [
            mach_ctr,
            mach_ctr**2,
            mach_ctr**3,
            aoa_deg**2,
            y**2,
            z**2,
            radius_yz**2,
            pi**2,
            mach_ctr * aoa_deg,
            mach_ctr * pi,
            mach_ctr * y,
            mach_ctr * z,
            mach_ctr * nz,
            aoa_deg * y,
            aoa_deg * z,
            y * pi,
            z * pi,
        ],
        axis=1,
    )
    return np.concatenate([base, extra], axis=1).astype(np.float64)


def _render_linear_equation(intercept: float, coefficients: np.ndarray, basis_names: list[str], top_k: int = 8) -> str:
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
    return "clip(" + "".join(parts) + ", 0, inf)"


def _load_teacher_gate_net(cfg: FullAircraftConfig) -> LegacyLatentGateNet | FullAircraftLatentGateNet:
    gate_config_path = cfg.models_dir / "latent_gate_config.json"
    gate_architecture = "legacy_hidden_plus_z"
    latent_dim = int(cfg.latent_dim)
    if gate_config_path.exists():
        gate_cfg = json.loads(gate_config_path.read_text())
        gate_input_dim = int(len(gate_cfg.get("gate_feature_indices", [])))
        gate_architecture = str(gate_cfg.get("gate_architecture", gate_architecture))
        latent_dim = int(gate_cfg.get("latent_dim", latent_dim))
    else:
        gate_input_dim = np.load(cfg.features_dir / "gate_features_train.npy", mmap_mode="r").shape[1]
    if gate_architecture == "latent_only_v1":
        gate_net = FullAircraftLatentGateNet(gate_input_dim=gate_input_dim, latent_dim=latent_dim, n_experts=cfg.n_experts)
    else:
        gate_net = LegacyLatentGateNet(gate_input_dim=gate_input_dim, latent_dim=latent_dim, n_experts=cfg.n_experts)

    state = torch.load(cfg.models_dir / "latent_sensor_moe.pth", map_location="cpu")
    gate_state = {}
    for key, value in state.items():
        if key.startswith("gate_net."):
            gate_state[key.removeprefix("gate_net.")] = value
    if not gate_state:
        raise ValueError("Could not find gate_net weights inside latent_sensor_moe.pth")

    gate_net.load_state_dict(gate_state)
    gate_net.to(cfg.device)
    gate_net.eval()
    return gate_net


def _teacher_outputs(
    cfg: FullAircraftConfig,
    gate_net: LegacyLatentGateNet | FullAircraftLatentGateNet,
    split: str,
    max_samples: int,
    seed: int,
    batch_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_features = np.load(cfg.features_dir / f"gate_features_{split}.npy", mmap_mode="r")
    x_raw = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")
    gate_config_path = cfg.models_dir / "latent_gate_config.json"
    if gate_config_path.exists():
        gate_cfg = json.loads(gate_config_path.read_text())
        gate_feature_indices = [int(idx) for idx in gate_cfg.get("gate_feature_indices", [])]
    else:
        gate_feature_indices = []

    idx = sample_indices(gate_features.shape[0], min(max_samples, int(gate_features.shape[0])), seed=seed)
    teacher_gates = np.zeros((idx.shape[0], cfg.n_experts), dtype=np.float32)

    with torch.no_grad():
        cursor = 0
        for start in range(0, idx.shape[0], batch_size):
            end = min(idx.shape[0], start + batch_size)
            batch_idx = idx[start:end]
            gate_batch_np = np.asarray(gate_features[batch_idx], dtype=np.float32)
            if gate_feature_indices:
                gate_batch_np = gate_batch_np[:, gate_feature_indices]
            gate_batch = torch.from_numpy(gate_batch_np).to(cfg.device, non_blocking=True)
            _, _, gates_t = gate_net(gate_batch)
            batch = end - start
            teacher_gates[cursor : cursor + batch] = gates_t.detach().cpu().numpy().astype(np.float32)
            cursor += batch

    x_sensor = np.asarray(x_raw[idx, : cfg.input_dim_raw], dtype=np.float32)
    return x_sensor, teacher_gates, idx.astype(np.int64)


def _fit_hybrid_symbolic_sensor(
    x_train: np.ndarray,
    teacher_gates_train: np.ndarray,
    cfg: FullAircraftConfig,
) -> dict[str, object]:
    basis_train = _build_symbolic_basis(x_train, cfg)

    band_lower, band_upper = _band_limits(cfg)
    mach_train = x_train[:, 6]
    in_band = (mach_train >= band_lower) & (mach_train <= band_upper)
    if not np.any(in_band):
        raise ValueError("Hybrid sensor fit produced an empty transonic band.")

    mean, scale, coefficients_std, coefficients_raw, intercepts_raw, equations = _solve_linear_symbolic_scores(
        basis_train[in_band],
        teacher_gates_train[in_band],
        cfg,
    )

    return {
        "type": "hybrid_linear_band",
        "description": "Hard symbolic routing outside the transonic band and calibrated enriched symbolic scores inside it.",
        "band_lower": band_lower,
        "band_upper": band_upper,
        "feature_indices": HYBRID_FEATURE_INDICES,
        "feature_names": SYMBOLIC_BASIS_NAMES,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "ridge_alpha": float(cfg.sensor_ridge_alpha),
        "feature_clip": float(cfg.sensor_feature_clip),
        "coefficients_std": coefficients_std.tolist(),
        "coefficients_raw": coefficients_raw.tolist(),
        "intercepts_raw": intercepts_raw,
        "equations": equations,
    }


def _fit_global_symbolic_sensor(
    x_train: np.ndarray,
    teacher_gates_train: np.ndarray,
    cfg: FullAircraftConfig,
) -> dict[str, object]:
    basis_train = _build_symbolic_basis(x_train, cfg)
    mean, scale, coefficients_std, coefficients_raw, intercepts_raw, equations = _solve_linear_symbolic_scores(
        basis_train,
        teacher_gates_train,
        cfg,
    )
    return {
        "type": "global_linear_scores",
        "description": "Global symbolic linear scores fitted over all rows for cluster-based routing.",
        "feature_indices": HYBRID_FEATURE_INDICES,
        "feature_names": SYMBOLIC_BASIS_NAMES,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "ridge_alpha": float(cfg.sensor_ridge_alpha),
        "feature_clip": float(cfg.sensor_feature_clip),
        "coefficients_std": coefficients_std.tolist(),
        "coefficients_raw": coefficients_raw.tolist(),
        "intercepts_raw": intercepts_raw,
        "equations": equations,
    }


def apply_hybrid_symbolic_sensor(
    x_raw: np.ndarray,
    artifact: dict[str, object],
    cfg: FullAircraftConfig,
) -> tuple[np.ndarray, np.ndarray]:
    basis = _build_symbolic_basis(x_raw, cfg)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
    coef_std = np.asarray(artifact["coefficients_std"], dtype=np.float64)
    clip_value = float(artifact.get("feature_clip", cfg.sensor_feature_clip))

    standardized = np.clip((basis - mean) / scale, -clip_value, clip_value)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    artifact_type = str(artifact.get("type", "hybrid_linear_band"))

    if artifact_type == "global_linear_scores":
        scores = design @ coef_std
    else:
        mach = np.asarray(x_raw[:, 6], dtype=np.float64)
        band_lower = float(artifact["band_lower"])
        band_upper = float(artifact["band_upper"])

        scores = np.zeros((x_raw.shape[0], cfg.n_experts), dtype=np.float64)
        lower_mask = mach < band_lower
        upper_mask = mach > band_upper
        band_mask = ~(lower_mask | upper_mask)

        scores[lower_mask, 0] = 1.0
        scores[upper_mask, min(2, cfg.n_experts - 1)] = 1.0

        if np.any(band_mask):
            scores[band_mask] = design[band_mask] @ coef_std

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    gates = _normalize_positive_scores(scores.astype(np.float32))
    return scores.astype(np.float32), gates


def distill_sensor(cfg: FullAircraftConfig) -> None:
    cfg.ensure_dirs()

    latent_model_path = cfg.models_dir / "latent_sensor_moe.pth"
    if not latent_model_path.exists():
        raise FileNotFoundError(f"Latent model not found: {latent_model_path}. Run 'train-latent' first.")

    gate_net = _load_teacher_gate_net(cfg)
    train_x, train_teacher_gates, _ = _teacher_outputs(cfg, gate_net, "train", cfg.sensor_max_samples, seed=21)
    test_x, test_teacher_gates, _ = _teacher_outputs(cfg, gate_net, "test", cfg.sensor_max_samples, seed=22)

    artifact = (
        _fit_hybrid_symbolic_sensor(train_x, train_teacher_gates, cfg)
        if cfg.expert_partition_mode == "mach"
        else _fit_global_symbolic_sensor(train_x, train_teacher_gates, cfg)
    )
    test_scores, test_symbolic_gates = apply_hybrid_symbolic_sensor(test_x, artifact, cfg)

    teacher_argmax_test = np.argmax(test_teacher_gates, axis=1)
    symbolic_argmax_test = np.argmax(test_symbolic_gates, axis=1)

    regime_names = expert_names(cfg)
    summary: dict[str, dict[str, float | str]] = {}
    equations = artifact["equations"]
    for dim, regime_name in enumerate(regime_names):
        gate_mae = float(np.mean(np.abs(test_symbolic_gates[:, dim] - test_teacher_gates[:, dim])))
        gate_rmse = float(np.sqrt(np.mean((test_symbolic_gates[:, dim] - test_teacher_gates[:, dim]) ** 2)))
        argmax_recall = float(np.mean(symbolic_argmax_test[teacher_argmax_test == dim] == dim)) if np.any(teacher_argmax_test == dim) else 0.0
        summary[regime_name] = {
            "equation": str(equations[dim]),
            "gate_mae_test": gate_mae,
            "gate_rmse_test": gate_rmse,
            "argmax_recall_test": argmax_recall,
            "mean_predicted_gate_test": float(test_symbolic_gates[:, dim].mean()),
            "feature_hint": ", ".join(SYMBOLIC_BASIS_NAMES[:8]),
        }

        plot_idx = sample_indices(test_x.shape[0], min(cfg.plot_sample_size, test_x.shape[0]), seed=100 + dim)
        teacher_gate_plot = test_teacher_gates[plot_idx, dim]
        pred_gate_plot = test_symbolic_gates[plot_idx, dim]

        plt.figure(figsize=(5.6, 5.6))
        plt.scatter(teacher_gate_plot, pred_gate_plot, s=4, alpha=0.3)
        lo = float(min(teacher_gate_plot.min(), pred_gate_plot.min()))
        hi = float(max(teacher_gate_plot.max(), pred_gate_plot.max()))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        plt.xlabel(f"teacher gate {regime_name}")
        plt.ylabel(f"symbolic gate {regime_name}")
        plt.title(f"Full-aircraft sensor distillation {regime_name}")
        plt.grid(True, alpha=0.25)
        plt.savefig(cfg.plots_dir / f"sensor_gate_{regime_name}_test.png", dpi=220, bbox_inches="tight")
        plt.close()

    overall = {
        "type": str(artifact["type"]),
        "partition_mode": cfg.expert_partition_mode,
        "train_rows": int(train_x.shape[0]),
        "test_rows": int(test_x.shape[0]),
        "ridge_alpha": float(artifact["ridge_alpha"]),
        "argmax_agreement_test": float(np.mean(symbolic_argmax_test == teacher_argmax_test)),
        "gate_mae_test": float(np.mean(np.abs(test_symbolic_gates - test_teacher_gates))),
        "gate_rmse_test": float(np.sqrt(np.mean((test_symbolic_gates - test_teacher_gates) ** 2))),
        "teacher_argmax_counts_test": np.bincount(teacher_argmax_test, minlength=cfg.n_experts).astype(int).tolist(),
        "symbolic_argmax_counts_test": np.bincount(symbolic_argmax_test, minlength=cfg.n_experts).astype(int).tolist(),
    }
    if "band_lower" in artifact and "band_upper" in artifact:
        overall["band_lower"] = float(artifact["band_lower"])
        overall["band_upper"] = float(artifact["band_upper"])

    with (cfg.sensor_dir / "sensor_hybrid.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    with (cfg.sensor_dir / "sensor_hybrid.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"type={artifact['type']}\n")
        if "band_lower" in artifact and "band_upper" in artifact:
            handle.write(f"band_lower={artifact['band_lower']:.6f}\n")
            handle.write(f"band_upper={artifact['band_upper']:.6f}\n")
        handle.write(
            f"ridge_alpha={artifact['ridge_alpha']:.6f}\n"
            f"feature_clip={artifact['feature_clip']:.6f}\n\n"
        )
        for regime_name, equation in zip(regime_names, equations):
            handle.write(f"{regime_name}: {equation}\n")

    with (cfg.sensor_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"overall": overall, "experts": summary}, handle, indent=2)

    np.savez_compressed(
        cfg.sensor_dir / "sensor_test_predictions.npz",
        teacher_gates=test_teacher_gates,
        symbolic_scores=test_scores,
        symbolic_gates=test_symbolic_gates,
        teacher_expert_id=teacher_argmax_test.astype(np.int64),
        symbolic_expert_id=symbolic_argmax_test.astype(np.int64),
    )
    print(f"[distill-sensor] Finished. Sensor artifacts stored in {cfg.sensor_dir}")
