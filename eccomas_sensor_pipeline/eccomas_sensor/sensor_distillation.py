from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import PipelineConfig
from .features import ENCODER_FEATURE_NAMES, build_encoder_features
from .models import LatentGateNet
from .utils import sample_indices

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


REGIME_NAMES = ["subsonic", "transonic", "supersonic"]
HYBRID_FEATURE_INDICES = [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 20]
HYBRID_FEATURE_NAMES = [ENCODER_FEATURE_NAMES[idx] for idx in HYBRID_FEATURE_INDICES]
BAND_EXTRA_FEATURE_NAMES = [
    "Mach_ctr",
    "Mach_ctr_sq",
    "Mach_ctr_cu",
    "AoA_deg_sq",
    "z_sq",
    "span_radius_sq",
    "Pi_sq",
    "Mach_ctr_AoA_deg",
    "Mach_ctr_Pi",
    "Mach_ctr_z",
    "Mach_ctr_nz",
    "Mach_ctr_span_radius",
    "AoA_deg_z",
    "AoA_deg_Pi",
    "z_Pi",
    "z_span_radius",
    "nz_Pi",
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


def _band_limits(cfg: PipelineConfig) -> tuple[float, float]:
    return float(cfg.mach_sub_max - cfg.expert_overlap_margin), float(cfg.mach_trans_max + cfg.expert_overlap_margin)


def _build_symbolic_basis(x_raw: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    encoder = build_encoder_features(x_raw, cfg.cut_threshold_y)
    base = encoder[:, HYBRID_FEATURE_INDICES].astype(np.float64)
    x = encoder[:, 0:1].astype(np.float64)
    z = encoder[:, 2:3].astype(np.float64)
    nz = encoder[:, 5:6].astype(np.float64)
    mach = encoder[:, 6:7].astype(np.float64)
    pi = encoder[:, 7:8].astype(np.float64)
    aoa_deg = encoder[:, 8:9].astype(np.float64)
    span_radius = encoder[:, 12:13].astype(np.float64)

    band_center = 0.5 * (_band_limits(cfg)[0] + _band_limits(cfg)[1])
    mach_ctr = mach - band_center
    extra = np.concatenate(
        [
            mach_ctr,
            mach_ctr**2,
            mach_ctr**3,
            aoa_deg**2,
            z**2,
            span_radius**2,
            pi**2,
            mach_ctr * aoa_deg,
            mach_ctr * pi,
            mach_ctr * z,
            mach_ctr * nz,
            mach_ctr * span_radius,
            aoa_deg * z,
            aoa_deg * pi,
            z * pi,
            z * span_radius,
            nz * pi,
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


def _load_teacher_gate_net(cfg: PipelineConfig) -> LatentGateNet:
    gate_input_dim = np.load(cfg.features_dir / "gate_features_train.npy", mmap_mode="r").shape[1]
    gate_net = LatentGateNet(gate_input_dim=gate_input_dim, latent_dim=cfg.latent_dim, n_experts=cfg.n_experts)

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
    cfg: PipelineConfig,
    gate_net: LatentGateNet,
    split: str,
    max_samples: int,
    seed: int,
    batch_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_features = np.load(cfg.features_dir / f"gate_features_{split}.npy", mmap_mode="r")
    x_raw = np.load(cfg.cut_data_dir / f"X_cut_{split}.npy", mmap_mode="r")

    idx = sample_indices(gate_features.shape[0], min(max_samples, int(gate_features.shape[0])), seed=seed)
    teacher_gates = np.zeros((idx.shape[0], cfg.n_experts), dtype=np.float32)

    with torch.no_grad():
        cursor = 0
        for start in range(0, idx.shape[0], batch_size):
            end = min(idx.shape[0], start + batch_size)
            batch_idx = idx[start:end]
            gate_batch = torch.from_numpy(np.asarray(gate_features[batch_idx], dtype=np.float32)).to(cfg.device, non_blocking=True)
            _, _, gates_t = gate_net(gate_batch)
            batch = end - start
            teacher_gates[cursor : cursor + batch] = gates_t.detach().cpu().numpy().astype(np.float32)
            cursor += batch

    x_sensor = np.asarray(x_raw[idx, : cfg.input_dim_raw], dtype=np.float32)
    return x_sensor, teacher_gates, idx.astype(np.int64)


def _fit_hybrid_symbolic_sensor(
    x_train: np.ndarray,
    teacher_gates_train: np.ndarray,
    cfg: PipelineConfig,
) -> dict[str, object]:
    basis_train = _build_symbolic_basis(x_train, cfg)
    mean = basis_train.mean(axis=0)
    scale = basis_train.std(axis=0) + 1e-6

    band_lower, band_upper = _band_limits(cfg)
    mach_train = x_train[:, 6]
    in_band = (mach_train >= band_lower) & (mach_train <= band_upper)
    if not np.any(in_band):
        raise ValueError("Hybrid sensor fit produced an empty transonic band.")

    standardized = np.clip((basis_train[in_band] - mean) / scale, -cfg.sensor_feature_clip, cfg.sensor_feature_clip)
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram = design.T @ design
    ridge = np.eye(gram.shape[0], dtype=np.float64)
    ridge[0, 0] = 0.0
    system = gram + float(cfg.sensor_ridge_alpha) * ridge

    coefficients_std: list[np.ndarray] = []
    coefficients_raw: list[np.ndarray] = []
    intercepts_raw: list[float] = []
    equations: list[str] = []

    for dim in range(cfg.n_experts):
        target = teacher_gates_train[in_band, dim].astype(np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            rhs = design.T @ target
        coef_std = np.linalg.solve(system, rhs)
        coef_raw = coef_std[1:] / scale
        intercept_raw = float(coef_std[0] - np.sum(coef_std[1:] * mean / scale))

        coefficients_std.append(coef_std.astype(np.float64))
        coefficients_raw.append(coef_raw.astype(np.float64))
        intercepts_raw.append(intercept_raw)
        equations.append(_render_linear_equation(intercept_raw, coef_raw, SYMBOLIC_BASIS_NAMES))

    return {
        "type": "hybrid_linear_band",
        "description": "Hard symbolic routing outside the transonic band and calibrated enriched symbolic scores inside it.",
        "band_lower": band_lower,
        "band_upper": band_upper,
        "feature_indices": HYBRID_FEATURE_INDICES,
        "feature_names": SYMBOLIC_BASIS_NAMES,
        "feature_mean": mean.astype(np.float64).tolist(),
        "feature_scale": scale.astype(np.float64).tolist(),
        "ridge_alpha": float(cfg.sensor_ridge_alpha),
        "feature_clip": float(cfg.sensor_feature_clip),
        "coefficients_std": np.stack(coefficients_std, axis=1).astype(np.float64).tolist(),
        "coefficients_raw": np.stack(coefficients_raw, axis=1).astype(np.float64).tolist(),
        "intercepts_raw": intercepts_raw,
        "equations": equations,
    }


def apply_hybrid_symbolic_sensor(
    x_raw: np.ndarray,
    artifact: dict[str, object],
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    basis = _build_symbolic_basis(x_raw, cfg)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
    coef_std = np.asarray(artifact["coefficients_std"], dtype=np.float64)
    clip_value = float(artifact.get("feature_clip", cfg.sensor_feature_clip))

    standardized = np.clip((basis - mean) / scale, -clip_value, clip_value)
    mach = np.asarray(x_raw[:, 6], dtype=np.float64)
    band_lower = float(artifact["band_lower"])
    band_upper = float(artifact["band_upper"])

    scores = np.zeros((x_raw.shape[0], cfg.n_experts), dtype=np.float64)
    lower_mask = mach < band_lower
    upper_mask = mach > band_upper
    band_mask = ~(lower_mask | upper_mask)

    scores[lower_mask, 0] = 1.0
    scores[upper_mask, 2] = 1.0

    if np.any(band_mask):
        design = np.concatenate([np.ones((int(np.count_nonzero(band_mask)), 1), dtype=np.float64), standardized[band_mask]], axis=1)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores[band_mask] = design @ coef_std

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    gates = _normalize_positive_scores(scores.astype(np.float32))
    return scores.astype(np.float32), gates


def distill_sensor(cfg: PipelineConfig) -> None:
    cfg.ensure_dirs()

    latent_model_path = cfg.models_dir / "latent_sensor_moe.pth"
    if not latent_model_path.exists():
        raise FileNotFoundError(f"Latent model not found: {latent_model_path}. Run 'train-latent' first.")

    gate_net = _load_teacher_gate_net(cfg)
    train_x, train_teacher_gates, _ = _teacher_outputs(cfg, gate_net, "train", cfg.sensor_max_samples, seed=21)
    test_x, test_teacher_gates, _ = _teacher_outputs(cfg, gate_net, "test", cfg.sensor_max_samples, seed=22)

    artifact = _fit_hybrid_symbolic_sensor(train_x, train_teacher_gates, cfg)
    test_scores, test_symbolic_gates = apply_hybrid_symbolic_sensor(test_x, artifact, cfg)

    teacher_argmax_test = np.argmax(test_teacher_gates, axis=1)
    symbolic_argmax_test = np.argmax(test_symbolic_gates, axis=1)

    summary: dict[str, dict[str, float | str]] = {}
    equations = artifact["equations"]
    for dim, regime_name in enumerate(REGIME_NAMES):
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
        plt.title(f"Sensor distillation {regime_name}")
        plt.grid(True, alpha=0.25)
        plt.savefig(cfg.plots_dir / f"sensor_gate_{regime_name}_test.png", dpi=220, bbox_inches="tight")
        plt.close()

    overall = {
        "type": str(artifact["type"]),
        "train_rows": int(train_x.shape[0]),
        "test_rows": int(test_x.shape[0]),
        "band_lower": float(artifact["band_lower"]),
        "band_upper": float(artifact["band_upper"]),
        "ridge_alpha": float(artifact["ridge_alpha"]),
        "argmax_agreement_test": float(np.mean(symbolic_argmax_test == teacher_argmax_test)),
        "gate_mae_test": float(np.mean(np.abs(test_symbolic_gates - test_teacher_gates))),
        "gate_rmse_test": float(np.sqrt(np.mean((test_symbolic_gates - test_teacher_gates) ** 2))),
        "teacher_argmax_counts_test": np.bincount(teacher_argmax_test, minlength=cfg.n_experts).astype(int).tolist(),
        "symbolic_argmax_counts_test": np.bincount(symbolic_argmax_test, minlength=cfg.n_experts).astype(int).tolist(),
    }

    with (cfg.sensor_dir / "sensor_hybrid.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    with (cfg.sensor_dir / "sensor_hybrid.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            f"type={artifact['type']}\n"
            f"band_lower={artifact['band_lower']:.6f}\n"
            f"band_upper={artifact['band_upper']:.6f}\n"
            f"ridge_alpha={artifact['ridge_alpha']:.6f}\n"
            f"feature_clip={artifact['feature_clip']:.6f}\n\n"
        )
        for regime_name, equation in zip(REGIME_NAMES, equations):
            handle.write(f"{regime_name}: {equation}\n")

    for regime_name, equation in zip(REGIME_NAMES, equations):
        with (cfg.sensor_dir / f"sensor_score_{regime_name}.txt").open("w", encoding="utf-8") as handle:
            handle.write(str(equation))

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
