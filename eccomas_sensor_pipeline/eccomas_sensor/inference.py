from __future__ import annotations

from pathlib import Path

import json
import joblib
import numpy as np
import torch

from .config import PipelineConfig
from .features import build_encoder_features, build_expert_features
from .models import CpExpertNet, LatentSensorMoE
from .sensor_distillation import apply_hybrid_symbolic_sensor


REGIME_NAMES = ["subsonic", "transonic", "supersonic"]


def _load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["mean"].astype(np.float32), payload["scale"].astype(np.float32)


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def _destandardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values * scale + mean).astype(np.float32)


def _softmax_numpy(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return (exp_scores / np.sum(exp_scores, axis=1, keepdims=True)).astype(np.float32)


def _normalize_positive_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.maximum(scores, 0.0).astype(np.float32)
    denom = scores.sum(axis=1, keepdims=True)
    denom = np.where(denom <= eps, 1.0, denom)
    gates = scores / denom
    zero_rows = np.flatnonzero(scores.sum(axis=1) <= eps)
    if zero_rows.size > 0:
        gates[zero_rows] = 1.0 / scores.shape[1]
    return gates.astype(np.float32)


def _mach_rule_scores(mach: np.ndarray, sub_max: float, trans_max: float, blend_width: float) -> np.ndarray:
    mach = np.asarray(mach, dtype=np.float32).reshape(-1, 1)
    if blend_width <= 0.0:
        sub = (mach < sub_max).astype(np.float32)
        sup = (mach > trans_max).astype(np.float32)
        trans = 1.0 - sub - sup
        return np.concatenate([sub, trans, sup], axis=1).astype(np.float32)

    lower = 2.0 * blend_width
    upper = 2.0 * blend_width
    sub = np.clip((sub_max + blend_width - mach) / max(lower, 1e-8), 0.0, 1.0)
    sup = np.clip((mach - (trans_max - blend_width)) / max(upper, 1e-8), 0.0, 1.0)
    trans = np.clip(1.0 - sub - sup, 0.0, 1.0)
    return np.concatenate([sub, trans, sup], axis=1).astype(np.float32)


def _expert_model_paths(cfg: PipelineConfig) -> list[Path]:
    return [cfg.models_dir / f"expert_{name}.pth" for name in REGIME_NAMES]


def _load_neural_model(cfg: PipelineConfig) -> LatentSensorMoE:
    expert_input_dim = np.load(cfg.features_dir / "expert_features_train.npy", mmap_mode="r").shape[1]
    gate_input_dim = np.load(cfg.features_dir / "gate_features_train.npy", mmap_mode="r").shape[1]
    model = LatentSensorMoE(
        gate_input_dim=gate_input_dim,
        expert_input_dim=expert_input_dim,
        latent_dim=cfg.latent_dim,
        expert_paths=_expert_model_paths(cfg),
    )
    state = torch.load(cfg.models_dir / "latent_sensor_moe.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()
    return model


def _load_expert_models(cfg: PipelineConfig) -> list[CpExpertNet]:
    expert_input_dim = np.load(cfg.features_dir / "expert_features_train.npy", mmap_mode="r").shape[1]
    models: list[CpExpertNet] = []
    for path in _expert_model_paths(cfg):
        if not path.exists():
            raise FileNotFoundError(f"Expert model not found: {path}")
        model = CpExpertNet(input_dim=expert_input_dim)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(cfg.device)
        model.eval()
        models.append(model)
    return models


def _load_sensor_models(cfg: PipelineConfig):
    hybrid_path = cfg.sensor_dir / "sensor_hybrid.json"
    if hybrid_path.exists():
        with hybrid_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    rule_path = cfg.sensor_dir / "sensor_rule.json"
    if rule_path.exists():
        with rule_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    sensor_models = []
    for regime_name in REGIME_NAMES:
        path = cfg.sensor_dir / f"sensor_score_{regime_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Sensor model not found: {path}. Run 'distill-sensor' first.")
        sensor_models.append(joblib.load(path))
    return sensor_models


def predict_array(
    cfg: PipelineConfig,
    x_raw: np.ndarray,
    mode: str = "neural",
    batch_size: int | None = None,
    max_rows: int | None = None,
) -> dict[str, np.ndarray]:
    if x_raw.ndim != 2 or x_raw.shape[1] < cfg.input_dim_raw:
        raise ValueError(f"Expected input shape [N, >= {cfg.input_dim_raw}], got {x_raw.shape}")

    n_rows = int(x_raw.shape[0] if max_rows is None else min(max_rows, int(x_raw.shape[0])))
    batch_size = int(batch_size or cfg.latent_batch_size)

    expert_mean, expert_scale = _load_scaler(cfg.scalers_dir / "expert_scaler.npz")
    gate_mean, gate_scale = _load_scaler(cfg.scalers_dir / "gate_scaler.npz")
    cp_mean, cp_scale = _load_scaler(cfg.scalers_dir / "cp_scaler.npz")

    cp_pred = np.zeros((n_rows, 1), dtype=np.float32)
    gates = np.zeros((n_rows, cfg.n_experts), dtype=np.float32)
    expert_id = np.zeros((n_rows,), dtype=np.int64)

    if mode == "neural":
        z = np.zeros((n_rows, cfg.latent_dim), dtype=np.float32)
        sensor_scores = np.zeros((n_rows, cfg.n_experts), dtype=np.float32)
        model = _load_neural_model(cfg)
    elif mode == "symbolic":
        z = None
        sensor_scores = np.zeros((n_rows, cfg.n_experts), dtype=np.float32)
        experts = _load_expert_models(cfg)
        sensor_artifact = _load_sensor_models(cfg)
    else:
        raise ValueError(f"Unsupported inference mode: {mode}")

    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(n_rows, start + batch_size)
            x_chunk = np.asarray(x_raw[start:end, : cfg.input_dim_raw], dtype=np.float32)

            expert_chunk = build_expert_features(x_chunk, cfg.cut_threshold_y)
            expert_chunk = _standardize(expert_chunk, expert_mean, expert_scale)
            expert_tensor = torch.from_numpy(expert_chunk).to(cfg.device, non_blocking=True)

            if mode == "neural":
                gate_chunk = build_encoder_features(x_chunk, cfg.cut_threshold_y)
                gate_chunk = _standardize(gate_chunk, gate_mean, gate_scale)
                gate_tensor = torch.from_numpy(gate_chunk).to(cfg.device, non_blocking=True)
                cp_norm_t, z_t, logits_t, gates_t = model(expert_tensor, gate_tensor)
                cp_pred[start:end] = _destandardize(cp_norm_t.detach().cpu().numpy(), cp_mean, cp_scale)
                z[start:end] = z_t.detach().cpu().numpy().astype(np.float32)
                sensor_scores[start:end] = logits_t.detach().cpu().numpy().astype(np.float32)
                gates[start:end] = gates_t.detach().cpu().numpy().astype(np.float32)
            else:
                if isinstance(sensor_artifact, dict):
                    if sensor_artifact.get("type") == "hybrid_linear_band":
                        score_chunk, gate_chunk = apply_hybrid_symbolic_sensor(x_chunk, sensor_artifact, cfg)
                    elif sensor_artifact.get("type") == "mach_rule":
                        mach = x_chunk[:, 6]
                        score_chunk = _mach_rule_scores(
                            mach,
                            float(sensor_artifact["mach_sub_max"]),
                            float(sensor_artifact["mach_trans_max"]),
                            float(sensor_artifact.get("blend_width", 0.0)),
                        )
                        gate_chunk = _normalize_positive_scores(score_chunk)
                    else:
                        raise ValueError(f"Unsupported symbolic sensor artifact type: {sensor_artifact.get('type')}")
                else:
                    encoder_chunk = build_encoder_features(x_chunk, cfg.cut_threshold_y)
                    if hasattr(sensor_artifact[0], "predict_proba"):
                        score_chunk = np.column_stack([model.predict_proba(encoder_chunk)[:, 1] for model in sensor_artifact]).astype(np.float32)
                        gate_chunk = _normalize_positive_scores(score_chunk)
                    else:
                        score_chunk = np.column_stack([model.predict(encoder_chunk) for model in sensor_artifact]).astype(np.float32)
                        gate_chunk = _normalize_positive_scores(score_chunk)

                expert_outputs = []
                for expert_model in experts:
                    expert_outputs.append(expert_model(expert_tensor).detach().cpu().numpy().astype(np.float32))
                expert_stack = np.stack(expert_outputs, axis=2)
                cp_norm = (expert_stack * gate_chunk[:, None, :]).sum(axis=2)

                cp_pred[start:end] = _destandardize(cp_norm, cp_mean, cp_scale)
                sensor_scores[start:end] = score_chunk
                gates[start:end] = gate_chunk

    expert_id[:] = np.argmax(gates, axis=1).astype(np.int64)

    payload = {
        "cp_pred": cp_pred,
        "gates": gates,
        "expert_id": expert_id,
        "sensor_scores": sensor_scores,
        "mode": np.array(mode),
    }
    if z is not None:
        payload["z"] = z

    return payload


def run_inference(
    cfg: PipelineConfig,
    input_path: Path,
    mode: str = "neural",
    output_path: Path | None = None,
    batch_size: int | None = None,
    max_rows: int | None = None,
) -> Path:
    cfg.ensure_dirs()

    if not input_path.exists():
        raise FileNotFoundError(f"Input array not found: {input_path}")

    x_raw = np.load(input_path, mmap_mode="r")
    payload = predict_array(cfg, x_raw, mode=mode, batch_size=batch_size, max_rows=max_rows)

    if output_path is None:
        suffix = f"_{mode}.npz"
        output_path = cfg.inference_dir / f"{Path(input_path).stem}{suffix}"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, **payload)
    print(f"[infer] Saved predictions to {output_path}")
    return output_path
