from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .features import ENCODER_FEATURE_NAMES, build_encoder_features
from .utils import sample_indices

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def _load_symbolic_backend():
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError as exc:
        raise ImportError("gplearn is required for 'distill-symbolic'. Install it in the active environment.") from exc
    return SymbolicRegressor


def distill_symbolic_encoder(cfg: PipelineConfig) -> None:
    SymbolicRegressor = _load_symbolic_backend()

    latent_train = np.load(cfg.latent_dir / "latent_train.npz")
    latent_test = np.load(cfg.latent_dir / "latent_test.npz")
    x_train = np.load(cfg.cut_data_dir / "X_cut_train.npy", mmap_mode="r")
    x_test = np.load(cfg.cut_data_dir / "X_cut_test.npy", mmap_mode="r")

    idx_train = sample_indices(x_train.shape[0], cfg.symbolic_max_samples, seed=21)
    idx_test = sample_indices(x_test.shape[0], min(cfg.symbolic_max_samples, x_test.shape[0]), seed=22)

    train_features = build_encoder_features(np.asarray(x_train[idx_train, : cfg.input_dim_raw], dtype=np.float32), cfg.cut_threshold_y)
    test_features = build_encoder_features(np.asarray(x_test[idx_test, : cfg.input_dim_raw], dtype=np.float32), cfg.cut_threshold_y)
    z_train = latent_train["z"][idx_train]
    z_test = latent_test["z"][idx_test]

    summary: dict[str, dict[str, float | str]] = {}
    for dim in range(cfg.latent_dim):
        reg = SymbolicRegressor(
            population_size=2500,
            generations=120,
            tournament_size=500,
            function_set=("add", "sub", "mul", "div", "sqrt", "log"),
            metric="mse",
            parsimony_coefficient="auto",
            max_samples=0.9,
            p_crossover=0.7,
            p_subtree_mutation=0.02,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            verbose=1,
            random_state=42 + dim,
            n_jobs=-1,
            feature_names=ENCODER_FEATURE_NAMES,
        )
        reg.fit(train_features, z_train[:, dim])
        pred_test = reg.predict(test_features)
        r2 = float(1.0 - np.sum((z_test[:, dim] - pred_test) ** 2) / np.sum((z_test[:, dim] - z_test[:, dim].mean()) ** 2))
        eq = str(reg._program)

        with (cfg.symbolic_dir / f"eq_z{dim + 1}.txt").open("w", encoding="utf-8") as handle:
            handle.write(eq)

        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(z_test[:, dim], pred_test, s=3, alpha=0.35)
        lo = min(float(z_test[:, dim].min()), float(pred_test.min()))
        hi = max(float(z_test[:, dim].max()), float(pred_test.max()))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        plt.xlabel(f"true z{dim + 1}")
        plt.ylabel(f"symbolic z{dim + 1}")
        plt.title(f"Symbolic distillation z{dim + 1} | R2={r2:.4f}")
        plt.grid(True, alpha=0.25)
        plt.savefig(cfg.plots_dir / f"symbolic_z{dim + 1}_test.png", dpi=220, bbox_inches="tight")
        plt.close()

        summary[f"z{dim + 1}"] = {"equation": eq, "test_r2": r2}

    with (cfg.symbolic_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[distill-symbolic] Finished. Equations stored in {cfg.symbolic_dir}")
