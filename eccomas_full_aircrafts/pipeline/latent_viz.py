from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import FullAircraftConfig
from .utils import sample_indices

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt


def plot_latent_summary(cfg: FullAircraftConfig, split: str) -> None:
    latent_path = cfg.latent_dir / f"latent_{split}.npz"
    data = np.load(latent_path)
    z = data["z"].astype(np.float32)
    expert_id = data["expert_id"].astype(np.int64)
    mach = data["mach"].astype(np.float32)
    aoa = data["aoa"].astype(np.float32)
    cp_pred = data["cp_pred"].astype(np.float32)

    idx = sample_indices(z.shape[0], cfg.latent_plot_sample_size, seed=11)
    z = z[idx]
    expert_id = expert_id[idx]
    mach = mach[idx]
    aoa = aoa[idx]
    cp_pred = cp_pred[idx]

    dims = z.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    z0 = z[:, 0]
    z1 = z[:, 1] if dims > 1 else np.zeros_like(z0)
    z2 = z[:, 2] if dims > 2 else np.zeros_like(z0)

    sc0 = axes[0, 0].scatter(z0, z1, c=expert_id, s=2.0, alpha=0.45, cmap="viridis")
    axes[0, 0].set_title(f"{split}: z1 vs z2 by expert")
    axes[0, 0].set_xlabel("z1")
    axes[0, 0].set_ylabel("z2")
    fig.colorbar(sc0, ax=axes[0, 0], shrink=0.8)

    sc1 = axes[0, 1].scatter(z0, z2, c=mach, s=2.0, alpha=0.45, cmap="turbo")
    axes[0, 1].set_title(f"{split}: z1 vs z3 by Mach")
    axes[0, 1].set_xlabel("z1")
    axes[0, 1].set_ylabel("z3")
    fig.colorbar(sc1, ax=axes[0, 1], shrink=0.8)

    sc2 = axes[1, 0].scatter(z1, z2, c=aoa, s=2.0, alpha=0.45, cmap="coolwarm")
    axes[1, 0].set_title(f"{split}: z2 vs z3 by AoA")
    axes[1, 0].set_xlabel("z2")
    axes[1, 0].set_ylabel("z3")
    fig.colorbar(sc2, ax=axes[1, 0], shrink=0.8)

    sc3 = axes[1, 1].scatter(z0, z1, c=cp_pred[:, 0], s=2.0, alpha=0.45, cmap="magma")
    axes[1, 1].set_title(f"{split}: z1 vs z2 by predicted Cp")
    axes[1, 1].set_xlabel("z1")
    axes[1, 1].set_ylabel("z2")
    fig.colorbar(sc3, ax=axes[1, 1], shrink=0.8)

    fig.savefig(cfg.plots_dir / f"latent_summary_{split}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if dims >= 3:
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(z0, z1, z2, c=expert_id, s=1.6, alpha=0.35, cmap="viridis")
        ax.set_title(f"{split}: latent 3D by expert")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        fig.colorbar(sc, ax=ax, shrink=0.8)
        fig.savefig(cfg.plots_dir / f"latent_3d_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
