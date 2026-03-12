import os
import numpy as np
import matplotlib.pyplot as plt

import config

# ===== Ajustes =====
LATENT_FILE = os.path.join(config.RESULTS_DIR, "latents", "latents_z_train.npz")
MAX_POINTS = 400_000  # submuestreo para no petar el plot

# Selector de color:
#   "expert" | "Mach" | "AoA" | "Pi" | "grad"
COLOR_BY = "grad"

# Opciones visuales
POINT_SIZE = 1
ALPHA = 0.35
# ===================


def _expert_id_from_mach(mach: np.ndarray) -> np.ndarray:
    exp = np.zeros(len(mach), dtype=np.int32)
    exp[(mach >= config.MACH_TRANS_MIN) & (mach <= config.MACH_TRANS_MAX)] = 1
    exp[(mach > config.MACH_TRANS_MAX)] = 2
    return exp


def _load_grad(split: str = "train") -> np.ndarray:
    """
    Carga gradCp_cut_{split}.npy desde config.DATA_DIR.
    Espera shape [N] o [N,1]. Devuelve [N].
    """
    if split == "train":
        g_path = os.path.join(config.DATA_DIR, "gradCp_cut_train.npy")
    else:
        g_path = os.path.join(config.DATA_DIR, "gradCp_cut_test.npy")

    if not os.path.exists(g_path):
        raise FileNotFoundError(
            f"No existe {g_path}. Necesito gradCp_cut_train.npy para COLOR_BY='grad'."
        )

    g = np.load(g_path).astype(np.float32)
    if g.ndim == 2:
        g = g[:, 0]
    return g  # [N]


def _get_color_values(X_raw: np.ndarray, idx: np.ndarray | None, color_by: str, split="train"):
    """
    X_raw tiene columnas:
      0 x, 1 y, 2 z, 3 nx, 4 ny, 5 nz, 6 Mach, 7 AoA, 8 Pi
    idx: índices del submuestreo (o None si no hay submuestreo)
    """
    color_by = color_by.lower().strip()

    if color_by == "mach":
        vals = X_raw[:, 6]
        label = "Mach"
        cmap = "turbo"
        is_discrete = False
        return vals, label, cmap, is_discrete

    if color_by == "aoa":
        vals = X_raw[:, 7]
        label = "AoA"
        cmap = "coolwarm"
        is_discrete = False
        return vals, label, cmap, is_discrete

    if color_by == "pi":
        vals = X_raw[:, 8]
        label = "Pi"
        cmap = "viridis"
        is_discrete = False
        return vals, label, cmap, is_discrete

    if color_by == "expert":
        mach = X_raw[:, 6]
        vals = _expert_id_from_mach(mach)
        label = "expert_id (0=sub,1=trans,2=sup)"
        cmap = "viridis"
        is_discrete = True
        return vals, label, cmap, is_discrete

    if color_by == "grad":
        # gradCp está guardado por punto en el mismo orden que X_cut_train
        g = _load_grad(split=split)
        if idx is not None:
            g = g[idx]
        # para visualizar mejor, usamos log1p
        vals = np.log1p(np.abs(g))
        label = "log(1 + |∇Cp|)"
        cmap = "magma"
        is_discrete = False
        return vals, label, cmap, is_discrete

    raise ValueError("COLOR_BY debe ser: 'expert', 'Mach', 'AoA', 'Pi' o 'grad'.")


def main():
    if not os.path.exists(LATENT_FILE):
        raise FileNotFoundError(
            f"No existe {LATENT_FILE}. Ejecuta antes dump_latents_z.py"
        )

    data = np.load(LATENT_FILE)
    Z = data["Z"].astype(np.float32)          # [N,3]
    X_raw = data["X_raw"].astype(np.float32)  # [N,9]

    N = Z.shape[0]
    print(f"[VIS2D] Cargado {LATENT_FILE} | N={N}")

    # Submuestreo
    idx = None
    if N > MAX_POINTS:
        idx = np.random.choice(N, MAX_POINTS, replace=False)
        Zp = Z[idx]
        Xp = X_raw[idx]
        print(f"[VIS2D] Submuestreo {MAX_POINTS}/{N}")
    else:
        Zp = Z
        Xp = X_raw

    z1, z2, z3 = Zp[:, 0], Zp[:, 1], Zp[:, 2]

    cvals, clabel, cmap, is_discrete = _get_color_values(Xp, idx, COLOR_BY, split="train")

    # ---------- FIG: 3 proyecciones 2D ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sc0 = axes[0].scatter(z1, z2, c=cvals, s=POINT_SIZE, alpha=ALPHA, cmap=cmap)
    axes[0].set_title(f"z1 vs z2 (color={COLOR_BY})")
    axes[0].set_xlabel("z1"); axes[0].set_ylabel("z2")
    axes[0].grid(True, alpha=0.2)

    sc1 = axes[1].scatter(z1, z3, c=cvals, s=POINT_SIZE, alpha=ALPHA, cmap=cmap)
    axes[1].set_title(f"z1 vs z3 (color={COLOR_BY})")
    axes[1].set_xlabel("z1"); axes[1].set_ylabel("z3")
    axes[1].grid(True, alpha=0.2)

    sc2 = axes[2].scatter(z2, z3, c=cvals, s=POINT_SIZE, alpha=ALPHA, cmap=cmap)
    axes[2].set_title(f"z2 vs z3 (color={COLOR_BY})")
    axes[2].set_xlabel("z2"); axes[2].set_ylabel("z3")
    axes[2].grid(True, alpha=0.2)

    # Colorbar
    cbar = fig.colorbar(sc2, ax=axes, shrink=0.9, pad=0.01)
    cbar.set_label(clabel)

    # Si es discreto (expert_id), ponemos ticks 0,1,2
    if is_discrete:
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["sub", "trans", "sup"])

    plt.show()


if __name__ == "__main__":
    main()