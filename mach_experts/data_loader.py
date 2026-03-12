import os
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import src.config as config


def explore_design_space(X_raw, save_dir):
    """
    Exploración básica del espacio de diseño:
      - Mach vs AoA
      - rangos, número de combinaciones, etc.
    Guarda una figura Mach vs AoA en save_dir.
    """
    mach = X_raw[:, 6]
    alpha = X_raw[:, 7]

    N = len(mach)
    mach_min, mach_max = float(mach.min()), float(mach.max())
    alpha_min, alpha_max = float(alpha.min()), float(alpha.max())

    mach_unique = np.unique(mach)
    alpha_unique = np.unique(alpha)

    print("\n[EDA] ================= EXPLORACIÓN ESPACIO DE DISEÑO =================")
    print(f"[EDA] Nº total de muestras              : {N}")
    print(f"[EDA] Mach   -> min={mach_min:.4f}, max={mach_max:.4f}")
    print(f"[EDA] AoA    -> min={alpha_min:.4f}, max={alpha_max:.4f}")
    print(f"[EDA] Nº valores únicos de Mach         : {len(mach_unique)}")
    print(f"[EDA] Nº valores únicos de AoA          : {len(alpha_unique)}")

    if len(mach_unique) <= 50:
        print(f"[EDA] Mach únicos: {np.round(mach_unique, 4)}")
    else:
        print(f"[EDA] Mach únicos (primeros 20): {np.round(mach_unique[:20], 4)} ...")

    if len(alpha_unique) <= 50:
        print(f"[EDA] AoA únicos: {np.round(alpha_unique, 4)}")
    else:
        print(f"[EDA] AoA únicos (primeros 20): {np.round(alpha_unique[:20], 4)} ...")

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "mach_alpha_design_space.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(mach, alpha, s=2, alpha=0.3)
    plt.xlabel("Mach")
    plt.ylabel("AoA")
    plt.title("Espacio de diseño: Mach vs AoA")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[EDA] Figura Mach vs AoA guardada en: {fig_path}")
    print("[EDA] ==========================================================\n")


def load_train_data():
    """
    Carga X_cut_train.npy / Y_cut_train.npy, explora el espacio de diseño,
    normaliza X e Y (solo Cp) y separa en 3 regímenes de Mach:
      - subsonic   : M < 0.6
      - transonic  : 0.6 <= M <= 0.85
      - supersonic : M > 0.85

    Devuelve:
      regime_data = {
        "subsonic":   {"X": X_sub,  "Y": Y_sub,  "idx": idx_sub},
        "transonic":  {"X": X_tr,   "Y": Y_tr,   "idx": idx_tr},
        "supersonic": {"X": X_sup,  "Y": Y_sup,  "idx": idx_sup},
      }
      scalers = {"x": scaler_x, "y": scaler_y}
    """
    x_path = os.path.join(config.DATA_DIR, "X_cut_train.npy")
    y_path = os.path.join(config.DATA_DIR, "Y_cut_train.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"No se encontró X_train en {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"No se encontró Y_train en {y_path}")

    X_raw = np.load(x_path).astype(np.float32)  # [N, >=9]
    Y_raw_all = np.load(y_path).astype(np.float32)  # [N, 4]

    if X_raw.shape[1] < 9:
        raise ValueError(f"Se esperaban al menos 9 columnas en X. Recibido: {X_raw.shape}")
    if Y_raw_all.shape[1] < 1:
        raise ValueError(f"Y debe tener al menos 1 columna (Cp). Recibido: {Y_raw_all.shape}")

    # Nos quedamos con las primeras 9 columnas (x,y,z,nx,ny,nz,Mach,AoA,Pi)
    X_raw = X_raw[:, :config.INPUT_DIM]

    # SOLO Cp (columna 0)
    Y_raw = Y_raw_all[:, [0]]  # shape [N, 1]

    # 1) Exploración de Mach y AoA
    explore_design_space(X_raw, save_dir=config.RESULTS_DIR)

    # Mach es la columna 6 (índice 6) en X_raw
    mach = X_raw[:, 6]

    # 2) Scalers globales
    scaler_x = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(Y_raw)

    X_norm = scaler_x.transform(X_raw).astype(np.float32)
    Y_norm = scaler_y.transform(Y_raw).astype(np.float32)

    # 3) Máscaras por régimen
    mask_sub   = (mach < config.MACH_SUB_MAX)
    mask_trans = (mach >= config.MACH_TRANS_MIN) & (mach <= config.MACH_TRANS_MAX)
    mask_sup   = (mach > config.MACH_TRANS_MAX)

    regime_data = {}

    def add_regime(name, mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(f"[WARN] No hay muestras para régimen '{name}'")
            regime_data[name] = {"X": None, "Y": None, "idx": idx}
        else:
            regime_data[name] = {
                "X": X_norm[idx],
                "Y": Y_norm[idx],
                "idx": idx
            }
            print(f"[INFO] Régimen {name}: N = {len(idx)}")

    add_regime("subsonic",   mask_sub)
    add_regime("transonic",  mask_trans)
    add_regime("supersonic", mask_sup)

    scalers = {"x": scaler_x, "y": scaler_y}

    joblib.dump(scalers["x"], os.path.join(config.SCALER_DIR, "scaler_x.bin"))
    joblib.dump(scalers["y"], os.path.join(config.SCALER_DIR, "scaler_y.bin"))

    print(f"[INFO] Scalers guardados en {config.SCALER_DIR}")
    return regime_data, scalers


def make_loader(X, Y, batch_size=None, shuffle=True):
    """
    Crea un DataLoader a partir de arrays normalizados X, Y.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    return loader