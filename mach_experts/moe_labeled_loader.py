# mach_experts/moe_labeled_loader.py

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler

import config


def _load_gradcp(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {path}. "
            "Necesito gradCp_cut_train.npy con la magnitud de grad(Cp) por punto."
        )
    g = np.load(path).astype(np.float32)
    if g.ndim == 1:
        g = g[:, None]
    return g  # [N, 1]


def make_global_labeled_loader_with_f(batch_size=None, shuffle=True):
    """
    Devuelve un DataLoader global con TODAS las muestras de train:

      - X_norm    : [N, 9]  (x,y,z,nx,ny,nz,Mach,AoA,Pi) normalizados
      - Y_norm    : [N, 1]  (Cp normalizado)
      - expert_id : [N]     (0=sub, 1=trans, 2=sup) según Mach crudo
      - F_norm    : [N, 4]  (log(1+|gradCp|), AoA, Mach, Pi) normalizados
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # ---- Rutas a los datos cortados ----
    x_path = os.path.join(config.DATA_DIR, "X_cut_train.npy")
    y_path = os.path.join(config.DATA_DIR, "Y_cut_train.npy")
    g_path = os.path.join(config.DATA_DIR, "gradCp_cut_train.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"No se encontró X_cut_train en {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"No se encontró Y_cut_train en {y_path}")

    # ---- Cargar datos crudos (ya cortados en y) ----
    X_raw = np.load(x_path).astype(np.float32)        # [N, >=9]
    Y_raw_all = np.load(y_path).astype(np.float32)    # [N, 4]
    gradCp = _load_gradcp(g_path)                     # [N, 1]

    if X_raw.shape[1] < config.INPUT_DIM:
        raise ValueError(f"X debe tener al menos {config.INPUT_DIM} columnas, shape={X_raw.shape}")
    if Y_raw_all.shape[1] < 1:
        raise ValueError(f"Y debe tener al menos 1 columna (Cp), shape={Y_raw_all.shape}")
    if gradCp.shape[0] != X_raw.shape[0]:
        raise ValueError(f"gradCp y X tienen distinto N: {gradCp.shape[0]} vs {X_raw.shape[0]}")

    # Nos quedamos con las primeras 9 columnas de X
    X_raw = X_raw[:, :config.INPUT_DIM]               # [N, 9]

    # SOLO Cp (columna 0 de Y)
    Y_raw = Y_raw_all[:, [0]]                         # [N, 1]

    # ---- Cargar scalers de expertos ----
    scaler_x_path = os.path.join(config.SCALER_DIR, "scaler_x.bin")
    scaler_y_path = os.path.join(config.SCALER_DIR, "scaler_y.bin")

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError(
            "No se encontraron los scalers en SCALER_DIR. "
            "Asegúrate de haber ejecutado antes el entrenamiento de expertos."
        )

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # ---- Normalizar X e Y con los mismos scalers ----
    X_norm = scaler_x.transform(X_raw).astype(np.float32)  # [N, 9]
    Y_norm = scaler_y.transform(Y_raw).astype(np.float32)  # [N, 1]

    # ---- Features físicas para el encoder/gating ----
    # Mach = col 6, AoA = col 7, Pi = col 8 (sin normalizar)
    mach = X_raw[:, 6:7]   # [N,1]
    aoa  = X_raw[:, 7:8]   # [N,1]
    pi   = X_raw[:, 8:9]   # [N,1]

    # log(1 + |gradCp|)
    grad_feat = np.log1p(np.abs(gradCp))  # [N,1]

    F_raw = np.concatenate([grad_feat, aoa, mach, pi], axis=1)  # [N,4]

    # Normalizamos F con SU PROPIO scaler
    scaler_f_path = os.path.join(config.SCALER_DIR, "scaler_f.bin")
    if os.path.exists(scaler_f_path):
        scaler_f = joblib.load(scaler_f_path)
    else:
        scaler_f = StandardScaler().fit(F_raw)   # <-- AQUÍ estaba el bug
        joblib.dump(scaler_f, scaler_f_path)
        print(f"[INFO] scaler_f (features físicas) guardado en {scaler_f_path}")

    F_norm = scaler_f.transform(F_raw).astype(np.float32)  # [N,4]

    # ---- Etiqueta de experto según Mach CRUDO ----
    mach_scalar = X_raw[:, 6]  # [N]

    expert_id = np.zeros_like(mach_scalar, dtype=np.int64)  # 0 = subsonic
    mask_trans = (mach_scalar >= config.MACH_TRANS_MIN) & (mach_scalar <= config.MACH_TRANS_MAX)
    expert_id[mask_trans] = 1
    mask_sup = (mach_scalar > config.MACH_TRANS_MAX)
    expert_id[mask_sup] = 2

    print("\n[LABELS] Distribución de expert_id (train):")
    print(f"  subsonic   (0): {np.sum(expert_id == 0)} muestras")
    print(f"  transonic  (1): {np.sum(expert_id == 1)} muestras")
    print(f"  supersonic (2): {np.sum(expert_id == 2)} muestras")

    dataset = TensorDataset(
        torch.from_numpy(X_norm),          # [N, 9]
        torch.from_numpy(Y_norm),          # [N, 1]
        torch.from_numpy(expert_id),       # [N]
        torch.from_numpy(F_norm),          # [N, 4]
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