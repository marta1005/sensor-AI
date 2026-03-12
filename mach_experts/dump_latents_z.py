import os
import numpy as np
import torch
import joblib

import config
from moe_latent_models import LatentGatedMoE

# ===== Ajustes =====
OUT_DIR = os.path.join(config.RESULTS_DIR, "latents")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 65536
SPLIT = "train"  # "train" o "test"
# ===================


def load_split(split: str):
    if split == "train":
        x_path = os.path.join(config.DATA_DIR, "X_cut_train.npy")
        g_path = os.path.join(config.DATA_DIR, "gradCp_cut_train.npy")
    elif split == "test":
        x_path = os.path.join(config.DATA_DIR, "X_cut_test.npy")
        g_path = os.path.join(config.DATA_DIR, "gradCp_cut_test.npy")
    else:
        raise ValueError("split debe ser 'train' o 'test'")

    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(g_path):
        raise FileNotFoundError(g_path)

    X_raw = np.load(x_path).astype(np.float32)[:, :config.INPUT_DIM]  # [N,9]
    gradCp = np.load(g_path).astype(np.float32)
    if gradCp.ndim == 1:
        gradCp = gradCp[:, None]
    if gradCp.shape[0] != X_raw.shape[0]:
        raise ValueError("gradCp y X no tienen el mismo N")

    # scalers
    scaler_x = joblib.load(os.path.join(config.SCALER_DIR, "scaler_x.bin"))
    scaler_f = joblib.load(os.path.join(config.SCALER_DIR, "scaler_f.bin"))

    # X_norm
    X_norm = scaler_x.transform(X_raw).astype(np.float32)

    # F_raw = [log1p|gradCp|, AoA, Mach, Pi]
    mach = X_raw[:, 6:7]
    aoa  = X_raw[:, 7:8]
    pi   = X_raw[:, 8:9]
    grad_feat = np.log1p(np.abs(gradCp))
    F_raw = np.concatenate([grad_feat, aoa, mach, pi], axis=1).astype(np.float32)

    F_norm = scaler_f.transform(F_raw).astype(np.float32)

    return X_raw, X_norm, F_norm


def main():
    device = config.DEVICE
    print("[DUMP] Device:", device)

    X_raw, X_norm, F_norm = load_split(SPLIT)
    N = X_norm.shape[0]
    print(f"[DUMP] Split={SPLIT} | N={N}")

    # Cargar modelo MoE (necesitamos el encoder)
    moe = LatentGatedMoE().to(device)
    moe_path = os.path.join(config.MODEL_DIR, "latent_gated_moe_phys.pth")
    if not os.path.exists(moe_path):
        raise FileNotFoundError(f"No existe {moe_path}. Entrena primero el moe_latent_train.py")
    moe.load_state_dict(torch.load(moe_path, map_location=device))
    moe.eval()

    Z = np.zeros((N, 3), dtype=np.float32)
    G = np.zeros((N, 3), dtype=np.float32)  # gates por si quieres mirar
    idx = np.arange(N, dtype=np.int64)

    with torch.no_grad():
        for s in range(0, N, BATCH):
            e = min(N, s + BATCH)
            bx = torch.from_numpy(X_norm[s:e]).to(device)
            bf = torch.from_numpy(F_norm[s:e]).to(device)

            # forward para obtener z y gates
            _, gates, z, _ = moe(bx, bf, return_all=True)
            Z[s:e] = z.detach().cpu().numpy().astype(np.float32)
            G[s:e] = gates.detach().cpu().numpy().astype(np.float32)

            if (s // BATCH) % 20 == 0:
                print(f"[DUMP] {s}/{N}")

    out_path = os.path.join(OUT_DIR, f"latents_z_{SPLIT}.npz")
    np.savez_compressed(
        out_path,
        X_raw=X_raw,     # [N,9] físico
        Z=Z,             # [N,3]
        G=G,             # [N,3] (opcional)
        idx=idx
    )
    print("[SAVED]", out_path)


if __name__ == "__main__":
    main()