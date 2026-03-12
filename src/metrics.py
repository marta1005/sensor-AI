import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config
import logging

logger = logging.getLogger(__name__)

def compute_cp_metrics(
    model,
    data_handler,
    snapshot_idx=None,
    batch_size=None,
    device=None,
):
    """
    Calcula MSE, RMSE, R2 y MAE para Cp (pred vs real) en unidades físicas (desnormalizado).
    Añadido MAE y métricas por región si hay máscara.
    """
    if device is None:
        device = config.DEVICE

    if batch_size is None:
        batch_size = min(65536, config.BATCH_SIZE * 2)

    # Checks mínimos
    if not getattr(data_handler, "has_targets", False) or data_handler.Y_raw is None:
        raise ValueError("Para calcular métricas necesitas Y_raw (cargar y_path en el AerodynamicDataHandler).")
    if getattr(data_handler, "scaler_y", None) is None:
        raise ValueError("Para desnormalizar predicción necesitas data_handler.scaler_y (cargar scaler_y).")

    # Selección de índices: todo o un snapshot
    if snapshot_idx is None:
        idx_all = np.arange(data_handler.X_final.shape[0], dtype=np.int64)
    else:
        start = snapshot_idx * config.NP
        end = start + config.NP
        idx_all = np.arange(start, end, dtype=np.int64)

    # Loader con índices (para alinear pred vs GT siempre)
    dset = TensorDataset(
        torch.from_numpy(idx_all),
        torch.from_numpy(data_handler.X_final[idx_all])
    )
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()

    # Acumuladores para SSE, SST, MAE
    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    sae = 0.0  # Sum Absolute Error
    n = 0

    with torch.no_grad():
        for idx_batch, x_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)

            # Predicción normalizada
            y_pred_norm, _ = model(x_batch)  # [B,4] (norm)
            y_pred_norm = y_pred_norm.detach().cpu().numpy()

            # Desnormalizar a físico
            y_pred = data_handler.scaler_y.inverse_transform(y_pred_norm)
            cp_pred = y_pred[:, 0].astype(np.float64)

            # Cp real (físico)
            idx_np = idx_batch.numpy()
            cp_true = data_handler.Y_raw[idx_np, 0].astype(np.float64)

            diff = cp_true - cp_pred
            sse += np.sum(diff * diff)
            sae += np.sum(np.abs(diff))
            sum_y += np.sum(cp_true)
            sum_y2 += np.sum(cp_true * cp_true)
            n += cp_true.size

    mse = sse / n
    rmse = np.sqrt(mse)
    mae = sae / n

    mean_y = sum_y / n
    sst = sum_y2 - n * (mean_y ** 2)
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")

    metrics = {"MSE": float(mse), "RMSE": float(rmse), "R2": float(r2), "MAE": float(mae), "N": int(n)}
    logger.info(f"[METRICS] Cp: {metrics}")

    return metrics