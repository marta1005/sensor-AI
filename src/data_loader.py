import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from . import config
from src.physics import compute_surface_gradients_batched

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging

logger = logging.getLogger(__name__)


class AerodynamicDataHandler:
    """Gestión de datos para entrenamiento e inferencia (sin validación)."""

    def __init__(
        self,
        x_path: str,
        y_path: str | None = None,
        is_train: bool = True,
        scalers: dict | None = None,
        compute_physics: bool | None = None,
        build_ae_inputs: bool | None = None,
        require_y_scaler_in_test: bool = True,
        require_grad_scaler_in_test: bool = False,
    ):
        if compute_physics is None:
            compute_physics = True if is_train else False
        if build_ae_inputs is None:
            build_ae_inputs = True if is_train else False

        self.is_train = is_train
        self.compute_physics = compute_physics
        self.build_ae_inputs = build_ae_inputs

        logger.info(f"\n[DATA] Cargando X desde: {os.path.basename(x_path)}")

        # 1) Cargar X
        self.X_raw = np.load(x_path).astype(np.float32)
        if self.X_raw.ndim != 2 or self.X_raw.shape[1] < 9:
            raise ValueError(f"X debe tener shape [N,>=9]. Recibido: {self.X_raw.shape}")
        if self.X_raw.shape[1] > 9:
            self.X_raw = self.X_raw[:, :9]

        # 2) Cargar Y (opcional)
        self.has_targets = False
        self.Y_raw = None
        if y_path is not None and os.path.exists(y_path):
            logger.info(f"[DATA] Cargando Y desde: {os.path.basename(y_path)}")
            self.Y_raw = np.load(y_path).astype(np.float32)
            if self.Y_raw.ndim != 2 or self.Y_raw.shape[1] != 4:
                raise ValueError(f"Y debe tener shape [N,4]. Recibido: {self.Y_raw.shape}")
            if self.Y_raw.shape[0] != self.X_raw.shape[0]:
                raise ValueError("X e Y deben tener el mismo número de filas.")
            self.has_targets = True
        else:
            if y_path is None:
                logger.info("[DATA] Sin ground truth (Modo Inferencia).")
            else:
                logger.info(f"[DATA] No se encontró Y en: {y_path}. (Modo Inferencia)")

        # 3) Scalers
        self.scalers = {}

        # X scaler (siempre)
        if is_train:
            self.scaler_x = StandardScaler().fit(self.X_raw)
        else:
            if scalers is None or "x" not in scalers:
                raise ValueError("En test/inferencia se debe pasar scalers con la clave 'x'.")
            self.scaler_x = scalers["x"]
        self.X_final = self.scaler_x.transform(self.X_raw).astype(np.float32)
        self.scalers["x"] = self.scaler_x

        # Inicializar opcionales
        self.scaler_y = None
        self.scaler_grad = None
        self.Y_norm = None
        self.grad_vals = None
        self.grad_norm = None
        self.Y_AE = None

        # Y scaler (si hay targets)
        if self.has_targets:
            if is_train:
                self.scaler_y = StandardScaler().fit(self.Y_raw)
            else:
                self.scaler_y = None if scalers is None else scalers.get("y", None)
                if require_y_scaler_in_test and self.scaler_y is None:
                    raise ValueError("En test con Y, falta scaler 'y'.")

            if self.scaler_y is not None:
                self.Y_norm = self.scaler_y.transform(self.Y_raw).astype(np.float32)
                self.scalers["y"] = self.scaler_y

        # Gradientes físicos (solo si hay targets y compute_physics)
        if self.has_targets and compute_physics:
            logger.info("[DATA] Calculando gradientes físicos (LSQ)...")
            self.grad_vals = compute_surface_gradients_batched(
                self.X_raw,
                self.Y_raw,
                np_points=config.NP,
                n_neighbors=20,
                reg=1e-4
            )
            if is_train:
                self.scaler_grad = StandardScaler().fit(self.grad_vals)
            else:
                self.scaler_grad = None if scalers is None else scalers.get("grad", None)
                if require_grad_scaler_in_test and self.scaler_grad is None:
                    raise ValueError("En test con gradientes, falta scaler 'grad'.")

            if self.scaler_grad is not None:
                self.grad_norm = self.scaler_grad.transform(self.grad_vals).astype(np.float32)
                self.scalers["grad"] = self.scaler_grad

        # Y_AE para AE (solo training)
        if build_ae_inputs:
            if not self.has_targets:
                raise ValueError("No se puede construir Y_AE sin targets (Y).")
            if self.grad_vals is None:
                raise ValueError("No se puede construir Y_AE sin gradientes físicos (activa compute_physics).")

            # ==========================
            # NUEVO INPUT DEL ENCODER:
            #   [ log1p(|∇Cp|), AoA, Mach, Pi ]
            # ==========================
            grad_cp = self.grad_vals[:, [0]]           # [N,1]
            log_grad_cp = np.log1p(grad_cp)           # [N,1]

            aoa  = self.X_raw[:, [7]]                  # AoA
            mach = self.X_raw[:, [6]]                  # Mach
            pi   = self.X_raw[:, [8]]                  # Pi

            self.Y_AE = np.hstack(
                [log_grad_cp, aoa, mach, pi]
            ).astype(np.float32)

            logger.info(
                f"[DATA] Construido Y_AE para AE con shape {self.Y_AE.shape} "
                f"(cols = log1p|∇Cp|, AoA, Mach, Pi)"
            )

    def save_scalers(self, save_dir: str | None = None):
        """Guarda los scalers disponibles."""
        if save_dir is None:
            save_dir = os.path.join(config.RESULTS_DIR, "scalers")
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.scaler_x, os.path.join(save_dir, "scaler_x.bin"))
        if "y" in self.scalers:
            joblib.dump(self.scalers["y"], os.path.join(save_dir, "scaler_y.bin"))
        if "grad" in self.scalers:
            joblib.dump(self.scalers["grad"], os.path.join(save_dir, "scaler_grad.bin"))

        logger.info(f"[DATA] Scalers guardados en {save_dir}")

    def get_loaders(self, shuffle_moe: bool | None = None):
        """
        Devuelve SOLO los loaders de entrenamiento (sin validación).
        - loader_ae: para el autoencoder (Y_AE)
        - loader_moe: para el MoE (X, Y_norm, placeholder)
        """
        if shuffle_moe is None:
            shuffle_moe = True if self.is_train else False

        ddp_on = dist.is_available() and dist.is_initialized()

        loader_ae = None
        loader_moe = None

        # Loader para AE (solo si hay targets y Y_AE)
        if self.has_targets and self.Y_AE is not None:
            dset_ae = TensorDataset(torch.from_numpy(self.Y_AE))
            sampler_ae = DistributedSampler(dset_ae, shuffle=True) if ddp_on else None
            loader_ae = DataLoader(
                dset_ae,
                batch_size=config.BATCH_SIZE,
                sampler=sampler_ae,
                shuffle=(sampler_ae is None),
                drop_last=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if 4 > 0 else False,
            )

        # Loader para MoE (placeholder; para nuestro nuevo MoE crearemos su propio loader en run.py)
        if self.has_targets:
            if self.Y_norm is not None:
                dset_moe = TensorDataset(
                    torch.from_numpy(self.X_final),
                    torch.from_numpy(self.Y_norm),
                    torch.from_numpy(np.zeros(len(self.Y_norm), dtype=np.float32))  # placeholder
                )
            else:
                dset_moe = TensorDataset(
                    torch.from_numpy(self.X_final),
                    torch.from_numpy(self.Y_raw),
                    torch.from_numpy(np.zeros(len(self.Y_raw), dtype=np.float32))
                )

            sampler_moe = DistributedSampler(dset_moe, shuffle=shuffle_moe) if ddp_on else None
            loader_moe = DataLoader(
                dset_moe,
                batch_size=config.BATCH_SIZE,
                sampler=sampler_moe,
                shuffle=(sampler_moe is None) and shuffle_moe,
                drop_last=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if 4 > 0 else False,
            )
        else:
            # Inferencia (solo X)
            dset_moe = TensorDataset(torch.from_numpy(self.X_final))
            loader_moe = DataLoader(
                dset_moe,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if 4 > 0 else False,
            )

        return loader_ae, loader_moe
