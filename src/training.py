import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist

import matplotlib.pyplot as plt
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)


# ===== Utilidades DDP =====

def _ddp_on():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _ddp_on() else 0


def _is_main():
    return _rank() == 0


def _reduce_mean(val: float) -> float:
    if not _ddp_on():
        return val
    t = torch.tensor(val, device=config.DEVICE, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()


# ===== KL VAE estándar (prior N(0,I)) =====

def kl_divergence(mu, logvar):
    """KL divergence promedio por batch contra N(0,I)."""
    var = logvar.exp()
    kl = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - logvar, dim=1)
    return kl.mean()


# ===== Entrenamiento Autoencoder / VAE =====

def train_autoencoder(model, loader_train):
    """
    Entrena AE o β-VAE sin validación.
    - Si el modelo tiene .encode → se asume VAE.
    - Loader de entrada: Y_AE (shape [N, input_dim]).
    """
    opt = optim.AdamW(
        model.parameters(),
        lr=config.LR_AE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(opt, T_max=config.EPOCHS_AE)
    mse = nn.MSELoss(reduction="mean")

    history_train: list[float] = []

    use_vae = hasattr(model, "encode")
    beta_max = config.VAE_BETA if use_vae else 0.0
    warmup = config.VAE_WARMUP_EPOCHS

    if _is_main():
        kind = "β-VAE" if use_vae else "AE"
        logger.info(
            f"\n[AE TRAIN] Iniciando entrenamiento {kind} "
            f"({config.EPOCHS_AE} épocas) - latent_dim={config.LATENT_DIM}"
        )

    for epoch in range(1, config.EPOCHS_AE + 1):
        model.train()
        if hasattr(loader_train, "sampler") and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        beta = beta_max * min(1.0, epoch / max(1, warmup)) if use_vae else 0.0

        total_loss = 0.0
        n_batches = 0
        last_kl_val = 0.0

        train_iter = tqdm(
            loader_train,
            desc=f"AE Epoch {epoch}/{config.EPOCHS_AE}",
            disable=not _is_main()
        )

        for batch in train_iter:
            # AE loader: (data,)
            batch = batch[0].to(config.DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_vae:
                rec, z, mu, logvar = model(batch, return_stats=True)
                recon_loss = mse(rec, batch)
                kl = kl_divergence(mu, logvar)
                loss = recon_loss + beta * kl
                last_kl_val = kl.item()
            else:
                rec, _ = model(batch)
                recon_loss = mse(rec, batch)
                loss = recon_loss

            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

            if _is_main():
                postfix = {"loss": f"{loss.item():.6f}"}
                if use_vae:
                    postfix["beta"] = f"{beta:.3f}"
                    postfix["KL"] = f"{last_kl_val:.4f}"
                train_iter.set_postfix(**postfix)

        avg_loss = total_loss / max(1, n_batches)
        avg_loss = _reduce_mean(avg_loss)
        history_train.append(avg_loss)

        scheduler.step()

        if _is_main() and ((epoch % 2 == 0) or (epoch == config.EPOCHS_AE)):
            if use_vae:
                logger.info(
                    f"[AE TRAIN] Epoch {epoch:3d}/{config.EPOCHS_AE} | "
                    f"Train loss={avg_loss:.6f} | beta={beta:.3f} | KL≈{last_kl_val:.4f}"
                )
            else:
                logger.info(
                    f"[AE TRAIN] Epoch {epoch:3d}/{config.EPOCHS_AE} | "
                    f"Train loss={avg_loss:.6f}"
                )

    # Gráfico final
    if _is_main():
        os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(history_train, label='Train loss')
        plt.title(f"{'β-VAE' if use_vae else 'AE'} - Pérdida de entrenamiento")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(config.RESULTS_DIR, "figures", "loss_ae.png"),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()

    logger.info(f"[AE TRAIN] Finalizado. Última loss: {history_train[-1]:.6f}")
    return history_train


# ===== Entrenamiento Mixture of Experts GUIADO POR GMM =====

def train_moe_clustered(model, loader_train, lambda_gate: float = 0.1):
    """
    Entrena un Mixture of Experts con K expertos guiado por pesos γ_teacher del GMM.
    loader_train debe devolver batches de (x, y, gamma_teacher):
      - x: [B, 9]   (X_inf normalizado)
      - y: [B, 4]   (Y_norm)
      - gamma_teacher: [B, K] (pesos de GMM por punto)
    """
    opt = optim.AdamW(model.parameters(), lr=config.LR_MOE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=config.EPOCHS_MOE)

    mse_reg = nn.MSELoss(reduction="mean")

    history_train: list[float] = []

    if _is_main():
        logger.info(f"\n[MoE TRAIN] Iniciando entrenamiento ClusteredMoE ({config.EPOCHS_MOE} épocas)")

    for epoch in range(1, config.EPOCHS_MOE + 1):
        model.train()
        if hasattr(loader_train, "sampler") and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        total_loss = 0.0
        n_batches = 0

        train_iter = tqdm(
            loader_train,
            desc=f"MoE Epoch {epoch}/{config.EPOCHS_MOE}",
            disable=not _is_main()
        )

        for batch in train_iter:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                raise ValueError("Se espera batch de 3 elementos: (x, y, gamma_teacher)")

            bx, by, bgamma = batch
            bx = bx.to(config.DEVICE, non_blocking=True)
            by = by.to(config.DEVICE, non_blocking=True)
            bgamma = bgamma.to(config.DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            y_pred, gates = model(bx, return_gates=True)

            reg_loss = mse_reg(y_pred, by)
            gate_loss = torch.mean((gates - bgamma) ** 2)

            loss = reg_loss + lambda_gate * gate_loss
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

            if _is_main():
                train_iter.set_postfix(
                    loss=f"{loss.item():.6f}",
                    reg=f"{reg_loss.item():.6f}",
                    gate=f"{gate_loss.item():.6f}"
                )

        avg_loss = total_loss / max(1, n_batches)
        avg_loss = _reduce_mean(avg_loss)
        history_train.append(avg_loss)

        scheduler.step()

        if _is_main() and ((epoch % 5 == 0) or (epoch == config.EPOCHS_MOE)):
            logger.info(f"[MoE TRAIN] Epoch {epoch:3d}/{config.EPOCHS_MOE} | Train loss: {avg_loss:.6f}")

    # Gráfico final
    if _is_main():
        os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(history_train, label='Train loss')
        plt.title("Clustered Mixture of Experts - Pérdida total")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(config.RESULTS_DIR, "figures", "loss_moe_clustered.png"),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()

    logger.info(f"[MoE TRAIN] Finalizado. Última loss: {history_train[-1]:.6f}")
    return history_train
