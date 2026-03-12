"""
Entrenamiento de la pipeline condicionada en features físicas:

  X_norm (9) -> expertos (fijos)
  F_norm (4 = [log(1+|gradCp|), AoA, Mach, Pi]) -> encoder -> z(3) -> gating(z) -> pesos expertos

Usa:
  - Datos: X_cut_train / Y_cut_train / gradCp_cut_train de config.DATA_DIR
  - Scalers: scaler_x.bin / scaler_y.bin / scaler_f.bin
  - Expertos pre-entrenados: expert_subsonic.pth, expert_transonic.pth, expert_supersonic.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from moe_labeled_loader import make_global_labeled_loader_with_f
from moe_latent_models import LatentGatedMoE


# Hiperparámetros específicos del MoE latente
LR_MOE       = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS_MOE   = 80
LAMBDA_GATE  = 0.5   # peso de la CrossEntropy de gating


def train_moe_latent():
    device = config.DEVICE
    print(f"[MoE-LS] Entrenando LatentGatedMoE (condicionado en gradCp, AoA, Mach, Pi) en {device}")

    # Loader global con X_norm, Y_norm, expert_id, F_norm
    loader = make_global_labeled_loader_with_f(
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # Modelo
    moe = LatentGatedMoE().to(device)

    # Optimizador y scheduler
    optimizer = optim.AdamW(
        moe.parameters(),
        lr=LR_MOE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_MOE)

    # Pérdidas
    reg_criterion = nn.SmoothL1Loss(beta=1.0)  # sobre Cp normalizado
    gate_criterion = nn.CrossEntropyLoss()     # sobre expert_id

    history_total = []
    history_reg   = []
    history_gate  = []

    for epoch in range(1, EPOCHS_MOE + 1):
        moe.train()
        epoch_loss      = 0.0
        epoch_loss_reg  = 0.0
        epoch_loss_gate = 0.0
        n_batches       = 0

        pbar = tqdm(loader, desc=f"[MoE-LS] Epoch {epoch}/{EPOCHS_MOE}")
        for bx, by, bexp, bf in pbar:
            bx   = bx.to(device, non_blocking=True)     # [B,9]
            by   = by.to(device, non_blocking=True)     # [B,1]
            bexp = bexp.to(device, non_blocking=True)   # [B]
            bf   = bf.to(device, non_blocking=True)     # [B,4]

            optimizer.zero_grad(set_to_none=True)

            y_pred, gates, z, logits = moe(bx, bf, return_all=True)

            # Pérdida de regresión en Cp normalizado
            loss_reg = reg_criterion(y_pred, by)

            # Pérdida de gating (clasificación de experto desde z)
            loss_gate = gate_criterion(logits, bexp)

            loss = loss_reg + LAMBDA_GATE * loss_gate
            loss.backward()
            optimizer.step()

            epoch_loss      += loss.item()
            epoch_loss_reg  += loss_reg.item()
            epoch_loss_gate += loss_gate.item()
            n_batches       += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                reg=f"{loss_reg.item():.4f}",
                gate=f"{loss_gate.item():.4f}"
            )

        avg_loss      = epoch_loss / max(1, n_batches)
        avg_loss_reg  = epoch_loss_reg / max(1, n_batches)
        avg_loss_gate = epoch_loss_gate / max(1, n_batches)

        history_total.append(avg_loss)
        history_reg.append(avg_loss_reg)
        history_gate.append(avg_loss_gate)

        scheduler.step()

        print(f"[MoE-LS] Epoch {epoch:03d} | "
              f"loss={avg_loss:.6f} | reg={avg_loss_reg:.6f} | gate={avg_loss_gate:.6f}")

    # Guardar modelo
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    moe_path = os.path.join(config.MODEL_DIR, "latent_gated_moe_phys.pth")
    torch.save(moe.state_dict(), moe_path)
    print(f"[MoE-LS] Modelo LatentGatedMoE guardado en: {moe_path}")

    # Guardar curvas
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history_total, label="Total")
    plt.plot(history_reg, label="Reg Cp")
    plt.plot(history_gate, label="Gate CE")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Entrenamiento LatentGatedMoE (físico)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = os.path.join(config.RESULTS_DIR, "loss_moe_latent_phys.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[MoE-LS] Curvas de loss guardadas en: {fig_path}")


def main():
    train_moe_latent()


if __name__ == "__main__":
    main()