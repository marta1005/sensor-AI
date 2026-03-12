import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

import config 
from models import MachExpertNet


def train_single_expert(X, Y, regime_name: str):
    """
    Entrena un experto específico (subsonic / transonic / supersonic)
    sobre los datos X, Y (ya normalizados, Y solo Cp).
    Guarda el modelo en config.MODEL_DIR con nombre expert_{regime}.pth
    """
    if X is None or Y is None or len(X) == 0:
        print(f"[WARN] No hay datos para el régimen {regime_name}. Saltando entrenamiento.")
        return None

    device = config.DEVICE
    print(f"\n[TRAIN] Entrenando experto para régimen '{regime_name}' en {device} | N={len(X)}")

    model = MachExpertNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    # CosineAnnealing en LR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # En vez de MSE pura, usamos SmoothL1 (Huber) para suavizar outliers
    criterion = nn.SmoothL1Loss(beta=1.0)  # beta=1 => transición a L2

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    history = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"[{regime_name}] Epoch {epoch}/{config.EPOCHS}")
        for bx, by in pbar:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(bx)
            loss = criterion(y_pred, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / max(1, n_batches)
        history.append(avg_loss)
        scheduler.step()

        print(f"[{regime_name}] Epoch {epoch:03d} | loss={avg_loss:.6e}")

    # Guardar modelo
    model_path = os.path.join(config.MODEL_DIR, f"expert_{regime_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[TRAIN] Experto '{regime_name}' guardado en: {model_path}")

    # Guardar curva de entrenamiento
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Época")
    plt.ylabel("SmoothL1 Loss (Cp normalizado)")
    plt.title(f"Loss entrenamiento - {regime_name}")
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join(config.RESULTS_DIR, f"loss_{regime_name}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[TRAIN] Curva de loss guardada en: {fig_path}")

    return model