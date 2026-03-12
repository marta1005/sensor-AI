import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models import MachExpertNet  # tus expertos ya definidos en models.py


LATENT_DIM = 3   # espacio latente 3D para poder visualizarlo


class LatentGatedMoE(nn.Module):
    """
    Encoder + Gating sobre espacio latente z, con expertos fijos ya entrenados.

    - encoder: F_norm (4) -> z (LATENT_DIM=3)
    - gating : z -> logits(3) -> softmax -> pesos sobre [sub,trans,sup]
    - expertos: 3 x MachExpertNet (cargados de disco y congelados)
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_experts: int = 3, gating_input_dim: int = 4):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_experts = n_experts

        input_dim_expert = config.INPUT_DIM   # 9 (para los expertos)
        output_dim_expert = config.OUTPUT_DIM # 1 (Cp)

        # ----- Encoder desde F_norm (4) a z (3) -----
        self.encoder = nn.Sequential(
            nn.Linear(gating_input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )

        # ----- Gating desde z a logits(3) -----
        self.gating = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_experts),
        )

        # ----- Expertos: cargamos tus tres redes entrenadas y las congelamos -----
        self.expert_sub   = MachExpertNet().to(config.DEVICE)
        self.expert_trans = MachExpertNet().to(config.DEVICE)
        self.expert_sup   = MachExpertNet().to(config.DEVICE)

        # Cargar pesos
        sub_path   = os.path.join(config.MODEL_DIR, "expert_subsonic.pth")
        trans_path = os.path.join(config.MODEL_DIR, "expert_transonic.pth")
        sup_path   = os.path.join(config.MODEL_DIR, "expert_supersonic.pth")

        if not os.path.exists(sub_path):
            raise FileNotFoundError(f"No se encontró experto subsonic en {sub_path}")
        if not os.path.exists(trans_path):
            raise FileNotFoundError(f"No se encontró experto transonic en {trans_path}")
        if not os.path.exists(sup_path):
            raise FileNotFoundError(f"No se encontró experto supersonic en {sup_path}")

        self.expert_sub.load_state_dict(torch.load(sub_path, map_location=config.DEVICE))
        self.expert_trans.load_state_dict(torch.load(trans_path, map_location=config.DEVICE))
        self.expert_sup.load_state_dict(torch.load(sup_path, map_location=config.DEVICE))

        # Congelar parámetros de los expertos
        for p in self.expert_sub.parameters():
            p.requires_grad = False
        for p in self.expert_trans.parameters():
            p.requires_grad = False
        for p in self.expert_sup.parameters():
            p.requires_grad = False

    def forward(self, x, f, return_all=False):
        """
        x: [B, 9] (X_norm, para los expertos)
        f: [B, 4] (F_norm = [log(1+|gradCp|), AoA, Mach, Pi], para encoder/gating)

        Devuelve:
          y_pred: [B, 1]
          (opcional) gates: [B, 3], z: [B, 3], logits: [B, 3]
        """
        # Encoder sobre las features físicas
        z = self.encoder(f)  # [B, 3]

        # Gating
        logits = self.gating(z)          # [B, 3]
        gates = F.softmax(logits, dim=1) # [B, 3]

        # Expertos (fijos) usan la X completa
        y_sub   = self.expert_sub(x)     # [B, 1]
        y_trans = self.expert_trans(x)   # [B, 1]
        y_sup   = self.expert_sup(x)     # [B, 1]

        # Stack: [B, 1, 3]
        y_stack = torch.stack([y_sub, y_trans, y_sup], dim=2)

        # gates: [B, 3] -> [B, 1, 3]
        gates_expanded = gates.unsqueeze(1)

        # Mezcla: [B, 1]
        y_pred = (y_stack * gates_expanded).sum(dim=2)

        if return_all:
            return y_pred, gates, z, logits
        return y_pred