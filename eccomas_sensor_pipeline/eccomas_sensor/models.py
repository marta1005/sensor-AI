from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CpExpertNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (512, 256, 128), dropout: float = 0.10):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentGateNet(nn.Module):
    def __init__(self, gate_input_dim: int, latent_dim: int, n_experts: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.gating = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_experts),
        )

    def forward(self, gate_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(gate_features)
        logits = self.gating(z)
        gates = F.softmax(logits, dim=1)
        return z, logits, gates


class LatentSensorMoE(nn.Module):
    def __init__(self, gate_input_dim: int, expert_input_dim: int, latent_dim: int, expert_paths: list[Path]):
        super().__init__()
        self.gate_net = LatentGateNet(gate_input_dim=gate_input_dim, latent_dim=latent_dim, n_experts=len(expert_paths))
        self.experts = nn.ModuleList([CpExpertNet(input_dim=expert_input_dim) for _ in expert_paths])

        for expert, expert_path in zip(self.experts, expert_paths):
            state = torch.load(expert_path, map_location="cpu")
            expert.load_state_dict(state)
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()

    def forward(self, expert_features: torch.Tensor, gate_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, logits, gates = self.gate_net(gate_features)
        expert_outputs = [expert(expert_features) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=2)
        mixed = (stacked * gates.unsqueeze(1)).sum(dim=2)
        return mixed, z, logits, gates
