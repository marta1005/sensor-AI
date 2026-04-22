from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eccomas_sensor_pipeline.eccomas_sensor.models import CpExpertNet


class FullAircraftLatentGateNet(nn.Module):
    """Gate net whose expert routing depends only on a compact latent code."""

    def __init__(
        self,
        gate_input_dim: int,
        latent_dim: int,
        n_experts: int,
        backbone_dim: int = 96,
        latent_hidden_dim: int = 48,
        gating_hidden_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(gate_input_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.SiLU(),
            nn.Linear(backbone_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(backbone_dim, latent_hidden_dim),
            nn.SiLU(),
            nn.Linear(latent_hidden_dim, latent_dim),
        )
        self.gating = nn.Sequential(
            nn.Linear(latent_dim, gating_hidden_dim),
            nn.LayerNorm(gating_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim, n_experts),
        )

    def forward(self, gate_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(gate_features)
        z = self.encoder(hidden)
        logits = self.gating(z)
        gates = F.softmax(logits, dim=1)
        return z, logits, gates


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = _ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class FullAircraftExpertUNet(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 16):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        self.in_block = _ConvBlock(input_channels, c1)
        self.down1 = _DownBlock(c1, c2)
        self.down2 = _DownBlock(c2, c3)
        self.down3 = _DownBlock(c3, c4)
        self.bottleneck = _ConvBlock(c4, c4)
        self.up3 = _UpBlock(c4, c4, c3)
        self.up2 = _UpBlock(c3, c3, c2)
        self.up1 = _UpBlock(c2, c2, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.in_block(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bottleneck(s4)
        x = self.up3(b, s4)
        x = self.up2(x, s3)
        x = self.up1(x, s2)
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = x + s1
        return self.head(x)


class FullAircraftLatentMixer(nn.Module):
    def __init__(self, gate_input_dim: int, latent_dim: int, n_experts: int):
        super().__init__()
        self.gate_net = FullAircraftLatentGateNet(
            gate_input_dim=gate_input_dim,
            latent_dim=latent_dim,
            n_experts=n_experts,
        )

    def forward(
        self,
        expert_stack: torch.Tensor,
        gate_features: torch.Tensor,
        return_expert_stack: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, logits, gates = self.gate_net(gate_features)
        mixed = (expert_stack * gates.unsqueeze(1)).sum(dim=2)
        if return_expert_stack:
            return mixed, z, logits, gates, expert_stack
        return mixed, z, logits, gates


class FullAircraftLatentSensorMoE(nn.Module):
    def __init__(self, gate_input_dim: int, expert_input_dim: int, latent_dim: int, expert_paths: list[Path]):
        super().__init__()
        self.gate_net = FullAircraftLatentGateNet(
            gate_input_dim=gate_input_dim,
            latent_dim=latent_dim,
            n_experts=len(expert_paths),
        )
        self.experts = nn.ModuleList([CpExpertNet(input_dim=expert_input_dim) for _ in expert_paths])

        for expert, expert_path in zip(self.experts, expert_paths):
            state = torch.load(expert_path, map_location="cpu")
            expert.load_state_dict(state)
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for expert in self.experts:
            expert.eval()
        return self

    def expert_stack(self, expert_features: torch.Tensor) -> torch.Tensor:
        expert_outputs = [expert(expert_features) for expert in self.experts]
        return torch.stack(expert_outputs, dim=2)

    def forward(
        self,
        expert_features: torch.Tensor,
        gate_features: torch.Tensor,
        return_expert_stack: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, logits, gates = self.gate_net(gate_features)
        stacked = self.expert_stack(expert_features)
        mixed = (stacked * gates.unsqueeze(1)).sum(dim=2)
        if return_expert_stack:
            return mixed, z, logits, gates, stacked
        return mixed, z, logits, gates
