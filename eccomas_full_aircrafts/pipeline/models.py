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
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
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


class LegacyLatentGateNet(nn.Module):
    def __init__(
        self,
        gate_input_dim: int,
        latent_dim: int,
        n_experts: int,
        backbone_dim: int = 128,
        gating_hidden_dim: int = 128,
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
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.SiLU(),
            nn.Linear(backbone_dim // 2, latent_dim),
        )
        self.gating = nn.Sequential(
            nn.Linear(backbone_dim + latent_dim, gating_hidden_dim),
            nn.LayerNorm(gating_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim, gating_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(gating_hidden_dim // 2, n_experts),
        )

    def forward(self, gate_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(gate_features)
        z = self.encoder(hidden)
        logits = self.gating(torch.cat([hidden, z], dim=1))
        gates = F.softmax(logits, dim=1)
        return z, logits, gates


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


class LegacyLatentSensorMoE(nn.Module):
    def __init__(self, gate_input_dim: int, expert_input_dim: int, latent_dim: int, expert_paths: list[Path]):
        super().__init__()
        self.gate_net = LegacyLatentGateNet(gate_input_dim=gate_input_dim, latent_dim=latent_dim, n_experts=len(expert_paths))
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


class DiffusionTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10_000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / max(half_dim - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


class _TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class ResidualDiffusionUNet(nn.Module):
    def __init__(self, cond_channels: int, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_embed = DiffusionTimeEmbedding(time_dim)
        in_channels = cond_channels + 1  # noisy residual is concatenated with the pre-packed conditioning tensor

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.in_block = _TimeResBlock(in_channels, c1, time_dim)
        self.down1 = _TimeResBlock(c1, c2, time_dim)
        self.down2 = _TimeResBlock(c2, c3, time_dim)
        self.mid = _TimeResBlock(c3, c3, time_dim)
        self.up2 = _TimeResBlock(c3 + c2, c2, time_dim)
        self.up1 = _TimeResBlock(c2 + c1, c1, time_dim)
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, noisy_residual: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        x = torch.cat([noisy_residual, cond], dim=1)

        s1 = self.in_block(x, t_emb)
        d1 = self.down1(self.pool(s1), t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        mid = self.mid(d2, t_emb)

        up2 = F.interpolate(mid, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(torch.cat([up2, d1], dim=1), t_emb)
        up1 = F.interpolate(up2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat([up1, s1], dim=1), t_emb)
        return self.head(up1)
