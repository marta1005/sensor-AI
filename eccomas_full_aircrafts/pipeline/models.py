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


class FullAircraftShockSplitUNet(nn.Module):
    """Shared global backbone with a compact latent bottleneck and two local experts."""

    def __init__(self, input_channels: int, base_channels: int = 16, latent_dim: int = 4):
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
        self.to_latent = nn.Conv2d(c4, latent_dim, kernel_size=1)
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, c4, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(c4), c4),
            nn.SiLU(),
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(latent_dim), latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, 1, kernel_size=1),
        )
        self.up3 = _UpBlock(c4, c4, c3)
        self.up2 = _UpBlock(c3, c3, c2)
        self.up1 = _UpBlock(c2, c2, c1)
        self.head = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.in_block(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bottleneck(s4)
        latent = self.to_latent(b)
        alpha_logits = self.alpha_head(latent)
        x = self.from_latent(latent)
        x = self.up3(x, s4)
        x = self.up2(x, s3)
        x = self.up1(x, s2)
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = x + s1
        heads = self.head(x)
        alpha_logits = F.interpolate(alpha_logits, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        return heads[:, 0:1], heads[:, 1:2], alpha_logits, latent


class _MeshMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, hidden_dim = x.shape
        n_edges = int(edge_src.shape[0])
        src_state = x[:, edge_src, :]
        dst_state = x[:, edge_dst, :]
        edge_feat = edge_attr.unsqueeze(0).expand(batch, n_edges, edge_attr.shape[1])
        msg_in = torch.cat([src_state, dst_state, edge_feat], dim=-1)
        msg = self.message_mlp(msg_in)

        agg = x.new_zeros((batch, n_nodes, hidden_dim))
        batch_offsets = (torch.arange(batch, device=x.device, dtype=torch.long).view(batch, 1) * n_nodes)
        flat_dst = (edge_dst.view(1, n_edges) + batch_offsets).reshape(-1)
        agg_flat = agg.reshape(batch * n_nodes, hidden_dim)
        agg_flat.index_add_(0, flat_dst, msg.reshape(batch * n_edges, hidden_dim))
        agg = agg_flat.view(batch, n_nodes, hidden_dim)

        updated = self.update_mlp(torch.cat([x, agg], dim=-1))
        return self.out_norm(x + updated)


class FullAircraftMeshTeacher(nn.Module):
    """Graph teacher with a local latent bottleneck and optional shock-gated Cp residual."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        latent_dim: int = 4,
        edge_dim: int = 6,
        message_passing_steps: int = 6,
        dropout: float = 0.05,
        use_shock_residual: bool = False,
    ):
        super().__init__()
        self.use_shock_residual = bool(use_shock_residual)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [_MeshMessagePassingLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(message_passing_steps)]
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        head_input_dim = hidden_dim + latent_dim
        self.cp_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2 if self.use_shock_residual else 1),
        )
        self.shock_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.node_encoder(node_features)
        for layer in self.layers:
            hidden = layer(hidden, edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_attr)
        latent = self.to_latent(hidden)
        head_input = torch.cat([hidden, latent], dim=-1)
        shock_logits = self.shock_head(head_input)
        cp_raw = self.cp_head(head_input)
        if self.use_shock_residual:
            cp_base = cp_raw[..., 0:1]
            cp_residual = cp_raw[..., 1:2]
            cp = cp_base + torch.sigmoid(shock_logits) * cp_residual
        else:
            cp = cp_raw
        return cp, shock_logits, latent


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
