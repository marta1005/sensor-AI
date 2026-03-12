import torch
import torch.nn as nn
import torch.nn.functional as F
import config 


class MachExpertNet(nn.Module):
    """
    MLP más potente para predecir Cp a partir de:
      [x,y,z,nx,ny,nz,Mach,AoA,Pi] normalizados.

    Arquitectura:
      in -> 512 -> 512 -> 256 -> 128 -> 1
    Con LayerNorm + SiLU + Dropout(0.1) en cada capa oculta.
    """
    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        hidden_sizes=(512, 512, 256, 128),
        dropout: float = 0.10
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # Inicialización tipo Kaiming para capas lineales
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Necesitamos importar math para la init
import math