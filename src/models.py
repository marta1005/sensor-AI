import torch
import torch.nn as nn
import numpy as np

from .config import LATENT_DIM
import logging

logger = logging.getLogger(__name__)


class VariationalLatentAutoencoder(nn.Module):
    """
    β-VAE sencillo para input_dim configurable.
    - forward(x, return_stats=True) → (recon, z, mu, logvar)
    - Usamos prior N(0,I); el clustering se hace luego con GMM.
    """
    def __init__(self, input_dim=4, latent_dim=LATENT_DIM, hidden_dims=(128, 64)):
        super().__init__()
        h1, h2 = hidden_dims

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
        )

        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.SiLU(),
            nn.Linear(h2, h1), nn.SiLU(),
            nn.Linear(h1, input_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_stats=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec = self.dec(z)

        if return_stats:
            return rec, z, mu, logvar
        return rec, z


class LatentAutoencoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 264), nn.ReLU(),
            nn.Linear(264, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 264), nn.ReLU(),
            nn.Linear(264, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class SymbolicSensorWrapper(nn.Module):
    """
    Wrapper para usar un modelo simbólico de gplearn dentro del MoE.
    - Evalúa gplearn en CPU (numpy).
    - Entrena solo scale (a) y bias (b) en PyTorch (calibración).
    """
    def __init__(self, gplearn_model, scaler_x=None, device=None):
        super().__init__()
        self.model = gplearn_model
        self.scaler_x = scaler_x
        self.device = device

        # Parámetros entrenables de calibración
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias  = nn.Parameter(torch.tensor(0.0))

        self.activation = nn.Sigmoid()

    @staticmethod
    def _logit(p, eps=1e-7):
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def forward(self, x_norm):
        x_cpu = x_norm.detach().cpu().numpy()

        if self.scaler_x is not None:
            x_cpu = self.scaler_x.inverse_transform(x_cpu)

        p_gp = self.model.predict_proba(x_cpu)[:, 1].astype(np.float32)

        z = self._logit(p_gp).astype(np.float32)

        z_t = torch.from_numpy(z).to(x_norm.device).unsqueeze(1)  # [B,1]
        return self.activation(self.scale * z_t + self.bias)


class MoE_Sensor(nn.Module):
    """
    Versión anterior basada en un sensor (posible wrapper simbólico).
    La dejamos por compatibilidad, aunque el nuevo pipeline usará ClusteredMoE.
    """
    def __init__(self, input_dim=9, output_dim=4, symbolic_sensor_module=None, scaler_x=None, device=None):
        super().__init__()
        
        # Sensor: Decide régimen de flujo
        if symbolic_sensor_module is not None:
            assert scaler_x is not None and device is not None, \
                "Si usas sensor simbólico con variables físicas, necesitas scaler_x y device."
            logger.info("[MODEL] MoE inicializado con SENSOR SIMBÓLICO (Interpretable).")
            self.sensor = SymbolicSensorWrapper(symbolic_sensor_module, scaler_x=scaler_x, device=device)
        else:
            logger.info("[MODEL] MoE inicializado con SENSOR NEURONAL (Black Box).")
            self.sensor = nn.Sequential(
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.SiLU(),
                nn.Linear(256, 128), nn.SiLU(),
                nn.Linear(128, 1), nn.Sigmoid() 
            )
        
        # Experto 1: Flujo Suave
        self.expert_smooth = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, output_dim)
        )
        
        # Experto 2: Onda de Choque (Tanh para gradientes fuertes)
        self.expert_shock = nn.Sequential(
            nn.Linear(input_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        prob = self.sensor(x)
        y_smooth = self.expert_smooth(x)
        y_shock = self.expert_shock(x)
        return (1 - prob) * y_smooth + prob * y_shock, prob


class ClusteredMoE(nn.Module):
    """
    Mixture of Experts guiado por GMM en el espacio latente.
    - K expertos, cada uno ve X_inf (9 vars normalizadas).
    - Gating produce pesos softmax \hat{γ}_{nk}.
    - Durante entrenamiento, el gating se entrena para imitar γ_teacher del GMM.
    """
    def __init__(self, input_dim=9, output_dim=4, n_experts=3, hidden_expert=256, hidden_gate=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_experts = n_experts

        # Gating network
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_gate),
            nn.SiLU(),
            nn.Linear(hidden_gate, n_experts)
        )

        # Expertos
        experts = []
        for k in range(n_experts):
            experts.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_expert),
                    nn.SiLU(),
                    nn.Linear(hidden_expert, hidden_expert),
                    nn.SiLU(),
                    nn.Linear(hidden_expert, output_dim)
                )
            )
        self.experts = nn.ModuleList(experts)

    def forward(self, x, return_gates=False):
        """
        x: [B, input_dim]
        returns:
            y_pred: [B, output_dim]
            gates : [B, n_experts] (si return_gates=True)
        """
        logits = self.gating(x)              # [B, K]
        gates = torch.softmax(logits, dim=1) # [B, K]

        # Expert outputs
        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x))    # [B, D]
        expert_stack = torch.stack(expert_outs, dim=2)  # [B, D, K]

        gates_unsq = gates.unsqueeze(1)      # [B, 1, K]
        y_pred = (expert_stack * gates_unsq).sum(dim=2)  # [B, D]

        if return_gates:
            return y_pred, gates
        return y_pred, gates
