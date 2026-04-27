from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class FullAircraftConfig:
    project_root: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    pipeline_root: Optional[Path] = None
    outputs_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    reduced_data_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    scalers_dir: Optional[Path] = None
    features_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None
    latent_dir: Optional[Path] = None
    inference_dir: Optional[Path] = None
    sensor_dir: Optional[Path] = None
    symbolic_dir: Optional[Path] = None
    diffusion_dir: Optional[Path] = None

    raw_points_per_condition: int = 260_774
    input_dim_raw: int = 9
    output_dim_cp: int = 1
    latent_dim: int = 4
    n_experts: int = 3
    cp_column: int = 0
    expert_partition_mode: str = "hybrid"
    cluster_algorithm: str = "kmeans"
    cluster_count: int = 3
    hybrid_source_clusters: int = 5

    x_bins: int = 1080
    y_bins: int = 540
    reduced_surface: str = "upper"

    expert_batch_size: int = 32_768
    expert_field_batch_size: int = 2
    latent_batch_size: int = 32_768
    num_workers: int = 0

    expert_epochs: int = 80
    latent_epochs: int = 24

    expert_model_architecture: str = "unet2d_compact_v1"
    expert_unet_base_channels: int = 16
    expert_lr: float = 1e-3
    latent_lr: float = 1e-3
    weight_decay: float = 1e-5
    latent_gate_architecture: str = "latent_only_v1"
    latent_gate_weight: float = 0.55
    latent_hard_gate_weight: float = 0.02
    latent_entropy_weight: float = 0.01
    latent_gate_noise_std: float = 0.01
    routing_soft_temperature: float = 0.12
    routing_oracle_margin: float = 0.015
    latent_train_max_samples: int = 2_000_000
    latent_test_max_samples: int = 500_000

    mach_sub_max: float = 0.65
    mach_trans_max: float = 0.85
    expert_overlap_margin: float = 0.05
    expert_overlap_min_weight: float = 0.25

    plot_sample_size: int = 40_000
    latent_plot_sample_size: int = 120_000
    sensor_max_samples: int = 250_000
    sensor_blend_width: float = 0.02
    sensor_ridge_alpha: float = 1e-2
    sensor_feature_clip: float = 8.0

    diffusion_train_mode: str = "symbolic"
    diffusion_batch_size: int = 2
    diffusion_epochs: int = 60
    diffusion_lr: float = 2e-4
    diffusion_weight_decay: float = 1e-5
    diffusion_timesteps: int = 200
    diffusion_sample_steps: int = 50
    diffusion_base_channels: int = 32
    diffusion_cond_feature_dim: int = 15
    diffusion_shock_weight: float = 2.0

    def __post_init__(self) -> None:
        default_project_root = Path(__file__).resolve().parents[2]
        project_root = self._normalize_path(self.project_root or default_project_root)
        pipeline_root = self._normalize_path(self.pipeline_root or (project_root / "eccomas_full_aircrafts"))
        raw_data_dir = self._normalize_path(self.raw_data_dir or (project_root / "data"))
        outputs_root = self._normalize_path(self.outputs_dir or (pipeline_root / "outputs"))
        results_root = self._normalize_path(self.results_dir or (pipeline_root / "results"))
        surface_outputs_root = outputs_root / self.reduced_surface

        plots_dir = self._normalize_path(self.plots_dir or (surface_outputs_root / "plots"))
        reduced_data_dir = self._normalize_path(self.reduced_data_dir or (surface_outputs_root / "reduced_data"))
        models_dir = self._normalize_path(self.models_dir or (surface_outputs_root / "models"))
        scalers_dir = self._normalize_path(self.scalers_dir or (surface_outputs_root / "scalers"))
        features_dir = self._normalize_path(self.features_dir or (surface_outputs_root / "features"))
        metrics_dir = self._normalize_path(self.metrics_dir or (surface_outputs_root / "metrics"))
        latent_dir = self._normalize_path(self.latent_dir or (surface_outputs_root / "latents"))
        inference_dir = self._normalize_path(self.inference_dir or (surface_outputs_root / "inference"))
        sensor_dir = self._normalize_path(self.sensor_dir or (surface_outputs_root / "sensor"))
        symbolic_dir = self._normalize_path(self.symbolic_dir or (surface_outputs_root / "symbolic"))
        diffusion_dir = self._normalize_path(self.diffusion_dir or (surface_outputs_root / "diffusion"))

        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "pipeline_root", pipeline_root)
        object.__setattr__(self, "raw_data_dir", raw_data_dir)
        object.__setattr__(self, "outputs_dir", outputs_root)
        object.__setattr__(self, "plots_dir", plots_dir)
        object.__setattr__(self, "results_dir", results_root)
        object.__setattr__(self, "reduced_data_dir", reduced_data_dir)
        object.__setattr__(self, "models_dir", models_dir)
        object.__setattr__(self, "scalers_dir", scalers_dir)
        object.__setattr__(self, "features_dir", features_dir)
        object.__setattr__(self, "metrics_dir", metrics_dir)
        object.__setattr__(self, "latent_dir", latent_dir)
        object.__setattr__(self, "inference_dir", inference_dir)
        object.__setattr__(self, "sensor_dir", sensor_dir)
        object.__setattr__(self, "symbolic_dir", symbolic_dir)
        object.__setattr__(self, "diffusion_dir", diffusion_dir)

    @staticmethod
    def _normalize_path(path: Path) -> Path:
        return Path(path).expanduser().resolve()

    @property
    def surfaces_dir(self) -> Path:
        return self.outputs_dir / "surfaces"

    @property
    def metadata_dir(self) -> Path:
        return self.outputs_dir / "metadata"

    @property
    def surface_outputs_dir(self) -> Path:
        return self.outputs_dir / self.reduced_surface

    @property
    def shared_results_dir(self) -> Path:
        return self.results_dir / "shared"

    def results_surface_dir(self, surface: str | None = None) -> Path:
        surface_name = surface or self.reduced_surface
        return self.results_dir / surface_name

    @property
    def cut_data_dir(self) -> Path:
        return self.reduced_data_dir

    @property
    def cut_threshold_y(self) -> float:
        return 0.0

    @property
    def device(self):
        if torch is None:
            raise ImportError("torch is required for training commands.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def surface_reference_path(self, surface: str | None = None) -> Path:
        surface_name = surface or self.reduced_surface
        return self.surfaces_dir / f"{surface_name}_surface_reference.npz"

    def ensure_dirs(self) -> None:
        for path in [
            self.pipeline_root,
            self.outputs_dir,
            self.surface_outputs_dir,
            self.plots_dir,
            self.results_dir,
            self.shared_results_dir,
            self.results_surface_dir(),
            self.reduced_data_dir,
            self.models_dir,
            self.scalers_dir,
            self.features_dir,
            self.metrics_dir,
            self.latent_dir,
            self.inference_dir,
            self.sensor_dir,
            self.symbolic_dir,
            self.diffusion_dir,
            self.surfaces_dir,
            self.metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
