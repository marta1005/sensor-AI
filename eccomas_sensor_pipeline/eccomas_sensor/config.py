from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - allows non-training commands without torch
    torch = None


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    cut_data_dir: Optional[Path] = None
    pipeline_root: Optional[Path] = None
    outputs_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    scalers_dir: Optional[Path] = None
    features_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None
    latent_dir: Optional[Path] = None
    symbolic_dir: Optional[Path] = None

    raw_points_per_condition: int = 260_774
    cut_threshold_y: float = 12.0
    keep_ge_threshold: bool = True

    input_dim_raw: int = 9
    output_dim_cp: int = 1
    latent_dim: int = 3
    n_experts: int = 3

    expert_batch_size: int = 16_384
    latent_batch_size: int = 16_384
    num_workers: int = 4

    expert_epochs: int = 80
    latent_epochs: int = 60

    expert_lr: float = 1e-3
    latent_lr: float = 1e-3
    weight_decay: float = 1e-5
    latent_gate_weight: float = 0.35

    mach_sub_max: float = 0.65
    mach_trans_max: float = 0.85

    plot_sample_size: int = 40_000
    latent_plot_sample_size: int = 120_000
    symbolic_max_samples: int = 250_000

    def __post_init__(self) -> None:
        default_project_root = Path(__file__).resolve().parents[2]
        project_root = self._normalize_path(self.project_root or default_project_root)
        pipeline_root = self._normalize_path(self.pipeline_root or (project_root / "eccomas_sensor_pipeline"))
        raw_data_dir = self._normalize_path(self.raw_data_dir or (project_root / "data"))
        cut_data_dir = self._normalize_path(self.cut_data_dir or (project_root / "data_cut_y12"))
        outputs_dir = self._normalize_path(self.outputs_dir or (pipeline_root / "outputs"))
        plots_dir = self._normalize_path(self.plots_dir or (outputs_dir / "plots"))
        models_dir = self._normalize_path(self.models_dir or (outputs_dir / "models"))
        scalers_dir = self._normalize_path(self.scalers_dir or (outputs_dir / "scalers"))
        features_dir = self._normalize_path(self.features_dir or (outputs_dir / "features"))
        metrics_dir = self._normalize_path(self.metrics_dir or (outputs_dir / "metrics"))
        latent_dir = self._normalize_path(self.latent_dir or (outputs_dir / "latents"))
        symbolic_dir = self._normalize_path(self.symbolic_dir or (outputs_dir / "symbolic"))

        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "pipeline_root", pipeline_root)
        object.__setattr__(self, "raw_data_dir", raw_data_dir)
        object.__setattr__(self, "cut_data_dir", cut_data_dir)
        object.__setattr__(self, "outputs_dir", outputs_dir)
        object.__setattr__(self, "plots_dir", plots_dir)
        object.__setattr__(self, "models_dir", models_dir)
        object.__setattr__(self, "scalers_dir", scalers_dir)
        object.__setattr__(self, "features_dir", features_dir)
        object.__setattr__(self, "metrics_dir", metrics_dir)
        object.__setattr__(self, "latent_dir", latent_dir)
        object.__setattr__(self, "symbolic_dir", symbolic_dir)

    @staticmethod
    def _normalize_path(path: Path) -> Path:
        return Path(path).expanduser().resolve()

    @property
    def device(self):
        if torch is None:
            raise ImportError("torch is required for training commands.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def cut_suffix(self) -> str:
        return "y12"

    def ensure_dirs(self) -> None:
        for path in [
            self.cut_data_dir,
            self.outputs_dir,
            self.plots_dir,
            self.models_dir,
            self.scalers_dir,
            self.features_dir,
            self.metrics_dir,
            self.latent_dir,
            self.symbolic_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
