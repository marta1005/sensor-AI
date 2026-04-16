from __future__ import annotations

from eccomas_sensor_pipeline.eccomas_sensor.train_latent import train_latent_pipeline as _train_latent_pipeline

from .config import FullAircraftConfig


def train_latent_pipeline(cfg: FullAircraftConfig) -> None:
    _train_latent_pipeline(cfg)
