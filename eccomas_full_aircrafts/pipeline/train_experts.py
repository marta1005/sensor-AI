from __future__ import annotations

from eccomas_sensor_pipeline.eccomas_sensor.train_experts import train_all_experts as _train_all_experts

from .config import FullAircraftConfig


def train_all_experts(cfg: FullAircraftConfig) -> None:
    _train_all_experts(cfg)
