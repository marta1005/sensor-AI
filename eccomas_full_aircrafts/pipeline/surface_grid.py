from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import FullAircraftConfig


@dataclass(frozen=True)
class CompactSurfaceGrid:
    height: int
    width: int
    n_points: int
    row_idx: np.ndarray
    col_idx: np.ndarray
    valid_mask: np.ndarray

    @classmethod
    def from_reference(cls, cfg: FullAircraftConfig) -> "CompactSurfaceGrid":
        ref = np.load(cfg.surface_reference_path(cfg.reduced_surface))
        x_bin = np.asarray(ref["x_bin"], dtype=np.int64)
        y_bin = np.asarray(ref["y_bin"], dtype=np.int64)
        unique_x, col_idx = np.unique(x_bin, return_inverse=True)
        unique_y, row_idx = np.unique(y_bin, return_inverse=True)
        height = int(unique_y.shape[0])
        width = int(unique_x.shape[0])
        valid_mask = np.zeros((height, width), dtype=np.float32)
        valid_mask[row_idx, col_idx] = 1.0
        return cls(
            height=height,
            width=width,
            n_points=int(x_bin.shape[0]),
            row_idx=row_idx.astype(np.int64),
            col_idx=col_idx.astype(np.int64),
            valid_mask=valid_mask,
        )

    def scatter_numpy(self, flat_values: np.ndarray) -> np.ndarray:
        if flat_values.ndim == 2:
            channels = int(flat_values.shape[1])
            grid = np.zeros((channels, self.height, self.width), dtype=np.float32)
            grid[:, self.row_idx, self.col_idx] = flat_values.T.astype(np.float32)
            return grid
        if flat_values.ndim == 3:
            batch, _, channels = flat_values.shape
            grid = np.zeros((batch, channels, self.height, self.width), dtype=np.float32)
            for idx in range(batch):
                grid[idx, :, self.row_idx, self.col_idx] = flat_values[idx].T.astype(np.float32)
            return grid
        raise ValueError(f"Expected [N,C] or [B,N,C], got {flat_values.shape}")

    def gather_numpy(self, grid_values: np.ndarray) -> np.ndarray:
        if grid_values.ndim == 3:
            return grid_values[:, self.row_idx, self.col_idx].T.astype(np.float32)
        if grid_values.ndim == 4:
            batch = grid_values.shape[0]
            gathered = np.zeros((batch, self.n_points, grid_values.shape[1]), dtype=np.float32)
            for idx in range(batch):
                gathered[idx] = grid_values[idx, :, self.row_idx, self.col_idx].T.astype(np.float32)
            return gathered
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {grid_values.shape}")

    def mask_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.from_numpy(self.valid_mask[None, ...])
        if device is not None:
            mask = mask.to(device=device)
        return mask
