from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import FullAircraftConfig


@dataclass(frozen=True)
class CompactSurfaceGraph:
    height: int
    width: int
    n_points: int
    row_idx: np.ndarray
    col_idx: np.ndarray
    coords: np.ndarray
    normals: np.ndarray
    valid_mask: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_attr: np.ndarray

    @classmethod
    def from_reference(cls, cfg: FullAircraftConfig) -> "CompactSurfaceGraph":
        ref = np.load(cfg.surface_reference_path(cfg.reduced_surface))
        x_bin = np.asarray(ref["x_bin"], dtype=np.int64)
        y_bin = np.asarray(ref["y_bin"], dtype=np.int64)
        unique_x, col_idx = np.unique(x_bin, return_inverse=True)
        unique_y, row_idx = np.unique(y_bin, return_inverse=True)
        height = int(unique_y.shape[0])
        width = int(unique_x.shape[0])
        valid_mask = np.zeros((height, width), dtype=np.float32)
        valid_mask[row_idx, col_idx] = 1.0

        coords = np.column_stack(
            [
                np.asarray(ref["x"], dtype=np.float32),
                np.asarray(ref["y"], dtype=np.float32),
                np.asarray(ref["z"], dtype=np.float32),
            ]
        ).astype(np.float32)
        normals = np.column_stack(
            [
                np.asarray(ref["nx"], dtype=np.float32),
                np.asarray(ref["ny"], dtype=np.float32),
                np.asarray(ref["nz"], dtype=np.float32),
            ]
        ).astype(np.float32)

        lookup = {(int(r), int(c)): idx for idx, (r, c) in enumerate(zip(row_idx.tolist(), col_idx.tolist()))}
        offsets: list[tuple[int, int]] = []
        for dc in cfg.mesh_graph_x_dilations:
            step = int(abs(dc))
            if step == 0:
                continue
            offsets.append((0, -step))
            offsets.append((0, step))
        offsets.extend([(-1, 0), (1, 0)])
        if cfg.mesh_graph_include_diagonals:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        offsets = list(dict.fromkeys(offsets))
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_attr: list[list[float]] = []
        edge_keys: set[tuple[int, int]] = set()

        def _add_edge(src_idx: int, dst_idx: int, dr: int, dc: int) -> None:
            if src_idx == dst_idx or (src_idx, dst_idx) in edge_keys:
                return
            edge_keys.add((src_idx, dst_idx))
            src_xyz = coords[src_idx]
            dst_xyz = coords[dst_idx]
            delta = dst_xyz - src_xyz
            delta_xy = delta[:2]
            distance = float(np.linalg.norm(delta_xy))
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_attr.append(
                [
                    float(delta[0]),
                    float(delta[1]),
                    0.0,
                    distance,
                    float(dr),
                    float(dc),
                ]
            )

        rows = row_idx.tolist()
        cols = col_idx.tolist()
        for src_idx, (r, c) in enumerate(zip(rows, cols)):
            for dr, dc in offsets:
                dst_idx = lookup.get((int(r + dr), int(c + dc)))
                if dst_idx is not None:
                    _add_edge(src_idx, dst_idx, int(dr), int(dc))

        for row in np.unique(row_idx):
            row_points = np.flatnonzero(row_idx == row)
            ordered = row_points[np.argsort(col_idx[row_points])]
            for left, right in zip(ordered[:-1], ordered[1:]):
                dc = int(col_idx[right] - col_idx[left])
                _add_edge(int(left), int(right), 0, dc)
                _add_edge(int(right), int(left), 0, -dc)

        for col in np.unique(col_idx):
            col_points = np.flatnonzero(col_idx == col)
            ordered = col_points[np.argsort(row_idx[col_points])]
            for lower, upper in zip(ordered[:-1], ordered[1:]):
                dr = int(row_idx[upper] - row_idx[lower])
                _add_edge(int(lower), int(upper), dr, 0)
                _add_edge(int(upper), int(lower), -dr, 0)

        return cls(
            height=height,
            width=width,
            n_points=int(coords.shape[0]),
            row_idx=row_idx.astype(np.int64),
            col_idx=col_idx.astype(np.int64),
            coords=coords,
            normals=normals,
            valid_mask=valid_mask,
            edge_src=np.asarray(edge_src, dtype=np.int64),
            edge_dst=np.asarray(edge_dst, dtype=np.int64),
            edge_attr=np.asarray(edge_attr, dtype=np.float32),
        )

    def edge_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = torch.from_numpy(self.edge_src).to(device=device, dtype=torch.long)
        dst = torch.from_numpy(self.edge_dst).to(device=device, dtype=torch.long)
        attr = torch.from_numpy(self.edge_attr).to(device=device, dtype=torch.float32)
        return src, dst, attr

    def scatter_numpy(self, flat_values: np.ndarray) -> np.ndarray:
        if flat_values.ndim != 2:
            raise ValueError(f"Expected [N,C], got {flat_values.shape}")
        channels = int(flat_values.shape[1])
        grid = np.zeros((channels, self.height, self.width), dtype=np.float32)
        grid[:, self.row_idx, self.col_idx] = flat_values.T.astype(np.float32)
        return grid

    def gather_numpy(self, grid_values: np.ndarray) -> np.ndarray:
        if grid_values.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got {grid_values.shape}")
        gathered = np.asarray(grid_values[:, self.row_idx, self.col_idx], dtype=np.float32)
        return gathered.T.astype(np.float32)

    def scatter_tensor(self, flat_values: torch.Tensor) -> torch.Tensor:
        if flat_values.ndim != 3:
            raise ValueError(f"Expected [B,N,C], got {tuple(flat_values.shape)}")
        batch, _, channels = flat_values.shape
        grid = flat_values.new_zeros((batch, channels, self.height, self.width))
        row_idx = torch.from_numpy(self.row_idx).to(device=flat_values.device, dtype=torch.long)
        col_idx = torch.from_numpy(self.col_idx).to(device=flat_values.device, dtype=torch.long)
        grid[:, :, row_idx, col_idx] = flat_values.permute(0, 2, 1)
        return grid

    def mask_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.from_numpy(self.valid_mask.astype(np.float32))[None, None, ...]
        if device is not None:
            mask = mask.to(device=device)
        return mask
