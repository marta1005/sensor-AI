from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class IndexedCpDataset(Dataset):
    def __init__(self, features_path: str, cp_path: str, indices: np.ndarray | None = None):
        self.features = np.load(features_path, mmap_mode="r")
        self.cp = np.load(cp_path, mmap_mode="r")
        self.indices = np.arange(self.features.shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        feat = torch.from_numpy(np.array(self.features[row], dtype=np.float32, copy=True))
        cp = torch.from_numpy(np.array(self.cp[row], dtype=np.float32, copy=True))
        return feat, cp


class WeightedIndexedCpDataset(Dataset):
    def __init__(self, features_path: str, cp_path: str, indices: np.ndarray, weights: np.ndarray):
        self.features = np.load(features_path, mmap_mode="r")
        self.cp = np.load(cp_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float32)
        if self.indices.shape[0] != self.weights.shape[0]:
            raise ValueError("indices and weights must have the same length")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        feat = torch.from_numpy(np.array(self.features[row], dtype=np.float32, copy=True))
        cp = torch.from_numpy(np.array(self.cp[row], dtype=np.float32, copy=True))
        weight = torch.tensor(float(self.weights[index]), dtype=torch.float32)
        return feat, cp, weight


class LatentMoEDataset(Dataset):
    def __init__(
        self,
        expert_path: str,
        gate_path: str,
        cp_path: str,
        expert_id_path: str,
        indices: np.ndarray | None = None,
    ):
        self.expert_features = np.load(expert_path, mmap_mode="r")
        self.gate_features = np.load(gate_path, mmap_mode="r")
        self.cp = np.load(cp_path, mmap_mode="r")
        self.expert_id = np.load(expert_id_path, mmap_mode="r")
        self.indices = np.arange(self.expert_features.shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        expert_feat = torch.from_numpy(np.array(self.expert_features[row], dtype=np.float32, copy=True))
        gate_feat = torch.from_numpy(np.array(self.gate_features[row], dtype=np.float32, copy=True))
        cp = torch.from_numpy(np.array(self.cp[row], dtype=np.float32, copy=True))
        expert_id = torch.tensor(int(self.expert_id[row]), dtype=torch.long)
        return expert_feat, gate_feat, cp, expert_id
