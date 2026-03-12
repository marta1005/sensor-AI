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


class LatentMoEDataset(Dataset):
    def __init__(self, expert_path: str, gate_path: str, cp_path: str, expert_id_path: str):
        self.expert_features = np.load(expert_path, mmap_mode="r")
        self.gate_features = np.load(gate_path, mmap_mode="r")
        self.cp = np.load(cp_path, mmap_mode="r")
        self.expert_id = np.load(expert_id_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.expert_features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_feat = torch.from_numpy(np.array(self.expert_features[index], dtype=np.float32, copy=True))
        gate_feat = torch.from_numpy(np.array(self.gate_features[index], dtype=np.float32, copy=True))
        cp = torch.from_numpy(np.array(self.cp[index], dtype=np.float32, copy=True))
        expert_id = torch.tensor(int(self.expert_id[index]), dtype=torch.long)
        return expert_feat, gate_feat, cp, expert_id
