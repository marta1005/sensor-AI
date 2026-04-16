from __future__ import annotations

import numpy as np


ENCODER_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "Mach",
    "Pi",
    "AoA_deg",
    "sin_AoA",
    "cos_AoA",
    "Mach_sq",
    "radius_yz",
    "Mach_Pi",
    "Mach_sin_AoA",
    "Mach_cos_AoA",
    "x_Mach",
    "y_Mach",
    "z_Mach",
    "Mach_nx",
    "Mach_ny",
    "Mach_nz",
]

EXPERT_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "Mach",
    "AoA_deg",
    "Pi",
    "sin_AoA",
    "cos_AoA",
    "Mach_sq",
    "radius_yz",
    "x_Mach",
    "y_Mach",
    "z_Mach",
    "Mach_nx",
    "Mach_ny",
    "Mach_nz",
]


def split_raw_columns(x_raw: np.ndarray) -> tuple[np.ndarray, ...]:
    x = x_raw[:, 0:1]
    y = x_raw[:, 1:2]
    z = x_raw[:, 2:3]
    nx = x_raw[:, 3:4]
    ny = x_raw[:, 4:5]
    nz = x_raw[:, 5:6]
    mach = x_raw[:, 6:7]
    aoa_deg = x_raw[:, 7:8]
    pi = x_raw[:, 8:9]
    return x, y, z, nx, ny, nz, mach, aoa_deg, pi


def build_encoder_features(x_raw: np.ndarray) -> np.ndarray:
    x, y, z, nx, ny, nz, mach, aoa_deg, pi = split_raw_columns(x_raw)
    aoa_rad = np.deg2rad(aoa_deg)
    sin_aoa = np.sin(aoa_rad)
    cos_aoa = np.cos(aoa_rad)
    mach_sq = mach * mach
    radius_yz = np.sqrt(y * y + z * z)

    feats = np.concatenate(
        [
            x,
            y,
            z,
            nx,
            ny,
            nz,
            mach,
            pi,
            aoa_deg,
            sin_aoa,
            cos_aoa,
            mach_sq,
            radius_yz,
            mach * pi,
            mach * sin_aoa,
            mach * cos_aoa,
            x * mach,
            y * mach,
            z * mach,
            mach * nx,
            mach * ny,
            mach * nz,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def build_expert_features(x_raw: np.ndarray) -> np.ndarray:
    x, y, z, nx, ny, nz, mach, aoa_deg, pi = split_raw_columns(x_raw)
    aoa_rad = np.deg2rad(aoa_deg)
    sin_aoa = np.sin(aoa_rad)
    cos_aoa = np.cos(aoa_rad)
    mach_sq = mach * mach
    radius_yz = np.sqrt(y * y + z * z)

    feats = np.concatenate(
        [
            x,
            y,
            z,
            nx,
            ny,
            nz,
            mach,
            aoa_deg,
            pi,
            sin_aoa,
            cos_aoa,
            mach_sq,
            radius_yz,
            x * mach,
            y * mach,
            z * mach,
            mach * nx,
            mach * ny,
            mach * nz,
        ],
        axis=1,
    )
    return feats.astype(np.float32)
