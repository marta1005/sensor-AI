from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PipelineConfig
from .inference import REGIME_NAMES, predict_array
from .utils import regime_from_mach, sample_indices, save_json

_PLOT_CACHE = Path(__file__).resolve().parents[1] / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _load_split_arrays(cfg: PipelineConfig, split: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    metadata_path = cfg.cut_data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cut-data metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if split not in metadata:
        raise ValueError(f"Split '{split}' not found in {metadata_path}")

    rows_per_condition = int(metadata[split]["kept_per_condition"])
    n_conditions = int(metadata[split]["n_conditions"])

    x_path = cfg.cut_data_dir / f"X_cut_{split}.npy"
    y_path = cfg.cut_data_dir / f"Y_cut_{split}.npy"
    x_split = np.load(x_path, mmap_mode="r")
    y_split = np.load(y_path, mmap_mode="r")
    return x_split, y_split, rows_per_condition, n_conditions


def _condition_metadata(x_split: np.ndarray, rows_per_condition: int) -> dict[str, np.ndarray]:
    first_rows = np.asarray(x_split[::rows_per_condition, :9], dtype=np.float32)
    mach = first_rows[:, 6]
    aoa = first_rows[:, 7]
    pi = first_rows[:, 8]
    regime_id = regime_from_mach(mach, sub_max=0.65, trans_max=0.85)
    return {
        "mach": mach,
        "aoa": aoa,
        "pi": pi,
        "regime_id": regime_id,
    }


def _select_representative_conditions(cond_meta: dict[str, np.ndarray]) -> list[int]:
    selected: list[int] = []
    for regime_id in range(3):
        candidates = np.flatnonzero(cond_meta["regime_id"] == regime_id)
        if candidates.size == 0:
            continue
        regime_mach = cond_meta["mach"][candidates]
        target_mach = float(np.median(regime_mach))
        score = (
            np.abs(cond_meta["mach"][candidates] - target_mach)
            + 0.12 * np.abs(cond_meta["aoa"][candidates])
            + 0.08 * np.abs(cond_meta["pi"][candidates] - 2.0)
        )
        selected.append(int(candidates[np.argmin(score)]))
    return selected


def _condition_slice(condition_index: int, rows_per_condition: int) -> slice:
    start = int(condition_index) * rows_per_condition
    end = start + rows_per_condition
    return slice(start, end)


def _mae_rmse(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    residual = pred - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return mae, rmse


def _scatter_surface(ax, xyz: np.ndarray, cp: np.ndarray, title: str, cmap: str, vmin: float, vmax: float):
    scatter = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=cp,
        s=0.45,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-118)
    return scatter


def _tb_camera_limits(xyz: np.ndarray) -> dict[str, object]:
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]
    return {
        "elevation": 90.0,
        "azimuth": 0.0,
        "zoom": 1.9,
        "xlim": [float(np.min(x_vals)), float(np.max(x_vals))],
        "ylim": [float(np.min(y_vals)), float(np.max(y_vals))],
        "zlim": [float(np.min(z_vals)), float(np.max(z_vals))],
        "xoffsets": [0.0, 0.0],
        "yoffsets": [0.0, 4.0],
        "zoffsets": [0.0, 0.0],
    }


def _split_surface_sides(xyz: np.ndarray, n_span_bins: int = 140) -> tuple[np.ndarray, np.ndarray]:
    y_vals = xyz[:, 1].astype(np.float32)
    edges = np.linspace(float(y_vals.min()), float(y_vals.max()), n_span_bins + 1, dtype=np.float32)
    upper_mask = np.zeros(xyz.shape[0], dtype=bool)
    lower_mask = np.zeros(xyz.shape[0], dtype=bool)

    for idx in range(n_span_bins):
        in_bin = (y_vals >= edges[idx]) & (y_vals < edges[idx + 1] if idx < n_span_bins - 1 else y_vals <= edges[idx + 1])
        bin_rows = np.flatnonzero(in_bin)
        if bin_rows.size < 12:
            continue

        points_xz = xyz[bin_rows][:, [0, 2]].astype(np.float32)
        try:
            _, _, _, _, _, signed_dist = _section_chord_frame(points_xz)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue

        upper_rows = bin_rows[signed_dist > 0.0]
        lower_rows = bin_rows[signed_dist < 0.0]
        upper_mask[upper_rows] = True
        lower_mask[lower_rows] = True

    # Fallback for rare unresolved rows near the chord line.
    unresolved = ~(upper_mask | lower_mask)
    if np.any(unresolved):
        median_z = np.median(xyz[:, 2])
        upper_mask[unresolved] = xyz[unresolved, 2] >= median_z
        lower_mask[unresolved] = ~upper_mask[unresolved]

    return upper_mask, lower_mask


def _upper_surface_projection(
    xyz: np.ndarray,
    values: np.ndarray,
    max_points: int = 35_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper_mask, _ = _split_surface_sides(xyz)
    xyz_upper = xyz[upper_mask]
    values_upper_raw = values[upper_mask]

    x_proj = xyz_upper[:, 0].astype(np.float32)
    y_proj = xyz_upper[:, 1].astype(np.float32)
    values_upper = values_upper_raw.astype(np.float32)

    if x_proj.size > max_points:
        idx = sample_indices(x_proj.size, max_points, seed=123)
        x_proj = x_proj[idx]
        y_proj = y_proj[idx]
        values_upper = values_upper[idx]

    return x_proj, y_proj, values_upper.astype(np.float32)


def _upper_surface_grid(
    xyz: np.ndarray,
    values: np.ndarray,
    max_cells: int = 18_000,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    upper_mask, _ = _split_surface_sides(xyz)
    xyz_upper = xyz[upper_mask]
    values_upper_raw = values[upper_mask]

    x_all = xyz_upper[:, 0].astype(np.float32)
    y_all = xyz_upper[:, 1].astype(np.float32)
    z_all = xyz_upper[:, 2].astype(np.float32)
    values_all = values_upper_raw.astype(np.float32)

    x_span = max(float(np.ptp(x_all)), 1e-6)
    y_span = max(float(np.ptp(y_all)), 1e-6)
    aspect = max(x_span / y_span, 1e-3)
    nx = int(np.clip(np.sqrt(max_cells * aspect), 90, 280))
    ny = int(np.clip(np.sqrt(max_cells / aspect), 60, 220))

    x_edges = np.linspace(float(np.min(x_all)), float(np.max(x_all)), nx + 1, dtype=np.float32)
    y_edges = np.linspace(float(np.min(y_all)), float(np.max(y_all)), ny + 1, dtype=np.float32)
    x_bin = np.clip(np.searchsorted(x_edges, x_all, side="right") - 1, 0, nx - 1)
    y_bin = np.clip(np.searchsorted(y_edges, y_all, side="right") - 1, 0, ny - 1)
    flat_bin = (y_bin * nx + x_bin).astype(np.int32)

    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    for bin_id in np.unique(flat_bin):
        rows = np.flatnonzero(flat_bin == bin_id)
        if rows.size == 0:
            continue

        z_cell = z_all[rows]
        z_threshold = float(np.max(z_cell) - max(0.01 * np.ptp(z_cell), 1e-5))
        top_rows = rows[z_cell >= z_threshold]
        if top_rows.size == 0:
            top_rows = rows[np.argmax(z_cell)][None]

        cell_y = int(bin_id // nx)
        cell_x = int(bin_id % nx)
        grid[cell_y, cell_x] = float(np.median(values_all[top_rows]))

    grid_masked = np.ma.masked_invalid(grid)
    return x_edges, y_edges, grid_masked


def _add_upper_surface_metric(
    fig: plt.Figure,
    subplot_spec,
    xyz: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    cmap,
    vmin: float,
    vmax: float,
) -> None:
    gs_metric = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[18, 1], hspace=0.03)
    ax = fig.add_subplot(gs_metric[0])
    x_proj, y_proj, values_upper = _upper_surface_projection(xyz, values)
    point_size = 3.4 if x_proj.size < 20_000 else 2.4
    artist = ax.scatter(
        x_proj,
        y_proj,
        c=values_upper,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
        alpha=0.95,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10, pad=4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    cax = fig.add_subplot(gs_metric[1])
    cbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, size=8)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_label_position("top")


def _add_upper_surface_metric_surface(
    fig: plt.Figure,
    subplot_spec,
    xyz: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    cmap,
    vmin: float,
    vmax: float,
) -> None:
    gs_metric = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[18, 1], hspace=0.03)
    ax = fig.add_subplot(gs_metric[0])
    x_edges, y_edges, grid = _upper_surface_grid(xyz, values)
    artist = ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=True,
    )
    ax.set_title(title, fontsize=10, pad=4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    cax = fig.add_subplot(gs_metric[1])
    cbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, size=8)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_label_position("top")


def _create_conference_style_cp_figure(
    cfg: PipelineConfig,
    condition_index: int,
    xyz: np.ndarray,
    cp_truth: np.ndarray,
    cp_symbolic: np.ndarray,
    mach: float,
    aoa: float,
    pi: float,
) -> Path:
    fig = plt.figure(figsize=(12, 4))
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.22)
    cp_cmap = plt.get_cmap("jet")
    err_cmap = plt.get_cmap("RdBu_r")

    cp_vmin = float(min(cp_truth.min(), cp_symbolic.min()))
    cp_vmax = float(max(cp_truth.max(), cp_symbolic.max()))
    err = cp_symbolic - cp_truth
    err_lim = float(max(np.max(np.abs(err)), 1e-6))

    _add_upper_surface_metric(
        fig,
        outer[0],
        xyz,
        cp_truth,
        title="CFD Upper Surface",
        cbar_label=r"$C_p$",
        cmap=cp_cmap,
        vmin=cp_vmin,
        vmax=cp_vmax,
    )
    _add_upper_surface_metric(
        fig,
        outer[1],
        xyz,
        cp_symbolic,
        title="Symbolic Upper Surface",
        cbar_label=r"$C_p$",
        cmap=cp_cmap,
        vmin=cp_vmin,
        vmax=cp_vmax,
    )
    _add_upper_surface_metric(
        fig,
        outer[2],
        xyz,
        err,
        title="Upper-Surface Signed Error",
        cbar_label=r"$C_{p,\mathrm{sym}} - C_{p,\mathrm{CFD}}$",
        cmap=err_cmap,
        vmin=-err_lim,
        vmax=err_lim,
    )

    fig.suptitle(rf"$p_i$ = {pi:.1f} 10⁵, $M_{{\infty}}$ = {mach:.2f}, AoA = {aoa:.1f}°", fontsize=12)
    fig.tight_layout(h_pad=2.0)

    out_path = cfg.results_dir / f"cp_field_truth_symbolic_error_tb_cond{condition_index}.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _create_upper_surface_cp_figure(
    cfg: PipelineConfig,
    condition_index: int,
    xyz: np.ndarray,
    cp_truth: np.ndarray,
    cp_symbolic: np.ndarray,
    mach: float,
    aoa: float,
    pi: float,
) -> Path:
    fig = plt.figure(figsize=(12, 4))
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.22)
    cp_cmap = plt.get_cmap("jet")
    err_cmap = plt.get_cmap("RdBu_r")

    cp_vmin = float(min(cp_truth.min(), cp_symbolic.min()))
    cp_vmax = float(max(cp_truth.max(), cp_symbolic.max()))
    err = cp_symbolic - cp_truth
    err_lim = float(max(np.max(np.abs(err)), 1e-6))

    _add_upper_surface_metric_surface(
        fig,
        outer[0],
        xyz,
        cp_truth,
        title="CFD Upper Surface",
        cbar_label=r"$C_p$",
        cmap=cp_cmap,
        vmin=cp_vmin,
        vmax=cp_vmax,
    )
    _add_upper_surface_metric_surface(
        fig,
        outer[1],
        xyz,
        cp_symbolic,
        title="Symbolic Upper Surface",
        cbar_label=r"$C_p$",
        cmap=cp_cmap,
        vmin=cp_vmin,
        vmax=cp_vmax,
    )
    _add_upper_surface_metric_surface(
        fig,
        outer[2],
        xyz,
        err,
        title="Upper-Surface Signed Error",
        cbar_label=r"$C_{p,\mathrm{sym}} - C_{p,\mathrm{CFD}}$",
        cmap=err_cmap,
        vmin=-err_lim,
        vmax=err_lim,
    )

    fig.suptitle(rf"$p_i$ = {pi:.1f} 10⁵, $M_{{\infty}}$ = {mach:.2f}, AoA = {aoa:.1f}°", fontsize=12)
    fig.tight_layout(h_pad=2.0)

    out_path = cfg.results_dir / f"cp_field_truth_symbolic_error_surface_cond{condition_index}.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _section_chord_frame(points_xz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points_xz.ndim != 2 or points_xz.shape[0] < 3 or points_xz.shape[1] != 2:
        raise ValueError("Need at least three 2D points to build a section chord frame.")
    if not np.all(np.isfinite(points_xz)):
        raise ValueError("Non-finite points found in section geometry.")

    center = points_xz.mean(axis=0).astype(np.float32)
    _, _, vh = np.linalg.svd(points_xz - center[None, :], full_matrices=False)
    chord_axis = vh[0].astype(np.float32)
    if chord_axis[0] < 0.0:
        chord_axis *= -1.0

    with np.errstate(invalid="raise", over="raise", divide="raise"):
        proj = (points_xz - center[None, :]) @ chord_axis
    span = float(proj.max() - proj.min())
    if not np.isfinite(span) or span <= 1e-8:
        raise ValueError("Degenerate section span while computing chord frame.")
    edge_tol = max(0.015 * span, 1e-4)

    le_mask = proj <= proj.min() + edge_tol
    te_mask = proj >= proj.max() - edge_tol
    leading_edge = np.median(points_xz[le_mask], axis=0).astype(np.float32)
    trailing_edge = np.median(points_xz[te_mask], axis=0).astype(np.float32)

    chord_vec = trailing_edge - leading_edge
    chord_len = max(float(np.linalg.norm(chord_vec)), 1e-8)
    if not np.isfinite(chord_len) or chord_len <= 1e-8:
        raise ValueError("Degenerate chord length while computing chord frame.")
    chord_hat = (chord_vec / chord_len).astype(np.float32)
    normal_hat = np.asarray([-chord_hat[1], chord_hat[0]], dtype=np.float32)

    with np.errstate(invalid="raise", over="raise", divide="raise"):
        signed_dist = (points_xz - leading_edge[None, :]) @ normal_hat
    # Keep positive distances on the geometrically upper side of the section.
    if np.corrcoef(points_xz[:, 1], signed_dist)[0, 1] < 0.0:
        normal_hat *= -1.0
        signed_dist *= -1.0

    with np.errstate(invalid="raise", over="raise", divide="raise"):
        chord_coord = (points_xz - leading_edge[None, :]) @ chord_hat
    x_over_c = np.clip(chord_coord / chord_len, 0.0, 1.0).astype(np.float32)
    return leading_edge, trailing_edge, chord_hat, normal_hat, x_over_c, signed_dist.astype(np.float32)


def _extract_section_branches(
    x_condition: np.ndarray,
    y_target: float,
    bandwidth: float,
    bins: int,
    surface_fraction: float = 0.10,
    min_points: int = 900,
    exact_y: bool = False,
) -> dict[str, np.ndarray | int]:
    y_distance_full = np.abs(x_condition[:, 1] - y_target).astype(np.float32)
    base_halfwidth = bandwidth / 2.0
    if exact_y:
        mask = np.isclose(x_condition[:, 1], y_target, rtol=0.0, atol=1e-7)
        adaptive_halfwidth = 0.0
    else:
        mask = y_distance_full <= base_halfwidth
        adaptive_halfwidth = float(base_halfwidth)

        if int(np.count_nonzero(mask)) < min_points:
            kth = min(max(min_points, 1), x_condition.shape[0])
            adaptive_halfwidth = float(np.partition(y_distance_full, kth - 1)[kth - 1])
            mask = y_distance_full <= adaptive_halfwidth

    if not np.any(mask):
        return {
            "mask": mask,
            "raw_xz": np.empty((0, 2), dtype=np.float32),
            "upper_x": np.empty((0,), dtype=np.float32),
            "upper_z": np.empty((0,), dtype=np.float32),
            "lower_x": np.empty((0,), dtype=np.float32),
            "lower_z": np.empty((0,), dtype=np.float32),
            "upper_rows": [],
            "lower_rows": [],
            "sample_weights": np.empty((0,), dtype=np.float32),
            "n_points": 0,
            "halfwidth": adaptive_halfwidth,
            "exact_y": bool(exact_y),
        }

    points = x_condition[mask]
    points_xz = points[:, [0, 2]].astype(np.float32)
    if points_xz.shape[0] < 3:
        return {
            "mask": mask,
            "raw_xz": points_xz,
            "x_norm": np.empty((0,), dtype=np.float32),
            "z": np.empty((0,), dtype=np.float32),
            "upper_x": np.empty((0,), dtype=np.float32),
            "upper_z": np.empty((0,), dtype=np.float32),
            "lower_x": np.empty((0,), dtype=np.float32),
            "lower_z": np.empty((0,), dtype=np.float32),
            "upper_rows": [],
            "lower_rows": [],
            "sample_weights": np.ones((points_xz.shape[0],), dtype=np.float32),
            "n_points": int(points_xz.shape[0]),
            "halfwidth": adaptive_halfwidth,
            "exact_y": bool(exact_y),
        }
    y_distance = y_distance_full[mask]
    sigma = max(adaptive_halfwidth * 0.55, base_halfwidth * 0.35, 1e-4)
    sample_weights = np.exp(-0.5 * (y_distance / sigma) ** 2).astype(np.float32)
    global_upper_mask, global_lower_mask = _split_surface_sides(np.asarray(x_condition[:, :3], dtype=np.float32))
    upper_mask_local = global_upper_mask[mask]
    lower_mask_local = global_lower_mask[mask]

    leading_edge, trailing_edge, chord_hat, normal_hat, x_norm, signed_dist = _section_chord_frame(points_xz)
    edges = np.linspace(0.0, 1.0, bins + 1)

    upper_x: list[float] = []
    upper_z: list[float] = []
    lower_x: list[float] = []
    lower_z: list[float] = []
    upper_rows_list: list[np.ndarray] = []
    lower_rows_list: list[np.ndarray] = []
    for idx in range(bins):
        in_bin = (x_norm >= edges[idx]) & (x_norm < edges[idx + 1] if idx < bins - 1 else x_norm <= edges[idx + 1])
        if not np.any(in_bin):
            continue

        bin_rows = np.flatnonzero(in_bin)
        if bin_rows.size < 2:
            continue

        # Keep the two sides strictly on opposite sides of the local chord line.
        upper_pool = bin_rows[upper_mask_local[bin_rows]]
        lower_pool = bin_rows[lower_mask_local[bin_rows]]
        if upper_pool.size == 0 or lower_pool.size == 0:
            upper_pool = bin_rows[signed_dist[bin_rows] > 0.0]
            lower_pool = bin_rows[signed_dist[bin_rows] < 0.0]
        if upper_pool.size == 0 or lower_pool.size == 0:
            continue

        # Use only the outer envelopes of each side in this x/c bin.
        k_upper = min(upper_pool.size, max(3, int(np.ceil(upper_pool.size * surface_fraction))))
        k_lower = min(lower_pool.size, max(3, int(np.ceil(lower_pool.size * surface_fraction))))

        upper_order = np.argsort(signed_dist[upper_pool])
        lower_order = np.argsort(signed_dist[lower_pool])
        upper_rows = upper_pool[upper_order[-k_upper:]]
        lower_rows = lower_pool[lower_order[:k_lower]]

        upper_w = sample_weights[upper_rows]
        lower_w = sample_weights[lower_rows]

        lower_x.append(float(np.average(x_norm[lower_rows], weights=lower_w)))
        lower_z.append(float(np.average(signed_dist[lower_rows], weights=lower_w)))
        lower_rows_list.append(lower_rows.astype(np.int32))

        upper_x.append(float(np.average(x_norm[upper_rows], weights=upper_w)))
        upper_z.append(float(np.average(signed_dist[upper_rows], weights=upper_w)))
        upper_rows_list.append(upper_rows.astype(np.int32))

    return {
        "mask": mask,
        "x_norm": x_norm.astype(np.float32),
        "z": signed_dist.astype(np.float32),
        "upper_x": np.asarray(upper_x, dtype=np.float32),
        "upper_z": np.asarray(upper_z, dtype=np.float32),
        "lower_x": np.asarray(lower_x, dtype=np.float32),
        "lower_z": np.asarray(lower_z, dtype=np.float32),
        "upper_rows": upper_rows_list,
        "lower_rows": lower_rows_list,
        "sample_weights": sample_weights,
        "leading_edge": leading_edge,
        "trailing_edge": trailing_edge,
        "chord_hat": chord_hat,
        "normal_hat": normal_hat,
        "n_points": int(np.count_nonzero(mask)),
        "halfwidth": adaptive_halfwidth,
        "exact_y": bool(exact_y),
        "raw_xz": points_xz,
    }


def _branch_cp_curve(
    cp_values: np.ndarray,
    section_geometry: dict[str, np.ndarray | int],
    branch: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp_section = cp_values[np.asarray(section_geometry["mask"])]
    sample_weights = np.asarray(section_geometry["sample_weights"], dtype=np.float32)
    x_key = f"{branch}_x"
    z_key = f"{branch}_z"
    rows_key = f"{branch}_rows"

    x_branch = np.asarray(section_geometry[x_key], dtype=np.float32)
    z_branch = np.asarray(section_geometry[z_key], dtype=np.float32)
    branch_rows = list(section_geometry[rows_key])
    cp_branch = np.asarray(
        [
            float(np.average(cp_section[rows], weights=sample_weights[rows]))
            for rows in branch_rows
        ],
        dtype=np.float32,
    )
    return x_branch, z_branch, cp_branch


def _trim_merged_edge_points(
    x_branch: np.ndarray,
    z_branch: np.ndarray,
    cp_branch: np.ndarray,
    merge_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_branch.size == 0:
        return x_branch, z_branch, cp_branch

    start = 0
    end = x_branch.size
    while start < end and abs(float(z_branch[start])) < merge_tol:
        start += 1
    while end > start and abs(float(z_branch[end - 1])) < merge_tol:
        end -= 1

    return x_branch[start:end], z_branch[start:end], cp_branch[start:end]


def _assemble_full_arc_curve(
    lower_x: np.ndarray,
    lower_z: np.ndarray,
    lower_cp: np.ndarray,
    upper_x: np.ndarray,
    upper_z: np.ndarray,
    upper_cp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    if lower_x.size == 0 and upper_x.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), None

    lower_order = np.argsort(lower_x)[::-1]
    upper_order = np.argsort(upper_x)

    x_arc = np.concatenate([lower_x[lower_order], upper_x[upper_order]]).astype(np.float32)
    z_arc = np.concatenate([lower_z[lower_order], upper_z[upper_order]]).astype(np.float32)
    cp_arc = np.concatenate([lower_cp[lower_order], upper_cp[upper_order]]).astype(np.float32)

    if x_arc.size == 1:
        return np.asarray([0.0], dtype=np.float32), z_arc, cp_arc, None

    ds = np.sqrt(np.diff(x_arc) ** 2 + np.diff(z_arc) ** 2).astype(np.float32)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s /= max(float(s[-1]), 1e-8)

    lower_count = int(lower_x.size)
    join_s = float(s[lower_count - 1]) if lower_count > 0 else None
    return s.astype(np.float32), z_arc, cp_arc, join_s


def _detect_shock_location(x_curve: np.ndarray, cp_curve: np.ndarray) -> float | None:
    if x_curve.size < 7:
        return None

    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(cp_curve, kernel, mode="same")
    gradient = np.gradient(smooth, x_curve)
    valid = (x_curve >= 0.15) & (x_curve <= 0.90) & (smooth <= -0.15)
    if not np.any(valid):
        return None

    valid_idx = np.flatnonzero(valid)
    best = valid_idx[int(np.argmax(gradient[valid]))]
    if float(gradient[best]) <= 0.0:
        return None
    return float(x_curve[best])


def _pick_transonic_condition(condition_indices: Iterable[int], cond_meta: dict[str, np.ndarray]) -> int | None:
    for condition_index in condition_indices:
        if int(cond_meta["regime_id"][condition_index]) == 1:
            return int(condition_index)

    transonic = np.flatnonzero(cond_meta["regime_id"] == 1)
    if transonic.size == 0:
        return None
    regime_mach = cond_meta["mach"][transonic]
    target_mach = float(np.median(regime_mach))
    score = (
        np.abs(cond_meta["mach"][transonic] - target_mach)
        + 0.12 * np.abs(cond_meta["aoa"][transonic])
        + 0.08 * np.abs(cond_meta["pi"][transonic] - 2.0)
    )
    return int(transonic[np.argmin(score)])


def _transonic_profile_score(
    x_condition: np.ndarray,
    cp_truth: np.ndarray,
    cut_y_values: Iterable[float],
    cut_bandwidth: float,
    profile_bins: int,
) -> float:
    best_score = -np.inf
    for y_target in cut_y_values:
        section_geometry = _extract_section_branches(x_condition, float(y_target), cut_bandwidth, profile_bins)
        upper_x, _, upper_cp = _branch_cp_curve(cp_truth, section_geometry, "upper")
        lower_x, _, lower_cp = _branch_cp_curve(cp_truth, section_geometry, "lower")
        if upper_x.size < 20 or lower_x.size < 20:
            continue

        shock_x = _detect_shock_location(upper_x, upper_cp)
        if shock_x is None:
            continue

        lift_sep = float(np.mean(-upper_cp) - np.mean(-lower_cp))
        smooth = np.convolve(upper_cp, np.asarray([0.2, 0.6, 0.2], dtype=np.float32), mode="same")
        shock_strength = float(np.max(np.gradient(smooth, upper_x)))
        mid_bonus = max(0.0, 1.0 - abs(float(shock_x) - 0.6) / 0.4)
        score = max(lift_sep, 0.0) + 0.01 * shock_strength + 0.4 * mid_bonus
        best_score = max(best_score, score)

    return float(best_score)


def _pick_shock_transonic_condition(
    x_split: np.ndarray,
    y_split: np.ndarray,
    rows_per_condition: int,
    cond_meta: dict[str, np.ndarray],
    cut_y_values: Iterable[float],
    cut_bandwidth: float,
    profile_bins: int,
) -> int | None:
    transonic = np.flatnonzero(cond_meta["regime_id"] == 1)
    if transonic.size == 0:
        return None

    best_condition: int | None = None
    best_score = -np.inf
    for condition_index in transonic:
        condition_slice = _condition_slice(int(condition_index), rows_per_condition)
        x_condition = np.asarray(x_split[condition_slice, :9], dtype=np.float32)
        cp_truth = np.asarray(y_split[condition_slice, 0], dtype=np.float32)
        score = _transonic_profile_score(x_condition, cp_truth, cut_y_values, cut_bandwidth, profile_bins)
        if score > best_score:
            best_score = score
            best_condition = int(condition_index)

    if best_condition is not None:
        return best_condition
    return _pick_transonic_condition([], cond_meta)


def _nearest_y_station(x_condition: np.ndarray, y_target: float) -> float:
    y_vals = np.asarray(x_condition[:, 1], dtype=np.float32)
    nearest_idx = int(np.argmin(np.abs(y_vals - y_target)))
    return float(y_vals[nearest_idx])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute a weighted quantile of an empty array.")

    quantile = float(np.clip(quantile, 0.0, 1.0))
    weights = np.maximum(weights, 0.0)
    if not np.any(weights > 0.0):
        return float(np.quantile(values, quantile))

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return float(np.interp(quantile, cdf, v))


def _local_poly_predict(
    x_support: np.ndarray,
    y_support: np.ndarray,
    values_support: np.ndarray,
    x_query: float,
    y_query: float,
    x_scale: float,
    y_scale: float,
    max_neighbors: int = 120,
    min_neighbors: int = 18,
) -> tuple[float | None, int]:
    if x_support.size < min_neighbors:
        return None, int(x_support.size)

    dx_scaled = (x_support - x_query) / max(x_scale, 1e-8)
    dy_scaled = (y_support - y_query) / max(y_scale, 1e-8)
    d2 = dx_scaled * dx_scaled + dy_scaled * dy_scaled
    order = np.argsort(d2)

    keep = order[: min(max_neighbors, order.size)]
    near = keep[d2[keep] <= 9.0]
    if near.size >= min_neighbors:
        keep = near
    elif keep.size < min_neighbors:
        keep = order[:min(min_neighbors, order.size)]

    dx = (x_support[keep] - x_query) / max(x_scale, 1e-8)
    dy = (y_support[keep] - y_query) / max(y_scale, 1e-8)
    values = values_support[keep]
    weights = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float64)
    if np.all(weights < 1e-8):
        return None, int(keep.size)

    design = np.column_stack(
        [
            np.ones_like(dx, dtype=np.float64),
            dx.astype(np.float64),
            dy.astype(np.float64),
            (dx * dx).astype(np.float64),
            (dx * dy).astype(np.float64),
        ]
    )
    ridge = 1e-6 * np.eye(design.shape[1], dtype=np.float64)
    lhs = design.T @ (weights[:, None] * design) + ridge
    rhs = design.T @ (weights * values.astype(np.float64))
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None, int(keep.size)
    return float(beta[0]), int(keep.size)


def _estimate_section_chord_bounds(
    x_condition: np.ndarray,
    y_target: float,
    y_scale: float,
) -> tuple[float, float]:
    xyz = np.asarray(x_condition[:, :3], dtype=np.float32)
    y_vals = xyz[:, 1]
    x_vals = xyz[:, 0]
    weights = np.exp(-0.5 * ((y_vals - y_target) / max(y_scale, 1e-8)) ** 2).astype(np.float64)
    x_le = _weighted_quantile(x_vals, weights, 0.01)
    x_te = _weighted_quantile(x_vals, weights, 0.99)
    if not np.isfinite(x_le) or not np.isfinite(x_te) or x_te <= x_le:
        nearest = np.argsort(np.abs(y_vals - y_target))[: max(300, min(2000, x_vals.size // 10))]
        x_le = float(np.quantile(x_vals[nearest], 0.01))
        x_te = float(np.quantile(x_vals[nearest], 0.99))
    return float(x_le), float(x_te)


def _reconstruct_section_branch(
    xyz_branch: np.ndarray,
    cp_branch: np.ndarray,
    x_query: np.ndarray,
    y_target: float,
    x_scale: float,
    y_scale: float,
    max_neighbors: int,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_support = np.asarray(xyz_branch[:, 0], dtype=np.float32)
    y_support = np.asarray(xyz_branch[:, 1], dtype=np.float32)
    z_support = np.asarray(xyz_branch[:, 2], dtype=np.float32)
    cp_support = np.asarray(cp_branch, dtype=np.float32)

    out_x: list[float] = []
    out_z: list[float] = []
    out_cp: list[float] = []
    out_n: list[int] = []
    for xq in np.asarray(x_query, dtype=np.float32):
        z_pred, n_geom = _local_poly_predict(
            x_support,
            y_support,
            z_support,
            float(xq),
            float(y_target),
            x_scale=x_scale,
            y_scale=y_scale,
            max_neighbors=max_neighbors,
            min_neighbors=min_neighbors,
        )
        cp_pred, n_cp = _local_poly_predict(
            x_support,
            y_support,
            cp_support,
            float(xq),
            float(y_target),
            x_scale=x_scale,
            y_scale=y_scale,
            max_neighbors=max_neighbors,
            min_neighbors=min_neighbors,
        )
        if z_pred is None or cp_pred is None:
            continue
        out_x.append(float(xq))
        out_z.append(float(z_pred))
        out_cp.append(float(cp_pred))
        out_n.append(int(min(n_geom, n_cp)))

    return (
        np.asarray(out_x, dtype=np.float32),
        np.asarray(out_z, dtype=np.float32),
        np.asarray(out_cp, dtype=np.float32),
        np.asarray(out_n, dtype=np.int32),
    )


def _reconstruct_section_curves(
    x_condition: np.ndarray,
    cp_truth: np.ndarray,
    cp_symbolic: np.ndarray,
    y_target: float,
    bins: int,
    y_scale: float = 0.18,
    x_scale_frac: float = 0.022,
    max_neighbors: int = 120,
    min_neighbors: int = 18,
) -> dict[str, np.ndarray | float | int | str]:
    xyz = np.asarray(x_condition[:, :3], dtype=np.float32)
    upper_mask, lower_mask = _split_surface_sides(xyz)
    x_le, x_te = _estimate_section_chord_bounds(x_condition, y_target, y_scale=y_scale)
    chord = max(float(x_te - x_le), 1e-6)
    x_query = np.linspace(x_le, x_te, bins, dtype=np.float32)
    x_scale = max(x_scale_frac * chord, 1e-4)

    upper_x_abs, upper_z, truth_upper_cp, upper_n = _reconstruct_section_branch(
        xyz[upper_mask],
        cp_truth[upper_mask],
        x_query,
        y_target,
        x_scale=x_scale,
        y_scale=y_scale,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    )
    _, _, symbolic_upper_cp, _ = _reconstruct_section_branch(
        xyz[upper_mask],
        cp_symbolic[upper_mask],
        x_query,
        y_target,
        x_scale=x_scale,
        y_scale=y_scale,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    )

    lower_x_abs, lower_z, truth_lower_cp, lower_n = _reconstruct_section_branch(
        xyz[lower_mask],
        cp_truth[lower_mask],
        x_query,
        y_target,
        x_scale=x_scale,
        y_scale=y_scale,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    )
    _, _, symbolic_lower_cp, _ = _reconstruct_section_branch(
        xyz[lower_mask],
        cp_symbolic[lower_mask],
        x_query,
        y_target,
        x_scale=x_scale,
        y_scale=y_scale,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    )

    upper_x = ((upper_x_abs - x_le) / chord).astype(np.float32)
    lower_x = ((lower_x_abs - x_le) / chord).astype(np.float32)
    if upper_x_abs.size >= 3 and lower_x_abs.size >= 3:
        section_points = np.vstack(
            [
                np.column_stack([upper_x_abs, upper_z]),
                np.column_stack([lower_x_abs, lower_z]),
            ]
        ).astype(np.float32)
        _, _, _, _, x_norm_all, signed_dist_all = _section_chord_frame(section_points)
        upper_count = upper_x_abs.size
        upper_x = x_norm_all[:upper_count].astype(np.float32)
        upper_z = signed_dist_all[:upper_count].astype(np.float32)
        lower_x = x_norm_all[upper_count:].astype(np.float32)
        lower_z = signed_dist_all[upper_count:].astype(np.float32)

    if upper_x.size and lower_x.size:
        common_x = np.intersect1d(np.round(upper_x, 6), np.round(lower_x, 6))
        if common_x.size:
            for x_val in common_x:
                iu = np.where(np.isclose(upper_x, x_val, atol=1e-6))[0]
                il = np.where(np.isclose(lower_x, x_val, atol=1e-6))[0]
                if iu.size == 0 or il.size == 0:
                    continue
                if upper_z[iu[0]] < lower_z[il[0]]:
                    mid = 0.5 * (upper_z[iu[0]] + lower_z[il[0]])
                    gap = max(abs(float(upper_z[iu[0]] - lower_z[il[0]])), 1e-4)
                    upper_z[iu[0]] = mid + 0.5 * gap
                    lower_z[il[0]] = mid - 0.5 * gap

    return {
        "method": "local_xy_interpolation",
        "y_target": float(y_target),
        "x_le": float(x_le),
        "x_te": float(x_te),
        "chord": float(chord),
        "x_scale": float(x_scale),
        "y_scale": float(y_scale),
        "upper_x": upper_x,
        "upper_z": upper_z.astype(np.float32),
        "lower_x": lower_x,
        "lower_z": lower_z.astype(np.float32),
        "truth_upper_cp": truth_upper_cp.astype(np.float32),
        "truth_lower_cp": truth_lower_cp.astype(np.float32),
        "symbolic_upper_cp": symbolic_upper_cp.astype(np.float32),
        "symbolic_lower_cp": symbolic_lower_cp.astype(np.float32),
        "upper_support_count": upper_n.astype(np.int32),
        "lower_support_count": lower_n.astype(np.int32),
        "n_upper_bins": int(upper_x.size),
        "n_lower_bins": int(lower_x.size),
    }


def generate_cp_field_plots(
    cfg: PipelineConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
) -> dict[str, Path | list[Path]]:
    cfg.ensure_dirs()

    x_split, y_split, rows_per_condition, n_conditions = _load_split_arrays(cfg, split)
    cond_meta = _condition_metadata(x_split, rows_per_condition)

    if not condition_indices:
        condition_indices = list(range(n_conditions))
    condition_indices = [int(idx) for idx in condition_indices]

    for condition_index in condition_indices:
        if condition_index < 0 or condition_index >= n_conditions:
            raise ValueError(f"Condition index out of range for split '{split}': {condition_index}")

    surface_summary: list[dict[str, float | int | str | list[float]]] = []
    field_paths: list[Path] = []

    for condition_index in condition_indices:
        condition_slice = _condition_slice(condition_index, rows_per_condition)
        x_condition = np.asarray(x_split[condition_slice, : cfg.input_dim_raw], dtype=np.float32)
        cp_truth = np.asarray(y_split[condition_slice, 0], dtype=np.float32)
        symbolic = predict_array(cfg, x_condition, mode="symbolic")
        cp_symbolic = symbolic["cp_pred"].reshape(-1).astype(np.float32)

        symbolic_mae, symbolic_rmse = _mae_rmse(cp_truth, cp_symbolic)
        mach = float(cond_meta["mach"][condition_index])
        aoa = float(cond_meta["aoa"][condition_index])
        pi = float(cond_meta["pi"][condition_index])
        regime_name = REGIME_NAMES[int(cond_meta["regime_id"][condition_index])]

        plot_idx = sample_indices(x_condition.shape[0], min(cfg.plot_sample_size, x_condition.shape[0]), seed=1000 + condition_index)
        xyz_plot = x_condition[plot_idx, :3]
        truth_plot = cp_truth[plot_idx]
        symbolic_plot = cp_symbolic[plot_idx]

        field_plot_path = _create_conference_style_cp_figure(
            cfg,
            condition_index=condition_index,
            xyz=xyz_plot,
            cp_truth=truth_plot,
            cp_symbolic=symbolic_plot,
            mach=mach,
            aoa=aoa,
            pi=pi,
        )
        field_paths.append(field_plot_path)

        surface_summary.append(
            {
                "condition_index": int(condition_index),
                "expected_regime": regime_name,
                "mach": mach,
                "aoa_deg": aoa,
                "pi": pi,
                "symbolic_mae": symbolic_mae,
                "symbolic_rmse": symbolic_rmse,
                "symbolic_gate_mean": symbolic["gates"].mean(axis=0).astype(float).tolist(),
                "field_plot_path": str(field_plot_path),
            }
        )

    summary_path = cfg.results_dir / f"cp_field_summary_symbolic_{split}.json"
    save_json(
        summary_path,
        {
            "split": split,
            "condition_indices": condition_indices,
            "surface_summary": surface_summary,
        },
    )
    return {
        "summary_path": summary_path,
        "field_paths": field_paths,
    }


def _create_section_y16_figure(
    out_dir: Path,
    condition_index: int,
    y_station: float,
    requested_y: float,
    mach: float,
    aoa: float,
    pi: float,
    section_geometry: dict[str, np.ndarray | int | float],
    truth_upper_x: np.ndarray,
    truth_upper_z: np.ndarray,
    truth_upper_cp: np.ndarray,
    truth_lower_x: np.ndarray,
    truth_lower_z: np.ndarray,
    truth_lower_cp: np.ndarray,
    symbolic_upper_x: np.ndarray,
    symbolic_upper_z: np.ndarray,
    symbolic_upper_cp: np.ndarray,
    symbolic_lower_x: np.ndarray,
    symbolic_lower_z: np.ndarray,
    symbolic_lower_cp: np.ndarray,
) -> Path:
    thickness_scale = float(
        max(
            np.max(np.abs(truth_upper_z)) if truth_upper_z.size else 0.0,
            np.max(np.abs(truth_lower_z)) if truth_lower_z.size else 0.0,
        )
    )
    merge_tol = max(0.05 * thickness_scale, 0.01)

    truth_upper_x, truth_upper_z, truth_upper_cp = _trim_merged_edge_points(truth_upper_x, truth_upper_z, truth_upper_cp, merge_tol)
    truth_lower_x, truth_lower_z, truth_lower_cp = _trim_merged_edge_points(truth_lower_x, truth_lower_z, truth_lower_cp, merge_tol)
    symbolic_upper_x, symbolic_upper_z, symbolic_upper_cp = _trim_merged_edge_points(symbolic_upper_x, symbolic_upper_z, symbolic_upper_cp, merge_tol)
    symbolic_lower_x, symbolic_lower_z, symbolic_lower_cp = _trim_merged_edge_points(symbolic_lower_x, symbolic_lower_z, symbolic_lower_cp, merge_tol)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.9]},
    )
    ax_cp, ax_profile = axes

    shock_truth = _detect_shock_location(truth_upper_x, truth_upper_cp)
    shock_symbolic = _detect_shock_location(symbolic_upper_x, symbolic_upper_cp)

    if truth_upper_x.size:
        ax_cp.plot(truth_upper_x, -truth_upper_cp, color="black", linewidth=2.0, label="Truth upper")
    if truth_lower_x.size:
        ax_cp.plot(truth_lower_x, -truth_lower_cp, color="black", linewidth=2.0, linestyle="--", label="Truth lower")
    if symbolic_upper_x.size:
        ax_cp.plot(symbolic_upper_x, -symbolic_upper_cp, color="#f58518", linewidth=1.8, label="Symbolic upper")
    if symbolic_lower_x.size:
        ax_cp.plot(symbolic_lower_x, -symbolic_lower_cp, color="#f58518", linewidth=1.8, linestyle="--", label="Symbolic lower")
    if shock_truth is not None:
        ax_cp.axvline(shock_truth, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if shock_symbolic is not None:
        ax_cp.axvline(shock_symbolic, color="#f58518", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_cp.axhline(0.0, color="0.82", linewidth=0.8)
    ax_cp.grid(True, alpha=0.22)
    ax_cp.set_ylabel("-Cp")
    if truth_upper_x.size or truth_lower_x.size or symbolic_upper_x.size or symbolic_lower_x.size:
        ax_cp.legend(loc="best", fontsize=9)
    ax_cp.set_title(
        f"Condition {condition_index} | y≈{y_station:.3f} m (target {requested_y:.1f}) | Mach={mach:.2f}, AoA={aoa:.1f}, Pi={pi:.1f}",
        fontsize=12,
    )
    info_text = (
        f"method: local interpolation at y={y_station:.6f} | "
        f"upper bins={int(section_geometry['n_upper_bins'])}, lower bins={int(section_geometry['n_lower_bins'])}"
    )
    ax_cp.text(0.02, 0.06, info_text, transform=ax_cp.transAxes, fontsize=9, color="0.35")

    if truth_upper_x.size:
        upper_order = np.argsort(truth_upper_x)
        ax_profile.plot(
            truth_upper_x[upper_order],
            truth_upper_z[upper_order],
            color="black",
            linewidth=1.6,
            marker="o",
            markersize=2.5,
        )
    if truth_lower_x.size:
        lower_order = np.argsort(truth_lower_x)
        ax_profile.plot(
            truth_lower_x[lower_order],
            truth_lower_z[lower_order],
            color="black",
            linewidth=1.4,
            linestyle="--",
            marker="o",
            markersize=2.5,
        )
    ax_profile.axhline(0.0, color="0.82", linewidth=0.8)
    ax_profile.grid(True, alpha=0.18)
    ax_profile.set_xlabel("x/c local")
    ax_profile.set_ylabel("section z")
    ax_profile.set_aspect("equal", adjustable="box")

    out_path = out_dir / f"minus_cp_chord_y16_cond{condition_index}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_section_y16_plots(
    cfg: PipelineConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    y_target: float = 16.0,
    cut_bandwidth: float = 0.2,
    profile_bins: int = 60,
    min_section_points: int = 900,
) -> dict[str, Path | list[Path]]:
    cfg.ensure_dirs()
    section_dir = cfg.pipeline_root / "results_section_y16"
    section_dir.mkdir(parents=True, exist_ok=True)

    x_split, y_split, rows_per_condition, n_conditions = _load_split_arrays(cfg, split)
    cond_meta = _condition_metadata(x_split, rows_per_condition)

    if not condition_indices:
        condition_indices = list(range(n_conditions))
    condition_indices = [int(idx) for idx in condition_indices]

    for condition_index in condition_indices:
        if condition_index < 0 or condition_index >= n_conditions:
            raise ValueError(f"Condition index out of range for split '{split}': {condition_index}")

    section_paths: list[Path] = []
    section_summary: list[dict[str, float | int | str | None]] = []

    for condition_index in condition_indices:
        condition_slice = _condition_slice(condition_index, rows_per_condition)
        x_condition = np.asarray(x_split[condition_slice, : cfg.input_dim_raw], dtype=np.float32)
        cp_truth = np.asarray(y_split[condition_slice, 0], dtype=np.float32)
        cp_symbolic = predict_array(cfg, x_condition, mode="symbolic")["cp_pred"].reshape(-1).astype(np.float32)
        y_station = _nearest_y_station(x_condition, float(y_target))
        section_geometry = _reconstruct_section_curves(
            x_condition,
            cp_truth,
            cp_symbolic,
            y_target=y_station,
            bins=profile_bins,
            y_scale=max(cut_bandwidth / 2.0, 0.18),
            x_scale_frac=0.022,
            max_neighbors=120,
            min_neighbors=18,
        )
        truth_upper_x = np.asarray(section_geometry["upper_x"], dtype=np.float32)
        truth_upper_z = np.asarray(section_geometry["upper_z"], dtype=np.float32)
        truth_upper_cp = np.asarray(section_geometry["truth_upper_cp"], dtype=np.float32)
        truth_lower_x = np.asarray(section_geometry["lower_x"], dtype=np.float32)
        truth_lower_z = np.asarray(section_geometry["lower_z"], dtype=np.float32)
        truth_lower_cp = np.asarray(section_geometry["truth_lower_cp"], dtype=np.float32)
        symbolic_upper_x = np.asarray(section_geometry["upper_x"], dtype=np.float32)
        symbolic_upper_z = np.asarray(section_geometry["upper_z"], dtype=np.float32)
        symbolic_upper_cp = np.asarray(section_geometry["symbolic_upper_cp"], dtype=np.float32)
        symbolic_lower_x = np.asarray(section_geometry["lower_x"], dtype=np.float32)
        symbolic_lower_z = np.asarray(section_geometry["lower_z"], dtype=np.float32)
        symbolic_lower_cp = np.asarray(section_geometry["symbolic_lower_cp"], dtype=np.float32)

        if truth_upper_x.size == 0 or truth_lower_x.size == 0:
            continue

        mach = float(cond_meta["mach"][condition_index])
        aoa = float(cond_meta["aoa"][condition_index])
        pi = float(cond_meta["pi"][condition_index])

        fig_path = _create_section_y16_figure(
            out_dir=section_dir,
            condition_index=condition_index,
            y_station=y_station,
            requested_y=float(y_target),
            mach=mach,
            aoa=aoa,
            pi=pi,
            section_geometry=section_geometry,
            truth_upper_x=truth_upper_x,
            truth_upper_z=truth_upper_z,
            truth_upper_cp=truth_upper_cp,
            truth_lower_x=truth_lower_x,
            truth_lower_z=truth_lower_z,
            truth_lower_cp=truth_lower_cp,
            symbolic_upper_x=symbolic_upper_x,
            symbolic_upper_z=symbolic_upper_z,
            symbolic_upper_cp=symbolic_upper_cp,
            symbolic_lower_x=symbolic_lower_x,
            symbolic_lower_z=symbolic_lower_z,
            symbolic_lower_cp=symbolic_lower_cp,
        )
        section_paths.append(fig_path)
        section_summary.append(
            {
                "condition_index": int(condition_index),
                "mach": mach,
                "aoa_deg": aoa,
                "pi": pi,
                "requested_y": float(y_target),
                "y_station": y_station,
                "method": str(section_geometry["method"]),
                "y_scale": float(section_geometry["y_scale"]),
                "x_scale": float(section_geometry["x_scale"]),
                "n_upper_bins": int(truth_upper_x.size),
                "n_lower_bins": int(truth_lower_x.size),
                "figure_path": str(fig_path),
            }
        )

    summary_path = section_dir / f"section_y16_summary_{split}.json"
    save_json(
        summary_path,
        {
            "split": split,
            "y_target": float(y_target),
            "method": "local_xy_interpolation",
            "condition_indices": condition_indices,
            "sections": section_summary,
        },
    )
    return {
        "summary_path": summary_path,
        "section_paths": section_paths,
    }


def generate_prediction_diagnostics(
    cfg: PipelineConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    transonic_condition_index: int | None = None,
    cut_y_values: list[float] | None = None,
    cut_bandwidth: float = 0.4,
    profile_bins: int = 60,
) -> dict[str, Path]:
    cfg.ensure_dirs()

    x_split, y_split, rows_per_condition, n_conditions = _load_split_arrays(cfg, split)
    cond_meta = _condition_metadata(x_split, rows_per_condition)

    auto_condition_indices = not condition_indices
    if not condition_indices:
        condition_indices = _select_representative_conditions(cond_meta)
    condition_indices = [int(idx) for idx in condition_indices]

    for condition_index in condition_indices:
        if condition_index < 0 or condition_index >= n_conditions:
            raise ValueError(f"Condition index out of range for split '{split}': {condition_index}")

    if cut_y_values is None:
        cut_y_values = [16.0, 23.0, 27.6]

    if transonic_condition_index is None:
        transonic_condition_index = _pick_shock_transonic_condition(
            x_split,
            y_split,
            rows_per_condition,
            cond_meta,
            cut_y_values,
            cut_bandwidth,
            profile_bins,
        )
    elif transonic_condition_index < 0 or transonic_condition_index >= n_conditions:
        raise ValueError(f"Transonic condition index out of range for split '{split}': {transonic_condition_index}")

    if auto_condition_indices and transonic_condition_index is not None:
        condition_indices = [
            int(transonic_condition_index) if int(cond_meta["regime_id"][condition_index]) == 1 else int(condition_index)
            for condition_index in condition_indices
        ]

    surface_summary: list[dict[str, float | int | str | list[float] | str]] = []
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    field_paths: list[Path] = []

    for row_idx, condition_index in enumerate(condition_indices):
        condition_slice = _condition_slice(condition_index, rows_per_condition)
        x_condition = np.asarray(x_split[condition_slice, : cfg.input_dim_raw], dtype=np.float32)
        cp_truth = np.asarray(y_split[condition_slice, 0], dtype=np.float32)

        symbolic = predict_array(cfg, x_condition, mode="symbolic")
        prediction_cache[condition_index] = {
            "x": x_condition,
            "truth": cp_truth,
            "symbolic": symbolic["cp_pred"].reshape(-1).astype(np.float32),
            "symbolic_gates": symbolic["gates"].astype(np.float32),
        }

        cp_symbolic = prediction_cache[condition_index]["symbolic"]
        symbolic_mae, symbolic_rmse = _mae_rmse(cp_truth, cp_symbolic)

        mach = float(cond_meta["mach"][condition_index])
        aoa = float(cond_meta["aoa"][condition_index])
        pi = float(cond_meta["pi"][condition_index])
        regime_name = REGIME_NAMES[int(cond_meta["regime_id"][condition_index])]
        plot_idx = sample_indices(x_condition.shape[0], min(cfg.plot_sample_size, x_condition.shape[0]), seed=1000 + condition_index)
        xyz_plot = x_condition[plot_idx, :3]
        truth_plot = cp_truth[plot_idx]
        symbolic_plot = cp_symbolic[plot_idx]
        field_plot_path = _create_conference_style_cp_figure(
            cfg,
            condition_index=condition_index,
            xyz=xyz_plot,
            cp_truth=truth_plot,
            cp_symbolic=symbolic_plot,
            mach=mach,
            aoa=aoa,
            pi=pi,
        )
        field_paths.append(field_plot_path)

        surface_summary.append(
            {
                "condition_index": int(condition_index),
                "expected_regime": regime_name,
                "mach": mach,
                "aoa_deg": aoa,
                "pi": pi,
                "symbolic_mae": symbolic_mae,
                "symbolic_rmse": symbolic_rmse,
                "symbolic_gate_mean": prediction_cache[condition_index]["symbolic_gates"].mean(axis=0).astype(float).tolist(),
                "field_plot_path": str(field_plot_path),
            }
        )

    profile_summary: list[dict[str, float | int | None]] = []
    profile_path: Path | None = None
    if transonic_condition_index is not None:
        if transonic_condition_index not in prediction_cache:
            condition_slice = _condition_slice(transonic_condition_index, rows_per_condition)
            x_condition = np.asarray(x_split[condition_slice, : cfg.input_dim_raw], dtype=np.float32)
            cp_truth = np.asarray(y_split[condition_slice, 0], dtype=np.float32)
            symbolic = predict_array(cfg, x_condition, mode="symbolic")
            prediction_cache[transonic_condition_index] = {
                "x": x_condition,
                "truth": cp_truth,
                "symbolic": symbolic["cp_pred"].reshape(-1).astype(np.float32),
                "symbolic_gates": symbolic["gates"].astype(np.float32),
            }

        transonic_payload = prediction_cache[transonic_condition_index]
        transonic_x = transonic_payload["x"]
        fig_profile, axes_profile = plt.subplots(
            len(cut_y_values),
            1,
            figsize=(9.2, 3.3 * max(1, len(cut_y_values))),
            constrained_layout=True,
            squeeze=False,
        )

        for row_idx, y_target in enumerate(cut_y_values):
            ax = axes_profile[row_idx, 0]
            section_geometry = _extract_section_branches(transonic_x, y_target, cut_bandwidth, profile_bins)
            n_points = int(section_geometry["n_points"])
            truth_upper_x, truth_upper_z, truth_upper_cp = _branch_cp_curve(transonic_payload["truth"], section_geometry, "upper")
            truth_lower_x, truth_lower_z, truth_lower_cp = _branch_cp_curve(transonic_payload["truth"], section_geometry, "lower")
            symbolic_upper_x, _, symbolic_upper_cp = _branch_cp_curve(transonic_payload["symbolic"], section_geometry, "upper")
            symbolic_lower_x, _, symbolic_lower_cp = _branch_cp_curve(transonic_payload["symbolic"], section_geometry, "lower")

            if truth_upper_x.size == 0 or truth_lower_x.size == 0 or symbolic_upper_x.size == 0:
                ax.text(0.5, 0.5, "No points for this cut", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                profile_summary.append(
                    {
                        "y_target": float(y_target),
                        "bandwidth": float(cut_bandwidth),
                        "n_points": int(n_points),
                        "shock_truth_upper_x_over_c": None,
                        "shock_symbolic_upper_x_over_c": None,
                    }
                )
                continue

            shock_truth = _detect_shock_location(truth_upper_x, truth_upper_cp)
            shock_symbolic = _detect_shock_location(symbolic_upper_x, symbolic_upper_cp)

            # Standard aerodynamics view: -Cp vs x/c.
            ax.plot(truth_upper_x, -truth_upper_cp, color="black", linewidth=2.0, label="Truth")
            ax.plot(truth_lower_x, -truth_lower_cp, color="black", linewidth=2.0, linestyle="--")
            ax.plot(symbolic_upper_x, -symbolic_upper_cp, color="#f58518", linewidth=1.8, label="Symbolic")
            ax.plot(symbolic_lower_x, -symbolic_lower_cp, color="#f58518", linewidth=1.8, linestyle="--")

            if shock_truth is not None:
                ax.axvline(shock_truth, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
            if shock_symbolic is not None:
                ax.axvline(shock_symbolic, color="#f58518", linestyle=":", linewidth=1.2, alpha=0.8)

            ax.axhline(0.0, color="0.75", linewidth=0.8)
            ax.text(0.02, 0.08, "dashed = lower", transform=ax.transAxes, fontsize=9, color="0.35")
            ax.text(0.02, 0.16, "solid = upper", transform=ax.transAxes, fontsize=9, color="0.35")
            ax.set_title(f"Transonic chord cut: y={y_target:.1f} ± {cut_bandwidth / 2.0:.2f} | points={n_points}")
            ax.set_ylabel("-Cp")
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.legend(loc="best")

            profile_summary.append(
                {
                    "y_target": float(y_target),
                    "bandwidth": float(cut_bandwidth),
                    "n_points": int(n_points),
                    "shock_truth_upper_x_over_c": shock_truth,
                    "shock_symbolic_upper_x_over_c": shock_symbolic,
                    "n_upper_bins": int(truth_upper_x.size),
                    "n_lower_bins": int(truth_lower_x.size),
                }
            )

        axes_profile[-1, 0].set_xlabel("x/c local")
        fig_profile.suptitle(
            f"Transonic chord cuts over the full section | truth vs symbolic | condition={transonic_condition_index}",
            fontsize=13,
        )
        profile_path = cfg.results_dir / f"minus_cp_chord_cuts_transonic_condition{transonic_condition_index}_symbolic.png"
        fig_profile.savefig(profile_path, dpi=220, bbox_inches="tight")
        plt.close(fig_profile)

    summary_path = cfg.results_dir / f"prediction_diagnostics_symbolic_{split}.json"
    save_json(
        summary_path,
        {
            "split": split,
            "condition_indices": condition_indices,
            "transonic_condition_index": transonic_condition_index,
            "surface_summary": surface_summary,
            "profile_summary": profile_summary,
        },
    )

    outputs = {
        "summary_path": summary_path,
        "field_paths": field_paths,
    }
    if profile_path is not None:
        outputs["profile_path"] = profile_path
    return outputs
