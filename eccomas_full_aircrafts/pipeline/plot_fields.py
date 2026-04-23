from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import FullAircraftConfig
from .utils import condition_start, condition_stop, raw_paths, save_json, sample_indices

_PLOT_CACHE = Path(__file__).resolve().parent / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _reference_surface_path(cfg: FullAircraftConfig, surface: str) -> Path:
    path = cfg.surfaces_dir / f"{surface}_surface_reference.npz"
    if not path.exists():
        raise FileNotFoundError(f"Reference surface not found: {path}. Run 'prepare-reference-surface' first.")
    return path


def _condition_records(cfg: FullAircraftConfig, split: str) -> list[dict[str, float]]:
    x_path, _ = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
    records: list[dict[str, float]] = []
    for idx in range(n_conditions):
        row = np.asarray(x_raw[idx * cfg.raw_points_per_condition, 6:9], dtype=np.float32)
        records.append(
            {
                "condition_index": int(idx),
                "Mach": float(row[0]),
                "AoA_deg": float(row[1]),
                "Pi": float(row[2]),
            }
        )
    return records


def _surface_grid(ref: np.lib.npyio.NpzFile, values: np.ndarray) -> np.ma.MaskedArray:
    x_bins = int(np.asarray(ref["x_edges"]).shape[0] - 1)
    y_bins = int(np.asarray(ref["y_edges"]).shape[0] - 1)
    grid = np.full((y_bins, x_bins), np.nan, dtype=np.float32)
    x_bin = np.asarray(ref["x_bin"], dtype=np.int64)
    y_bin = np.asarray(ref["y_bin"], dtype=np.int64)
    grid[y_bin, x_bin] = values.astype(np.float32, copy=False)
    return np.ma.masked_invalid(grid)


def generate_raw_cp_field_plots(
    cfg: FullAircraftConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    surface: str = "upper",
    max_plotted_points: int = 60_000,
    mode: str = "points",
) -> dict:
    cfg.ensure_dirs()
    result_dir = cfg.results_surface_dir(surface)
    result_dir.mkdir(parents=True, exist_ok=True)
    ref = np.load(_reference_surface_path(cfg, surface))
    local_idx = np.asarray(ref["local_idx"], dtype=np.int64)
    x_ref = np.asarray(ref["x"], dtype=np.float32)
    y_ref = np.asarray(ref["y"], dtype=np.float32)
    x_edges = np.asarray(ref["x_edges"], dtype=np.float32)
    y_edges = np.asarray(ref["y_edges"], dtype=np.float32)

    x_path, y_path = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    y_raw = np.load(y_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
    condition_indices = list(range(n_conditions)) if condition_indices is None else [int(idx) for idx in condition_indices]
    records = _condition_records(cfg, split)

    plot_idx = sample_indices(local_idx.size, min(max_plotted_points, local_idx.size), seed=17)
    selected_local = local_idx[plot_idx]
    selected_x = x_ref[plot_idx]
    selected_y = y_ref[plot_idx]

    summary: list[dict[str, float | str]] = []
    for cond_idx in condition_indices:
        start = condition_start(cfg.raw_points_per_condition, cond_idx)
        stop = condition_stop(cfg.raw_points_per_condition, cond_idx)
        cp_block = np.asarray(y_raw[start:stop, cfg.cp_column], dtype=np.float32)
        cp_surface = cp_block[selected_local]
        cp_full_surface = cp_block[local_idx]

        rec = records[cond_idx]
        fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
        if mode == "surface":
            cp_grid = _surface_grid(ref, cp_full_surface)
            sc = ax.pcolormesh(x_edges, y_edges, cp_grid, cmap="jet", shading="flat")
            plotted_count = int(cp_full_surface.size)
        elif mode == "points":
            sc = ax.scatter(selected_x, selected_y, c=cp_surface, s=2.6, cmap="jet", linewidths=0)
            plotted_count = int(selected_local.size)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Full-aircraft {surface} {mode} Cp | cond {cond_idx} | "
            f"Mach={rec['Mach']:.2f}, AoA={rec['AoA_deg']:.1f}, Pi={rec['Pi']:.1f}"
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.9)
        cb.set_label("Cp")
        out_path = result_dir / f"cp_full_aircraft_{surface}_{mode}_cond{cond_idx}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        summary.append(
            {
                "condition_index": int(cond_idx),
                "surface": surface,
                "mode": mode,
                "Mach": float(rec["Mach"]),
                "AoA_deg": float(rec["AoA_deg"]),
                "Pi": float(rec["Pi"]),
                "points_plotted": plotted_count,
                "cp_min": float(cp_full_surface.min()),
                "cp_max": float(cp_full_surface.max()),
                "path": str(out_path),
            }
        )

    payload = {
        "split": split,
        "surface": surface,
        "mode": mode,
        "n_conditions": len(summary),
        "results": summary,
    }
    save_json(result_dir / f"cp_full_aircraft_{surface}_{mode}_{split}_summary.json", payload)
    return payload


def _eval_tb_cp_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cp: np.ndarray,
    rec: dict[str, float],
    title_prefix: str,
    out_path: Path,
) -> None:
    camera_settings = {
        "elevation": 90.0,
        "azimuth": 0.0,
        "zoom": 1.9,
        "xoffsets": [0.0, 0.0],
        "yoffsets": [0.0, 4.0],
        "zoffsets": [0.0, 0.0],
    }
    cp_min = float(np.min(cp))
    cp_max = float(np.max(cp))

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))

    fig = plt.figure(figsize=(4.4, 4.6))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[20, 1], hspace=0.06)
    gs_views = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.0)

    scatter = None
    for j in range(2):
        ax = fig.add_subplot(gs_views[0, j], projection="3d")
        scatter = ax.scatter3D(
            x,
            y,
            z,
            c=cp,
            s=1.0,
            cmap="jet",
            vmin=cp_min,
            vmax=cp_max,
            linewidths=0,
            clip_on=False,
        )
        ax.view_init(
            elev=camera_settings["elevation"] * ((-1) ** (j + 1)),
            azim=camera_settings["azimuth"] + 180 * abs(j - 1),
        )
        ax.set(
            xlim=(xmin + camera_settings["xoffsets"][0], xmax + camera_settings["xoffsets"][1]),
            ylim=(ymin + camera_settings["yoffsets"][0], ymax + camera_settings["yoffsets"][1]),
            zlim=(zmin + camera_settings["zoffsets"][0], zmax + camera_settings["zoffsets"][1]),
        )
        ax.set_axis_off()
        limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
        ax.set_box_aspect(np.ptp(limits, axis=1), zoom=camera_settings["zoom"])
        if j == 0:
            ax.text((xmax - xmin) / 4 + 0.5, (ymax - ymin) / 2 + 7.0, 0.0, "bottom", fontsize=6)
        else:
            ax.text((xmax - xmin) / 4, (ymax - ymin) / 2 - 5.0, 0.0, "top", fontsize=6)

    cax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_label("$C_p$", size=8)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_label_position("top")

    fig.suptitle(
        f"{title_prefix} $p_i$ = {rec['Pi']:.1f} 10⁵, $M_{{\\infty}}$ = {rec['Mach']:.2f}, AoA = {rec['AoA_deg']:.1f}°",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89], h_pad=2.0)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _eval_tb_inference_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    rec: dict[str, float],
    title_prefix: str,
    out_path: Path,
    include_error: bool = True,
    view: str = "tb",
) -> None:
    camera_settings = {
        "elevation": 90.0,
        "azimuth": 0.0,
        "zoom": 1.75,
        "xoffsets": [0.0, 0.0],
        "yoffsets": [0.0, 4.0],
        "zoffsets": [0.0, 0.0],
    }
    cp_min = float(min(np.min(cp_true), np.min(cp_pred)))
    cp_max = float(max(np.max(cp_true), np.max(cp_pred)))
    signed_err = (cp_pred - cp_true).astype(np.float32)
    abs_err = np.abs(signed_err).astype(np.float32)
    err_lim = float(np.max(np.abs(signed_err)))

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))

    fields = [
        ("Truth $C_p$", cp_true, "jet", cp_min, cp_max),
        ("Predicted $C_p$", cp_pred, "jet", cp_min, cp_max),
    ]
    if include_error:
        err_vmin = -err_lim if err_lim > 0.0 else -1.0
        err_vmax = err_lim if err_lim > 0.0 else 1.0
        fields.append((f"Error [{err_vmin:.3f}, {err_vmax:.3f}]", signed_err, "RdBu_r", err_vmin, err_vmax))

    ncols = len(fields)
    use_tb = view == "tb"
    plot_zoom = camera_settings["zoom"] if use_tb else 1.32
    fig_width = 8.8 if ncols == 2 else 13.2
    if not use_tb:
        fig_width = 8.4 if ncols == 2 else 12.8
    fig = plt.figure(figsize=(fig_width, 4.8))
    outer = gridspec.GridSpec(
        nrows=2,
        ncols=ncols,
        height_ratios=[20, 1],
        hspace=0.14 if use_tb else 0.20,
        wspace=0.16 if use_tb else 0.28,
    )

    for col, (label, values, cmap, vmin, vmax) in enumerate(fields):
        inner_cols = 2 if use_tb else 1
        inner = gridspec.GridSpecFromSubplotSpec(
            1,
            inner_cols,
            subplot_spec=outer[0, col],
            wspace=0.0 if use_tb else 0.12,
        )
        scatter = None
        for j in range(inner_cols):
            ax = fig.add_subplot(inner[0, j], projection="3d")
            scatter = ax.scatter3D(
                x,
                y,
                z,
                c=values,
                s=1.0,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                clip_on=False,
            )
            if use_tb:
                ax.view_init(
                    elev=camera_settings["elevation"] * ((-1) ** (j + 1)),
                    azim=camera_settings["azimuth"] + 180 * abs(j - 1),
                )
            else:
                ax.view_init(elev=90.0, azim=180.0)
            ax.set(
                xlim=(xmin + camera_settings["xoffsets"][0], xmax + camera_settings["xoffsets"][1]),
                ylim=(ymin + camera_settings["yoffsets"][0], ymax + camera_settings["yoffsets"][1]),
                zlim=(zmin + camera_settings["zoffsets"][0], zmax + camera_settings["zoffsets"][1]),
            )
            ax.set_axis_off()
            limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
            ax.set_box_aspect(np.ptp(limits, axis=1), zoom=plot_zoom)
            if use_tb:
                if j == 0:
                    ax.text((xmax - xmin) / 4 + 0.5, (ymax - ymin) / 2 + 7.0, 0.0, "bottom", fontsize=6)
                else:
                    ax.text((xmax - xmin) / 4, (ymax - ymin) / 2 - 5.0, 0.0, "top", fontsize=6)
            else:
                ax.text((xmax - xmin) / 4, (ymax - ymin) / 2 - 6.5, 0.0, "upper", fontsize=6)
            if j == 0:
                ax.set_title(label, fontsize=10, pad=8)

        cax = fig.add_subplot(outer[1, col])
        cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
        cbar.set_label(label, size=8)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.xaxis.set_label_position("top")

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((cp_pred - cp_true) ** 2)))
    fig.suptitle(
        f"{title_prefix} $p_i$ = {rec['Pi']:.1f} 10⁵, $M_{{\\infty}}$ = {rec['Mach']:.2f}, "
        f"AoA = {rec['AoA_deg']:.1f}°, MAE = {mae:.4f}, RMSE = {rmse:.4f}",
        fontsize=12,
        y=0.985,
    )
    if use_tb:
        fig.subplots_adjust(left=0.03, right=0.985, bottom=0.08, top=0.84, wspace=0.18)
    else:
        fig.subplots_adjust(left=0.03, right=0.985, bottom=0.12, top=0.76, wspace=0.26)
    fig.savefig(out_path, dpi=360, bbox_inches=None, pad_inches=0.22)
    plt.close(fig)


def generate_inference_cp_grid_plot(
    cfg: FullAircraftConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    prediction_path: Path | None = None,
    max_plotted_points: int = 120_000,
    view: str = "tb",
) -> dict:
    cfg.ensure_dirs()
    result_dir = cfg.results_surface_dir(cfg.reduced_surface)
    result_dir.mkdir(parents=True, exist_ok=True)

    x_path = cfg.reduced_data_dir / f"X_cut_{split}.npy"
    y_path = cfg.reduced_data_dir / f"Y_cut_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Reduced arrays not found for split={split} in {cfg.reduced_data_dir}")

    if prediction_path is None:
        prediction_path = cfg.inference_dir / f"{x_path.stem}_symbolic.npz"
    prediction_path = Path(prediction_path).expanduser().resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}. Run 'infer --mode symbolic' first.")

    x_red = np.load(x_path, mmap_mode="r")
    y_red = np.load(y_path, mmap_mode="r")
    pred_payload = np.load(prediction_path)
    cp_pred = np.asarray(pred_payload["cp_pred"], dtype=np.float32).reshape(-1)

    records = _condition_records(cfg, split)
    n_conditions = len(records)
    condition_indices = [int(idx) for idx in (condition_indices or list(range(min(4, n_conditions))))]
    reduced_points_per_condition = int(x_red.shape[0] // max(n_conditions, 1))

    camera_settings = {
        "elevation": 90.0,
        "azimuth": 0.0,
        "zoom": 1.55,
        "xoffsets": [0.0, 0.0],
        "yoffsets": [0.0, 4.0],
        "zoffsets": [0.0, 0.0],
    }
    use_tb = view == "tb"
    plot_zoom = camera_settings["zoom"] if use_tb else 1.28

    def _draw_pair(fig, subplot_spec, xyz, values, cmap, vmin, vmax, label):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        zmin, zmax = float(np.min(z)), float(np.max(z))
        inner_cols = 2 if use_tb else 1
        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            inner_cols,
            subplot_spec=subplot_spec,
            height_ratios=[20, 1],
            hspace=0.04 if use_tb else 0.16,
            wspace=0.0 if use_tb else 0.12,
        )
        scatter = None
        for j in range(inner_cols):
            ax = fig.add_subplot(inner[0, j], projection="3d")
            scatter = ax.scatter3D(
                x,
                y,
                z,
                c=values,
                s=0.7,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                clip_on=False,
            )
            if use_tb:
                ax.view_init(
                    elev=camera_settings["elevation"] * ((-1) ** (j + 1)),
                    azim=camera_settings["azimuth"] + 180 * abs(j - 1),
                )
            else:
                ax.view_init(elev=90.0, azim=180.0)
            ax.set(
                xlim=(xmin + camera_settings["xoffsets"][0], xmax + camera_settings["xoffsets"][1]),
                ylim=(ymin + camera_settings["yoffsets"][0], ymax + camera_settings["yoffsets"][1]),
                zlim=(zmin + camera_settings["zoffsets"][0], zmax + camera_settings["zoffsets"][1]),
            )
            ax.set_axis_off()
            limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
            ax.set_box_aspect(np.ptp(limits, axis=1), zoom=plot_zoom)
            if use_tb:
                if j == 0:
                    ax.text((xmax - xmin) / 4 + 0.5, (ymax - ymin) / 2 + 7.0, 0.0, "bottom", fontsize=5)
                else:
                    ax.text((xmax - xmin) / 4, (ymax - ymin) / 2 - 5.0, 0.0, "top", fontsize=5)
            else:
                ax.text((xmax - xmin) / 4, (ymax - ymin) / 2 - 6.5, 0.0, "upper", fontsize=5)
            if j == 0:
                ax.set_title(label, fontsize=8, pad=6)
        cax = fig.add_subplot(inner[1, :])
        cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
        cbar.set_label(label, size=7)
        cbar.ax.tick_params(labelsize=5)
        cbar.ax.xaxis.set_label_position("top")

    fig_width = 13.4 if use_tb else 12.2
    fig = plt.figure(figsize=(fig_width, max(4.8, 4.0 * len(condition_indices))))
    outer = gridspec.GridSpec(
        len(condition_indices),
        3,
        figure=fig,
        hspace=0.34 if use_tb else 0.46,
        wspace=0.16 if use_tb else 0.30,
    )
    if use_tb:
        fig.subplots_adjust(left=0.03, right=0.985, bottom=0.04, top=0.95)
    else:
        fig.subplots_adjust(left=0.03, right=0.985, bottom=0.06, top=0.91)
    summary: list[dict[str, float | str]] = []

    for row, cond_idx in enumerate(condition_indices):
        start = condition_start(reduced_points_per_condition, cond_idx)
        stop = condition_stop(reduced_points_per_condition, cond_idx)
        x_block = np.asarray(x_red[start:stop, : cfg.input_dim_raw], dtype=np.float32)
        cp_true_block = np.asarray(y_red[start:stop, cfg.cp_column], dtype=np.float32)
        cp_pred_block = np.asarray(cp_pred[start:stop], dtype=np.float32)
        n_plot = min(max_plotted_points, x_block.shape[0])
        plot_idx = sample_indices(x_block.shape[0], n_plot, seed=404 + cond_idx)
        xyz = x_block[plot_idx, 0:3]
        cp_true_plot = cp_true_block[plot_idx]
        cp_pred_plot = cp_pred_block[plot_idx]
        signed_err_plot = cp_pred_plot - cp_true_plot
        cp_min = float(min(np.min(cp_true_plot), np.min(cp_pred_plot)))
        cp_max = float(max(np.max(cp_true_plot), np.max(cp_pred_plot)))
        err_lim = float(np.max(np.abs(signed_err_plot)))
        rec = records[cond_idx]
        mae = float(np.mean(np.abs(cp_pred_block - cp_true_block)))
        rmse = float(np.sqrt(np.mean((cp_pred_block - cp_true_block) ** 2)))
        row_text = (
            f"cond {cond_idx} | M={rec['Mach']:.2f} | AoA={rec['AoA_deg']:.1f} | "
            f"Pi={rec['Pi']:.1f} | MAE={mae:.4f}"
        )

        _draw_pair(fig, outer[row, 0], xyz, cp_true_plot, "jet", cp_min, cp_max, "Truth $C_p$")
        _draw_pair(fig, outer[row, 1], xyz, cp_pred_plot, "jet", cp_min, cp_max, "Predicted $C_p$")
        err_vmin = -err_lim if err_lim > 0.0 else -1.0
        err_vmax = err_lim if err_lim > 0.0 else 1.0
        _draw_pair(
            fig,
            outer[row, 2],
            xyz,
            signed_err_plot,
            "RdBu_r",
            err_vmin,
            err_vmax,
            f"Error [{err_vmin:.3f}, {err_vmax:.3f}]",
        )

        bbox_left = outer[row, 0].get_position(fig)
        bbox_right = outer[row, 2].get_position(fig)
        x_center = 0.5 * (bbox_left.x0 + bbox_right.x1)
        y_centered_label = bbox_left.y1 + (0.010 if use_tb else 0.016)
        fig.text(x_center, y_centered_label, row_text, ha="center", va="bottom", fontsize=7.0)

        summary.append(
            {
                "condition_index": int(cond_idx),
                "split": split,
                "surface": cfg.reduced_surface,
                "Mach": float(rec["Mach"]),
                "AoA_deg": float(rec["AoA_deg"]),
                "Pi": float(rec["Pi"]),
                "mae": mae,
                "rmse": rmse,
            }
        )

    fig.suptitle(
        f"Full-aircraft {cfg.reduced_surface} symbolic inference | multiple test conditions",
        fontsize=13,
        y=0.99,
    )
    cond_label = "_".join(str(idx) for idx in condition_indices)
    out_path = result_dir / f"cp_full_aircraft_inference_tb_{cfg.reduced_surface}_{split}_truth_pred_error_grid_{cond_label}.png"
    fig.savefig(out_path, dpi=320, bbox_inches=None, pad_inches=0.22)
    plt.close(fig)

    payload = {
        "split": split,
        "surface": cfg.reduced_surface,
        "prediction_path": str(prediction_path),
        "view": view,
        "condition_indices": condition_indices,
        "path": str(out_path),
        "results": summary,
    }
    save_json(result_dir / f"cp_full_aircraft_inference_tb_{cfg.reduced_surface}_{split}_truth_pred_error_grid_summary.json", payload)
    return payload


def generate_eval_cp_plots(
    cfg: FullAircraftConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    max_plotted_points: int = 120_000,
) -> dict:
    cfg.ensure_dirs()
    result_dir = cfg.shared_results_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    x_path, y_path = raw_paths(cfg.raw_data_dir, split)
    x_raw = np.load(x_path, mmap_mode="r")
    y_raw = np.load(y_path, mmap_mode="r")
    n_conditions = int(x_raw.shape[0] // cfg.raw_points_per_condition)
    condition_indices = list(range(n_conditions)) if condition_indices is None else [int(idx) for idx in condition_indices]
    records = _condition_records(cfg, split)

    summary: list[dict[str, float | str]] = []
    for cond_idx in condition_indices:
        start = condition_start(cfg.raw_points_per_condition, cond_idx)
        stop = condition_stop(cfg.raw_points_per_condition, cond_idx)
        x_block = np.asarray(x_raw[start:stop, : cfg.input_dim_raw], dtype=np.float32)
        cp_block = np.asarray(y_raw[start:stop, cfg.cp_column], dtype=np.float32)

        n_plot = min(max_plotted_points, x_block.shape[0])
        plot_idx = sample_indices(x_block.shape[0], n_plot, seed=17 + cond_idx)
        xyz = x_block[plot_idx, 0:3]
        cp_plot = cp_block[plot_idx]

        rec = records[cond_idx]
        out_path = result_dir / f"cp_full_aircraft_eval_tb_{split}_cond{cond_idx}.png"
        _eval_tb_cp_figure(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            cp_plot,
            rec,
            title_prefix="Full-aircraft raw Cp",
            out_path=out_path,
        )

        summary.append(
            {
                "condition_index": int(cond_idx),
                "split": split,
                "Mach": float(rec["Mach"]),
                "AoA_deg": float(rec["AoA_deg"]),
                "Pi": float(rec["Pi"]),
                "points_plotted": int(n_plot),
                "cp_min": float(cp_block.min()),
                "cp_max": float(cp_block.max()),
                "path": str(out_path),
            }
        )

    payload = {
        "split": split,
        "n_conditions": len(summary),
        "results": summary,
    }
    save_json(result_dir / f"cp_full_aircraft_eval_tb_{split}_summary.json", payload)
    return payload


def generate_inference_cp_plots(
    cfg: FullAircraftConfig,
    split: str = "test",
    condition_indices: list[int] | None = None,
    prediction_path: Path | None = None,
    max_plotted_points: int = 120_000,
    layout: str = "truth-pred",
    view: str = "tb",
) -> dict:
    cfg.ensure_dirs()
    result_dir = cfg.results_surface_dir(cfg.reduced_surface)
    result_dir.mkdir(parents=True, exist_ok=True)

    x_path = cfg.reduced_data_dir / f"X_cut_{split}.npy"
    y_path = cfg.reduced_data_dir / f"Y_cut_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Reduced arrays not found for split={split} in {cfg.reduced_data_dir}")

    if prediction_path is None:
        prediction_path = cfg.inference_dir / f"{x_path.stem}_symbolic.npz"
    prediction_path = Path(prediction_path).expanduser().resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}. Run 'infer --mode symbolic' first.")
    include_error = layout == "truth-pred-error"

    x_red = np.load(x_path, mmap_mode="r")
    y_red = np.load(y_path, mmap_mode="r")
    pred_payload = np.load(prediction_path)
    cp_pred = np.asarray(pred_payload["cp_pred"], dtype=np.float32).reshape(-1)

    records = _condition_records(cfg, split)
    n_conditions = len(records)
    reduced_points_per_condition = int(x_red.shape[0] // max(n_conditions, 1))
    condition_indices = list(range(n_conditions)) if condition_indices is None else [int(idx) for idx in condition_indices]

    summary: list[dict[str, float | str]] = []
    for cond_idx in condition_indices:
        start = condition_start(reduced_points_per_condition, cond_idx)
        stop = condition_stop(reduced_points_per_condition, cond_idx)

        x_block = np.asarray(x_red[start:stop, : cfg.input_dim_raw], dtype=np.float32)
        cp_true_block = np.asarray(y_red[start:stop, cfg.cp_column], dtype=np.float32)
        cp_pred_block = np.asarray(cp_pred[start:stop], dtype=np.float32)

        n_plot = min(max_plotted_points, x_block.shape[0])
        plot_idx = sample_indices(x_block.shape[0], n_plot, seed=101 + cond_idx)
        xyz = x_block[plot_idx, 0:3]
        cp_true_plot = cp_true_block[plot_idx]
        cp_pred_plot = cp_pred_block[plot_idx]

        rec = records[cond_idx]
        suffix = "truth_pred_error" if include_error else "truth_pred"
        out_path = result_dir / f"cp_full_aircraft_inference_tb_{cfg.reduced_surface}_{split}_{suffix}_cond{cond_idx}.png"
        _eval_tb_inference_figure(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            cp_true_plot,
            cp_pred_plot,
            rec,
            title_prefix=f"Full-aircraft {cfg.reduced_surface} symbolic inference",
            out_path=out_path,
            include_error=include_error,
            view=view,
        )

        abs_err = np.abs(cp_pred_block - cp_true_block)
        rmse = np.sqrt(np.mean((cp_pred_block - cp_true_block) ** 2))
        summary.append(
            {
                "condition_index": int(cond_idx),
                "split": split,
                "surface": cfg.reduced_surface,
                "Mach": float(rec["Mach"]),
                "AoA_deg": float(rec["AoA_deg"]),
                "Pi": float(rec["Pi"]),
                "points_plotted": int(n_plot),
                "mae": float(np.mean(abs_err)),
                "rmse": float(rmse),
                "max_abs_error": float(np.max(abs_err)),
                "cp_true_min": float(cp_true_block.min()),
                "cp_true_max": float(cp_true_block.max()),
                "cp_pred_min": float(cp_pred_block.min()),
                "cp_pred_max": float(cp_pred_block.max()),
                "path": str(out_path),
            }
        )

    payload = {
        "split": split,
        "surface": cfg.reduced_surface,
        "layout": layout,
        "view": view,
        "prediction_path": str(prediction_path),
        "n_conditions": len(summary),
        "results": summary,
    }
    suffix = "truth_pred_error" if include_error else "truth_pred"
    save_json(result_dir / f"cp_full_aircraft_inference_tb_{cfg.reduced_surface}_{split}_{suffix}_summary.json", payload)
    return payload
