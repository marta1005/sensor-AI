from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eccomas_full_aircrafts.pipeline.config import FullAircraftConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-aircraft ONERA reduced-data pipeline")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", type=str, default=None, help="Base project root. Used only as default for derived paths.")
    common.add_argument("--raw-data-dir", type=str, default=None, help="Folder containing the full ONERA arrays.")
    common.add_argument("--pipeline-root", type=str, default=None, help="Folder where full-aircraft outputs will be written.")
    common.add_argument(
        "--reduced-surface",
        type=str,
        choices=["upper", "lower"],
        default="upper",
        help="Active reduced surface for the train/latent/sensor/inference pipeline.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect-raw", parents=[common], help="Summarize full-aircraft raw arrays and condition ranges")

    surface_parser = subparsers.add_parser(
        "prepare-reference-surface",
        parents=[common],
        help="Build a simplified full-aircraft upper/lower surface reference from one raw condition",
    )
    surface_parser.add_argument("--reference-split", type=str, choices=["train", "test"], default="train")
    surface_parser.add_argument("--reference-condition-index", type=int, default=0)
    surface_parser.add_argument("--x-bins", type=int, default=None, help="Optional number of bins along x used to simplify the surface.")
    surface_parser.add_argument("--y-bins", type=int, default=None, help="Optional number of bins along y used to simplify the surface.")

    reduced_parser = subparsers.add_parser(
        "prepare-reduced-data",
        parents=[common],
        help="Extract reduced full-aircraft arrays using the prepared reference surface indices",
    )
    reduced_parser.add_argument("--surface", type=str, choices=["upper", "lower", "both"], default="upper")
    reduced_parser.add_argument("--splits", type=str, nargs="+", choices=["train", "test"], default=["train", "test"])
    reduced_parser.add_argument("--max-conditions", type=int, default=None, help="Optional cap for debugging or smoke tests.")

    subparsers.add_parser(
        "prepare-features",
        parents=[common],
        help="Build standardized expert/gate features over the reduced full-aircraft arrays",
    )

    subparsers.add_parser(
        "train-experts",
        parents=[common],
        help="Train the three Mach experts over the reduced full-aircraft features",
    )

    subparsers.add_parser(
        "train-latent",
        parents=[common],
        help="Train the latent gate/MoE teacher over the reduced full-aircraft features",
    )

    subparsers.add_parser(
        "distill-sensor",
        parents=[common],
        help="Distill the latent gate into a symbolic full-aircraft sensor",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        parents=[common],
        help="Run neural or symbolic inference over a reduced raw array",
    )
    infer_parser.add_argument("--input-path", type=str, default=None, help="Path to the raw/reduced input array. Defaults to reduced test data.")
    infer_parser.add_argument("--mode", type=str, choices=["neural", "symbolic"], default="symbolic")
    infer_parser.add_argument("--output-path", type=str, default=None)
    infer_parser.add_argument("--batch-size", type=int, default=None)
    infer_parser.add_argument("--max-rows", type=int, default=None)

    field_parser = subparsers.add_parser(
        "plot-raw-fields",
        parents=[common],
        help="Plot raw Cp fields over the simplified full-aircraft surface reference",
    )
    field_parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    field_parser.add_argument("--surface", type=str, choices=["upper", "lower"], default="upper")
    field_parser.add_argument("--mode", type=str, choices=["points", "surface"], default="points")
    field_parser.add_argument("--condition-indices", type=int, nargs="*", default=None)
    field_parser.add_argument("--max-plotted-points", type=int, default=60_000)

    eval_parser = subparsers.add_parser(
        "plot-eval-cp",
        parents=[common],
        help="Plot full-aircraft raw Cp in the eval_score_plot_fields.py top/bottom style",
    )
    eval_parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    eval_parser.add_argument("--condition-indices", type=int, nargs="*", default=None)
    eval_parser.add_argument("--max-plotted-points", type=int, default=120_000)

    return parser


def _build_cfg(args: argparse.Namespace, surface: str | None = None) -> FullAircraftConfig:
    return FullAircraftConfig(
        project_root=Path(args.project_root) if getattr(args, "project_root", None) else None,
        raw_data_dir=Path(args.raw_data_dir) if getattr(args, "raw_data_dir", None) else None,
        pipeline_root=Path(args.pipeline_root) if getattr(args, "pipeline_root", None) else None,
        reduced_surface=surface or getattr(args, "reduced_surface", "upper"),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _build_cfg(args)

    if args.command == "inspect-raw":
        from eccomas_full_aircrafts.pipeline.prepare_surface import inspect_raw_dataset

        inspect_raw_dataset(cfg)
    elif args.command == "prepare-reference-surface":
        from eccomas_full_aircrafts.pipeline.prepare_surface import prepare_reference_surface

        prepare_reference_surface(
            cfg,
            reference_split=args.reference_split,
            reference_condition_index=args.reference_condition_index,
            x_bins=args.x_bins,
            y_bins=args.y_bins,
        )
    elif args.command == "prepare-reduced-data":
        from eccomas_full_aircrafts.pipeline.prepare_data import prepare_reduced_data

        target_surfaces = ["upper", "lower"] if args.surface == "both" else [args.surface]
        for surface_name in target_surfaces:
            surface_cfg = _build_cfg(args, surface=surface_name)
            prepare_reduced_data(
                surface_cfg,
                surface=surface_name,
                splits=tuple(args.splits),
                max_conditions=args.max_conditions,
            )
    elif args.command == "prepare-features":
        from eccomas_full_aircrafts.pipeline.feature_store import prepare_feature_store

        prepare_feature_store(cfg)
    elif args.command == "train-experts":
        from eccomas_full_aircrafts.pipeline.train_experts import train_all_experts

        train_all_experts(cfg)
    elif args.command == "train-latent":
        from eccomas_full_aircrafts.pipeline.train_latent import train_latent_pipeline

        train_latent_pipeline(cfg)
    elif args.command == "distill-sensor":
        from eccomas_full_aircrafts.pipeline.sensor_distillation import distill_sensor

        distill_sensor(cfg)
    elif args.command == "infer":
        from eccomas_full_aircrafts.pipeline.inference import run_inference

        input_path = Path(args.input_path).expanduser().resolve() if args.input_path else (cfg.reduced_data_dir / "X_cut_test.npy")
        output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
        run_inference(
            cfg,
            input_path=input_path,
            mode=args.mode,
            output_path=output_path,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
        )
    elif args.command == "plot-raw-fields":
        from eccomas_full_aircrafts.pipeline.plot_fields import generate_raw_cp_field_plots

        generate_raw_cp_field_plots(
            cfg,
            split=args.split,
            condition_indices=args.condition_indices,
            surface=args.surface,
            max_plotted_points=args.max_plotted_points,
            mode=args.mode,
        )
    elif args.command == "plot-eval-cp":
        from eccomas_full_aircrafts.pipeline.plot_fields import generate_eval_cp_plots

        generate_eval_cp_plots(
            cfg,
            split=args.split,
            condition_indices=args.condition_indices,
            max_plotted_points=args.max_plotted_points,
        )
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
