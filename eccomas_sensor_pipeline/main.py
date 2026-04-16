from __future__ import annotations

import argparse
from pathlib import Path

from eccomas_sensor.config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ordered Eccomas symbolic sensor pipeline")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", type=str, default=None, help="Base project root. Used only as default for derived paths.")
    common.add_argument("--raw-data-dir", type=str, default=None, help="Folder containing X_train.npy, X_test.npy, Ytrain.npy, Ytest.npy, and optionally dataset.csv.")
    common.add_argument("--cut-data-dir", type=str, default=None, help="Folder where data_cut_y12 artifacts will be written/read.")
    common.add_argument("--pipeline-root", type=str, default=None, help="Folder where pipeline outputs will be written.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-data", parents=[common], help="Cut geometry at y>=12 and generate validation plots")
    subparsers.add_parser("plot-design-space", parents=[common], help="Plot AoA vs Mach for each Pi using raw-condition metadata")
    subparsers.add_parser("prepare-features", parents=[common], help="Build and normalize feature stores for train/test")
    subparsers.add_parser("train-experts", parents=[common], help="Train Cp experts per Mach regime")
    subparsers.add_parser("train-latent", parents=[common], help="Train latent encoder + gating and export latent plots")
    subparsers.add_parser(
        "distill-sensor",
        parents=[common],
        aliases=["distill-symbolic"],
        help="Fit a symbolic sensor that maps raw inputs to expert routing scores",
    )
    infer_parser = subparsers.add_parser("infer", parents=[common], help="Run end-to-end Cp inference from raw X inputs")
    infer_parser.add_argument("--input-path", type=str, required=True, help="Path to a raw X .npy array with at least 9 columns.")
    infer_parser.add_argument("--output-path", type=str, default=None, help="Optional output .npz path for predictions.")
    infer_parser.add_argument("--mode", type=str, choices=["neural", "symbolic"], default="neural", help="Use the neural gate or the distilled symbolic sensor.")
    infer_parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size for inference.")
    infer_parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick smoke tests.")
    field_parser = subparsers.add_parser("plot-fields", parents=[common], help="Generate cp_field plots for one or more conditions without profile cuts")
    field_parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Split used to build the field plots.")
    field_parser.add_argument("--condition-indices", type=int, nargs="*", default=None, help="Optional condition indices to plot. Defaults to every condition in the split.")
    section_parser = subparsers.add_parser("plot-sections-y16", parents=[common], help="Generate y=16 m section figures using local interpolation in the y=cte plane")
    section_parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Split used to build the section plots.")
    section_parser.add_argument("--condition-indices", type=int, nargs="*", default=None, help="Optional condition indices to plot. Defaults to every condition in the split.")
    section_parser.add_argument("--cut-bandwidth", type=float, default=0.2, help="Local y-support scale used by the section interpolation around y=16 m.")
    section_parser.add_argument("--profile-bins", type=int, default=60, help="Number of x/c stations used to reconstruct each section.")
    plot_parser = subparsers.add_parser("plot-inference", parents=[common], help="Generate surface and profile-cut diagnostics from truth, neural, and symbolic Cp predictions")
    plot_parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Split used to build the diagnostics.")
    plot_parser.add_argument("--condition-indices", type=int, nargs="*", default=None, help="Optional condition indices to plot on the wing surface.")
    plot_parser.add_argument("--transonic-condition-index", type=int, default=None, help="Optional condition index used for the transonic profile cuts.")
    plot_parser.add_argument("--cut-y", type=float, nargs="*", default=None, help="Optional y locations for the transonic profile cuts.")
    plot_parser.add_argument("--cut-bandwidth", type=float, default=0.2, help="Thickness of the y-slab used for each profile cut.")
    plot_parser.add_argument("--profile-bins", type=int, default=60, help="Number of x/c bins used to build the profile curves.")
    subparsers.add_parser("run-all", parents=[common], help="Run every stage in sequence")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = PipelineConfig(
        project_root=Path(args.project_root) if getattr(args, "project_root", None) else None,
        raw_data_dir=Path(args.raw_data_dir) if getattr(args, "raw_data_dir", None) else None,
        cut_data_dir=Path(args.cut_data_dir) if getattr(args, "cut_data_dir", None) else None,
        pipeline_root=Path(args.pipeline_root) if getattr(args, "pipeline_root", None) else None,
    )

    if args.command == "prepare-data":
        from eccomas_sensor.prepare_data import prepare_cut_data

        prepare_cut_data(cfg)
    elif args.command == "plot-design-space":
        from eccomas_sensor.prepare_data import plot_design_space

        plot_design_space(cfg)
    elif args.command == "prepare-features":
        from eccomas_sensor.feature_store import prepare_feature_store

        prepare_feature_store(cfg)
    elif args.command == "train-experts":
        from eccomas_sensor.train_experts import train_all_experts

        train_all_experts(cfg)
    elif args.command == "train-latent":
        from eccomas_sensor.train_latent import train_latent_pipeline

        train_latent_pipeline(cfg)
    elif args.command in {"distill-sensor", "distill-symbolic"}:
        from eccomas_sensor.sensor_distillation import distill_sensor

        distill_sensor(cfg)
    elif args.command == "infer":
        from eccomas_sensor.inference import run_inference

        run_inference(
            cfg,
            input_path=Path(args.input_path),
            mode=args.mode,
            output_path=Path(args.output_path) if args.output_path else None,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
        )
    elif args.command == "plot-inference":
        from eccomas_sensor.diagnostics import generate_prediction_diagnostics

        generate_prediction_diagnostics(
            cfg,
            split=args.split,
            condition_indices=args.condition_indices,
            transonic_condition_index=args.transonic_condition_index,
            cut_y_values=args.cut_y,
            cut_bandwidth=args.cut_bandwidth,
            profile_bins=args.profile_bins,
        )
    elif args.command == "plot-fields":
        from eccomas_sensor.diagnostics import generate_cp_field_plots

        generate_cp_field_plots(
            cfg,
            split=args.split,
            condition_indices=args.condition_indices,
        )
    elif args.command == "plot-sections-y16":
        from eccomas_sensor.diagnostics import generate_section_y16_plots

        generate_section_y16_plots(
            cfg,
            split=args.split,
            condition_indices=args.condition_indices,
            cut_bandwidth=args.cut_bandwidth,
            profile_bins=args.profile_bins,
        )
    elif args.command == "run-all":
        from eccomas_sensor.prepare_data import prepare_cut_data
        from eccomas_sensor.feature_store import prepare_feature_store
        from eccomas_sensor.train_experts import train_all_experts
        from eccomas_sensor.train_latent import train_latent_pipeline
        from eccomas_sensor.sensor_distillation import distill_sensor

        prepare_cut_data(cfg)
        prepare_feature_store(cfg)
        train_all_experts(cfg)
        train_latent_pipeline(cfg)
        distill_sensor(cfg)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
