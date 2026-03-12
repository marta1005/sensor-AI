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
    subparsers.add_parser("distill-symbolic", parents=[common], help="Fit symbolic regressors for latent coordinates")
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
    elif args.command == "distill-symbolic":
        from eccomas_sensor.symbolic_distillation import distill_symbolic_encoder

        distill_symbolic_encoder(cfg)
    elif args.command == "run-all":
        from eccomas_sensor.prepare_data import prepare_cut_data
        from eccomas_sensor.feature_store import prepare_feature_store
        from eccomas_sensor.train_experts import train_all_experts
        from eccomas_sensor.train_latent import train_latent_pipeline
        from eccomas_sensor.symbolic_distillation import distill_symbolic_encoder

        prepare_cut_data(cfg)
        prepare_feature_store(cfg)
        train_all_experts(cfg)
        train_latent_pipeline(cfg)
        distill_symbolic_encoder(cfg)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
