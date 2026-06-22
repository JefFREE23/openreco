from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openreco.external.acts_fatras_loader import (
    ActsFatrasLoaderConfig,
    load_fatras_dataset,
)
from openreco.external.acts_loader import load_acts_dataset
from openreco.external.reconstruction import run_external_dataset_reconstruction
from openreco.validation.external_metrics import (
    compute_external_validation_metrics,
    format_external_validation_metrics,
)
from openreco.validation.report import write_external_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenReco v2 reconstruction on ACTS-style external files."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/acts_small",
        help="Path to external dataset directory.",
    )

    parser.add_argument(
        "--input-format",
        type=str,
        default="acts-style",
        choices=("acts-style", "acts-fatras"),
        help=(
            "Input dataset format. "
            "'acts-style' expects truth_particles.csv + measurements.csv. "
            "'acts-fatras' expects official ACTS/Fatras event*-hits.csv "
            "+ event*-particles_initial.csv files."
        ),
    )

    parser.add_argument(
        "--fatras-length-scale",
        type=float,
        default=0.1,
        help=(
            "Length scale for ACTS/Fatras coordinates. "
            "Default 0.1 maps mm-like ACTS coordinates to cm-like OpenReco units."
        ),
    )

    parser.add_argument(
        "--fatras-radius-merge-tolerance",
        type=float,
        default=0.5,
        help=(
            "Merge ACTS/Fatras surfaces into the same simplified layer "
            "if their radii are this close after scaling."
        ),
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process.",
    )

    parser.add_argument(
        "--seed-mode",
        type=str,
        default="hole-aware",
        choices=("strict", "hole-aware"),
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--chi2-threshold",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--no-ekf-fit",
        action="store_true",
        help="Disable EKF fitting and run only seeding + track finding.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="Directory where CSV reports and plots are saved.",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write CSV report files.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write validation plots.",
    )

    args = parser.parse_args()

    if args.input_format == "acts-style":
        dataset = load_acts_dataset(args.dataset)
        reconstruction_max_events = args.max_events

    elif args.input_format == "acts-fatras":
        dataset = load_fatras_dataset(
            args.dataset,
            max_events=args.max_events,
            config=ActsFatrasLoaderConfig(
                length_scale=args.fatras_length_scale,
                radius_merge_tolerance=args.fatras_radius_merge_tolerance,
            ),
        )
        reconstruction_max_events = None

    else:
        raise ValueError(f"Unknown input format: {args.input_format}")

    summary = run_external_dataset_reconstruction(
        dataset,
        max_events=reconstruction_max_events,
        seed_mode=args.seed_mode,
        min_hits=args.min_hits,
        chi2_threshold=args.chi2_threshold,
        use_ekf_fit=not args.no_ekf_fit,
    )

    metrics = compute_external_validation_metrics(summary)

    print(format_external_validation_metrics(metrics))

    if not args.no_report:
        paths = write_external_validation_report(
            summary,
            args.output_dir,
            make_plots=not args.no_plots,
        )

        print("")
        print(f"summary CSV:               {paths.summary_csv}")
        print(f"tracks CSV:                {paths.tracks_csv}")

        if paths.efficiency_plot is not None:
            print(f"efficiency plot:           {paths.efficiency_plot}")

        if paths.momentum_residual_plot is not None:
            print(f"momentum residual plot:    {paths.momentum_residual_plot}")


if __name__ == "__main__":
    main()
