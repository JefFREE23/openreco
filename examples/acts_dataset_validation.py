from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openreco.external.acts_loader import load_acts_dataset
from openreco.external.reconstruction import (
    format_external_reco_summary,
    run_external_dataset_reconstruction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenReco v2 reconstruction on ACTS-style external files."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/acts_small",
        help="Path to ACTS-style dataset directory.",
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

    args = parser.parse_args()

    dataset = load_acts_dataset(args.dataset)

    summary = run_external_dataset_reconstruction(
        dataset,
        max_events=args.max_events,
        seed_mode=args.seed_mode,
        min_hits=args.min_hits,
        chi2_threshold=args.chi2_threshold,
        use_ekf_fit=not args.no_ekf_fit,
    )

    print(format_external_reco_summary(summary))


if __name__ == "__main__":
    main()
