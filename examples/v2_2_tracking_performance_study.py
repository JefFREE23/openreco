"""OpenReco v2.2 tracking-performance study entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openreco.analysis.plotting import ensure_figure_dir
from openreco.analysis.scans import (
    make_scan_grid,
    run_and_write_tracking_performance_scan,
)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list."""

    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""

    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v2.2 tracking-performance study."
    )

    parser.add_argument(
        "--output-dir",
        default="docs/reports/v2_2_tracking_performance",
        help="Directory where v2.2 performance outputs will be written.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned scan grid without running reconstruction.",
    )

    parser.add_argument(
        "--n-particles",
        default="1,2,5",
        help="Comma-separated particle multiplicities to scan.",
    )

    parser.add_argument(
        "--noise-hits",
        default="0,1",
        help="Comma-separated noise hits per layer values to scan.",
    )

    parser.add_argument(
        "--hit-efficiencies",
        default="1.0,0.95",
        help="Comma-separated hit efficiencies to scan.",
    )

    parser.add_argument(
        "--n-events",
        type=int,
        default=50,
        help="Number of events per scan point.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used by the scan runner.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    figure_dir = ensure_figure_dir(output_dir / "figures")
    summary_csv = output_dir / "tracking_performance_summary.csv"

    configs = make_scan_grid(
        n_particles=_parse_int_list(args.n_particles),
        noise_hits_per_layer=_parse_int_list(args.noise_hits),
        hit_efficiencies=_parse_float_list(args.hit_efficiencies),
        n_events=args.n_events,
        seed=args.seed,
        seed_mode="hole-aware",
    )

    print("OpenReco v2.2 tracking performance study")
    print(f"output directory: {output_dir}")
    print(f"figure directory: {figure_dir}")
    print(f"summary CSV:      {summary_csv}")
    print(f"scan points:      {len(configs)}")

    if args.dry_run:
        print()
        print("Planned scan grid:")
        for config in configs:
            print(
                "  "
                f"n_particles={config.n_particles}, "
                f"noise/layer={config.noise_hits_per_layer}, "
                f"hit_eff={config.hit_efficiency:.2f}, "
                f"events={config.n_events}, "
                f"seed_mode={config.seed_mode}"
            )
        return 0

    print()
    print("Running tracking-performance scan...")
    results = run_and_write_tracking_performance_scan(
        output_path=summary_csv,
        configs=configs,
    )

    print(f"completed scan points: {len(results)}")
    print(f"CSV saved:             {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())