"""OpenReco v2.2 tracking-performance study entry point.

Chunk 1 provides the analysis-layer skeleton and dry-run configuration display.
The actual reconstruction scan runner is added in the next v2.2 chunk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openreco.analysis.plotting import ensure_figure_dir
from openreco.analysis.scans import default_v2_2_scan_grid


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    figure_dir = ensure_figure_dir(output_dir / "figures")
    configs = default_v2_2_scan_grid()

    print("OpenReco v2.2 tracking performance study")
    print(f"output directory: {output_dir}")
    print(f"figure directory: {figure_dir}")
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
    print("Scan execution will be implemented in v2.2 Chunk 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())