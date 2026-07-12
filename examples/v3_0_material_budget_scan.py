"""OpenReco v3.0 material-budget scan.

This example studies how increasing detector material affects event-level
tracking performance when truth hits include simplified multiple scattering.

At this stage, scattering is applied during event generation only. The
reconstruction fit still uses the existing clean-detector covariance model.
This is intentional: it creates the first controlled setup where material
effects can degrade residuals, chi2/ndof, and momentum resolution.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.detector_effects import DetectorEffectsConfig


@dataclass(frozen=True)
class MaterialBudgetScanResult:
    x_over_x0_per_layer: float
    n_events: int
    n_particles: int
    min_hits: int
    events_processed: int
    truth_particles_generated: int
    measurements_mean: float
    seeds_mean: float
    reconstructed_tracks_mean: float
    tracking_efficiency_mean: float
    fake_rate_mean: float
    duplicate_rate_mean: float
    mean_hits_per_track: float
    mean_holes_per_track: float
    mean_chi2_ndof: float
    covariance_valid_rate_mean: float
    momentum_residual_mean: float
    momentum_residual_std: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def _safe_mean(values: list[float]) -> float:
    finite_values = [value for value in values if isfinite(value)]
    return mean(finite_values) if finite_values else float("nan")


def _safe_pstdev(values: list[float]) -> float:
    finite_values = [value for value in values if isfinite(value)]
    return pstdev(finite_values) if len(finite_values) >= 2 else float("nan")


def _write_csv(path: str | Path, results: list[MaterialBudgetScanResult]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MaterialBudgetScanResult.fieldnames(),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def run_single_material_budget_point(
    *,
    x_over_x0_per_layer: float,
    n_events: int,
    n_particles: int,
    min_hits: int,
    seed: int,
) -> MaterialBudgetScanResult:
    config = DetectorEffectsConfig.with_uniform_material(
        layer_ids=range(6),
        x_over_x0=x_over_x0_per_layer,
    )

    measurement_counts: list[float] = []
    seed_counts: list[float] = []
    track_counts: list[float] = []
    tracking_efficiencies: list[float] = []
    fake_rates: list[float] = []
    duplicate_rates: list[float] = []
    hits_per_track: list[float] = []
    holes_per_track: list[float] = []
    chi2_ndof_values: list[float] = []
    covariance_valid_rates: list[float] = []
    momentum_residuals: list[float] = []

    for event_index in range(n_events):
        result = run_multi_track_reconstruction(
            event_id=event_index,
            n_particles=n_particles,
            hit_efficiency=1.0,
            noise_hits_per_layer=0,
            detector_effects=config,
            random_seed=seed + event_index,
            seed_mode="hole-aware",
            min_hits=min_hits,
            make_plot=False,
        )

        event = result.event
        validation = result.validation
        tracks = result.tracks

        measurement_counts.append(float(len(event.measurements)))
        seed_counts.append(float(len(result.seeds)))
        track_counts.append(float(len(tracks)))

        tracking_efficiencies.append(float(validation.tracking_efficiency))
        fake_rates.append(float(validation.fake_rate))
        duplicate_rates.append(float(validation.duplicate_rate))

        if tracks:
            hits_per_track.append(
                mean(float(len(track.used_measurements)) for track in tracks)
            )
            holes_per_track.append(mean(float(track.n_holes) for track in tracks))
            chi2_ndof_values.extend(float(track.chi2_ndof) for track in tracks)
            covariance_valid_rates.append(
                mean(1.0 if track.covariance_valid else 0.0 for track in tracks)
            )
        else:
            hits_per_track.append(float("nan"))
            holes_per_track.append(float("nan"))
            covariance_valid_rates.append(float("nan"))

        momentum_residuals.extend(
            float(value) for value in result.momentum_relative_errors
        )

    return MaterialBudgetScanResult(
        x_over_x0_per_layer=x_over_x0_per_layer,
        n_events=n_events,
        n_particles=n_particles,
        min_hits=min_hits,
        events_processed=n_events,
        truth_particles_generated=n_events * n_particles,
        measurements_mean=_safe_mean(measurement_counts),
        seeds_mean=_safe_mean(seed_counts),
        reconstructed_tracks_mean=_safe_mean(track_counts),
        tracking_efficiency_mean=_safe_mean(tracking_efficiencies),
        fake_rate_mean=_safe_mean(fake_rates),
        duplicate_rate_mean=_safe_mean(duplicate_rates),
        mean_hits_per_track=_safe_mean(hits_per_track),
        mean_holes_per_track=_safe_mean(holes_per_track),
        mean_chi2_ndof=_safe_mean(chi2_ndof_values),
        covariance_valid_rate_mean=_safe_mean(covariance_valid_rates),
        momentum_residual_mean=_safe_mean(momentum_residuals),
        momentum_residual_std=_safe_pstdev(momentum_residuals),
    )


def run_material_budget_scan(
    *,
    x_over_x0_values: tuple[float, ...],
    n_events: int,
    n_particles: int,
    min_hits: int,
    seed: int,
    output_path: str | Path,
) -> list[MaterialBudgetScanResult]:
    results = [
        run_single_material_budget_point(
            x_over_x0_per_layer=x_over_x0,
            n_events=n_events,
            n_particles=n_particles,
            min_hits=min_hits,
            seed=seed + 1000 * index,
        )
        for index, x_over_x0 in enumerate(x_over_x0_values)
    ]

    _write_csv(output_path, results)
    return results


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.0 material-budget scan."
    )

    parser.add_argument(
        "--x-over-x0",
        default="0.0,0.001,0.005,0.01,0.02,0.05",
        help="Comma-separated material thickness values per layer.",
    )

    parser.add_argument(
        "--n-events",
        type=int,
        default=10,
        help="Number of events per scan point.",
    )

    parser.add_argument(
        "--n-particles",
        type=int,
        default=5,
        help="Number of generated particles per event.",
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=4,
        help="Minimum hits required by the track finder.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )

    parser.add_argument(
        "--output-csv",
        default="docs/reports/v3_0_detector_effects/material_budget_scan.csv",
        help="Output CSV path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    x_over_x0_values = _parse_float_list(args.x_over_x0)

    print("OpenReco v3.0 material-budget scan")
    print(f"x/X0 per layer:   {x_over_x0_values}")
    print(f"events/point:     {args.n_events}")
    print(f"particles/event:  {args.n_particles}")
    print(f"min hits:         {args.min_hits}")
    print(f"output CSV:       {args.output_csv}")
    print()

    results = run_material_budget_scan(
        x_over_x0_values=x_over_x0_values,
        n_events=args.n_events,
        n_particles=args.n_particles,
        min_hits=args.min_hits,
        seed=args.seed,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"x/X0={result.x_over_x0_per_layer:.4f}, "
            f"eff={result.tracking_efficiency_mean:.3f}, "
            f"fake={result.fake_rate_mean:.3f}, "
            f"dup={result.duplicate_rate_mean:.3f}, "
            f"tracks/event={result.reconstructed_tracks_mean:.2f}, "
            f"holes/track={result.mean_holes_per_track:.2f}, "
            f"chi2/ndof={result.mean_chi2_ndof:.3f}, "
            f"mom_res_std={result.momentum_residual_std:.4f}"
        )

    print()
    print(f"CSV saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())