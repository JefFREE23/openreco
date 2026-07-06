"""OpenReco v3.0 noise-occupancy scan.

This example studies how random noise hits affect event-level tracking
performance.

It uses the existing OpenReco reconstruction chain:

    event generation
    -> triplet seeding
    -> greedy track finding
    -> EKF fitting/smoothing
    -> truth matching
    -> CSV summary
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
from openreco.detector_effects import (
    DetectorEffectsConfig,
    InefficiencyModel,
    NoiseOccupancyModel,
)


@dataclass(frozen=True)
class NoiseOccupancyScanResult:
    mean_noise_hits_per_layer: float
    n_events: int
    n_particles: int
    hit_efficiency: float
    min_hits: int
    events_processed: int
    truth_particles_generated: int
    measurements_mean: float
    noise_hits_mean: float
    real_hits_mean: float
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


def _write_csv(path: str | Path, results: list[NoiseOccupancyScanResult]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=NoiseOccupancyScanResult.fieldnames(),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def run_single_noise_occupancy_point(
    *,
    mean_noise_hits_per_layer: float,
    n_events: int,
    n_particles: int,
    hit_efficiency: float,
    min_hits: int,
    seed: int,
) -> NoiseOccupancyScanResult:
    config = DetectorEffectsConfig(
        inefficiency=InefficiencyModel(hit_efficiency=hit_efficiency),
        noise_occupancy=NoiseOccupancyModel(
            mean_noise_hits_per_layer=mean_noise_hits_per_layer
        ),
    )

    measurement_counts: list[float] = []
    noise_hit_counts: list[float] = []
    real_hit_counts: list[float] = []
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

        n_noise_hits = sum(1 for hit in event.measurements if hit.is_noise)
        n_real_hits = sum(1 for hit in event.measurements if not hit.is_noise)

        measurement_counts.append(float(len(event.measurements)))
        noise_hit_counts.append(float(n_noise_hits))
        real_hit_counts.append(float(n_real_hits))
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

    return NoiseOccupancyScanResult(
        mean_noise_hits_per_layer=mean_noise_hits_per_layer,
        n_events=n_events,
        n_particles=n_particles,
        hit_efficiency=hit_efficiency,
        min_hits=min_hits,
        events_processed=n_events,
        truth_particles_generated=n_events * n_particles,
        measurements_mean=_safe_mean(measurement_counts),
        noise_hits_mean=_safe_mean(noise_hit_counts),
        real_hits_mean=_safe_mean(real_hit_counts),
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


def run_noise_occupancy_scan(
    *,
    mean_noise_hits_per_layer_values: tuple[float, ...],
    n_events: int,
    n_particles: int,
    hit_efficiency: float,
    min_hits: int,
    seed: int,
    output_path: str | Path,
) -> list[NoiseOccupancyScanResult]:
    results = [
        run_single_noise_occupancy_point(
            mean_noise_hits_per_layer=mean_noise_hits_per_layer,
            n_events=n_events,
            n_particles=n_particles,
            hit_efficiency=hit_efficiency,
            min_hits=min_hits,
            seed=seed + 1000 * index,
        )
        for index, mean_noise_hits_per_layer in enumerate(
            mean_noise_hits_per_layer_values
        )
    ]

    _write_csv(output_path, results)
    return results


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.0 noise-occupancy scan."
    )

    parser.add_argument(
        "--mean-noise-hits-per-layer",
        default="0.0,0.5,1.0,2.0,5.0",
        help="Comma-separated mean random noise hits per detector layer.",
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
        "--hit-efficiency",
        type=float,
        default=1.0,
        help="Truth hit efficiency.",
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
        default="docs/reports/v3_0_detector_effects/noise_occupancy_scan.csv",
        help="Output CSV path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    mean_noise_values = _parse_float_list(args.mean_noise_hits_per_layer)

    print("OpenReco v3.0 noise-occupancy scan")
    print(f"mean noise/layer:  {mean_noise_values}")
    print(f"events/point:      {args.n_events}")
    print(f"particles/event:   {args.n_particles}")
    print(f"hit efficiency:    {args.hit_efficiency}")
    print(f"min hits:          {args.min_hits}")
    print(f"output CSV:        {args.output_csv}")
    print()

    results = run_noise_occupancy_scan(
        mean_noise_hits_per_layer_values=mean_noise_values,
        n_events=args.n_events,
        n_particles=args.n_particles,
        hit_efficiency=args.hit_efficiency,
        min_hits=args.min_hits,
        seed=args.seed,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"mean_noise/layer={result.mean_noise_hits_per_layer:.2f}, "
            f"noise/event={result.noise_hits_mean:.2f}, "
            f"eff={result.tracking_efficiency_mean:.3f}, "
            f"fake={result.fake_rate_mean:.3f}, "
            f"dup={result.duplicate_rate_mean:.3f}, "
            f"tracks/event={result.reconstructed_tracks_mean:.2f}, "
            f"seeds/event={result.seeds_mean:.2f}, "
            f"chi2/ndof={result.mean_chi2_ndof:.3f}"
        )

    print()
    print(f"CSV saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())