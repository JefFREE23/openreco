"""OpenReco v3.0 hit-resolution scan.

This example studies how the configured hit resolution affects event-level
tracking performance. It uses the existing OpenReco reconstruction chain:

    event generation
    -> triplet seeding
    -> greedy track finding
    -> EKF fitting/smoothing
    -> truth matching
    -> CSV summary

This is the first v3.0 detector-effects study script.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openreco.detector_effects import DetectorEffectsConfig, HitResolutionModel
from examples.multi_track_reconstruction import run_multi_track_reconstruction


@dataclass(frozen=True)
class HitResolutionScanResult:
    sigma_phi: float
    sigma_z: float
    n_events: int
    n_particles: int
    hit_efficiency: float
    noise_hits_per_layer: int
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
    finite_values = [value for value in values if value == value]
    return mean(finite_values) if finite_values else float("nan")


def _safe_pstdev(values: list[float]) -> float:
    finite_values = [value for value in values if value == value]
    return pstdev(finite_values) if len(finite_values) >= 2 else float("nan")


def _write_csv(path: str | Path, results: list[HitResolutionScanResult]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HitResolutionScanResult.fieldnames(),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def run_single_hit_resolution_point(
    *,
    sigma_phi: float,
    sigma_z: float,
    n_events: int,
    n_particles: int,
    hit_efficiency: float,
    noise_hits_per_layer: int,
    seed: int,
) -> HitResolutionScanResult:
    config = DetectorEffectsConfig(
        hit_resolution=HitResolutionModel(
            sigma_phi=sigma_phi,
            sigma_z=sigma_z,
        )
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
            hit_efficiency=hit_efficiency,
            noise_hits_per_layer=noise_hits_per_layer,
            detector_effects=config,
            random_seed=seed + event_index,
            seed_mode="hole-aware",
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

        momentum_residuals.extend(float(value) for value in result.momentum_relative_errors)

    return HitResolutionScanResult(
        sigma_phi=sigma_phi,
        sigma_z=sigma_z,
        n_events=n_events,
        n_particles=n_particles,
        hit_efficiency=hit_efficiency,
        noise_hits_per_layer=noise_hits_per_layer,
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


def run_hit_resolution_scan(
    *,
    sigma_phi_values: tuple[float, ...],
    sigma_z_values: tuple[float, ...],
    n_events: int,
    n_particles: int,
    hit_efficiency: float,
    noise_hits_per_layer: int,
    seed: int,
    output_path: str | Path,
) -> list[HitResolutionScanResult]:
    if len(sigma_phi_values) != len(sigma_z_values):
        raise ValueError("sigma_phi_values and sigma_z_values must have the same length")

    results = [
        run_single_hit_resolution_point(
            sigma_phi=sigma_phi,
            sigma_z=sigma_z,
            n_events=n_events,
            n_particles=n_particles,
            hit_efficiency=hit_efficiency,
            noise_hits_per_layer=noise_hits_per_layer,
            seed=seed + 1000 * index,
        )
        for index, (sigma_phi, sigma_z) in enumerate(
            zip(sigma_phi_values, sigma_z_values)
        )
    ]

    _write_csv(output_path, results)
    return results


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.0 hit-resolution scan."
    )

    parser.add_argument(
        "--sigma-phi",
        default="0.0005,0.001,0.002,0.005",
        help="Comma-separated phi measurement resolutions.",
    )

    parser.add_argument(
        "--sigma-z",
        default="0.05,0.10,0.20,0.50",
        help="Comma-separated z measurement resolutions.",
    )

    parser.add_argument(
        "--n-events",
        type=int,
        default=30,
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
        "--noise-hits-per-layer",
        type=int,
        default=0,
        help="Number of random noise hits per layer.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )

    parser.add_argument(
        "--output-csv",
        default="docs/reports/v3_0_detector_effects/hit_resolution_scan.csv",
        help="Output CSV path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    sigma_phi_values = _parse_float_list(args.sigma_phi)
    sigma_z_values = _parse_float_list(args.sigma_z)

    print("OpenReco v3.0 hit-resolution scan")
    print(f"sigma_phi values: {sigma_phi_values}")
    print(f"sigma_z values:   {sigma_z_values}")
    print(f"events/point:     {args.n_events}")
    print(f"particles/event:  {args.n_particles}")
    print(f"output CSV:       {args.output_csv}")
    print()

    results = run_hit_resolution_scan(
        sigma_phi_values=sigma_phi_values,
        sigma_z_values=sigma_z_values,
        n_events=args.n_events,
        n_particles=args.n_particles,
        hit_efficiency=args.hit_efficiency,
        noise_hits_per_layer=args.noise_hits_per_layer,
        seed=args.seed,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"sigma_phi={result.sigma_phi:.4g}, "
            f"sigma_z={result.sigma_z:.4g}, "
            f"eff={result.tracking_efficiency_mean:.3f}, "
            f"fake={result.fake_rate_mean:.3f}, "
            f"dup={result.duplicate_rate_mean:.3f}, "
            f"mom_res_std={result.momentum_residual_std:.4f}, "
            f"chi2/ndof={result.mean_chi2_ndof:.3f}"
        )

    print()
    print(f"CSV saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())