"""OpenReco v3.0 process-noise scale scan.

This example studies reconstruction-side uncertainty calibration.

Truth-side material scattering is applied during event generation through
DetectorEffectsConfig.layer_materials. Reconstruction then repeats the same
sample with different process_noise_scale values:

    process_noise_scale = 0.0  -> reconstruction ignores material process noise
    process_noise_scale > 0.0  -> EKF covariance is inflated for material

The main observables are chi2/ndof, covariance size, momentum uncertainty,
and momentum residual width.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.detector_effects import DetectorEffectsConfig
from openreco.uncertainty_calibration import (
    CalibrationChoice,
    find_best_calibration_scale,
)


@dataclass(frozen=True)
class ProcessNoiseScanResult:
    process_noise_scale: float
    x_over_x0_per_layer: float
    n_events: int
    n_particles: int
    min_hits: int
    events_processed: int
    truth_particles_generated: int
    reconstructed_tracks_mean: float
    tracking_efficiency_mean: float
    fake_rate_mean: float
    duplicate_rate_mean: float
    mean_hits_per_track: float
    mean_holes_per_track: float
    mean_chi2_ndof: float
    covariance_valid_rate_mean: float
    covariance_trace_mean: float
    momentum_uncertainty_mean: float
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


def _write_csv(path: str | Path, results: list[ProcessNoiseScanResult]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ProcessNoiseScanResult.fieldnames(),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path

def choose_best_process_noise_scale(
    results: list[ProcessNoiseScanResult],
) -> CalibrationChoice:
    """Choose the process-noise scale with mean chi2/ndof closest to 1."""

    return find_best_calibration_scale(
        results,
        scale_key="process_noise_scale",
        metric_key="mean_chi2_ndof",
        target_value=1.0,
    )


def run_single_process_noise_point(
    *,
    process_noise_scale: float,
    x_over_x0_per_layer: float,
    n_events: int,
    n_particles: int,
    min_hits: int,
    seed: int,
) -> ProcessNoiseScanResult:
    config = DetectorEffectsConfig.with_uniform_material(
        layer_ids=range(6),
        x_over_x0=x_over_x0_per_layer,
    )

    track_counts: list[float] = []
    tracking_efficiencies: list[float] = []
    fake_rates: list[float] = []
    duplicate_rates: list[float] = []
    hits_per_track: list[float] = []
    holes_per_track: list[float] = []
    chi2_ndof_values: list[float] = []
    covariance_valid_rates: list[float] = []
    covariance_traces: list[float] = []
    momentum_uncertainties: list[float] = []
    momentum_residuals: list[float] = []

    for event_index in range(n_events):
        result = run_multi_track_reconstruction(
            event_id=event_index,
            n_particles=n_particles,
            hit_efficiency=1.0,
            noise_hits_per_layer=0,
            detector_effects=config,
            process_noise_scale=process_noise_scale,
            random_seed=seed + event_index,
            seed_mode="hole-aware",
            min_hits=min_hits,
            max_fit_chi2_ndof=None,
            make_plot=False,
        )

        tracks = result.tracks
        validation = result.validation

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

            for track in tracks:
                if track.final_covariance is not None:
                    covariance_traces.append(float(np.trace(track.final_covariance)))

                momentum_uncertainties.append(float(track.momentum_uncertainty))
        else:
            hits_per_track.append(float("nan"))
            holes_per_track.append(float("nan"))
            covariance_valid_rates.append(float("nan"))

        momentum_residuals.extend(
            float(value) for value in result.momentum_relative_errors
        )

    return ProcessNoiseScanResult(
        process_noise_scale=process_noise_scale,
        x_over_x0_per_layer=x_over_x0_per_layer,
        n_events=n_events,
        n_particles=n_particles,
        min_hits=min_hits,
        events_processed=n_events,
        truth_particles_generated=n_events * n_particles,
        reconstructed_tracks_mean=_safe_mean(track_counts),
        tracking_efficiency_mean=_safe_mean(tracking_efficiencies),
        fake_rate_mean=_safe_mean(fake_rates),
        duplicate_rate_mean=_safe_mean(duplicate_rates),
        mean_hits_per_track=_safe_mean(hits_per_track),
        mean_holes_per_track=_safe_mean(holes_per_track),
        mean_chi2_ndof=_safe_mean(chi2_ndof_values),
        covariance_valid_rate_mean=_safe_mean(covariance_valid_rates),
        covariance_trace_mean=_safe_mean(covariance_traces),
        momentum_uncertainty_mean=_safe_mean(momentum_uncertainties),
        momentum_residual_mean=_safe_mean(momentum_residuals),
        momentum_residual_std=_safe_pstdev(momentum_residuals),
    )


def run_process_noise_scan(
    *,
    process_noise_scales: tuple[float, ...],
    x_over_x0_per_layer: float,
    n_events: int,
    n_particles: int,
    min_hits: int,
    seed: int,
    output_path: str | Path,
) -> list[ProcessNoiseScanResult]:
    results = [
        run_single_process_noise_point(
            process_noise_scale=process_noise_scale,
            x_over_x0_per_layer=x_over_x0_per_layer,
            n_events=n_events,
            n_particles=n_particles,
            min_hits=min_hits,
            seed=seed,
        )
        for process_noise_scale in process_noise_scales
    ]

    _write_csv(output_path, results)
    return results


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.0 process-noise scale scan."
    )

    parser.add_argument(
        "--process-noise-scales",
        default="0.0,0.5,1.0,2.0,5.0,10.0,20.0",
        help="Comma-separated process-noise scale factors.",
    )

    parser.add_argument(
        "--x-over-x0",
        type=float,
        default=0.02,
        help="Truth material thickness per detector layer.",
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
        help="Base random seed. Same seed is reused for every scale.",
    )

    parser.add_argument(
        "--output-csv",
        default="docs/reports/v3_0_detector_effects/process_noise_scan.csv",
        help="Output CSV path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    process_noise_scales = _parse_float_list(args.process_noise_scales)

    print("OpenReco v3.0 process-noise scale scan")
    print(f"process-noise scales: {process_noise_scales}")
    print(f"x/X0 per layer:       {args.x_over_x0}")
    print(f"events/point:         {args.n_events}")
    print(f"particles/event:      {args.n_particles}")
    print(f"min hits:             {args.min_hits}")
    print(f"output CSV:           {args.output_csv}")
    print()

    results = run_process_noise_scan(
        process_noise_scales=process_noise_scales,
        x_over_x0_per_layer=args.x_over_x0,
        n_events=args.n_events,
        n_particles=args.n_particles,
        min_hits=args.min_hits,
        seed=args.seed,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"scale={result.process_noise_scale:.2f}, "
            f"eff={result.tracking_efficiency_mean:.3f}, "
            f"tracks/event={result.reconstructed_tracks_mean:.2f}, "
            f"chi2/ndof={result.mean_chi2_ndof:.3f}, "
            f"cov_trace={result.covariance_trace_mean:.4e}, "
            f"p_unc={result.momentum_uncertainty_mean:.4f}, "
            f"mom_res_std={result.momentum_residual_std:.4f}"
        )
    best_choice = choose_best_process_noise_scale(results)

    print()
    print("Best process-noise calibration by chi2/ndof:")
    print(
        f"scale={best_choice.scale:.2f}, "
        f"mean_chi2/ndof={best_choice.metric_value:.3f}, "
        f"target={best_choice.target_value:.3f}, "
        f"abs_distance={best_choice.absolute_distance:.3f}"
    )

    print()
    print(f"CSV saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())