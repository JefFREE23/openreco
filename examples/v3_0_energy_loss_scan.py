"""OpenReco v3.0 energy-loss scan.

This example studies how deterministic truth-side energy loss affects tracking.

Energy loss is applied during event generation through
LayerMaterial.energy_loss_mev. Reconstruction still fits tracks with the
existing constant-q/p model, so large energy loss can degrade chi2/ndof and
momentum residuals.
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
from openreco.detector_effects import DetectorEffectsConfig, LayerMaterial


@dataclass(frozen=True)
class EnergyLossScanResult:
    energy_loss_mev_per_layer: float
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


def _write_csv(path: str | Path, results: list[EnergyLossScanResult]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EnergyLossScanResult.fieldnames(),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def _energy_loss_config(energy_loss_mev_per_layer: float) -> DetectorEffectsConfig:
    return DetectorEffectsConfig(
        layer_materials=tuple(
            LayerMaterial(
                layer_id=layer_id,
                energy_loss_mev=energy_loss_mev_per_layer,
            )
            for layer_id in range(6)
        )
    )


def run_single_energy_loss_point(
    *,
    energy_loss_mev_per_layer: float,
    n_events: int,
    n_particles: int,
    min_hits: int,
    seed: int,
) -> EnergyLossScanResult:
    config = _energy_loss_config(energy_loss_mev_per_layer)

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
    momentum_uncertainties: list[float] = []
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
            max_fit_chi2_ndof=None,
            make_plot=False,
        )

        event = result.event
        tracks = result.tracks
        validation = result.validation

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
            momentum_uncertainties.extend(
                float(track.momentum_uncertainty) for track in tracks
            )
        else:
            hits_per_track.append(float("nan"))
            holes_per_track.append(float("nan"))
            covariance_valid_rates.append(float("nan"))

        momentum_residuals.extend(
            float(value) for value in result.momentum_relative_errors
        )

    return EnergyLossScanResult(
        energy_loss_mev_per_layer=energy_loss_mev_per_layer,
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
        momentum_uncertainty_mean=_safe_mean(momentum_uncertainties),
        momentum_residual_mean=_safe_mean(momentum_residuals),
        momentum_residual_std=_safe_pstdev(momentum_residuals),
    )


def run_energy_loss_scan(
    *,
    energy_loss_mev_values: tuple[float, ...],
    n_events: int,
    n_particles: int,
    min_hits: int,
    seed: int,
    output_path: str | Path,
) -> list[EnergyLossScanResult]:
    results = [
        run_single_energy_loss_point(
            energy_loss_mev_per_layer=energy_loss_mev,
            n_events=n_events,
            n_particles=n_particles,
            min_hits=min_hits,
            seed=seed + 1000 * index,
        )
        for index, energy_loss_mev in enumerate(energy_loss_mev_values)
    ]

    _write_csv(output_path, results)
    return results


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.0 energy-loss scan."
    )

    parser.add_argument(
        "--energy-loss-mev",
        default="0.0,1.0,5.0,10.0,25.0,50.0,100.0",
        help="Comma-separated deterministic energy loss values per layer in MeV.",
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
        default="docs/reports/v3_0_detector_effects/energy_loss_scan.csv",
        help="Output CSV path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    energy_loss_values = _parse_float_list(args.energy_loss_mev)

    print("OpenReco v3.0 energy-loss scan")
    print(f"energy loss/layer: {energy_loss_values} MeV")
    print(f"events/point:      {args.n_events}")
    print(f"particles/event:   {args.n_particles}")
    print(f"min hits:          {args.min_hits}")
    print(f"output CSV:        {args.output_csv}")
    print()

    results = run_energy_loss_scan(
        energy_loss_mev_values=energy_loss_values,
        n_events=args.n_events,
        n_particles=args.n_particles,
        min_hits=args.min_hits,
        seed=args.seed,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"dE/layer={result.energy_loss_mev_per_layer:.1f} MeV, "
            f"eff={result.tracking_efficiency_mean:.3f}, "
            f"tracks/event={result.reconstructed_tracks_mean:.2f}, "
            f"chi2/ndof={result.mean_chi2_ndof:.3f}, "
            f"p_unc={result.momentum_uncertainty_mean:.4f}, "
            f"mom_res_mean={result.momentum_residual_mean:.4f}, "
            f"mom_res_std={result.momentum_residual_std:.4f}"
        )

    print()
    print(f"CSV saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())