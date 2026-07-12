"""OpenReco v3.0 hit-inefficiency and dead-layer scan.

This example studies how detector hit inefficiency and dead detector layers
affect event-level tracking performance.

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
    DeadLayerModel,
    DetectorEffectsConfig,
    InefficiencyModel,
)


@dataclass(frozen=True)
class InefficiencyDeadLayerScanResult:
    hit_efficiency: float
    dead_layer_scenario: str
    dead_layers: str
    n_dead_layers: int
    n_events: int
    n_particles: int
    noise_hits_per_layer: int
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


def _dead_layers_to_string(dead_layers: tuple[int, ...]) -> str:
    if not dead_layers:
        return "none"
    return ",".join(str(layer) for layer in dead_layers)


def _write_csv(
    path: str | Path,
    results: list[InefficiencyDeadLayerScanResult],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=InefficiencyDeadLayerScanResult.fieldnames(),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def run_single_inefficiency_dead_layer_point(
    *,
    hit_efficiency: float,
    dead_layer_scenario: str,
    dead_layers: tuple[int, ...],
    n_events: int,
    n_particles: int,
    noise_hits_per_layer: int,
    min_hits: int,
    seed: int,
) -> InefficiencyDeadLayerScanResult:
    config = DetectorEffectsConfig(
        inefficiency=InefficiencyModel(hit_efficiency=hit_efficiency),
        dead_layers=DeadLayerModel(dead_layers=list(dead_layers)),
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
            noise_hits_per_layer=noise_hits_per_layer,
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

    return InefficiencyDeadLayerScanResult(
        hit_efficiency=hit_efficiency,
        dead_layer_scenario=dead_layer_scenario,
        dead_layers=_dead_layers_to_string(dead_layers),
        n_dead_layers=len(dead_layers),
        n_events=n_events,
        n_particles=n_particles,
        noise_hits_per_layer=noise_hits_per_layer,
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


def run_inefficiency_dead_layer_scan(
    *,
    hit_efficiency_values: tuple[float, ...],
    dead_layer_scenarios: tuple[tuple[str, tuple[int, ...]], ...],
    n_events: int,
    n_particles: int,
    noise_hits_per_layer: int,
    min_hits: int,
    seed: int,
    output_path: str | Path,
) -> list[InefficiencyDeadLayerScanResult]:
    results: list[InefficiencyDeadLayerScanResult] = []

    scan_index = 0
    for hit_efficiency in hit_efficiency_values:
        for scenario_name, dead_layers in dead_layer_scenarios:
            results.append(
                run_single_inefficiency_dead_layer_point(
                    hit_efficiency=hit_efficiency,
                    dead_layer_scenario=scenario_name,
                    dead_layers=dead_layers,
                    n_events=n_events,
                    n_particles=n_particles,
                    noise_hits_per_layer=noise_hits_per_layer,
                    min_hits=min_hits,
                    seed=seed + 1000 * scan_index,
                )
            )
            scan_index += 1

    _write_csv(output_path, results)
    return results


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _parse_dead_layer_scenarios(
    raw: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    scenarios: list[tuple[str, tuple[int, ...]]] = []

    for raw_scenario in raw.split(";"):
        item = raw_scenario.strip()

        if not item or item.lower() == "none":
            scenarios.append(("none", ()))
            continue

        dead_layers = tuple(
            int(layer.strip())
            for layer in item.split(",")
            if layer.strip()
        )
        name = "dead_" + "_".join(str(layer) for layer in dead_layers)
        scenarios.append((name, dead_layers))

    return tuple(scenarios)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.0 inefficiency/dead-layer scan."
    )

    parser.add_argument(
        "--hit-efficiencies",
        default="1.0,0.95,0.9,0.8",
        help="Comma-separated truth hit efficiencies.",
    )

    parser.add_argument(
        "--dead-layer-scenarios",
        default="none;2;1,4",
        help=(
            "Semicolon-separated dead-layer scenarios. "
            "Examples: 'none;2;1,4'"
        ),
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
        "--noise-hits-per-layer",
        type=int,
        default=0,
        help="Number of random noise hits per layer.",
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
        default="docs/reports/v3_0_detector_effects/inefficiency_dead_layer_scan.csv",
        help="Output CSV path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    hit_efficiency_values = _parse_float_list(args.hit_efficiencies)
    dead_layer_scenarios = _parse_dead_layer_scenarios(args.dead_layer_scenarios)

    print("OpenReco v3.0 inefficiency/dead-layer scan")
    print(f"hit efficiencies:     {hit_efficiency_values}")
    print(f"dead-layer scenarios: {dead_layer_scenarios}")
    print(f"events/point:         {args.n_events}")
    print(f"particles/event:      {args.n_particles}")
    print(f"min hits:             {args.min_hits}")
    print(f"output CSV:           {args.output_csv}")
    print()

    results = run_inefficiency_dead_layer_scan(
        hit_efficiency_values=hit_efficiency_values,
        dead_layer_scenarios=dead_layer_scenarios,
        n_events=args.n_events,
        n_particles=args.n_particles,
        noise_hits_per_layer=args.noise_hits_per_layer,
        min_hits=args.min_hits,
        seed=args.seed,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"hit_eff={result.hit_efficiency:.2f}, "
            f"scenario={result.dead_layer_scenario}, "
            f"eff={result.tracking_efficiency_mean:.3f}, "
            f"fake={result.fake_rate_mean:.3f}, "
            f"dup={result.duplicate_rate_mean:.3f}, "
            f"tracks/event={result.reconstructed_tracks_mean:.2f}, "
            f"holes/track={result.mean_holes_per_track:.2f}, "
            f"mom_res_std={result.momentum_residual_std:.4f}"
        )

    print()
    print(f"CSV saved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())