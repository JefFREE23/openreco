"""OpenReco v3.1 mass-resolution scan.

This example propagates controlled detector effects into a downstream physics
observable: the reconstructed invariant mass of a toy J/psi -> mu+ mu- decay.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from statistics import mean

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.detector_effects import (
    DetectorEffectsConfig,
    HitResolutionModel,
    MagneticFieldScale,
)
from openreco.invariant_mass import (
    JPSI_MASS_GEV,
    summarize_reconstructed_masses,
)
from openreco.resonance import generate_two_body_resonance_decay
from openreco.resonance_reconstruction import select_best_mass_candidate


DEFAULT_SCAN_POINTS: tuple[tuple[str, float], ...] = (
    ("baseline", 0.0),
    ("hit_resolution_scale", 0.5),
    ("hit_resolution_scale", 1.0),
    ("hit_resolution_scale", 2.0),
    ("hit_resolution_scale", 5.0),
    ("material_budget", 0.005),
    ("material_budget", 0.020),
    ("bfield_reco_scale", 0.98),
    ("bfield_reco_scale", 1.02),
)


@dataclass(frozen=True)
class MassResolutionScanResult:
    """Summary result for one v3.1 mass-resolution scan point."""

    study_name: str
    scan_value: float
    n_events: int
    events_with_candidate: int
    candidate_efficiency: float
    tracking_efficiency_mean: float
    reconstructed_tracks_mean: float
    truth_mass: float
    mass_mean: float
    mass_width: float
    residual_mean: float
    residual_width: float
    abs_residual_mean: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def _safe_mean(values: list[float]) -> float:
    finite_values = [value for value in values if isfinite(value)]
    return mean(finite_values) if finite_values else float("nan")


def _write_csv(
    path: str | Path,
    results: list[MassResolutionScanResult],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MassResolutionScanResult.fieldnames(),
        )
        writer.writeheader()

        for result in results:
            writer.writerow(result.to_dict())

    return output_path


def detector_effects_for_scan_point(
    *,
    study_name: str,
    scan_value: float,
) -> DetectorEffectsConfig:
    """Build a detector-effects config for one mass-resolution scan point."""

    if study_name == "baseline":
        return DetectorEffectsConfig.default()

    if study_name == "hit_resolution_scale":
        if scan_value <= 0.0:
            raise ValueError("hit_resolution_scale must be positive")

        return DetectorEffectsConfig(
            hit_resolution=HitResolutionModel(
                sigma_phi=1.0e-3 * scan_value,
                sigma_z=0.10 * scan_value,
            )
        )

    if study_name == "material_budget":
        return DetectorEffectsConfig.with_uniform_material(
            layer_ids=range(6),
            x_over_x0=scan_value,
        )

    if study_name == "bfield_reco_scale":
        return DetectorEffectsConfig(
            b_field_scale=MagneticFieldScale(
                truth_scale=1.0,
                reco_scale=scan_value,
            )
        )

    raise ValueError(f"unknown mass-resolution study: {study_name!r}")


def run_single_mass_resolution_point(
    *,
    study_name: str,
    scan_value: float,
    n_events: int,
    seed: int,
    min_hits: int,
) -> MassResolutionScanResult:
    """Run one detector-effect scan point."""

    detector_effects = detector_effects_for_scan_point(
        study_name=study_name,
        scan_value=scan_value,
    )

    reconstructed_masses: list[float] = []
    residuals: list[float] = []
    abs_residuals: list[float] = []
    tracking_efficiencies: list[float] = []
    reconstructed_track_counts: list[float] = []

    events_with_candidate = 0

    for event_index in range(n_events):
        decay = generate_two_body_resonance_decay(
            rng=np.random.default_rng(seed + 10_000 * event_index),
        )

        result = run_multi_track_reconstruction(
            event_id=event_index,
            n_particles=999,
            truth_particles=decay.truth_particles,
            hit_efficiency=1.0,
            noise_hits_per_layer=0,
            detector_effects=detector_effects,
            random_seed=seed + event_index,
            chi2_threshold=100.0,
            min_hits=min_hits,
            use_ekf_fit=False,
            make_plot=False,
        )

        tracking_efficiencies.append(float(result.validation.tracking_efficiency))
        reconstructed_track_counts.append(float(len(result.tracks)))

        candidate = select_best_mass_candidate(
            result.tracks,
            truth_mass=JPSI_MASS_GEV,
        )

        if candidate is None:
            continue

        events_with_candidate += 1
        reconstructed_masses.append(float(candidate.mass))
        residuals.append(float(candidate.mass_residual))
        abs_residuals.append(abs(float(candidate.mass_residual)))

    mass_summary = summarize_reconstructed_masses(
        reconstructed_masses,
        truth_mass=JPSI_MASS_GEV,
    )

    candidate_efficiency = (
        events_with_candidate / n_events if n_events > 0 else float("nan")
    )

    return MassResolutionScanResult(
        study_name=study_name,
        scan_value=float(scan_value),
        n_events=int(n_events),
        events_with_candidate=int(events_with_candidate),
        candidate_efficiency=float(candidate_efficiency),
        tracking_efficiency_mean=_safe_mean(tracking_efficiencies),
        reconstructed_tracks_mean=_safe_mean(reconstructed_track_counts),
        truth_mass=JPSI_MASS_GEV,
        mass_mean=mass_summary.mass_mean,
        mass_width=mass_summary.mass_width,
        residual_mean=mass_summary.residual_mean,
        residual_width=mass_summary.residual_width,
        abs_residual_mean=_safe_mean(abs_residuals),
    )


def run_mass_resolution_scan(
    *,
    scan_points: tuple[tuple[str, float], ...] = DEFAULT_SCAN_POINTS,
    n_events: int = 20,
    seed: int = 12345,
    min_hits: int = 6,
    output_path: str | Path = (
        "docs/reports/v3_1_toy_resonance/mass_resolution_scan.csv"
    ),
) -> list[MassResolutionScanResult]:
    """Run the v3.1 toy-resonance mass-resolution scan."""

    results = [
        run_single_mass_resolution_point(
            study_name=study_name,
            scan_value=scan_value,
            n_events=n_events,
            seed=seed + 1000 * index,
            min_hits=min_hits,
        )
        for index, (study_name, scan_value) in enumerate(scan_points)
    ]

    _write_csv(output_path, results)

    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v3.1 mass-resolution scan."
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=20,
        help="Number of toy resonance events per scan point.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=6,
        help="Minimum hits required by the track finder.",
    )
    parser.add_argument(
        "--output-csv",
        default="docs/reports/v3_1_toy_resonance/mass_resolution_scan.csv",
        help="Output CSV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    print("OpenReco v3.1 mass-resolution scan")
    print("------------------------------------")
    print(f"events/point: {args.n_events}")
    print(f"min hits: {args.min_hits}")
    print(f"output CSV: {args.output_csv}")
    print()

    results = run_mass_resolution_scan(
        n_events=args.n_events,
        seed=args.seed,
        min_hits=args.min_hits,
        output_path=args.output_csv,
    )

    for result in results:
        print(
            f"{result.study_name:>20s} "
            f"value={result.scan_value:>7.4f} "
            f"cand_eff={result.candidate_efficiency:.3f} "
            f"mass_mean={result.mass_mean:.6f} "
            f"mass_width={result.mass_width:.6f} "
            f"res_mean={result.residual_mean:+.6f} "
            f"res_width={result.residual_width:.6f}"
        )

    print()
    print(f"CSV saved: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())