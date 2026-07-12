from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional


import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openreco.detector_effects import DetectorEffectsConfig

from openreco.field import UniformMagneticField

from openreco.event_generation import (
    Event,
    count_noise_hits,
    count_real_hits,
    generate_event,
)
from openreco.seeding import (
    TripletSeed,
    build_triplet_seeds,
    build_triplet_seeds_for_layer_sets,
)
from openreco.track_finding import ReconstructedTrack, find_tracks_from_seeds
from openreco.track_fitting import fit_reconstructed_tracks_with_ekf
from openreco.truth_matching import ValidationSummary, validate_reconstructed_tracks


@dataclass(frozen=True)
class MultiTrackDemoResult:
    event: Event
    seeds: list[TripletSeed]
    tracks: list[ReconstructedTrack]
    validation: ValidationSummary
    plot_path: Optional[Path]
    momentum_relative_errors: list[float]


def run_multi_track_reconstruction(
    *,
    event_id: int = 0,
    n_particles: int = 5,
    hit_efficiency: float = 1.0,
    noise_hits_per_layer: int = 1,
    detector_effects: DetectorEffectsConfig | None = None,
    process_noise_scale: float = 0.0,
    random_seed: int = 123,
    chi2_threshold: float = 25.0,
    min_hits: int = 6,
    seed_mode: str = "strict",
    use_ekf_fit: bool = True,
    max_fit_chi2_ndof: float | None = 50.0,
    make_plot: bool = True,
    output_path: str | Path = "docs/images/v1_multi_track_event.png",
) -> MultiTrackDemoResult:
    """
    Run the full OpenReco v1 minimum reconstruction chain on one event.

    Chain:
        event generation
        -> triplet seeding
        -> greedy track finding
        -> optional EKF fitting/smoothing
        -> truth matching
        -> validation metrics
        -> optional XY event plot
    """

    rng = np.random.default_rng(random_seed)

    event = generate_event(
        event_id=event_id,
        n_particles=n_particles,
        hit_efficiency=hit_efficiency,
        noise_hits_per_layer=noise_hits_per_layer,
        measurement_sigma_phi=1.0e-3,
        measurement_sigma_z=0.10,
        detector_effects=detector_effects,
        pt_range=(2.0, 5.0),
        tan_lambda_range=(-0.8, 0.8),
        rng=rng,
    )

    if seed_mode == "strict":
        seeds = build_triplet_seeds(event.measurements_by_layer)
    elif seed_mode == "hole-aware":
        seeds = build_triplet_seeds_for_layer_sets(event.measurements_by_layer)
    else:
        raise ValueError("seed_mode must be either 'strict' or 'hole-aware'")

    raw_tracks = find_tracks_from_seeds(
        seeds,
        event.measurements_by_layer,
        chi2_threshold=chi2_threshold,
        min_hits=min_hits,
        allow_shared_hits=False,
        max_tracks=n_particles,
    )

    if use_ekf_fit:
        reco_bz = 2.0

        if detector_effects is not None:
            reco_bz *= detector_effects.b_field_scale.reco_scale

        tracks = fit_reconstructed_tracks_with_ekf(
            raw_tracks,
            fail_safely=True,
            field=UniformMagneticField(bz=reco_bz),
            detector_effects=detector_effects,
            process_noise_scale=process_noise_scale,
        )

        if max_fit_chi2_ndof is not None:
            tracks = [
                track
                for track in tracks
                if track.chi2_ndof <= max_fit_chi2_ndof
            ]

        tracks = [
            track
            for track in tracks
            if track.covariance_valid
            and math.isfinite(track.q_over_p)
            and math.isfinite(track.pt_estimate)
            and math.isfinite(track.p_estimate)
        ]
    else:
        tracks = raw_tracks

    validation = validate_reconstructed_tracks(
        tracks,
        n_truth_particles=len(event.truth_particles),
    )

    event.seeds = seeds
    event.reconstructed_tracks = tracks
    event.validation_results = {
        "tracking_efficiency": validation.tracking_efficiency,
        "fake_rate": validation.fake_rate,
        "duplicate_rate": validation.duplicate_rate,
        "matched_tracks": validation.n_matched_tracks,
        "fake_tracks": validation.n_fake_tracks,
        "duplicate_tracks": validation.n_duplicate_tracks,
    }

    momentum_relative_errors = compute_momentum_relative_errors(
        event=event,
        tracks=tracks,
        validation=validation,
    )

    plot_path = None
    if make_plot:
        plot_path = Path(output_path)
        plot_event_xy(
            event=event,
            tracks=tracks,
            output_path=plot_path,
        )

    return MultiTrackDemoResult(
        event=event,
        seeds=seeds,
        tracks=tracks,
        validation=validation,
        plot_path=plot_path,
        momentum_relative_errors=momentum_relative_errors,
    )


def compute_momentum_relative_errors(
    *,
    event: Event,
    tracks: list[ReconstructedTrack],
    validation: ValidationSummary,
) -> list[float]:
    """
    Compute (p_reco - p_truth) / p_truth for matched, non-duplicate tracks.
    """

    truth_by_id = {
        particle.truth_particle_id: particle
        for particle in event.truth_particles
    }

    errors: list[float] = []

    for match in validation.matches:
        if not match.is_matched:
            continue

        if match.is_duplicate:
            continue

        if match.matched_truth_particle_id is None:
            continue

        track = tracks[match.object_id]
        truth = truth_by_id[match.matched_truth_particle_id]

        errors.append((track.p_estimate - truth.p) / truth.p)

    return errors


def format_demo_summary(result: MultiTrackDemoResult) -> str:
    event = result.event
    validation = result.validation

    chi2_values = [
        track.chi2_ndof
        for track in result.tracks
        if np.isfinite(track.chi2_ndof)
    ]

    mean_chi2_ndof = mean(chi2_values) if chi2_values else float("nan")

    covariance_flags = [
        1.0 if getattr(track, "covariance_valid", False) else 0.0
        for track in result.tracks
    ]

    covariance_valid_rate = (
        mean(covariance_flags)
        if covariance_flags
        else float("nan")
    )

    hole_values = [
        track.n_holes
        for track in result.tracks
    ]
    mean_holes_per_track = mean(hole_values) if hole_values else float("nan")

    momentum_errors = result.momentum_relative_errors
    momentum_error_mean = mean(momentum_errors) if momentum_errors else float("nan")
    momentum_error_std = pstdev(momentum_errors) if len(momentum_errors) > 1 else 0.0

    lines = [
        "OpenReco v1 multi-track reconstruction",
        "",
        f"truth particles:        {len(event.truth_particles)}",
        f"measurements:           {len(event.measurements)}",
        f"real measurements:      {count_real_hits(event)}",
        f"noise measurements:     {count_noise_hits(event)}",
        f"seeds built:            {len(result.seeds)}",
        f"reconstructed tracks:   {len(result.tracks)}",
        f"matched tracks:         {validation.n_matched_tracks}",
        f"fake tracks:            {validation.n_fake_tracks}",
        f"duplicate tracks:       {validation.n_duplicate_tracks}",
        "",
        f"tracking efficiency:    {validation.tracking_efficiency:.3f}",
        f"fake rate:              {validation.fake_rate:.3f}",
        f"duplicate rate:         {validation.duplicate_rate:.3f}",
        f"mean chi2/ndof:         {mean_chi2_ndof:.3f}",
        f"covariance valid rate:  {covariance_valid_rate:.3f}",
        f"mean holes/track:       {mean_holes_per_track:.3f}",
        f"momentum rel error:     mean={momentum_error_mean:.4f}, std={momentum_error_std:.4f}",
    ]

    if result.plot_path is not None:
        lines.append(f"plot saved:             {result.plot_path}")

    return "\n".join(lines)


def plot_event_xy(
    *,
    event: Event,
    tracks: list[ReconstructedTrack],
    output_path: str | Path,
) -> None:
    """
    Save a simple XY event display.

    Shows:
    - all measurements
    - noise hits
    - reconstructed track hit sequences
    - cylindrical detector layers
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    layer_radii = sorted({hit.radius for hit in event.measurements})
    for radius in layer_radii:
        circle = plt.Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.35,
        )
        ax.add_patch(circle)

    real_hits = [hit for hit in event.measurements if not hit.is_noise]
    noise_hits = [hit for hit in event.measurements if hit.is_noise]

    if real_hits:
        ax.scatter(
            [hit.x for hit in real_hits],
            [hit.y for hit in real_hits],
            s=20,
            label="real hits",
        )

    if noise_hits:
        ax.scatter(
            [hit.x for hit in noise_hits],
            [hit.y for hit in noise_hits],
            s=20,
            marker="x",
            label="noise hits",
        )

    for track in tracks:
        hits = sorted(track.used_measurements, key=lambda hit: hit.radius)
        ax.plot(
            [hit.x for hit in hits],
            [hit.y for hit in hits],
            marker="o",
            linewidth=1.5,
            label=f"track {track.track_id}",
        )

    max_radius = max(layer_radii) if layer_radii else 1.0
    padding = 0.15 * max_radius

    ax.set_xlim(-max_radius - padding, max_radius + padding)
    ax.set_ylim(-max_radius - padding, max_radius + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("OpenReco v1 multi-track event display")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the OpenReco v1 multi-track reconstruction demo."
    )

    parser.add_argument("--event-id", type=int, default=0)
    parser.add_argument("--n-particles", type=int, default=5)
    parser.add_argument("--hit-efficiency", type=float, default=1.0)
    parser.add_argument("--noise-hits-per-layer", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--chi2-threshold", type=float, default=25.0)
    parser.add_argument("--min-hits", type=int, default=6)
    parser.add_argument(
        "--seed-mode",
        type=str,
        default="strict",
        choices=("strict", "hole-aware"),
        help="Use strict first-three-layer seeds or hole-aware multi-layer seeds.",
    )
    parser.add_argument(
        "--no-ekf-fit",
        action="store_true",
        help="Run the demo without EKF fitting/smoothing.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="docs/images/v1_multi_track_event.png",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run reconstruction without saving the event display.",
    )

    args = parser.parse_args()

    result = run_multi_track_reconstruction(
        event_id=args.event_id,
        n_particles=args.n_particles,
        hit_efficiency=args.hit_efficiency,
        noise_hits_per_layer=args.noise_hits_per_layer,
        random_seed=args.random_seed,
        chi2_threshold=args.chi2_threshold,
        min_hits=args.min_hits,
        seed_mode=args.seed_mode,
        use_ekf_fit=not args.no_ekf_fit,
        make_plot=not args.no_plot,
        output_path=args.output_path,
    )

    print(format_demo_summary(result))


if __name__ == "__main__":
    main()
