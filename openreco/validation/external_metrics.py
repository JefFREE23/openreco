"""Validation metrics for OpenReco v2 external ACTS-style datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from openreco.external.acts_schema import ActsTruthParticle
from openreco.external.reconstruction import (
    ExternalDatasetRecoSummary,
    ExternalRecoResult,
)


@dataclass(frozen=True)
class ExternalValidationMetrics:
    """Dataset-level validation metrics for external reconstruction."""

    n_events: int
    n_truth_particles: int
    n_reco_tracks: int
    n_matched_tracks: int
    n_unique_matched_truth_particles: int
    n_fake_tracks: int
    n_duplicate_tracks: int

    unique_tracking_efficiency: float
    raw_matched_track_efficiency: float
    fake_rate: float
    duplicate_rate: float

    mean_chi2_ndof: float
    covariance_valid_rate: float
    mean_momentum_relative_residual: float
    std_momentum_relative_residual: float
    runtime_per_event: float


def compute_external_validation_metrics(
    summary: ExternalDatasetRecoSummary,
) -> ExternalValidationMetrics:
    """Compute robust dataset-level metrics.

    The important difference from a naive matched-track count is that
    unique_tracking_efficiency counts unique truth particles, not matched
    reconstructed tracks. Therefore it cannot exceed 1.
    """

    n_events = summary.n_events
    n_truth_particles = summary.n_truth_particles
    n_reco_tracks = summary.n_reco_tracks

    all_matches = [
        match
        for result in summary.results
        for match in result.validation.matches
    ]

    n_matched_tracks = sum(1 for match in all_matches if match.is_matched)
    n_fake_tracks = sum(1 for match in all_matches if match.is_fake)
    n_duplicate_tracks = sum(1 for match in all_matches if match.is_duplicate)

    unique_truth_keys = {
        (result.event_id, match.matched_truth_particle_id)
        for result in summary.results
        for match in result.validation.matches
        if match.is_matched and match.matched_truth_particle_id is not None
    }

    n_unique_matched_truth_particles = len(unique_truth_keys)

    unique_tracking_efficiency = _safe_divide(
        n_unique_matched_truth_particles,
        n_truth_particles,
    )

    raw_matched_track_efficiency = _safe_divide(
        n_matched_tracks,
        n_truth_particles,
    )

    fake_rate = _safe_divide(n_fake_tracks, n_reco_tracks)
    duplicate_rate = _safe_divide(n_duplicate_tracks, n_reco_tracks)

    chi2_values = [
        track.chi2_ndof
        for result in summary.results
        for track in result.tracks
        if math.isfinite(track.chi2_ndof)
    ]

    covariance_flags = [
        1.0 if getattr(track, "covariance_valid", False) else 0.0
        for result in summary.results
        for track in result.tracks
    ]

    momentum_residuals = compute_momentum_relative_residuals(summary)

    return ExternalValidationMetrics(
        n_events=n_events,
        n_truth_particles=n_truth_particles,
        n_reco_tracks=n_reco_tracks,
        n_matched_tracks=n_matched_tracks,
        n_unique_matched_truth_particles=n_unique_matched_truth_particles,
        n_fake_tracks=n_fake_tracks,
        n_duplicate_tracks=n_duplicate_tracks,
        unique_tracking_efficiency=unique_tracking_efficiency,
        raw_matched_track_efficiency=raw_matched_track_efficiency,
        fake_rate=fake_rate,
        duplicate_rate=duplicate_rate,
        mean_chi2_ndof=mean(chi2_values) if chi2_values else float("nan"),
        covariance_valid_rate=mean(covariance_flags) if covariance_flags else float("nan"),
        mean_momentum_relative_residual=(
            mean(momentum_residuals) if momentum_residuals else float("nan")
        ),
        std_momentum_relative_residual=(
            pstdev(momentum_residuals) if len(momentum_residuals) > 1 else 0.0
        ),
        runtime_per_event=summary.runtime_per_event,
    )


def compute_momentum_relative_residuals(
    summary: ExternalDatasetRecoSummary,
) -> list[float]:
    """Compute (p_reco - p_truth) / p_truth for matched non-duplicate tracks."""

    residuals: list[float] = []

    for result in summary.results:
        truth_by_id = {
            particle.particle_id: particle
            for particle in result.adapted_event.truth_particles
        }

        for match in result.validation.matches:
            if not match.is_matched:
                continue

            if match.is_duplicate:
                continue

            if match.matched_truth_particle_id is None:
                continue

            if match.object_id >= len(result.tracks):
                continue

            truth = truth_by_id.get(match.matched_truth_particle_id)
            if truth is None:
                continue

            track = result.tracks[match.object_id]
            truth_p = truth_momentum(truth)

            if truth_p <= 0:
                continue

            if not math.isfinite(track.p_estimate):
                continue

            residuals.append((track.p_estimate - truth_p) / truth_p)

    return residuals


def event_summary_rows(
    summary: ExternalDatasetRecoSummary,
) -> list[dict[str, Any]]:
    """Return per-event validation rows suitable for CSV output."""

    rows: list[dict[str, Any]] = []

    for result in summary.results:
        rows.append(_event_summary_row(result))

    return rows


def track_summary_rows(
    summary: ExternalDatasetRecoSummary,
) -> list[dict[str, Any]]:
    """Return per-track validation rows suitable for CSV output."""

    rows: list[dict[str, Any]] = []

    for result in summary.results:
        rows.extend(_track_summary_rows_for_event(result))

    return rows


def truth_momentum(particle: ActsTruthParticle) -> float:
    """Return total truth momentum from external px, py, pz."""

    return math.sqrt(
        particle.px**2
        + particle.py**2
        + particle.pz**2
    )


def format_external_validation_metrics(
    metrics: ExternalValidationMetrics,
) -> str:
    """Format validation metrics for CLI output."""

    lines = [
        "OpenReco v2 external validation metrics",
        "",
        f"events processed:             {metrics.n_events}",
        f"truth particles:              {metrics.n_truth_particles}",
        f"reconstructed tracks:         {metrics.n_reco_tracks}",
        f"matched tracks:               {metrics.n_matched_tracks}",
        f"unique matched truth:         {metrics.n_unique_matched_truth_particles}",
        f"fake tracks:                  {metrics.n_fake_tracks}",
        f"duplicate tracks:             {metrics.n_duplicate_tracks}",
        "",
        f"unique tracking efficiency:   {metrics.unique_tracking_efficiency:.3f}",
        f"raw matched-track efficiency: {metrics.raw_matched_track_efficiency:.3f}",
        f"fake rate:                    {metrics.fake_rate:.3f}",
        f"duplicate rate:               {metrics.duplicate_rate:.3f}",
        f"mean chi2/ndof:               {metrics.mean_chi2_ndof:.3f}",
        f"covariance valid rate:        {metrics.covariance_valid_rate:.3f}",
        (
            "momentum rel residual:        "
            f"mean={metrics.mean_momentum_relative_residual:.4f}, "
            f"std={metrics.std_momentum_relative_residual:.4f}"
        ),
        f"runtime/event:                {metrics.runtime_per_event:.6f} s",
    ]

    return "\n".join(lines)


def _event_summary_row(result: ExternalRecoResult) -> dict[str, Any]:
    n_truth = len(result.adapted_event.truth_particles)
    n_reco = len(result.tracks)

    matches = result.validation.matches

    n_matched = sum(1 for match in matches if match.is_matched)
    n_fake = sum(1 for match in matches if match.is_fake)
    n_duplicate = sum(1 for match in matches if match.is_duplicate)

    unique_truth_ids = {
        match.matched_truth_particle_id
        for match in matches
        if match.is_matched and match.matched_truth_particle_id is not None
    }

    chi2_values = [
        track.chi2_ndof
        for track in result.tracks
        if math.isfinite(track.chi2_ndof)
    ]

    covariance_flags = [
        1.0 if getattr(track, "covariance_valid", False) else 0.0
        for track in result.tracks
    ]

    return {
        "event_id": result.event_id,
        "n_truth_particles": n_truth,
        "n_measurements": len(result.adapted_event.measurements),
        "n_seeds": len(result.seeds),
        "n_reco_tracks": n_reco,
        "n_matched_tracks": n_matched,
        "n_unique_matched_truth_particles": len(unique_truth_ids),
        "n_fake_tracks": n_fake,
        "n_duplicate_tracks": n_duplicate,
        "unique_tracking_efficiency": _safe_divide(len(unique_truth_ids), n_truth),
        "raw_matched_track_efficiency": _safe_divide(n_matched, n_truth),
        "fake_rate": _safe_divide(n_fake, n_reco),
        "duplicate_rate": _safe_divide(n_duplicate, n_reco),
        "mean_chi2_ndof": mean(chi2_values) if chi2_values else float("nan"),
        "covariance_valid_rate": mean(covariance_flags) if covariance_flags else float("nan"),
        "runtime_seconds": result.runtime_seconds,
    }


def _track_summary_rows_for_event(result: ExternalRecoResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    truth_by_id = {
        particle.particle_id: particle
        for particle in result.adapted_event.truth_particles
    }

    matches_by_object_id = {
        match.object_id: match
        for match in result.validation.matches
    }

    for track_index, track in enumerate(result.tracks):
        match = matches_by_object_id.get(track_index)

        matched_truth_particle_id = None
        matched_fraction = 0.0
        is_matched = False
        is_fake = True
        is_duplicate = False

        truth_p = float("nan")
        momentum_relative_residual = float("nan")

        if match is not None:
            matched_truth_particle_id = match.matched_truth_particle_id
            matched_fraction = match.matched_fraction
            is_matched = match.is_matched
            is_fake = match.is_fake
            is_duplicate = match.is_duplicate

            if matched_truth_particle_id is not None:
                truth = truth_by_id.get(matched_truth_particle_id)
                if truth is not None:
                    truth_p = truth_momentum(truth)
                    if truth_p > 0 and math.isfinite(track.p_estimate):
                        momentum_relative_residual = (
                            track.p_estimate - truth_p
                        ) / truth_p

        rows.append(
            {
                "event_id": result.event_id,
                "track_index": track_index,
                "track_id": track.track_id,
                "seed_id": track.seed_id,
                "n_used_hits": len(track.used_measurements),
                "matched_truth_particle_id": matched_truth_particle_id,
                "matched_fraction": matched_fraction,
                "is_matched": is_matched,
                "is_fake": is_fake,
                "is_duplicate": is_duplicate,
                "chi2": track.chi2,
                "ndof": track.ndof,
                "chi2_ndof": track.chi2_ndof,
                "q_over_p": track.q_over_p,
                "pt_estimate": track.pt_estimate,
                "p_estimate": track.p_estimate,
                "truth_p": truth_p,
                "momentum_relative_residual": momentum_relative_residual,
                "covariance_valid": getattr(track, "covariance_valid", False),
                "momentum_uncertainty": getattr(track, "momentum_uncertainty", float("nan")),
                "n_holes": getattr(track, "n_holes", 0),
                "fit_status": getattr(track, "fit_status", ""),
            }
        )

    return rows


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator
