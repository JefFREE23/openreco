"""Run the OpenReco v1 reconstruction chain on adapted ACTS-style events."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np

from openreco.event_generation import EventHit
from openreco.external.acts_adapter import (
    AdaptedOpenRecoEvent,
    ActsAdapterConfig,
    convert_acts_event_to_openreco,
)
from openreco.external.acts_schema import ActsDataset, ActsEvent
from openreco.seeding import (
    TripletSeed,
    build_triplet_seeds,
    build_triplet_seeds_for_layer_sets,
)
from openreco.track_finding import ReconstructedTrack, find_tracks_from_seeds
from openreco.track_fitting import fit_reconstructed_tracks_with_ekf
from openreco.truth_matching import ValidationSummary, validate_reconstructed_tracks


@dataclass(frozen=True)
class ExternalRecoResult:
    """Result of running OpenReco v1 reconstruction on one external event."""

    event_id: int
    adapted_event: AdaptedOpenRecoEvent
    measurements_by_layer: dict[str, list[EventHit]]
    seeds: list[TripletSeed]
    tracks: list[ReconstructedTrack]
    validation: ValidationSummary
    runtime_seconds: float


@dataclass(frozen=True)
class ExternalDatasetRecoSummary:
    """Summary over an external ACTS-style dataset."""

    results: list[ExternalRecoResult]

    @property
    def n_events(self) -> int:
        return len(self.results)

    @property
    def n_truth_particles(self) -> int:
        return sum(len(result.adapted_event.truth_particles) for result in self.results)

    @property
    def n_measurements(self) -> int:
        return sum(len(result.adapted_event.measurements) for result in self.results)

    @property
    def n_seeds(self) -> int:
        return sum(len(result.seeds) for result in self.results)

    @property
    def n_reco_tracks(self) -> int:
        return sum(len(result.tracks) for result in self.results)

    @property
    def n_matched_tracks(self) -> int:
        return sum(result.validation.n_matched_tracks for result in self.results)

    @property
    def n_fake_tracks(self) -> int:
        return sum(result.validation.n_fake_tracks for result in self.results)

    @property
    def n_duplicate_tracks(self) -> int:
        return sum(result.validation.n_duplicate_tracks for result in self.results)

    @property
    def tracking_efficiency(self) -> float:
        if self.n_truth_particles == 0:
            return float("nan")
        return self.n_matched_tracks / self.n_truth_particles

    @property
    def fake_rate(self) -> float:
        if self.n_reco_tracks == 0:
            return 0.0
        return self.n_fake_tracks / self.n_reco_tracks

    @property
    def duplicate_rate(self) -> float:
        if self.n_reco_tracks == 0:
            return 0.0
        return self.n_duplicate_tracks / self.n_reco_tracks

    @property
    def mean_chi2_ndof(self) -> float:
        values = [
            track.chi2_ndof
            for result in self.results
            for track in result.tracks
            if math.isfinite(track.chi2_ndof)
        ]
        return mean(values) if values else float("nan")

    @property
    def runtime_per_event(self) -> float:
        if not self.results:
            return float("nan")
        return mean(result.runtime_seconds for result in self.results)


def event_hits_from_adapted_event(
    adapted_event: AdaptedOpenRecoEvent,
) -> dict[str, list[EventHit]]:
    """Convert adapted OpenReco measurements/source rows into EventHit objects.

    v1 seeding and track finding operate on EventHit objects grouped by layer.
    """

    layer_index_by_name = {
        layer.name: index
        for index, layer in enumerate(adapted_event.detector.layers)
    }

    radius_by_name = {
        layer.name: layer.radius
        for layer in adapted_event.detector.layers
    }

    hits_by_layer: dict[str, list[EventHit]] = defaultdict(list)

    for source_measurement, measurement in zip(
        adapted_event.source_measurements,
        adapted_event.measurements,
        strict=True,
    ):
        layer_name = measurement.layer_name

        hit = EventHit(
            hit_id=source_measurement.measurement_id,
            layer_index=layer_index_by_name[layer_name],
            layer_name=layer_name,
            radius=radius_by_name[layer_name],
            phi=float(measurement.values[0]),
            z=float(measurement.values[1]),
            covariance=measurement.covariance,
            truth_particle_id=source_measurement.particle_id,
            is_noise=source_measurement.is_noise,
        )

        hits_by_layer[layer_name].append(hit)

    return dict(hits_by_layer)


def run_external_event_reconstruction(
    event: ActsEvent,
    *,
    adapter_config: ActsAdapterConfig | None = None,
    seed_mode: str = "strict",
    chi2_threshold: float = 25.0,
    min_hits: int = 5,
    allow_shared_hits: bool = False,
    max_tracks: int | None = None,
    use_ekf_fit: bool = True,
    max_fit_chi2_ndof: float | None = 50.0,
) -> ExternalRecoResult:
    """Run the v1 reconstruction chain on one ACTS-style external event."""

    start = time.perf_counter()

    adapted_event = convert_acts_event_to_openreco(
        event,
        config=adapter_config,
    )

    measurements_by_layer = event_hits_from_adapted_event(adapted_event)

    if seed_mode == "strict":
        seeds = build_triplet_seeds(measurements_by_layer)
    elif seed_mode == "hole-aware":
        seeds = build_triplet_seeds_for_layer_sets(measurements_by_layer)
    else:
        raise ValueError("seed_mode must be either 'strict' or 'hole-aware'.")

    raw_tracks = find_tracks_from_seeds(
        seeds,
        measurements_by_layer,
        chi2_threshold=chi2_threshold,
        min_hits=min_hits,
        allow_shared_hits=allow_shared_hits,
        max_tracks=max_tracks,
    )

    if use_ekf_fit:
        tracks = fit_reconstructed_tracks_with_ekf(
            raw_tracks,
            fail_safely=True,
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
            if track.fit_status == "accepted"
            and math.isfinite(track.q_over_p)
            and math.isfinite(track.pt_estimate)
            and math.isfinite(track.p_estimate)
        ]
    else:
        tracks = raw_tracks

    validation = validate_reconstructed_tracks(
        tracks,
        n_truth_particles=len(adapted_event.truth_particles),
    )

    runtime_seconds = time.perf_counter() - start

    return ExternalRecoResult(
        event_id=event.event_id,
        adapted_event=adapted_event,
        measurements_by_layer=measurements_by_layer,
        seeds=seeds,
        tracks=tracks,
        validation=validation,
        runtime_seconds=runtime_seconds,
    )


def run_external_dataset_reconstruction(
    dataset: ActsDataset,
    *,
    max_events: int | None = None,
    **kwargs,
) -> ExternalDatasetRecoSummary:
    """Run external reconstruction over an ACTS-style dataset."""

    events = dataset.events[:max_events] if max_events is not None else dataset.events

    results = [
        run_external_event_reconstruction(event, **kwargs)
        for event in events
    ]

    return ExternalDatasetRecoSummary(results=results)


def format_external_reco_summary(summary: ExternalDatasetRecoSummary) -> str:
    """Format a human-readable summary for CLI examples."""

    lines = [
        "OpenReco v2 ACTS-style external validation",
        "",
        f"events processed:       {summary.n_events}",
        f"truth particles:        {summary.n_truth_particles}",
        f"measurements:           {summary.n_measurements}",
        f"seeds built:            {summary.n_seeds}",
        f"reconstructed tracks:   {summary.n_reco_tracks}",
        f"matched tracks:         {summary.n_matched_tracks}",
        f"fake tracks:            {summary.n_fake_tracks}",
        f"duplicate tracks:       {summary.n_duplicate_tracks}",
        "",
        f"tracking efficiency:    {summary.tracking_efficiency:.3f}",
        f"fake rate:              {summary.fake_rate:.3f}",
        f"duplicate rate:         {summary.duplicate_rate:.3f}",
        f"mean chi2/ndof:         {summary.mean_chi2_ndof:.3f}",
        f"runtime/event:          {summary.runtime_per_event:.6f} s",
    ]

    return "\n".join(lines)
