from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TruthMatchResult:
    """
    Result of matching one seed or reconstructed track to truth.

    matched:
        The largest truth contribution is at least min_fraction.

    fake:
        No truth particle contributes enough hits.

    duplicate:
        More than one reconstructed object is matched to the same truth particle.
        This is filled by mark_duplicates().
    """

    object_id: int
    n_hits: int
    truth_counts: dict[int, int]
    matched_truth_particle_id: Optional[int]
    matched_fraction: float
    is_matched: bool
    is_fake: bool
    is_duplicate: bool = False


@dataclass(frozen=True)
class ValidationSummary:
    """
    Event-level validation summary for reconstructed tracks.
    """

    n_truth_particles: int
    n_reconstructed_tracks: int
    n_matched_tracks: int
    n_fake_tracks: int
    n_duplicate_tracks: int
    tracking_efficiency: float
    fake_rate: float
    duplicate_rate: float
    matches: list[TruthMatchResult]


def match_hits_to_truth(
    hits: list[Any],
    *,
    object_id: int = 0,
    min_fraction: float = 0.50,
) -> TruthMatchResult:
    """
    Match one seed/track to the truth particle contributing most of its hits.

    Rules:
    - Count only real hits with truth_particle_id not None.
    - If largest contribution >= min_fraction of all hits, object is matched.
    - Otherwise it is fake.

    Example:
    - 5/6 hits from truth particle 3 -> matched to 3
    - 2/6, 2/6, 2/6 from different particles -> fake
    """

    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_fraction must be between 0 and 1.")

    n_hits = len(hits)

    if n_hits == 0:
        return TruthMatchResult(
            object_id=object_id,
            n_hits=0,
            truth_counts={},
            matched_truth_particle_id=None,
            matched_fraction=0.0,
            is_matched=False,
            is_fake=True,
            is_duplicate=False,
        )

    truth_ids = [
        getattr(hit, "truth_particle_id")
        for hit in hits
        if getattr(hit, "truth_particle_id", None) is not None
    ]

    truth_counts_counter = Counter(truth_ids)
    truth_counts = dict(truth_counts_counter)

    if not truth_counts:
        return TruthMatchResult(
            object_id=object_id,
            n_hits=n_hits,
            truth_counts={},
            matched_truth_particle_id=None,
            matched_fraction=0.0,
            is_matched=False,
            is_fake=True,
            is_duplicate=False,
        )

    matched_truth_particle_id, n_matched_hits = truth_counts_counter.most_common(1)[0]
    matched_fraction = n_matched_hits / n_hits
    is_matched = matched_fraction >= min_fraction

    return TruthMatchResult(
        object_id=object_id,
        n_hits=n_hits,
        truth_counts=truth_counts,
        matched_truth_particle_id=int(matched_truth_particle_id) if is_matched else None,
        matched_fraction=float(matched_fraction),
        is_matched=bool(is_matched),
        is_fake=not bool(is_matched),
        is_duplicate=False,
    )


def match_object_to_truth(
    obj: Any,
    *,
    object_id: int | None = None,
    min_fraction: float = 0.50,
) -> TruthMatchResult:
    """
    Match a generic seed or track object to truth.

    Supported object shapes:
    - obj.hits
    - obj.used_measurements
    - obj.measurements

    This keeps the function useful for TripletSeed now and ReconstructedTrack later.
    """

    hits = extract_hits(obj)

    if object_id is None:
        object_id = _extract_object_id(obj)

    return match_hits_to_truth(
        hits,
        object_id=object_id,
        min_fraction=min_fraction,
    )


def mark_duplicates(matches: list[TruthMatchResult]) -> list[TruthMatchResult]:
    """
    Mark duplicate reconstructed objects.

    Rule:
    If multiple objects match the same truth particle, keep the first one as non-duplicate
    and mark the later ones as duplicates.
    """

    seen_truth_ids: set[int] = set()
    output: list[TruthMatchResult] = []

    for match in matches:
        is_duplicate = False

        if match.is_matched and match.matched_truth_particle_id is not None:
            if match.matched_truth_particle_id in seen_truth_ids:
                is_duplicate = True
            else:
                seen_truth_ids.add(match.matched_truth_particle_id)

        output.append(
            TruthMatchResult(
                object_id=match.object_id,
                n_hits=match.n_hits,
                truth_counts=match.truth_counts,
                matched_truth_particle_id=match.matched_truth_particle_id,
                matched_fraction=match.matched_fraction,
                is_matched=match.is_matched,
                is_fake=match.is_fake,
                is_duplicate=is_duplicate,
            )
        )

    return output


def validate_reconstructed_tracks(
    reconstructed_tracks: list[Any],
    *,
    n_truth_particles: int,
    min_fraction: float = 0.50,
) -> ValidationSummary:
    """
    Validate one event of reconstructed tracks.

    Metrics:
    - tracking efficiency = unique matched truth particles / generated truth particles
    - fake rate = fake tracks / reconstructed tracks
    - duplicate rate = duplicate tracks / reconstructed tracks
    """

    raw_matches = [
        match_object_to_truth(
            track,
            object_id=i,
            min_fraction=min_fraction,
        )
        for i, track in enumerate(reconstructed_tracks)
    ]

    matches = mark_duplicates(raw_matches)

    matched_truth_ids = {
        match.matched_truth_particle_id
        for match in matches
        if match.is_matched and match.matched_truth_particle_id is not None
    }

    n_reconstructed_tracks = len(reconstructed_tracks)
    n_matched_tracks = sum(match.is_matched for match in matches)
    n_fake_tracks = sum(match.is_fake for match in matches)
    n_duplicate_tracks = sum(match.is_duplicate for match in matches)

    tracking_efficiency = (
        len(matched_truth_ids) / n_truth_particles
        if n_truth_particles > 0
        else 0.0
    )

    fake_rate = (
        n_fake_tracks / n_reconstructed_tracks
        if n_reconstructed_tracks > 0
        else 0.0
    )

    duplicate_rate = (
        n_duplicate_tracks / n_reconstructed_tracks
        if n_reconstructed_tracks > 0
        else 0.0
    )

    return ValidationSummary(
        n_truth_particles=n_truth_particles,
        n_reconstructed_tracks=n_reconstructed_tracks,
        n_matched_tracks=n_matched_tracks,
        n_fake_tracks=n_fake_tracks,
        n_duplicate_tracks=n_duplicate_tracks,
        tracking_efficiency=float(tracking_efficiency),
        fake_rate=float(fake_rate),
        duplicate_rate=float(duplicate_rate),
        matches=matches,
    )


def extract_hits(obj: Any) -> list[Any]:
    """
    Extract hit-like objects from a seed or reconstructed track.

    This avoids hard-coding one future Track class too early.
    """

    for attr in ("hits", "used_measurements", "measurements"):
        if hasattr(obj, attr):
            hits = getattr(obj, attr)
            return list(hits)

    raise TypeError(
        "Object has no hits, used_measurements, or measurements attribute."
    )


def _extract_object_id(obj: Any) -> int:
    for attr in ("track_id", "seed_id", "object_id"):
        if hasattr(obj, attr):
            return int(getattr(obj, attr))

    return 0
