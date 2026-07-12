"""Reconstructed toy-resonance helpers for OpenReco v3.1.

This module connects reconstructed tracks to a downstream physics observable:
the two-body invariant mass.

It intentionally stays compact:
- convert reconstructed track parameters into Cartesian momentum,
- select opposite-charge track pairs,
- compute reconstructed invariant mass,
- report mass residual relative to a known truth resonance mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Any, Iterable

import numpy as np

from openreco.invariant_mass import (
    JPSI_MASS_GEV,
    MUON_MASS_GEV,
    invariant_mass_from_momenta,
    mass_residual,
)


@dataclass(frozen=True)
class ReconstructedResonanceCandidate:
    """Opposite-charge reconstructed two-track mass candidate."""

    track_indices: tuple[int, int]
    charges: tuple[int, int]
    mass: float
    mass_residual: float


def charge_from_track(track: Any) -> int:
    """Infer charge sign from reconstructed q/p."""

    q_over_p = float(getattr(track, "q_over_p"))

    if not isfinite(q_over_p):
        raise ValueError("track q_over_p must be finite")

    if np.isclose(q_over_p, 0.0):
        raise ValueError("track q_over_p must be non-zero")

    return 1 if q_over_p > 0.0 else -1


def track_direction_angle(track: Any) -> float:
    """Return best available transverse direction angle for a track."""

    for states_attr in ("smoothed_states", "filtered_states"):
        states = getattr(track, states_attr, ())

        if states:
            state = states[-1]

            if hasattr(state, "dir0"):
                direction = float(state.dir0)

                if isfinite(direction):
                    return direction

    if hasattr(track, "seed") and hasattr(track.seed, "alpha"):
        direction = float(track.seed.alpha)

        if isfinite(direction):
            return direction

    raise ValueError("track has no finite direction angle")


def momentum_vector_from_track(track: Any) -> np.ndarray:
    """Convert reconstructed track parameters into Cartesian momentum."""

    p = float(getattr(track, "p_estimate"))
    pt = float(getattr(track, "pt_estimate"))
    tan_lambda = float(getattr(track, "tan_lambda"))

    if not isfinite(p) or p <= 0.0:
        raise ValueError("track p_estimate must be positive and finite")

    if not isfinite(pt) or pt <= 0.0:
        raise ValueError("track pt_estimate must be positive and finite")

    if not isfinite(tan_lambda):
        raise ValueError("track tan_lambda must be finite")

    alpha = track_direction_angle(track)

    px = pt * cos(alpha)
    py = pt * sin(alpha)
    pz = pt * tan_lambda

    momentum = np.array([px, py, pz], dtype=float)

    if not np.all(np.isfinite(momentum)):
        raise ValueError("track momentum vector must be finite")

    return momentum


def build_opposite_charge_mass_candidates(
    tracks: Iterable[Any],
    *,
    truth_mass: float = JPSI_MASS_GEV,
    daughter_mass: float = MUON_MASS_GEV,
) -> list[ReconstructedResonanceCandidate]:
    """Build all opposite-charge two-track invariant-mass candidates."""

    track_list = list(tracks)

    if truth_mass <= 0.0:
        raise ValueError("truth_mass must be positive")

    if daughter_mass < 0.0:
        raise ValueError("daughter_mass must be non-negative")

    candidates: list[ReconstructedResonanceCandidate] = []

    for first_index in range(len(track_list)):
        for second_index in range(first_index + 1, len(track_list)):
            first_track = track_list[first_index]
            second_track = track_list[second_index]

            first_charge = charge_from_track(first_track)
            second_charge = charge_from_track(second_track)

            if first_charge * second_charge >= 0:
                continue

            first_momentum = momentum_vector_from_track(first_track)
            second_momentum = momentum_vector_from_track(second_track)

            candidate_mass = invariant_mass_from_momenta(
                first_momentum,
                second_momentum,
                mass_1=daughter_mass,
                mass_2=daughter_mass,
            )

            candidates.append(
                ReconstructedResonanceCandidate(
                    track_indices=(first_index, second_index),
                    charges=(first_charge, second_charge),
                    mass=float(candidate_mass),
                    mass_residual=mass_residual(
                        reconstructed_mass=float(candidate_mass),
                        truth_mass=truth_mass,
                    ),
                )
            )

    return candidates


def select_best_mass_candidate(
    tracks: Iterable[Any],
    *,
    truth_mass: float = JPSI_MASS_GEV,
    daughter_mass: float = MUON_MASS_GEV,
) -> ReconstructedResonanceCandidate | None:
    """Select the opposite-charge candidate closest to the truth mass."""

    candidates = build_opposite_charge_mass_candidates(
        tracks,
        truth_mass=truth_mass,
        daughter_mass=daughter_mass,
    )

    if not candidates:
        return None

    return min(candidates, key=lambda candidate: abs(candidate.mass_residual))