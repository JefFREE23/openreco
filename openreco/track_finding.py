from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np

from openreco.event_generation import EventHit
from openreco.seeding import TripletSeed


@dataclass(frozen=True)
class ReconstructedTrack:
    """
    Simple v1 reconstructed track candidate.

    This is the first pattern-recognition object in OpenReco v1:
    seed hits + greedily selected compatible outer-layer hits.
    """

    track_id: int
    seed_id: int
    seed: TripletSeed
    used_measurements: tuple[EventHit, ...]

    chi2: float
    ndof: int
    chi2_ndof: float

    final_phi: float
    final_z: float
    tan_lambda: float
    q_over_p: float
    pt_estimate: float
    p_estimate: float

    fit_status: str = "accepted"

    @property
    def hits(self) -> tuple[EventHit, ...]:
        return self.used_measurements

    @property
    def hit_ids(self) -> tuple[int, ...]:
        return tuple(hit.hit_id for hit in self.used_measurements)


@dataclass(frozen=True)
class _LinearSeedModel:
    """
    Lightweight local model for Chunk 4.

    We use this only for greedy hit selection.
    The real EKF fitting/smoothing remains the v0 machinery and will be wrapped later.
    """

    phi_slope: float
    phi_intercept: float
    z_slope: float
    z_intercept: float

    @classmethod
    def fit(cls, hits: list[EventHit] | tuple[EventHit, ...]) -> "_LinearSeedModel":
        if len(hits) < 2:
            raise ValueError("Need at least two hits to fit a seed model.")

        radii = np.array([hit.radius for hit in hits], dtype=float)
        phis = np.unwrap(np.array([hit.phi for hit in hits], dtype=float))
        zs = np.array([hit.z for hit in hits], dtype=float)

        phi_slope, phi_intercept = np.polyfit(radii, phis, deg=1)
        z_slope, z_intercept = np.polyfit(radii, zs, deg=1)

        return cls(
            phi_slope=float(phi_slope),
            phi_intercept=float(phi_intercept),
            z_slope=float(z_slope),
            z_intercept=float(z_intercept),
        )

    def predict(self, radius: float) -> tuple[float, float]:
        phi = self.phi_slope * radius + self.phi_intercept
        z = self.z_slope * radius + self.z_intercept
        return _wrap_phi(phi), float(z)


def find_tracks_from_seeds(
    seeds: list[TripletSeed],
    measurements_by_layer: dict[str, list[EventHit]],
    *,
    chi2_threshold: float = 25.0,
    min_hits: int = 5,
    allow_shared_hits: bool = False,
    max_tracks: int | None = None,
) -> list[ReconstructedTrack]:
    """
    Greedy v1 track finding.

    Algorithm:
    1. Start from a triplet seed.
    2. Fit a simple local phi(r), z(r) prediction model.
    3. Go outward layer by layer.
    4. Compare prediction to all hits on that layer.
    5. Choose the hit with the smallest chi-square.
    6. Accept it only if chi-square < threshold.
    7. Reject duplicate candidates that reuse hits from already accepted tracks.

    This is deliberately not full CKF branching yet.
    """

    if chi2_threshold <= 0.0:
        raise ValueError("chi2_threshold must be positive.")

    if min_hits <= 0:
        raise ValueError("min_hits must be positive.")

    sorted_layer_items = _sorted_layer_items(measurements_by_layer)

    accepted_tracks: list[ReconstructedTrack] = []
    globally_used_hit_ids: set[int] = set()

    sorted_seeds = sorted(seeds, key=lambda seed: (seed.quality, seed.seed_id))

    for seed in sorted_seeds:
        if max_tracks is not None and len(accepted_tracks) >= max_tracks:
            break

        seed_hit_ids = set(seed.hit_ids)

        if not allow_shared_hits and seed_hit_ids & globally_used_hit_ids:
            continue

        selected_hits: list[EventHit] = list(seed.hits)
        selected_hit_ids: set[int] = set(seed.hit_ids)
        total_chi2 = 0.0
        n_added_outer_hits = 0

        model = _LinearSeedModel.fit(selected_hits)

        max_seed_layer_index = max(hit.layer_index for hit in seed.hits)

        for _layer_name, layer_hits in sorted_layer_items:
            if not layer_hits:
                continue

            layer_index = layer_hits[0].layer_index

            if layer_index <= max_seed_layer_index:
                continue

            candidates = [
                hit for hit in layer_hits
                if hit.hit_id not in selected_hit_ids
            ]

            if not allow_shared_hits:
                candidates = [
                    hit for hit in candidates
                    if hit.hit_id not in globally_used_hit_ids
                ]

            if not candidates:
                continue

            best_hit, best_chi2 = _best_compatible_hit(
                model=model,
                candidates=candidates,
            )

            if best_hit is None:
                continue

            if best_chi2 > chi2_threshold:
                continue

            selected_hits.append(best_hit)
            selected_hit_ids.add(best_hit.hit_id)
            total_chi2 += best_chi2
            n_added_outer_hits += 1

            model = _LinearSeedModel.fit(selected_hits)

        if len(selected_hits) < min_hits:
            continue

        ndof = max(2 * n_added_outer_hits, 1)
        chi2_ndof = total_chi2 / ndof

        last_radius = max(hit.radius for hit in selected_hits)
        final_phi, final_z = model.predict(last_radius)

        track = ReconstructedTrack(
            track_id=len(accepted_tracks),
            seed_id=seed.seed_id,
            seed=seed,
            used_measurements=tuple(selected_hits),
            chi2=float(total_chi2),
            ndof=int(ndof),
            chi2_ndof=float(chi2_ndof),
            final_phi=float(final_phi),
            final_z=float(final_z),
            tan_lambda=float(model.z_slope),
            q_over_p=float(seed.q_over_p),
            pt_estimate=float(seed.pt_estimate),
            p_estimate=float(seed.p_estimate),
            fit_status="accepted",
            filtered_states=(),
            smoothed_states=(),
            final_covariance=None,
            covariance_valid=False,
            momentum_uncertainty=float("nan"),
        )

        accepted_tracks.append(track)

        if not allow_shared_hits:
            globally_used_hit_ids.update(track.hit_ids)

    return accepted_tracks


def _best_compatible_hit(
    *,
    model: _LinearSeedModel,
    candidates: list[EventHit],
) -> tuple[EventHit | None, float]:
    best_hit: EventHit | None = None
    best_chi2 = float("inf")

    for hit in candidates:
        predicted_phi, predicted_z = model.predict(hit.radius)
        chi2 = _measurement_chi2(hit, predicted_phi, predicted_z)

        if chi2 < best_chi2:
            best_hit = hit
            best_chi2 = chi2

    return best_hit, float(best_chi2)


def _measurement_chi2(
    hit: EventHit,
    predicted_phi: float,
    predicted_z: float,
) -> float:
    dphi = _angle_difference(hit.phi, predicted_phi)
    dz = hit.z - predicted_z

    covariance = hit.covariance
    sigma_phi2 = float(covariance[0, 0])
    sigma_z2 = float(covariance[1, 1])

    if sigma_phi2 <= 0.0 or sigma_z2 <= 0.0:
        raise ValueError("Hit covariance must have positive diagonal entries.")

    return float((dphi * dphi) / sigma_phi2 + (dz * dz) / sigma_z2)


def _sorted_layer_items(
    measurements_by_layer: dict[str, list[EventHit]],
) -> list[tuple[str, list[EventHit]]]:
    def layer_sort_key(item: tuple[str, list[EventHit]]) -> int:
        layer_name, hits = item

        if hits:
            return hits[0].layer_index

        if layer_name.startswith("barrel_"):
            return int(layer_name.split("_")[-1])

        return 10**9

    return sorted(measurements_by_layer.items(), key=layer_sort_key)


def _angle_difference(phi_a: float, phi_b: float) -> float:
    return float((phi_a - phi_b + pi) % (2.0 * pi) - pi)


def _wrap_phi(phi: float) -> float:
    return float(phi % (2.0 * pi))
