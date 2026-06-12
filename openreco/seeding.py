from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import cos, pi, sin, sqrt
from typing import Optional

import numpy as np

from openreco.event_generation import EventHit


@dataclass(frozen=True)
class TripletSeed:
    """
    Rough track candidate built from three hits on three inner layers.

    This is not a final fitted track.
    It is only an initial candidate good enough to start later track finding / EKF fitting.
    """

    seed_id: int
    hit_ids: tuple[int, int, int]
    layer_names: tuple[str, str, str]
    hits: tuple[EventHit, EventHit, EventHit]

    phi: float
    z: float
    alpha: float
    tan_lambda: float
    q_over_p: float

    pt_estimate: float
    p_estimate: float
    circle_radius: float
    circle_center_x: float
    circle_center_y: float
    quality: float

    truth_particle_id: Optional[int] = None


def build_triplet_seeds(
    measurements_by_layer: dict[str, list[EventHit]],
    *,
    layer_names: tuple[str, str, str] = ("barrel_0", "barrel_1", "barrel_2"),
    bz: float = 2.0,
    curvature_scale: float = 0.003,
    max_circle_residual: float = 1.0,
) -> list[TripletSeed]:
    """
    Build triplet seeds from three cylindrical layers.

    v1 minimum logic:
    - take all hit combinations from barrel_0, barrel_1, barrel_2
    - fit an x-y circle through the three global hit positions
    - estimate curvature and q/p
    - estimate tan_lambda from z versus radius
    - keep simple seed quality information

    This is intentionally simple. Later chunks will add pruning and better selection.
    """

    if bz == 0.0:
        raise ValueError("Triplet seeding needs nonzero bz to estimate q/p sign.")

    layer_hits = []
    for name in layer_names:
        if name not in measurements_by_layer:
            raise KeyError(f"Missing layer {name!r} in measurements_by_layer.")
        layer_hits.append(measurements_by_layer[name])

    seeds: list[TripletSeed] = []
    seed_id = 0

    for h0, h1, h2 in product(layer_hits[0], layer_hits[1], layer_hits[2]):
        maybe_seed = _make_seed_from_hits(
            seed_id=seed_id,
            hits=(h0, h1, h2),
            bz=bz,
            curvature_scale=curvature_scale,
        )

        if maybe_seed is None:
            continue

        if maybe_seed.quality > max_circle_residual:
            continue

        seeds.append(maybe_seed)
        seed_id += 1

    return seeds


def _make_seed_from_hits(
    *,
    seed_id: int,
    hits: tuple[EventHit, EventHit, EventHit],
    bz: float,
    curvature_scale: float,
) -> TripletSeed | None:
    points = np.array([[hit.x, hit.y] for hit in hits], dtype=float)

    circle = _fit_circle_three_points(points)
    if circle is None:
        return None

    cx, cy, radius = circle

    if not np.isfinite(radius) or radius <= 0.0:
        return None

    curvature = 1.0 / radius
    pt_estimate = curvature_scale * abs(bz) / curvature

    if not np.isfinite(pt_estimate) or pt_estimate <= 0.0:
        return None

    radii = np.array([hit.radius for hit in hits], dtype=float)
    z_values = np.array([hit.z for hit in hits], dtype=float)

    tan_lambda = _fit_tan_lambda(radii, z_values)

    p_estimate = pt_estimate * sqrt(1.0 + tan_lambda * tan_lambda)

    if not np.isfinite(p_estimate) or p_estimate <= 0.0:
        return None

    charge_sign = _estimate_charge_sign_from_bending(hits, bz)
    q_over_p = charge_sign / p_estimate

    phi0 = _wrap_phi(hits[0].phi)
    z0 = float(hits[0].z)

    alpha = _estimate_initial_direction(points)

    quality = _circle_residual_quality(points, cx, cy, radius)

    truth_particle_id = _common_truth_id(hits)

    return TripletSeed(
        seed_id=seed_id,
        hit_ids=(hits[0].hit_id, hits[1].hit_id, hits[2].hit_id),
        layer_names=(hits[0].layer_name, hits[1].layer_name, hits[2].layer_name),
        hits=hits,
        phi=phi0,
        z=z0,
        alpha=alpha,
        tan_lambda=float(tan_lambda),
        q_over_p=float(q_over_p),
        pt_estimate=float(pt_estimate),
        p_estimate=float(p_estimate),
        circle_radius=float(radius),
        circle_center_x=float(cx),
        circle_center_y=float(cy),
        quality=float(quality),
        truth_particle_id=truth_particle_id,
    )


def _fit_circle_three_points(points: np.ndarray) -> tuple[float, float, float] | None:
    """
    Fit circle through three x-y points.

    Circle equation:
        x^2 + y^2 + D x + E y + F = 0

    Center:
        cx = -D / 2
        cy = -E / 2
    """

    if points.shape != (3, 2):
        raise ValueError("points must have shape (3, 2)")

    x = points[:, 0]
    y = points[:, 1]

    matrix = np.column_stack([x, y, np.ones(3)])
    rhs = -(x * x + y * y)

    try:
        d, e, f = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None

    cx = -0.5 * d
    cy = -0.5 * e

    radius_squared = cx * cx + cy * cy - f
    if radius_squared <= 0.0:
        return None

    return float(cx), float(cy), float(sqrt(radius_squared))


def _fit_tan_lambda(radii: np.ndarray, z_values: np.ndarray) -> float:
    """
    Estimate tan(lambda) from a simple z-r line fit.

    This is an approximation. It is good enough for the seed stage.
    """

    slope, _intercept = np.polyfit(radii, z_values, deg=1)
    return float(slope)


def _estimate_charge_sign_from_bending(
    hits: tuple[EventHit, EventHit, EventHit],
    bz: float,
) -> int:
    """
    Estimate charge sign using phi bending across increasing radius.

    In our v1 simulator, for positive Bz:
        positive charge -> phi increases with radius
        negative charge -> phi decreases with radius

    Therefore:
        sign(q) = sign(delta_phi / Bz)
    """

    phis = np.unwrap(np.array([hit.phi for hit in hits], dtype=float))
    delta_phi = float(phis[-1] - phis[0])

    if abs(delta_phi) < 1.0e-12:
        return 1

    sign = np.sign(delta_phi / bz)
    return 1 if sign >= 0.0 else -1


def _estimate_initial_direction(points: np.ndarray) -> float:
    """
    Estimate initial transverse direction using the first two seed hits.
    """

    dx = float(points[1, 0] - points[0, 0])
    dy = float(points[1, 1] - points[0, 1])

    return _wrap_phi(float(np.arctan2(dy, dx)))


def _circle_residual_quality(
    points: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> float:
    distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    residuals = distances - radius
    return float(np.sqrt(np.mean(residuals * residuals)))


def _common_truth_id(hits: tuple[EventHit, EventHit, EventHit]) -> Optional[int]:
    truth_ids = [hit.truth_particle_id for hit in hits]

    if truth_ids[0] is None:
        return None

    if truth_ids[0] == truth_ids[1] == truth_ids[2]:
        return truth_ids[0]

    return None


def _wrap_phi(phi: float) -> float:
    return float(phi % (2.0 * pi))
