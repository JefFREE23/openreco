from __future__ import annotations

from dataclasses import dataclass, field
from math import asin, cos, pi, sin, sqrt
from typing import Any, Optional
from openreco.detector_effects import DetectorEffectsConfig

import numpy as np


@dataclass(frozen=True)
class TruthParticle:
    """
    Truth-level charged particle used by the OpenReco v1 simulator.

    Coordinates are intentionally simple:
    - pt is transverse momentum
    - phi is the initial azimuthal direction
    - tan_lambda = pz / pt
    - q_over_p is signed inverse momentum
    """

    truth_particle_id: int
    pt: float
    phi: float
    z0: float
    tan_lambda: float
    charge: int
    q_over_p: float
    p: float


@dataclass(frozen=True)
class EventHit:
    """
    A detector hit/measurement belonging to one event.

    Real simulated hits have truth_particle_id set to the particle that made them.
    Noise hits have truth_particle_id = None.
    """

    hit_id: int
    layer_index: int
    layer_name: str
    radius: float
    phi: float
    z: float
    covariance: np.ndarray = field(repr=False)
    truth_particle_id: Optional[int] = None
    is_noise: bool = False

    @property
    def x(self) -> float:
        return self.radius * cos(self.phi)

    @property
    def y(self) -> float:
        return self.radius * sin(self.phi)

    @property
    def global_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Event:
    """
    Full event container for v1 reconstruction.

    v1 will gradually fill:
    - seeds
    - reconstructed_tracks
    - validation_results
    """

    event_id: int
    truth_particles: list[TruthParticle]
    measurements_by_layer: dict[str, list[EventHit]]
    seeds: list[Any] = field(default_factory=list)
    reconstructed_tracks: list[Any] = field(default_factory=list)
    validation_results: dict[str, Any] = field(default_factory=dict)

    @property
    def measurements(self) -> list[EventHit]:
        hits: list[EventHit] = []
        for layer_hits in self.measurements_by_layer.values():
            hits.extend(layer_hits)
        return hits


@dataclass(frozen=True)
class SimpleCylinderLayer:
    layer_index: int
    name: str
    radius: float


@dataclass(frozen=True)
class SimpleBarrelDetector:
    layers: tuple[SimpleCylinderLayer, ...]


def make_default_barrel(
    radii: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
) -> SimpleBarrelDetector:
    """
    Lightweight v1 barrel definition.

    This is only for the event-generation tests and early v1 demos.
    Later, we can pass your existing v0 BarrelDetector directly.
    """

    layers = tuple(
        SimpleCylinderLayer(layer_index=i, name=f"barrel_{i}", radius=float(r))
        for i, r in enumerate(radii)
    )
    return SimpleBarrelDetector(layers=layers)


def generate_truth_particles(
    n_particles: int,
    *,
    pt_range: tuple[float, float] = (0.5, 5.0),
    phi_range: tuple[float, float] = (0.0, 2.0 * pi),
    z0_range: tuple[float, float] = (0.0, 0.0),
    tan_lambda_range: tuple[float, float] = (-1.5, 1.5),
    charge_choices: tuple[int, ...] = (-1, 1),
    rng: np.random.Generator | None = None,
) -> list[TruthParticle]:
    """
    Generate several independent charged truth particles.
    """

    if n_particles < 0:
        raise ValueError("n_particles must be non-negative")

    rng = np.random.default_rng() if rng is None else rng
    particles: list[TruthParticle] = []

    for truth_particle_id in range(n_particles):
        pt = float(rng.uniform(*pt_range))
        phi = float(rng.uniform(*phi_range))
        z0 = float(rng.uniform(*z0_range))
        tan_lambda = float(rng.uniform(*tan_lambda_range))
        charge = int(rng.choice(charge_choices))

        p = pt * sqrt(1.0 + tan_lambda * tan_lambda)
        q_over_p = charge / p

        particles.append(
            TruthParticle(
                truth_particle_id=truth_particle_id,
                pt=pt,
                phi=_wrap_phi(phi),
                z0=z0,
                tan_lambda=tan_lambda,
                charge=charge,
                q_over_p=q_over_p,
                p=p,
            )
        )

    return particles


def generate_event(
    *,
    event_id: int = 0,
    n_particles: int = 5,
    detector: Any | None = None,
    hit_efficiency: float = 0.95,
    noise_hits_per_layer: int = 0,
    measurement_sigma_phi: float = 1.0e-3,
    measurement_sigma_z: float = 0.10,
    detector_effects: DetectorEffectsConfig | None = None,
    pt_range: tuple[float, float] = (0.5, 5.0),
    phi_range: tuple[float, float] = (0.0, 2.0 * pi),
    z0_range: tuple[float, float] = (0.0, 0.0),
    tan_lambda_range: tuple[float, float] = (-1.5, 1.5),
    charge_choices: tuple[int, ...] = (-1, 1),
    bz: float = 2.0,
    curvature_scale: float = 0.003,
    noise_z_range: tuple[float, float] = (-100.0, 100.0),
    rng: np.random.Generator | None = None,
) -> Event:
    """
    Generate one OpenReco v1 simulated event.

    The output is a mixed hit collection:
        measurements_by_layer = {
            "barrel_0": [hit, hit, ...],
            "barrel_1": [hit, hit, ...],
            ...
        }

    Real hits are truth-labelled.
    Noise hits are not truth-labelled.

    If detector_effects is supplied, it overrides the legacy hit-resolution
    and hit-efficiency arguments and can suppress selected dead layers.
    """

    if noise_hits_per_layer < 0:
        raise ValueError("noise_hits_per_layer must be non-negative")

    noise_mean_per_layer = float(noise_hits_per_layer)

    if detector_effects is not None:
        measurement_sigma_phi = detector_effects.hit_resolution.sigma_phi
        measurement_sigma_z = detector_effects.hit_resolution.sigma_z
        hit_efficiency = detector_effects.inefficiency.hit_efficiency
        noise_mean_per_layer = detector_effects.noise_occupancy.mean_noise_hits_per_layer

    if not 0.0 <= hit_efficiency <= 1.0:
        raise ValueError("hit_efficiency must be between 0 and 1")

    if noise_mean_per_layer < 0.0:
        raise ValueError("noise_mean_per_layer must be non-negative")

    if measurement_sigma_phi <= 0.0:
        raise ValueError("measurement_sigma_phi must be positive")

    if measurement_sigma_z <= 0.0:
        raise ValueError("measurement_sigma_z must be positive")

    rng = np.random.default_rng() if rng is None else rng
    detector = make_default_barrel() if detector is None else detector
    layers = _extract_layers(detector)

    truth_particles = generate_truth_particles(
        n_particles,
        pt_range=pt_range,
        phi_range=phi_range,
        z0_range=z0_range,
        tan_lambda_range=tan_lambda_range,
        charge_choices=charge_choices,
        rng=rng,
    )

    measurements_by_layer: dict[str, list[EventHit]] = {}

    for i, layer in enumerate(layers):
        layer_name = _layer_name(layer, i)
        measurements_by_layer[layer_name] = []

    hit_id = 0
    covariance = np.diag([measurement_sigma_phi**2, measurement_sigma_z**2])

    for particle in truth_particles:
        for i, layer in enumerate(layers):
            if detector_effects is not None and detector_effects.dead_layers.is_dead(i):
                continue

            if rng.random() > hit_efficiency:
                continue

            radius = _layer_radius(layer)
            expected = _expected_cylindrical_hit(
                particle,
                radius=radius,
                bz=bz,
                curvature_scale=curvature_scale,
            )

            if expected is None:
                continue

            true_phi, true_z = expected
            measured_phi = _wrap_phi(
                float(true_phi + rng.normal(0.0, measurement_sigma_phi))
            )
            measured_z = float(true_z + rng.normal(0.0, measurement_sigma_z))

            layer_name = _layer_name(layer, i)
            measurements_by_layer[layer_name].append(
                EventHit(
                    hit_id=hit_id,
                    layer_index=i,
                    layer_name=layer_name,
                    radius=radius,
                    phi=measured_phi,
                    z=measured_z,
                    covariance=covariance.copy(),
                    truth_particle_id=particle.truth_particle_id,
                    is_noise=False,
                )
            )
            hit_id += 1

    for i, layer in enumerate(layers):
        layer_name = _layer_name(layer, i)

        if detector_effects is not None and detector_effects.dead_layers.is_dead(i):
            continue

        radius = _layer_radius(layer)

        n_noise_hits_this_layer = (
            int(rng.poisson(noise_mean_per_layer))
            if detector_effects is not None
            else int(noise_mean_per_layer)
        )

        for _ in range(n_noise_hits_this_layer):
            measurements_by_layer[layer_name].append(
                EventHit(
                    hit_id=hit_id,
                    layer_index=i,
                    layer_name=layer_name,
                    radius=radius,
                    phi=float(rng.uniform(0.0, 2.0 * pi)),
                    z=float(rng.uniform(*noise_z_range)),
                    covariance=covariance.copy(),
                    truth_particle_id=None,
                    is_noise=True,
                )
            )
            hit_id += 1

    return Event(
        event_id=event_id,
        truth_particles=truth_particles,
        measurements_by_layer=measurements_by_layer,
    )


def count_real_hits(event: Event) -> int:
    return sum(1 for hit in event.measurements if not hit.is_noise)


def count_noise_hits(event: Event) -> int:
    return sum(1 for hit in event.measurements if hit.is_noise)


def _expected_cylindrical_hit(
    particle: TruthParticle,
    *,
    radius: float,
    bz: float,
    curvature_scale: float,
) -> tuple[float, float] | None:
    """
    Approximate intersection of a charged particle with a cylindrical layer.

    This is intentionally simple for v1 event generation.
    The v0 propagator remains the real fitter/transport model.

    curvature_scale defaults to 0.003 for GeV, Tesla, and centimeter-like radii.
    """

    if radius <= 0.0:
        raise ValueError("layer radius must be positive")

    kappa = curvature_scale * particle.charge * bz / particle.pt

    if abs(kappa) < 1.0e-14:
        return _wrap_phi(particle.phi), particle.z0 + radius * particle.tan_lambda

    arg = 0.5 * kappa * radius

    if abs(arg) >= 1.0:
        return None

    delta = 2.0 * asin(arg)
    path_length = delta / kappa

    phi_at_radius = _wrap_phi(particle.phi + 0.5 * delta)
    z_at_radius = particle.z0 + path_length * particle.tan_lambda

    return phi_at_radius, z_at_radius


def _wrap_phi(phi: float) -> float:
    return float(phi % (2.0 * pi))


def _extract_layers(detector: Any) -> list[Any]:
    """
    Accept either the lightweight v1 detector or your existing v0 detector.

    Expected possibilities:
    - detector.layers
    - detector.cylindrical_layers
    - detector.barrel_layers
    """

    for attr in ("layers", "cylindrical_layers", "barrel_layers"):
        if hasattr(detector, attr):
            layers = list(getattr(detector, attr))
            if layers:
                return layers

    raise TypeError(
        "Could not extract detector layers. Expected detector.layers, "
        "detector.cylindrical_layers, or detector.barrel_layers."
    )


def _layer_name(layer: Any, index: int) -> str:
    for attr in ("name", "surface_name", "layer_name"):
        if hasattr(layer, attr):
            return str(getattr(layer, attr))
    return f"barrel_{index}"


def _layer_radius(layer: Any) -> float:
    for attr in ("radius", "r"):
        if hasattr(layer, attr):
            return float(getattr(layer, attr))
    raise TypeError("Layer must have radius or r attribute.")
