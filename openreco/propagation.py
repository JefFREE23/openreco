"""
Propagation tools for OpenReco.

For OpenReco v0, this module provides a simple charged-particle propagation
model in a homogeneous magnetic field along z.

Main purpose:
    - propagate truth particles in uniform Bz
    - find intersections with cylindrical barrel layers

This is not an ACTS-level propagator yet. It is a clean, minimal propagation
model for the first cylindrical OpenReco v0 demo.
"""

from dataclasses import dataclass

import numpy as np

from openreco.field import UniformMagneticField
from openreco.geometry import BarrelDetector, CylindricalLayer
from openreco.particle_gun import Particle


@dataclass(frozen=True)
class PropagationResult:
    """
    Result of propagating a particle to a detector layer.

    Parameters
    ----------
    layer_name:
        Name of the detector layer.
    position:
        3D position [x, y, z] at the intersection.
    momentum:
        3D momentum [px, py, pz] at the intersection.
    path_length_xy:
        Transverse path length traveled in the xy plane.
    """

    layer_name: str
    position: np.ndarray
    momentum: np.ndarray
    path_length_xy: float

    def __post_init__(self):
        position = np.asarray(self.position, dtype=float)
        momentum = np.asarray(self.momentum, dtype=float)

        if not isinstance(self.layer_name, str):
            raise TypeError("layer_name must be a string")

        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")

        if momentum.shape != (3,):
            raise ValueError("momentum must have shape (3,)")

        if self.path_length_xy < 0.0:
            raise ValueError("path_length_xy must be non-negative")

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "momentum", momentum)
        object.__setattr__(self, "path_length_xy", float(self.path_length_xy))


def transverse_momentum(momentum: np.ndarray) -> float:
    """
    Return transverse momentum sqrt(px^2 + py^2).
    """
    momentum = np.asarray(momentum, dtype=float)

    if momentum.shape != (3,):
        raise ValueError("momentum must have shape (3,)")

    px, py, _ = momentum
    pt = np.sqrt(px**2 + py**2)

    if pt <= 0.0:
        raise ValueError("transverse momentum must be positive")

    return float(pt)


def curvature_from_particle(
    particle: Particle,
    field: UniformMagneticField,
    curvature_scale: float = 0.003,
) -> float:
    """
    Return signed transverse curvature.

    The simple model uses:

        curvature = curvature_scale * q * Bz / pt

    The curvature_scale keeps the toy geometry numerically gentle.
    """

    if not field.is_along_z():
        raise ValueError("This simple propagator only supports fields along z")

    if curvature_scale < 0.0:
        raise ValueError("curvature_scale must be non-negative")

    pt = particle.pt

    if pt <= 0.0:
        raise ValueError("particle transverse momentum must be positive")

    return float(curvature_scale * particle.charge * field.bz / pt)


def helix_position_at_s(
    particle: Particle,
    field: UniformMagneticField,
    s_xy: float,
    curvature_scale: float = 0.003,
) -> np.ndarray:
    """
    Return particle position after transverse path length s_xy.

    s_xy is the arc length traveled in the transverse xy plane.
    """

    if s_xy < 0.0:
        raise ValueError("s_xy must be non-negative")

    x0, y0, z0 = particle.position
    px, py, pz = particle.momentum

    pt = transverse_momentum(particle.momentum)
    phi0 = np.arctan2(py, px)
    kappa = curvature_from_particle(
        particle=particle,
        field=field,
        curvature_scale=curvature_scale,
    )

    if np.isclose(kappa, 0.0):
        x = x0 + np.cos(phi0) * s_xy
        y = y0 + np.sin(phi0) * s_xy
    else:
        phi = phi0 + kappa * s_xy
        x = x0 + (np.sin(phi) - np.sin(phi0)) / kappa
        y = y0 - (np.cos(phi) - np.cos(phi0)) / kappa

    z = z0 + (pz / pt) * s_xy

    return np.array([x, y, z], dtype=float)


def helix_momentum_at_s(
    particle: Particle,
    field: UniformMagneticField,
    s_xy: float,
    curvature_scale: float = 0.003,
) -> np.ndarray:
    """
    Return particle momentum direction after transverse path length s_xy.

    Momentum magnitude is kept constant.
    """

    if s_xy < 0.0:
        raise ValueError("s_xy must be non-negative")

    px, py, pz = particle.momentum

    pt = transverse_momentum(particle.momentum)
    phi0 = np.arctan2(py, px)
    kappa = curvature_from_particle(
        particle=particle,
        field=field,
        curvature_scale=curvature_scale,
    )

    phi = phi0 + kappa * s_xy

    new_px = pt * np.cos(phi)
    new_py = pt * np.sin(phi)

    return np.array([new_px, new_py, pz], dtype=float)


def radial_distance(position: np.ndarray) -> float:
    """
    Return radial distance sqrt(x^2 + y^2).
    """
    position = np.asarray(position, dtype=float)

    if position.shape != (3,):
        raise ValueError("position must have shape (3,)")

    x, y, _ = position
    return float(np.sqrt(x**2 + y**2))


def find_cylinder_intersection_s(
    particle: Particle,
    field: UniformMagneticField,
    layer: CylindricalLayer,
    curvature_scale: float = 0.003,
    max_s: float = 10000.0,
    n_scan: int = 1000,
) -> float:
    """
    Find transverse path length where the helix intersects a cylindrical layer.

    The method scans for the first crossing and then refines it with bisection.
    """

    if max_s <= 0.0:
        raise ValueError("max_s must be positive")

    if n_scan < 2:
        raise ValueError("n_scan must be at least 2")

    start_radius = radial_distance(particle.position)

    if np.isclose(start_radius, layer.radius):
        return 0.0

    s_values = np.linspace(0.0, max_s, n_scan)

    previous_s = s_values[0]
    previous_value = start_radius - layer.radius

    bracket = None

    for current_s in s_values[1:]:
        current_position = helix_position_at_s(
            particle=particle,
            field=field,
            s_xy=current_s,
            curvature_scale=curvature_scale,
        )
        current_value = radial_distance(current_position) - layer.radius

        if previous_value * current_value <= 0.0:
            bracket = (previous_s, current_s)
            break

        previous_s = current_s
        previous_value = current_value

    if bracket is None:
        raise RuntimeError(
            f"No intersection found with cylinder radius {layer.radius}"
        )

    low_s, high_s = bracket

    for _ in range(80):
        mid_s = 0.5 * (low_s + high_s)

        low_position = helix_position_at_s(
            particle=particle,
            field=field,
            s_xy=low_s,
            curvature_scale=curvature_scale,
        )
        mid_position = helix_position_at_s(
            particle=particle,
            field=field,
            s_xy=mid_s,
            curvature_scale=curvature_scale,
        )

        low_value = radial_distance(low_position) - layer.radius
        mid_value = radial_distance(mid_position) - layer.radius

        if low_value * mid_value <= 0.0:
            high_s = mid_s
        else:
            low_s = mid_s

    return float(0.5 * (low_s + high_s))


def propagate_to_cylindrical_layer(
    particle: Particle,
    field: UniformMagneticField,
    layer: CylindricalLayer,
    curvature_scale: float = 0.003,
    max_s: float = 10000.0,
    n_scan: int = 1000,
) -> PropagationResult:
    """
    Propagate a particle to a cylindrical detector layer.
    """

    s_xy = find_cylinder_intersection_s(
        particle=particle,
        field=field,
        layer=layer,
        curvature_scale=curvature_scale,
        max_s=max_s,
        n_scan=n_scan,
    )

    position = helix_position_at_s(
        particle=particle,
        field=field,
        s_xy=s_xy,
        curvature_scale=curvature_scale,
    )

    if not layer.contains_z(position[2]):
        raise RuntimeError(
            f"Intersection with {layer.name} is outside cylinder z length"
        )

    momentum = helix_momentum_at_s(
        particle=particle,
        field=field,
        s_xy=s_xy,
        curvature_scale=curvature_scale,
    )

    return PropagationResult(
        layer_name=layer.name,
        position=position,
        momentum=momentum,
        path_length_xy=s_xy,
    )


def propagate_to_barrel_detector(
    particle: Particle,
    field: UniformMagneticField,
    detector: BarrelDetector,
    curvature_scale: float = 0.003,
    max_s: float = 10000.0,
    n_scan: int = 1000,
) -> list[PropagationResult]:
    """
    Propagate a particle through all layers of a BarrelDetector.
    """

    results = []

    for layer in detector.layers:
        result = propagate_to_cylindrical_layer(
            particle=particle,
            field=field,
            layer=layer,
            curvature_scale=curvature_scale,
            max_s=max_s,
            n_scan=n_scan,
        )
        results.append(result)

    return results
