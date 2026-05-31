"""
Detector geometry for OpenReco.

This module contains simple detector surface descriptions.

Current geometry support:
    - Planar layers at fixed z positions
    - Cylindrical barrel layers at fixed radii

Important:
    The planar geometry is an early checkpoint.
    The final OpenReco v0 demo will use cylindrical tracker layers.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DetectorLayer:
    """
    A simple planar detector layer perpendicular to the z axis.

    Parameters
    ----------
    name:
        Unique layer name.
    z:
        Position of the layer along the z axis.
    """

    name: str
    z: float

    surface_type: str = "plane"

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("DetectorLayer name must be a string")

        object.__setattr__(self, "z", float(self.z))


@dataclass(frozen=True)
class CylindricalLayer:
    """
    A simple cylindrical detector layer around the beam axis.

    The cylinder is centered on the z axis and has fixed radius.

    Parameters
    ----------
    name:
        Unique layer name.
    radius:
        Cylinder radius.
    half_length:
        Half-length of the cylinder along z.
    """

    name: str
    radius: float
    half_length: float

    surface_type: str = "cylinder"

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("CylindricalLayer name must be a string")

        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "half_length", float(self.half_length))

        if self.radius <= 0.0:
            raise ValueError("CylindricalLayer radius must be positive")

        if self.half_length <= 0.0:
            raise ValueError("CylindricalLayer half_length must be positive")

    def contains_z(self, z: float) -> bool:
        """
        Return True if a z position lies inside the cylinder length.
        """
        return abs(float(z)) <= self.half_length


class SimpleDetector:
    """
    A detector made of planar layers sorted by increasing z.

    This is a temporary checkpoint geometry, not the final OpenReco v0
    cylindrical detector.
    """

    def __init__(self, layers: Iterable[DetectorLayer]):
        self.layers = list(layers)

        if len(self.layers) == 0:
            raise ValueError("SimpleDetector must contain at least one layer")

        self.layers.sort(key=lambda layer: layer.z)

        self._check_unique_layer_names()
        self._check_unique_z_positions()

    def _check_unique_layer_names(self):
        names = [layer.name for layer in self.layers]

        if len(names) != len(set(names)):
            raise ValueError("Detector layer names must be unique")

    def _check_unique_z_positions(self):
        z_positions = [layer.z for layer in self.layers]

        if len(z_positions) != len(set(z_positions)):
            raise ValueError("Detector layer z positions must be unique")

    @property
    def z_positions(self) -> np.ndarray:
        """
        Return all layer z positions as a NumPy array.
        """
        return np.array([layer.z for layer in self.layers], dtype=float)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> DetectorLayer:
        return self.layers[index]


class BarrelDetector:
    """
    A cylindrical barrel detector made of layers sorted by increasing radius.

    This is the geometry direction required for the final OpenReco v0 demo.
    """

    def __init__(self, layers: Iterable[CylindricalLayer]):
        self.layers = list(layers)

        if len(self.layers) == 0:
            raise ValueError("BarrelDetector must contain at least one layer")

        self.layers.sort(key=lambda layer: layer.radius)

        self._check_unique_layer_names()
        self._check_unique_radii()

    def _check_unique_layer_names(self):
        names = [layer.name for layer in self.layers]

        if len(names) != len(set(names)):
            raise ValueError("Cylindrical layer names must be unique")

    def _check_unique_radii(self):
        radii = [layer.radius for layer in self.layers]

        if len(radii) != len(set(radii)):
            raise ValueError("Cylindrical layer radii must be unique")

    @property
    def radii(self) -> np.ndarray:
        """
        Return all cylinder radii as a NumPy array.
        """
        return np.array([layer.radius for layer in self.layers], dtype=float)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> CylindricalLayer:
        return self.layers[index]


def make_uniform_detector(
    n_layers: int,
    z_min: float,
    z_max: float,
    name_prefix: str = "layer",
) -> SimpleDetector:
    """
    Create a detector with uniformly spaced planar layers.
    """

    if n_layers < 2:
        raise ValueError("n_layers must be at least 2")

    z_values = np.linspace(z_min, z_max, n_layers)

    layers = [
        DetectorLayer(name=f"{name_prefix}_{i}", z=z)
        for i, z in enumerate(z_values)
    ]

    return SimpleDetector(layers)


def make_barrel_detector(
    radii: Iterable[float],
    half_length: float,
    name_prefix: str = "barrel",
) -> BarrelDetector:
    """
    Create a cylindrical barrel detector from a list of radii.
    """

    radii = list(radii)

    if len(radii) == 0:
        raise ValueError("radii must contain at least one radius")

    layers = [
        CylindricalLayer(
            name=f"{name_prefix}_{i}",
            radius=radius,
            half_length=half_length,
        )
        for i, radius in enumerate(radii)
    ]

    return BarrelDetector(layers)
