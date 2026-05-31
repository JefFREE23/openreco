"""
Basic detector geometry for OpenReco.

This file starts with simple planar layers at fixed z positions.

Important:
    These planar layers are only an early geometry checkpoint.
    The final OpenReco v0 detector will use cylindrical tracker layers.

For now, this module gives us:
    - DetectorLayer
    - SimpleDetector
    - make_uniform_detector

Later, we will extend this file with:
    - CylindricalLayer
    - BarrelDetector
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

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("DetectorLayer name must be a string")

        object.__setattr__(self, "z", float(self.z))


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
