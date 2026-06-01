"""
Magnetic field models for OpenReco.

For OpenReco v0, we start with a homogeneous magnetic field.

The main v0 use case is a uniform field along the beam axis:

    B = [0, 0, Bz]

This is the field used later for the charged-particle uniform-B
cylindrical tracker demo.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UniformMagneticField:
    """
    Constant magnetic field.

    Parameters
    ----------
    bx:
        Magnetic field x component.
    by:
        Magnetic field y component.
    bz:
        Magnetic field z component.
    """

    bx: float = 0.0
    by: float = 0.0
    bz: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "bx", float(self.bx))
        object.__setattr__(self, "by", float(self.by))
        object.__setattr__(self, "bz", float(self.bz))

    @property
    def vector(self) -> np.ndarray:
        """
        Return the magnetic field vector [Bx, By, Bz].
        """
        return np.array([self.bx, self.by, self.bz], dtype=float)

    @property
    def magnitude(self) -> float:
        """
        Return the magnetic field magnitude.
        """
        return float(np.linalg.norm(self.vector))

    def at(self, position: np.ndarray) -> np.ndarray:
        """
        Return the magnetic field at a given position.

        For a uniform field, the result does not depend on position.
        The position argument is kept because later field models may
        depend on position.
        """

        position = np.asarray(position, dtype=float)

        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")

        return self.vector.copy()

    def is_along_z(self) -> bool:
        """
        Return True if the field has only a z component.
        """
        return np.isclose(self.bx, 0.0) and np.isclose(self.by, 0.0)
