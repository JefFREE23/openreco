"""Invariant-mass utilities for OpenReco v3.1.

This module provides small physics-observable helpers used by the toy
resonance study.

All masses, energies, and momenta are expressed in GeV.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, pstdev
from typing import Iterable

import numpy as np


MUON_MASS_GEV = 0.1056583745
JPSI_MASS_GEV = 3.0969


@dataclass(frozen=True)
class FourVector:
    """Simple Cartesian four-vector."""

    energy: float
    px: float
    py: float
    pz: float

    def __post_init__(self) -> None:
        values = (self.energy, self.px, self.py, self.pz)

        if not all(isfinite(value) for value in values):
            raise ValueError("four-vector components must be finite")

        if self.energy < 0.0:
            raise ValueError("energy must be non-negative")

    @property
    def spatial_momentum_squared(self) -> float:
        return float(self.px * self.px + self.py * self.py + self.pz * self.pz)

    @property
    def spatial_momentum(self) -> float:
        return float(sqrt(self.spatial_momentum_squared))

    @property
    def mass_squared(self) -> float:
        return float(self.energy * self.energy - self.spatial_momentum_squared)

    @property
    def mass(self) -> float:
        return float(sqrt(max(0.0, self.mass_squared)))

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            energy=self.energy + other.energy,
            px=self.px + other.px,
            py=self.py + other.py,
            pz=self.pz + other.pz,
        )


@dataclass(frozen=True)
class MassSummary:
    """Summary statistics for reconstructed invariant masses."""

    n: int
    truth_mass: float
    mass_mean: float
    mass_width: float
    residual_mean: float
    residual_width: float


def energy_from_momentum(
    *,
    px: float,
    py: float,
    pz: float,
    mass: float,
) -> float:
    """Return relativistic total energy from Cartesian momentum and mass."""

    values = (px, py, pz, mass)

    if not all(isfinite(value) for value in values):
        raise ValueError("momentum components and mass must be finite")

    if mass < 0.0:
        raise ValueError("mass must be non-negative")

    return float(sqrt(px * px + py * py + pz * pz + mass * mass))


def make_four_vector(
    *,
    px: float,
    py: float,
    pz: float,
    mass: float = MUON_MASS_GEV,
) -> FourVector:
    """Build a mass-constrained four-vector from Cartesian momentum."""

    return FourVector(
        energy=energy_from_momentum(px=px, py=py, pz=pz, mass=mass),
        px=float(px),
        py=float(py),
        pz=float(pz),
    )


def invariant_mass(four_vectors: Iterable[FourVector]) -> float:
    """Return invariant mass of one or more four-vectors."""

    total: FourVector | None = None

    for vector in four_vectors:
        if total is None:
            total = vector
        else:
            total = total + vector

    if total is None:
        raise ValueError("at least one four-vector is required")

    return total.mass


def invariant_mass_from_momenta(
    momentum_1: Iterable[float],
    momentum_2: Iterable[float],
    *,
    mass_1: float = MUON_MASS_GEV,
    mass_2: float = MUON_MASS_GEV,
) -> float:
    """Return two-body invariant mass from two Cartesian momenta."""

    p1 = np.asarray(list(momentum_1), dtype=float)
    p2 = np.asarray(list(momentum_2), dtype=float)

    if p1.shape != (3,):
        raise ValueError("momentum_1 must have shape (3,)")

    if p2.shape != (3,):
        raise ValueError("momentum_2 must have shape (3,)")

    vector_1 = make_four_vector(
        px=float(p1[0]),
        py=float(p1[1]),
        pz=float(p1[2]),
        mass=mass_1,
    )
    vector_2 = make_four_vector(
        px=float(p2[0]),
        py=float(p2[1]),
        pz=float(p2[2]),
        mass=mass_2,
    )

    return invariant_mass((vector_1, vector_2))


def mass_residual(
    *,
    reconstructed_mass: float,
    truth_mass: float,
) -> float:
    """Return reconstructed_mass - truth_mass."""

    if not isfinite(reconstructed_mass):
        raise ValueError("reconstructed_mass must be finite")

    if not isfinite(truth_mass):
        raise ValueError("truth_mass must be finite")

    return float(reconstructed_mass - truth_mass)


def summarize_reconstructed_masses(
    reconstructed_masses: Iterable[float],
    *,
    truth_mass: float,
) -> MassSummary:
    """Summarize reconstructed invariant masses and mass residuals."""

    masses = [float(value) for value in reconstructed_masses if isfinite(float(value))]

    if truth_mass <= 0.0:
        raise ValueError("truth_mass must be positive")

    if not masses:
        return MassSummary(
            n=0,
            truth_mass=float(truth_mass),
            mass_mean=float("nan"),
            mass_width=float("nan"),
            residual_mean=float("nan"),
            residual_width=float("nan"),
        )

    residuals = [mass - truth_mass for mass in masses]

    return MassSummary(
        n=len(masses),
        truth_mass=float(truth_mass),
        mass_mean=mean(masses),
        mass_width=pstdev(masses) if len(masses) >= 2 else float("nan"),
        residual_mean=mean(residuals),
        residual_width=pstdev(residuals) if len(residuals) >= 2 else float("nan"),
    )