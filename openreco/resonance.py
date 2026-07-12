"""Toy two-body resonance generation for OpenReco v3.1.

This module generates simple truth-level two-body decays for downstream
physics-observable studies.

The default channel is:

    J/psi -> mu+ mu-

The generated daughters are returned as OpenReco TruthParticle objects so they
can later be passed into the existing detector-hit generation and reconstruction
chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi, sqrt
from typing import Iterable

import numpy as np

from openreco.event_generation import TruthParticle
from openreco.invariant_mass import (
    JPSI_MASS_GEV,
    MUON_MASS_GEV,
    invariant_mass_from_momenta,
)


@dataclass(frozen=True)
class ToyResonanceDecay:
    """Truth-level two-body resonance decay."""

    resonance_id: int
    resonance_mass: float
    daughter_mass: float
    daughters: tuple[TruthParticle, TruthParticle]
    daughter_momenta: tuple[np.ndarray, np.ndarray]

    @property
    def truth_particles(self) -> tuple[TruthParticle, TruthParticle]:
        """Return the two daughter truth particles."""

        return self.daughters

    @property
    def truth_invariant_mass(self) -> float:
        """Return invariant mass computed from daughter momenta."""

        return invariant_mass_from_momenta(
            self.daughter_momenta[0],
            self.daughter_momenta[1],
            mass_1=self.daughter_mass,
            mass_2=self.daughter_mass,
        )

    @property
    def total_momentum(self) -> np.ndarray:
        """Return vector sum of daughter momenta."""

        return self.daughter_momenta[0] + self.daughter_momenta[1]


def two_body_momentum_magnitude(
    *,
    parent_mass: float,
    daughter_mass_1: float,
    daughter_mass_2: float,
) -> float:
    """Return daughter momentum magnitude in the parent rest frame.

    Formula:

        p = sqrt((M^2 - (m1 + m2)^2)(M^2 - (m1 - m2)^2)) / (2M)
    """

    values = (parent_mass, daughter_mass_1, daughter_mass_2)

    if not all(isfinite(value) for value in values):
        raise ValueError("masses must be finite")

    if parent_mass <= 0.0:
        raise ValueError("parent_mass must be positive")

    if daughter_mass_1 < 0.0 or daughter_mass_2 < 0.0:
        raise ValueError("daughter masses must be non-negative")

    if parent_mass < daughter_mass_1 + daughter_mass_2:
        raise ValueError("parent mass is too small for this two-body decay")

    first = parent_mass**2 - (daughter_mass_1 + daughter_mass_2) ** 2
    second = parent_mass**2 - (daughter_mass_1 - daughter_mass_2) ** 2

    return float(sqrt(max(0.0, first * second)) / (2.0 * parent_mass))


def random_unit_vector(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random unit vector with isotropic angular distribution."""

    cos_theta = float(rng.uniform(-1.0, 1.0))
    cos_theta = float(np.clip(cos_theta, -1.0 + 1.0e-12, 1.0 - 1.0e-12))

    sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = float(rng.uniform(0.0, 2.0 * pi))

    return np.array(
        [
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ],
        dtype=float,
    )


def truth_particle_from_momentum(
    *,
    truth_particle_id: int,
    momentum: Iterable[float],
    charge: int,
    z0: float = 0.0,
) -> TruthParticle:
    """Build an OpenReco TruthParticle from a Cartesian momentum vector."""

    momentum_vector = np.asarray(list(momentum), dtype=float)

    if momentum_vector.shape != (3,):
        raise ValueError("momentum must have shape (3,)")

    if charge not in (-1, 1):
        raise ValueError("charge must be either -1 or +1")

    px, py, pz = momentum_vector
    pt = float(sqrt(px * px + py * py))
    p = float(sqrt(px * px + py * py + pz * pz))

    if p <= 0.0:
        raise ValueError("momentum magnitude must be positive")

    if pt <= 0.0:
        raise ValueError("transverse momentum must be positive")

    phi = float(atan2(py, px))
    tan_lambda = float(pz / pt)
    q_over_p = float(charge / p)

    return TruthParticle(
        truth_particle_id=truth_particle_id,
        pt=pt,
        phi=phi,
        z0=float(z0),
        tan_lambda=tan_lambda,
        charge=int(charge),
        q_over_p=q_over_p,
        p=p,
    )


def generate_two_body_resonance_decay(
    *,
    resonance_id: int = 0,
    resonance_mass: float = JPSI_MASS_GEV,
    daughter_mass: float = MUON_MASS_GEV,
    first_truth_particle_id: int = 0,
    rng: np.random.Generator | None = None,
) -> ToyResonanceDecay:
    """Generate a toy two-body resonance decay at rest.

    The default is J/psi -> mu+ mu-. The two daughters have opposite
    three-momenta and opposite charges.
    """

    rng = np.random.default_rng() if rng is None else rng

    daughter_p = two_body_momentum_magnitude(
        parent_mass=resonance_mass,
        daughter_mass_1=daughter_mass,
        daughter_mass_2=daughter_mass,
    )

    direction = random_unit_vector(rng)

    momentum_plus = daughter_p * direction
    momentum_minus = -momentum_plus

    positive_daughter = truth_particle_from_momentum(
        truth_particle_id=first_truth_particle_id,
        momentum=momentum_plus,
        charge=1,
    )
    negative_daughter = truth_particle_from_momentum(
        truth_particle_id=first_truth_particle_id + 1,
        momentum=momentum_minus,
        charge=-1,
    )

    return ToyResonanceDecay(
        resonance_id=int(resonance_id),
        resonance_mass=float(resonance_mass),
        daughter_mass=float(daughter_mass),
        daughters=(positive_daughter, negative_daughter),
        daughter_momenta=(momentum_plus, momentum_minus),
    )