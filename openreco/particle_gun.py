"""
Particle gun for OpenReco.

For OpenReco v0, we use a simple single-particle source instead of a full
event generator.

This module creates truth particles with known initial position, momentum,
and charge. These truth particles will later be propagated through the
cylindrical tracker in a uniform magnetic field.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Particle:
    """
    Truth-level particle.

    Parameters
    ----------
    position:
        Initial 3D position [x, y, z].
    momentum:
        Initial 3D momentum [px, py, pz].
    charge:
        Particle charge in units of elementary charge.
    """

    position: np.ndarray
    momentum: np.ndarray
    charge: float

    def __post_init__(self):
        position = np.asarray(self.position, dtype=float)
        momentum = np.asarray(self.momentum, dtype=float)
        charge = float(self.charge)

        if position.shape != (3,):
            raise ValueError("Particle position must have shape (3,)")

        if momentum.shape != (3,):
            raise ValueError("Particle momentum must have shape (3,)")

        if np.linalg.norm(momentum) <= 0.0:
            raise ValueError("Particle momentum magnitude must be positive")

        if charge == 0.0:
            raise ValueError("Particle charge must be nonzero")

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "momentum", momentum)
        object.__setattr__(self, "charge", charge)

    @property
    def p(self) -> float:
        """
        Total momentum magnitude.
        """
        return float(np.linalg.norm(self.momentum))

    @property
    def pt(self) -> float:
        """
        Transverse momentum magnitude.
        """
        px, py, _ = self.momentum
        return float(np.sqrt(px**2 + py**2))

    @property
    def direction(self) -> np.ndarray:
        """
        Unit direction vector.
        """
        return self.momentum / self.p

    @property
    def q_over_p(self) -> float:
        """
        Charge divided by total momentum.
        """
        return self.charge / self.p

    def copy(self) -> "Particle":
        """
        Return a deep copy of the particle.
        """
        return Particle(
            position=self.position.copy(),
            momentum=self.momentum.copy(),
            charge=self.charge,
        )


class ParticleGun:
    """
    Simple single-particle generator.

    The gun samples particles around configurable momentum and angle ranges.
    This is intentionally simple and deterministic when an RNG seed is used.
    """

    def __init__(
        self,
        position: np.ndarray | None = None,
        momentum_range: tuple[float, float] = (1.0, 5.0),
        theta_range: tuple[float, float] = (0.4, 1.2),
        phi_range: tuple[float, float] = (0.0, 2.0 * np.pi),
        charge_choices: tuple[float, ...] = (-1.0, 1.0),
        rng: np.random.Generator | None = None,
    ):
        if position is None:
            position = np.zeros(3)

        position = np.asarray(position, dtype=float)

        if position.shape != (3,):
            raise ValueError("ParticleGun position must have shape (3,)")

        if momentum_range[0] <= 0.0 or momentum_range[1] <= 0.0:
            raise ValueError("momentum_range values must be positive")

        if momentum_range[1] < momentum_range[0]:
            raise ValueError("momentum_range upper value must be >= lower value")

        if theta_range[1] < theta_range[0]:
            raise ValueError("theta_range upper value must be >= lower value")

        if phi_range[1] < phi_range[0]:
            raise ValueError("phi_range upper value must be >= lower value")

        if len(charge_choices) == 0:
            raise ValueError("charge_choices must not be empty")

        if any(float(charge) == 0.0 for charge in charge_choices):
            raise ValueError("charge_choices must not contain zero charge")

        self.position = position
        self.momentum_range = tuple(float(v) for v in momentum_range)
        self.theta_range = tuple(float(v) for v in theta_range)
        self.phi_range = tuple(float(v) for v in phi_range)
        self.charge_choices = tuple(float(v) for v in charge_choices)

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

    def shoot(self) -> Particle:
        """
        Generate one truth particle.
        """
        p = self.rng.uniform(*self.momentum_range)
        theta = self.rng.uniform(*self.theta_range)
        phi = self.rng.uniform(*self.phi_range)
        charge = self.rng.choice(self.charge_choices)

        px = p * np.sin(theta) * np.cos(phi)
        py = p * np.sin(theta) * np.sin(phi)
        pz = p * np.cos(theta)

        momentum = np.array([px, py, pz], dtype=float)

        return Particle(
            position=self.position.copy(),
            momentum=momentum,
            charge=charge,
        )


def make_fixed_particle(
    position: np.ndarray,
    momentum: np.ndarray,
    charge: float,
) -> Particle:
    """
    Convenience helper for creating one fixed truth particle.
    """
    return Particle(
        position=position,
        momentum=momentum,
        charge=charge,
    )
