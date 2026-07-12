"""Simple energy-loss helpers for OpenReco v3.0.

This module provides a compact deterministic energy-loss model for controlled
detector-effects studies.

The model is intentionally simple:

    total energy before material:
        E = sqrt(p^2 + m^2)

    total energy after material:
        E' = max(E - dE, m)

    momentum after material:
        p' = sqrt(E'^2 - m^2)

All energies and momenta are expressed in GeV, except energy_loss_mev inputs,
which are expressed in MeV for convenience.
"""

from __future__ import annotations

from math import isfinite, sqrt


PION_MASS_GEV = 0.13957039


def mev_to_gev(value_mev: float) -> float:
    """Convert MeV to GeV."""

    if not isfinite(value_mev):
        raise ValueError("value_mev must be finite")

    return float(value_mev) * 1.0e-3


def total_energy_from_momentum(
    *,
    p_gev: float,
    mass_gev: float = PION_MASS_GEV,
) -> float:
    """Return relativistic total energy from momentum and mass."""

    if not isfinite(p_gev):
        raise ValueError("p_gev must be finite")

    if not isfinite(mass_gev):
        raise ValueError("mass_gev must be finite")

    if p_gev < 0.0:
        raise ValueError("p_gev must be non-negative")

    if mass_gev < 0.0:
        raise ValueError("mass_gev must be non-negative")

    return float(sqrt(p_gev**2 + mass_gev**2))


def momentum_from_total_energy(
    *,
    energy_gev: float,
    mass_gev: float = PION_MASS_GEV,
) -> float:
    """Return momentum magnitude from relativistic total energy and mass."""

    if not isfinite(energy_gev):
        raise ValueError("energy_gev must be finite")

    if not isfinite(mass_gev):
        raise ValueError("mass_gev must be finite")

    if mass_gev < 0.0:
        raise ValueError("mass_gev must be non-negative")

    if energy_gev < mass_gev:
        raise ValueError("energy_gev must be greater than or equal to mass_gev")

    return float(sqrt(max(0.0, energy_gev**2 - mass_gev**2)))


def momentum_after_energy_loss(
    *,
    p_gev: float,
    energy_loss_mev: float,
    mass_gev: float = PION_MASS_GEV,
) -> float:
    """Return momentum after a deterministic energy loss.

    If the requested energy loss would reduce the particle below rest energy,
    the returned momentum is clamped to zero.
    """

    if not isfinite(energy_loss_mev):
        raise ValueError("energy_loss_mev must be finite")

    if energy_loss_mev < 0.0:
        raise ValueError("energy_loss_mev must be non-negative")

    initial_energy = total_energy_from_momentum(
        p_gev=p_gev,
        mass_gev=mass_gev,
    )

    final_energy = max(mass_gev, initial_energy - mev_to_gev(energy_loss_mev))

    return momentum_from_total_energy(
        energy_gev=final_energy,
        mass_gev=mass_gev,
    )


def q_over_p_after_energy_loss(
    *,
    q_over_p: float,
    energy_loss_mev: float,
    mass_gev: float = PION_MASS_GEV,
) -> float:
    """Return q/p after deterministic energy loss.

    The charge sign is preserved. Since p decreases after energy loss,
    |q/p| usually increases.
    """

    if not isfinite(q_over_p):
        raise ValueError("q_over_p must be finite")

    if q_over_p == 0.0:
        raise ValueError("q_over_p must be non-zero")

    charge_sign = 1.0 if q_over_p > 0.0 else -1.0
    p_gev = 1.0 / abs(q_over_p)

    final_p_gev = momentum_after_energy_loss(
        p_gev=p_gev,
        energy_loss_mev=energy_loss_mev,
        mass_gev=mass_gev,
    )

    if final_p_gev == 0.0:
        return charge_sign * float("inf")

    return float(charge_sign / final_p_gev)