import numpy as np

from openreco.energy_loss import (
    PION_MASS_GEV,
    mev_to_gev,
    momentum_after_energy_loss,
    momentum_from_total_energy,
    q_over_p_after_energy_loss,
    total_energy_from_momentum,
)


def test_mev_to_gev():
    assert mev_to_gev(1000.0) == 1.0
    assert mev_to_gev(1.0) == 0.001


def test_total_energy_from_momentum():
    energy = total_energy_from_momentum(
        p_gev=1.0,
        mass_gev=PION_MASS_GEV,
    )

    assert energy > 1.0


def test_momentum_from_total_energy_inverse_relation():
    p_gev = 2.0
    energy = total_energy_from_momentum(
        p_gev=p_gev,
        mass_gev=PION_MASS_GEV,
    )
    recovered_p = momentum_from_total_energy(
        energy_gev=energy,
        mass_gev=PION_MASS_GEV,
    )

    assert np.isclose(recovered_p, p_gev)


def test_momentum_after_zero_energy_loss_is_unchanged():
    final_p = momentum_after_energy_loss(
        p_gev=2.0,
        energy_loss_mev=0.0,
    )

    assert np.isclose(final_p, 2.0)


def test_momentum_after_energy_loss_decreases():
    final_p = momentum_after_energy_loss(
        p_gev=2.0,
        energy_loss_mev=50.0,
    )

    assert final_p < 2.0
    assert final_p > 0.0


def test_q_over_p_after_energy_loss_preserves_charge_sign_and_increases_magnitude():
    initial_q_over_p = 0.5

    final_q_over_p = q_over_p_after_energy_loss(
        q_over_p=initial_q_over_p,
        energy_loss_mev=50.0,
    )

    assert final_q_over_p > 0.0
    assert abs(final_q_over_p) > abs(initial_q_over_p)

    negative_final_q_over_p = q_over_p_after_energy_loss(
        q_over_p=-initial_q_over_p,
        energy_loss_mev=50.0,
    )

    assert negative_final_q_over_p < 0.0
    assert abs(negative_final_q_over_p) > abs(initial_q_over_p)


def test_energy_loss_rejects_invalid_inputs():
    try:
        momentum_after_energy_loss(
            p_gev=1.0,
            energy_loss_mev=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for negative energy_loss_mev")

    try:
        q_over_p_after_energy_loss(
            q_over_p=0.0,
            energy_loss_mev=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero q_over_p")