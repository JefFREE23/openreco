import numpy as np

from openreco.invariant_mass import JPSI_MASS_GEV, MUON_MASS_GEV
from openreco.resonance import (
    generate_two_body_resonance_decay,
    random_unit_vector,
    truth_particle_from_momentum,
    two_body_momentum_magnitude,
)


def test_two_body_momentum_magnitude_for_jpsi_to_mumu():
    daughter_p = two_body_momentum_magnitude(
        parent_mass=JPSI_MASS_GEV,
        daughter_mass_1=MUON_MASS_GEV,
        daughter_mass_2=MUON_MASS_GEV,
    )

    assert daughter_p > 0.0
    assert daughter_p < 0.5 * JPSI_MASS_GEV


def test_two_body_momentum_rejects_impossible_decay():
    try:
        two_body_momentum_magnitude(
            parent_mass=1.0,
            daughter_mass_1=1.0,
            daughter_mass_2=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for impossible decay")


def test_random_unit_vector_has_unit_norm():
    rng = np.random.default_rng(123)

    direction = random_unit_vector(rng)

    assert direction.shape == (3,)
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_truth_particle_from_momentum_converts_cartesian_momentum():
    particle = truth_particle_from_momentum(
        truth_particle_id=7,
        momentum=(1.0, 1.0, 2.0),
        charge=1,
    )

    expected_pt = np.sqrt(2.0)
    expected_p = np.sqrt(6.0)

    assert particle.truth_particle_id == 7
    assert particle.charge == 1
    assert np.isclose(particle.pt, expected_pt)
    assert np.isclose(particle.p, expected_p)
    assert np.isclose(particle.q_over_p, 1.0 / expected_p)
    assert np.isclose(particle.tan_lambda, 2.0 / expected_pt)


def test_truth_particle_from_momentum_rejects_zero_transverse_momentum():
    try:
        truth_particle_from_momentum(
            truth_particle_id=0,
            momentum=(0.0, 0.0, 1.0),
            charge=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero transverse momentum")


def test_generate_two_body_resonance_decay_has_opposite_charges():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    first, second = decay.truth_particles

    assert first.charge == 1
    assert second.charge == -1
    assert first.truth_particle_id == 0
    assert second.truth_particle_id == 1


def test_generate_two_body_resonance_decay_conserves_momentum():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    assert np.allclose(decay.total_momentum, np.zeros(3), atol=1.0e-12)


def test_generate_two_body_resonance_decay_reconstructs_truth_mass():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    assert np.isclose(decay.truth_invariant_mass, JPSI_MASS_GEV)


def test_generate_two_body_resonance_decay_is_reproducible_for_same_seed():
    first = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )
    second = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    assert np.allclose(first.daughter_momenta[0], second.daughter_momenta[0])
    assert np.allclose(first.daughter_momenta[1], second.daughter_momenta[1])