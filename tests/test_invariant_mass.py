import numpy as np

from openreco.invariant_mass import (
    FourVector,
    JPSI_MASS_GEV,
    MUON_MASS_GEV,
    energy_from_momentum,
    invariant_mass,
    invariant_mass_from_momenta,
    make_four_vector,
    mass_residual,
    summarize_reconstructed_masses,
)


def test_energy_from_momentum_rest_particle_equals_mass():
    energy = energy_from_momentum(
        px=0.0,
        py=0.0,
        pz=0.0,
        mass=MUON_MASS_GEV,
    )

    assert np.isclose(energy, MUON_MASS_GEV)


def test_make_four_vector_has_expected_mass():
    vector = make_four_vector(
        px=1.0,
        py=0.0,
        pz=0.0,
        mass=MUON_MASS_GEV,
    )

    assert np.isclose(vector.mass, MUON_MASS_GEV)


def test_four_vector_addition():
    first = FourVector(energy=1.0, px=0.2, py=0.0, pz=0.0)
    second = FourVector(energy=2.0, px=-0.1, py=0.3, pz=0.0)

    total = first + second

    assert total.energy == 3.0
    assert total.px == 0.1
    assert total.py == 0.3
    assert total.pz == 0.0


def test_invariant_mass_of_back_to_back_jpsi_daughters():
    daughter_energy = 0.5 * JPSI_MASS_GEV
    daughter_p = np.sqrt(daughter_energy**2 - MUON_MASS_GEV**2)

    mass = invariant_mass_from_momenta(
        (daughter_p, 0.0, 0.0),
        (-daughter_p, 0.0, 0.0),
        mass_1=MUON_MASS_GEV,
        mass_2=MUON_MASS_GEV,
    )

    assert np.isclose(mass, JPSI_MASS_GEV)


def test_invariant_mass_accepts_four_vectors():
    daughter_energy = 0.5 * JPSI_MASS_GEV
    daughter_p = np.sqrt(daughter_energy**2 - MUON_MASS_GEV**2)

    first = make_four_vector(
        px=daughter_p,
        py=0.0,
        pz=0.0,
        mass=MUON_MASS_GEV,
    )
    second = make_four_vector(
        px=-daughter_p,
        py=0.0,
        pz=0.0,
        mass=MUON_MASS_GEV,
    )

    assert np.isclose(invariant_mass((first, second)), JPSI_MASS_GEV)


def test_mass_residual():
    residual = mass_residual(
        reconstructed_mass=3.10,
        truth_mass=3.0969,
    )

    assert np.isclose(residual, 0.0031)


def test_summarize_reconstructed_masses():
    summary = summarize_reconstructed_masses(
        [3.0, 3.1, 3.2],
        truth_mass=3.1,
    )

    assert summary.n == 3
    assert np.isclose(summary.mass_mean, 3.1)
    assert np.isclose(summary.residual_mean, 0.0)
    assert summary.mass_width > 0.0
    assert summary.residual_width > 0.0


def test_invariant_mass_rejects_empty_input():
    try:
        invariant_mass([])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty invariant-mass input")


def test_invariant_mass_from_momenta_rejects_wrong_shape():
    try:
        invariant_mass_from_momenta(
            (1.0, 0.0),
            (-1.0, 0.0, 0.0),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong momentum shape")