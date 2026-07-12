import numpy as np

from examples.v3_1_toy_resonance_baseline import run_toy_resonance_baseline
from openreco.invariant_mass import JPSI_MASS_GEV


def test_toy_resonance_baseline_reconstructs_candidate():
    result = run_toy_resonance_baseline(
        random_seed=123,
        event_id=31,
        use_ekf_fit=False,
    )

    assert result.best_candidate is not None
    assert len(result.decay.truth_particles) == 2
    assert len(result.reconstruction_result.tracks) == 2
    assert len(result.candidates) == 1
    assert result.reconstruction_result.validation.n_matched_tracks == 2
    assert result.reconstruction_result.validation.tracking_efficiency == 1.0
    assert np.isfinite(result.reconstructed_mass)
    assert result.reconstructed_mass > 0.0
    assert abs(result.mass_residual) < 1.0


def test_toy_resonance_baseline_truth_mass_matches_jpsi_constant():
    result = run_toy_resonance_baseline(
        random_seed=123,
        event_id=32,
        use_ekf_fit=False,
    )

    assert np.isclose(result.truth_mass, JPSI_MASS_GEV)
    assert np.isclose(result.decay.truth_invariant_mass, JPSI_MASS_GEV)


def test_toy_resonance_baseline_is_reproducible():
    first = run_toy_resonance_baseline(
        random_seed=123,
        event_id=33,
        use_ekf_fit=False,
    )
    second = run_toy_resonance_baseline(
        random_seed=123,
        event_id=33,
        use_ekf_fit=False,
    )

    assert first.best_candidate is not None
    assert second.best_candidate is not None
    assert np.isclose(first.reconstructed_mass, second.reconstructed_mass)
    assert np.isclose(first.mass_residual, second.mass_residual)