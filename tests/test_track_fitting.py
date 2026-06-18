import numpy as np

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.track_fitting import (
    fit_reconstructed_track_with_ekf,
    fit_track_candidate_with_ekf,
)


def test_ekf_fit_returns_filtered_and_smoothed_states():
    result = run_multi_track_reconstruction(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=123,
        make_plot=False,
        use_ekf_fit=False,
    )

    track = result.tracks[0]
    fit_result = fit_track_candidate_with_ekf(track)

    assert len(fit_result.filtered_states) == len(track.used_measurements)
    assert len(fit_result.smoothed_states) == len(track.used_measurements)

    assert fit_result.final_state.surface_type == "cylinder"
    assert fit_result.final_covariance.shape == (5, 5)


def test_ekf_fit_covariance_is_valid_for_clean_track():
    result = run_multi_track_reconstruction(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=123,
        make_plot=False,
        use_ekf_fit=False,
    )

    track = result.tracks[0]
    fitted_track = fit_reconstructed_track_with_ekf(track)

    assert fitted_track.covariance_valid is True
    assert fitted_track.final_covariance is not None
    assert fitted_track.final_covariance.shape == (5, 5)


def test_ekf_fit_updates_track_momentum_fields():
    result = run_multi_track_reconstruction(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=123,
        make_plot=False,
        use_ekf_fit=False,
    )

    track = result.tracks[0]
    fitted_track = fit_reconstructed_track_with_ekf(track)

    assert np.isfinite(fitted_track.q_over_p)
    assert np.isfinite(fitted_track.pt_estimate)
    assert np.isfinite(fitted_track.p_estimate)
    assert fitted_track.p_estimate > 0.0
    assert fitted_track.fit_status == "accepted"


def test_multi_track_reconstruction_attaches_ekf_outputs_by_default():
    result = run_multi_track_reconstruction(
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=123,
        make_plot=False,
    )

    assert len(result.tracks) == 2

    for track in result.tracks:
        assert len(track.filtered_states) == len(track.used_measurements)
        assert len(track.smoothed_states) == len(track.used_measurements)
        assert track.final_covariance is not None
        assert track.covariance_valid is True
