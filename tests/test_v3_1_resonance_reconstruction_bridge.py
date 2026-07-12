import numpy as np

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.resonance import generate_two_body_resonance_decay


def test_multi_track_reconstruction_accepts_resonance_truth_particles():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    result = run_multi_track_reconstruction(
        event_id=11,
        n_particles=999,
        truth_particles=decay.truth_particles,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=456,
        chi2_threshold=100.0,
        min_hits=6,
        use_ekf_fit=False,
        make_plot=False,
    )

    assert result.event.event_id == 11
    assert len(result.event.truth_particles) == 2
    assert len(result.tracks) == 2
    assert result.validation.n_matched_tracks == 2
    assert result.validation.tracking_efficiency == 1.0
    assert result.validation.fake_rate == 0.0