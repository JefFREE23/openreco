import numpy as np

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.detector_effects import DetectorEffectsConfig


def _mean_covariance_trace(result) -> float:
    traces = [
        float(np.trace(track.final_covariance))
        for track in result.tracks
        if track.final_covariance is not None
    ]

    if not traces:
        return float("nan")

    return float(np.mean(traces))


def test_v3_0_process_noise_scale_changes_fitted_covariance():
    config = DetectorEffectsConfig.with_uniform_material(
        layer_ids=range(6),
        x_over_x0=0.05,
    )

    common_kwargs = dict(
        event_id=0,
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        detector_effects=config,
        random_seed=123,
        seed_mode="hole-aware",
        min_hits=4,
        max_fit_chi2_ndof=None,
        make_plot=False,
    )

    no_process_noise = run_multi_track_reconstruction(
        **common_kwargs,
        process_noise_scale=0.0,
    )

    with_process_noise = run_multi_track_reconstruction(
        **common_kwargs,
        process_noise_scale=20.0,
    )

    assert no_process_noise.tracks
    assert with_process_noise.tracks

    no_process_noise_trace = _mean_covariance_trace(no_process_noise)
    with_process_noise_trace = _mean_covariance_trace(with_process_noise)

    assert np.isfinite(no_process_noise_trace)
    assert np.isfinite(with_process_noise_trace)
    assert not np.isclose(no_process_noise_trace, with_process_noise_trace)