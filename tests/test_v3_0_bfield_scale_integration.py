import numpy as np

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.detector_effects import DetectorEffectsConfig, MagneticFieldScale


def _mean_chi2_ndof(result) -> float:
    values = [float(track.chi2_ndof) for track in result.tracks]

    if not values:
        return float("nan")

    return float(np.mean(values))


def test_v3_0_reco_b_field_scale_changes_fit_quality():
    nominal_config = DetectorEffectsConfig(
        b_field_scale=MagneticFieldScale(
            truth_scale=1.0,
            reco_scale=1.0,
        )
    )
    shifted_reco_config = DetectorEffectsConfig(
        b_field_scale=MagneticFieldScale(
            truth_scale=1.0,
            reco_scale=1.05,
        )
    )

    common_kwargs = dict(
        event_id=0,
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=123,
        seed_mode="hole-aware",
        min_hits=4,
        max_fit_chi2_ndof=None,
        make_plot=False,
    )

    nominal_result = run_multi_track_reconstruction(
        **common_kwargs,
        detector_effects=nominal_config,
    )
    shifted_result = run_multi_track_reconstruction(
        **common_kwargs,
        detector_effects=shifted_reco_config,
    )

    assert nominal_result.tracks
    assert shifted_result.tracks

    nominal_chi2 = _mean_chi2_ndof(nominal_result)
    shifted_chi2 = _mean_chi2_ndof(shifted_result)

    assert np.isfinite(nominal_chi2)
    assert np.isfinite(shifted_chi2)
    assert not np.isclose(nominal_chi2, shifted_chi2)