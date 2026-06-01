import numpy as np
import pytest

from openreco.field import UniformMagneticField
from openreco.geometry import make_barrel_detector
from openreco.kalman import (
    KalmanUpdateResult,
    filter_cylindrical_track,
)
from openreco.measurements import Measurement
from openreco.smoothing import (
    SmoothingResult,
    compute_smoothing_gain,
    smooth_track,
    smoothed_positions,
    smoothed_states,
)
from openreco.state import make_cylindrical_state


def make_initial_state():
    return make_cylindrical_state(
        phi=0.0,
        z=0.0,
        dir0=0.0,
        dir1=1.0,
        q_over_p=0.5,
        covariance=np.eye(5) * 0.1,
        surface_radius=1e-6,
        surface_name="seed",
    )


def make_test_results():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=100.0,
    )

    field = UniformMagneticField(bz=2.0)

    measurements = [
        Measurement(
            values=np.array([0.0, 10.0]),
            covariance=np.diag([0.01**2, 0.1**2]),
            layer_name="barrel_0",
            surface_type="cylinder",
        ),
        Measurement(
            values=np.array([0.0, 20.0]),
            covariance=np.diag([0.01**2, 0.1**2]),
            layer_name="barrel_1",
            surface_type="cylinder",
        ),
        Measurement(
            values=np.array([0.0, 30.0]),
            covariance=np.diag([0.01**2, 0.1**2]),
            layer_name="barrel_2",
            surface_type="cylinder",
        ),
    ]

    return filter_cylindrical_track(
        initial_state=make_initial_state(),
        measurements=measurements,
        detector=detector,
        field=field,
        curvature_scale=0.0,
    )


def test_compute_smoothing_gain_shape():
    gain = compute_smoothing_gain(
        filtered_covariance=np.eye(5),
        transport_jacobian=np.eye(5),
        predicted_next_covariance=np.eye(5) * 2.0,
    )

    assert gain.shape == (5, 5)


def test_compute_smoothing_gain_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compute_smoothing_gain(
            filtered_covariance=np.eye(4),
            transport_jacobian=np.eye(5),
            predicted_next_covariance=np.eye(5),
        )

    with pytest.raises(ValueError):
        compute_smoothing_gain(
            filtered_covariance=np.eye(5),
            transport_jacobian=np.eye(4),
            predicted_next_covariance=np.eye(5),
        )

    with pytest.raises(ValueError):
        compute_smoothing_gain(
            filtered_covariance=np.eye(5),
            transport_jacobian=np.eye(5),
            predicted_next_covariance=np.eye(4),
        )


def test_smooth_track_returns_result_for_each_filter_result():
    results = make_test_results()

    smoothing_results = smooth_track(results)

    assert len(smoothing_results) == len(results)
    assert all(isinstance(result, SmoothingResult) for result in smoothing_results)


def test_last_smoothed_state_equals_last_filtered_state():
    results = make_test_results()

    smoothing_results = smooth_track(results)

    np.testing.assert_allclose(
        smoothing_results[-1].smoothed_state.parameters,
        results[-1].filtered_state.parameters,
    )

    np.testing.assert_allclose(
        smoothing_results[-1].smoothed_state.covariance,
        results[-1].filtered_state.covariance,
    )


def test_smoothed_covariances_are_symmetric():
    results = make_test_results()

    smoothing_results = smooth_track(results)

    for result in smoothing_results:
        np.testing.assert_allclose(
            result.smoothed_state.covariance,
            result.smoothed_state.covariance.T,
            atol=1e-10,
        )


def test_smoothed_states_keep_surface_information():
    results = make_test_results()

    smoothing_results = smooth_track(results)

    for smoothing_result, kalman_result in zip(smoothing_results, results):
        assert smoothing_result.smoothed_state.surface_type == "cylinder"
        assert smoothing_result.smoothed_state.surface_name == kalman_result.filtered_state.surface_name
        assert smoothing_result.smoothed_state.surface_radius == kalman_result.filtered_state.surface_radius


def test_smoothed_states_helper():
    results = make_test_results()

    smoothing_results = smooth_track(results)
    states = smoothed_states(smoothing_results)

    assert len(states) == len(results)


def test_smoothed_positions_helper():
    results = make_test_results()

    smoothing_results = smooth_track(results)
    positions = smoothed_positions(smoothing_results)

    assert positions.shape == (len(results), 3)


def test_smooth_track_rejects_empty_results():
    with pytest.raises(ValueError):
        smooth_track([])


def test_smooth_track_requires_transport_jacobian():
    state = make_cylindrical_state(
        phi=0.0,
        z=10.0,
        dir0=0.0,
        dir1=1.0,
        q_over_p=0.5,
        covariance=np.eye(5),
        surface_radius=10.0,
        surface_name="barrel_0",
    )

    result_0 = KalmanUpdateResult(
        predicted_state=state,
        filtered_state=state,
        residual=np.zeros(2),
        residual_covariance=np.eye(2),
        kalman_gain=np.zeros((5, 2)),
        chi2=0.0,
        layer_name="barrel_0",
    )

    result_1 = KalmanUpdateResult(
        predicted_state=state,
        filtered_state=state,
        residual=np.zeros(2),
        residual_covariance=np.eye(2),
        kalman_gain=np.zeros((5, 2)),
        chi2=0.0,
        layer_name="barrel_1",
    )

    with pytest.raises(ValueError):
        smooth_track([result_0, result_1])
