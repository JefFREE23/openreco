import numpy as np
import pytest

from openreco.field import UniformMagneticField
from openreco.geometry import CylindricalLayer, make_barrel_detector
from openreco.kalman import (
    KalmanPredictionResult,
    KalmanUpdateResult,
    cylindrical_full_residual,
    cylindrical_phi_jacobian,
    cylindrical_phi_prediction,
    filter_cylindrical_track,
    make_process_noise,
    numerical_jacobian,
    predict_to_cylindrical_layer,
    state_to_particle,
    total_chi2,
    update_state,
    update_with_cylindrical_measurement,
    wrap_angle,
)
from openreco.measurements import Measurement
from openreco.state import TrackState


def make_test_state():
    return TrackState(
        parameters=np.array([0.0, 0.0, 1.0, 0.0, 0.5]),
        covariance=np.eye(5) * 0.1,
        z=0.0,
    )


def test_make_process_noise():
    noise = make_process_noise([1.0, 2.0, 3.0, 4.0, 5.0])

    np.testing.assert_allclose(noise, np.diag([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_make_process_noise_rejects_wrong_shape():
    with pytest.raises(ValueError):
        make_process_noise([1.0, 2.0])


def test_make_process_noise_rejects_negative_values():
    with pytest.raises(ValueError):
        make_process_noise([1.0, 2.0, -1.0, 4.0, 5.0])


def test_wrap_angle():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(2.0 * np.pi) == pytest.approx(0.0)
    assert wrap_angle(-2.0 * np.pi) == pytest.approx(0.0)
    assert wrap_angle(2.0 * np.pi - 0.01) == pytest.approx(-0.01)


def test_state_to_particle():
    state = TrackState(
        parameters=np.array([1.0, 2.0, 1.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=3.0,
    )

    particle = state_to_particle(state)

    np.testing.assert_allclose(particle.position, np.array([1.0, 2.0, 3.0]))
    assert particle.charge == 1.0
    assert particle.p == pytest.approx(2.0)


def test_state_to_particle_negative_charge():
    state = TrackState(
        parameters=np.array([1.0, 2.0, 1.0, 0.0, -0.5]),
        covariance=np.eye(5),
        z=3.0,
    )

    particle = state_to_particle(state)

    assert particle.charge == -1.0
    assert particle.p == pytest.approx(2.0)


def test_state_to_particle_rejects_zero_q_over_p():
    state = TrackState(
        parameters=np.array([1.0, 2.0, 1.0, 0.0, 0.0]),
        covariance=np.eye(5),
        z=3.0,
    )

    with pytest.raises(ValueError):
        state_to_particle(state)


def test_numerical_jacobian():
    def function(parameters):
        x, y = parameters
        return np.array([x**2, x + y])

    jacobian = numerical_jacobian(
        function=function,
        parameters=np.array([2.0, 3.0]),
        step=1e-6,
    )

    expected = np.array(
        [
            [4.0, 0.0],
            [1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(jacobian, expected, atol=1e-5)


def test_numerical_jacobian_rejects_bad_step():
    def function(parameters):
        return parameters

    with pytest.raises(ValueError):
        numerical_jacobian(function, np.array([1.0, 2.0]), step=0.0)


def test_predict_to_cylindrical_layer():
    state = make_test_state()
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    prediction = predict_to_cylindrical_layer(
        state=state,
        field=field,
        layer=layer,
        curvature_scale=0.0,
    )

    assert isinstance(prediction, KalmanPredictionResult)
    assert prediction.predicted_state.z == pytest.approx(10.0)
    assert prediction.transport_jacobian.shape == (5, 5)
    assert prediction.propagation_result.layer_name == "barrel_0"


def test_predict_to_cylindrical_layer_hits_radius():
    state = make_test_state()
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    prediction = predict_to_cylindrical_layer(
        state=state,
        field=field,
        layer=layer,
    )

    predicted_position = np.array(
        [
            prediction.predicted_state.x,
            prediction.predicted_state.y,
            prediction.predicted_state.z,
        ]
    )

    assert predicted_position[:2].shape == (2,)
    assert np.sqrt(predicted_position[0] ** 2 + predicted_position[1] ** 2) == pytest.approx(10.0)


def test_predict_to_cylindrical_layer_with_process_noise():
    state = make_test_state()
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)
    noise = make_process_noise([0.01, 0.01, 0.0, 0.0, 0.0])

    prediction_without_noise = predict_to_cylindrical_layer(
        state=state,
        field=field,
        layer=layer,
        process_noise=None,
    )

    prediction_with_noise = predict_to_cylindrical_layer(
        state=state,
        field=field,
        layer=layer,
        process_noise=noise,
    )

    assert (
        prediction_with_noise.predicted_state.covariance[0, 0]
        > prediction_without_noise.predicted_state.covariance[0, 0]
    )


def test_predict_to_cylindrical_layer_rejects_bad_process_noise():
    state = make_test_state()
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    with pytest.raises(ValueError):
        predict_to_cylindrical_layer(
            state=state,
            field=field,
            layer=layer,
            process_noise=np.eye(4),
        )


def test_cylindrical_phi_prediction():
    state = TrackState(
        parameters=np.array([0.0, 10.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    prediction = cylindrical_phi_prediction(state)

    np.testing.assert_allclose(prediction, np.array([np.pi / 2.0]))


def test_cylindrical_phi_jacobian():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    jacobian = cylindrical_phi_jacobian(state)

    expected = np.array([[0.0, 0.1, 0.0, 0.0, 0.0]])

    np.testing.assert_allclose(jacobian, expected)


def test_cylindrical_phi_jacobian_rejects_origin():
    state = TrackState(
        parameters=np.array([0.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    with pytest.raises(ValueError):
        cylindrical_phi_jacobian(state)


def test_cylindrical_full_residual():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    measurement = Measurement(
        values=np.array([0.01, 5.2]),
        covariance=np.diag([0.001**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
    )

    residual = cylindrical_full_residual(state, measurement)

    np.testing.assert_allclose(residual, np.array([0.01, 0.2]))


def test_update_state():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    result = update_state(
        predicted_state=state,
        measurement_values=np.array([0.01]),
        measurement_covariance=np.array([[0.001**2]]),
        measurement_matrix=cylindrical_phi_jacobian(state),
        predicted_measurement=cylindrical_phi_prediction(state),
        layer_name="barrel_0",
        wrap_first_residual=True,
    )

    assert isinstance(result, KalmanUpdateResult)
    assert result.residual.shape == (1,)
    assert result.residual_covariance.shape == (1, 1)
    assert result.kalman_gain.shape == (5, 1)
    assert result.layer_name == "barrel_0"
    assert result.chi2 >= 0.0


def test_update_state_reduces_phi_residual():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    measurement = Measurement(
        values=np.array([0.01, 5.0]),
        covariance=np.diag([0.001**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
    )

    residual_before = abs(cylindrical_full_residual(state, measurement)[0])

    result = update_with_cylindrical_measurement(
        predicted_state=state,
        measurement=measurement,
    )

    residual_after = abs(
        cylindrical_full_residual(result.filtered_state, measurement)[0]
    )

    assert residual_after < residual_before


def test_update_with_cylindrical_measurement_wraps_phi():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    measurement = Measurement(
        values=np.array([2.0 * np.pi - 0.01, 5.0]),
        covariance=np.diag([0.001**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
    )

    result = update_with_cylindrical_measurement(state, measurement)

    assert result.residual[0] == pytest.approx(-0.01)


def test_update_with_cylindrical_measurement_rejects_wrong_surface_type():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.0, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    measurement = Measurement(
        values=np.array([0.0, 5.0]),
        covariance=np.eye(2),
        layer_name="layer_0",
        surface_type="plane",
    )

    with pytest.raises(ValueError):
        update_with_cylindrical_measurement(state, measurement)


def test_filter_cylindrical_track():
    initial_state = make_test_state()
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

    results = filter_cylindrical_track(
        initial_state=initial_state,
        measurements=measurements,
        detector=detector,
        field=field,
        curvature_scale=0.0,
    )

    assert len(results) == 3
    assert results[0].layer_name == "barrel_0"
    assert results[1].layer_name == "barrel_1"
    assert results[2].layer_name == "barrel_2"
    assert total_chi2(results) >= 0.0


def test_filter_cylindrical_track_rejects_length_mismatch():
    initial_state = make_test_state()
    detector = make_barrel_detector(
        radii=[10.0, 20.0],
        half_length=100.0,
    )
    field = UniformMagneticField(bz=2.0)

    measurement = Measurement(
        values=np.array([0.0, 10.0]),
        covariance=np.diag([0.01**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
    )

    with pytest.raises(ValueError):
        filter_cylindrical_track(
            initial_state=initial_state,
            measurements=[measurement],
            detector=detector,
            field=field,
        )
