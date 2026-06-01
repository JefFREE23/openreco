import numpy as np
import pytest

from openreco.field import UniformMagneticField
from openreco.geometry import CylindricalLayer, make_barrel_detector
from openreco.kalman import (
    KalmanPredictionResult,
    KalmanUpdateResult,
    cylindrical_full_residual,
    cylindrical_measurement_matrix,
    cylindrical_measurement_prediction,
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
from openreco.state import make_cylindrical_state, make_planar_state


def make_test_state():
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


def make_state_on_layer():
    return make_cylindrical_state(
        phi=0.0,
        z=5.0,
        dir0=0.0,
        dir1=1.0,
        q_over_p=0.5,
        covariance=np.eye(5),
        surface_radius=10.0,
        surface_name="barrel_0",
    )


def make_measurement():
    return Measurement(
        values=np.array([0.01, 5.2]),
        covariance=np.diag([0.001**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
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
    state = make_state_on_layer()

    particle = state_to_particle(state)

    np.testing.assert_allclose(particle.position, np.array([10.0, 0.0, 5.0]))
    assert particle.charge == 1.0
    assert particle.p == pytest.approx(2.0)


def test_state_to_particle_negative_charge():
    state = make_cylindrical_state(
        phi=0.0,
        z=5.0,
        dir0=0.0,
        dir1=1.0,
        q_over_p=-0.5,
        covariance=np.eye(5),
        surface_radius=10.0,
    )

    particle = state_to_particle(state)

    assert particle.charge == -1.0
    assert particle.p == pytest.approx(2.0)


def test_state_to_particle_rejects_plane():
    state = make_planar_state(
        x=1.0,
        y=2.0,
        tx=0.1,
        ty=0.2,
        q_over_p=0.5,
        covariance=np.eye(5),
    )

    with pytest.raises(ValueError):
        state_to_particle(state)


def test_state_to_particle_rejects_zero_q_over_p():
    state = make_cylindrical_state(
        phi=0.0,
        z=5.0,
        dir0=0.0,
        dir1=1.0,
        q_over_p=0.0,
        covariance=np.eye(5),
        surface_radius=10.0,
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
    assert prediction.predicted_state.surface_type == "cylinder"
    assert prediction.predicted_state.surface_name == "barrel_0"
    assert prediction.predicted_state.surface_radius == 10.0
    assert prediction.predicted_state.z == pytest.approx(10.0, abs=1e-5)
    assert prediction.transport_jacobian.shape == (5, 5)


def test_predict_to_cylindrical_layer_rejects_plane_state():
    state = make_planar_state(
        x=0.0,
        y=0.0,
        tx=1.0,
        ty=0.0,
        q_over_p=0.5,
        covariance=np.eye(5),
    )
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    with pytest.raises(ValueError):
        predict_to_cylindrical_layer(state, field, layer)


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


def test_cylindrical_measurement_prediction():
    state = make_state_on_layer()

    prediction = cylindrical_measurement_prediction(state)

    np.testing.assert_allclose(prediction, np.array([0.0, 5.0]))


def test_cylindrical_measurement_prediction_rejects_plane():
    state = make_planar_state(
        x=1.0,
        y=2.0,
        tx=0.1,
        ty=0.2,
        q_over_p=0.5,
        covariance=np.eye(5),
    )

    with pytest.raises(ValueError):
        cylindrical_measurement_prediction(state)


def test_cylindrical_measurement_matrix():
    matrix = cylindrical_measurement_matrix()

    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(matrix, expected)


def test_cylindrical_full_residual():
    state = make_state_on_layer()
    measurement = make_measurement()

    residual = cylindrical_full_residual(state, measurement)

    np.testing.assert_allclose(residual, np.array([0.01, 0.2]))


def test_update_state():
    state = make_state_on_layer()

    result = update_state(
        predicted_state=state,
        measurement_values=np.array([0.01, 5.2]),
        measurement_covariance=np.diag([0.001**2, 0.1**2]),
        measurement_matrix=cylindrical_measurement_matrix(),
        predicted_measurement=cylindrical_measurement_prediction(state),
        layer_name="barrel_0",
        wrap_first_residual=True,
    )

    assert isinstance(result, KalmanUpdateResult)
    assert result.residual.shape == (2,)
    assert result.residual_covariance.shape == (2, 2)
    assert result.kalman_gain.shape == (5, 2)
    assert result.layer_name == "barrel_0"
    assert result.chi2 >= 0.0


def test_update_with_cylindrical_measurement_reduces_local_residual():
    state = make_state_on_layer()
    measurement = make_measurement()

    residual_before = np.linalg.norm(cylindrical_full_residual(state, measurement))

    result = update_with_cylindrical_measurement(
        predicted_state=state,
        measurement=measurement,
    )

    residual_after = np.linalg.norm(
        cylindrical_full_residual(result.filtered_state, measurement)
    )

    assert residual_after < residual_before


def test_update_with_cylindrical_measurement_wraps_phi():
    state = make_state_on_layer()

    measurement = Measurement(
        values=np.array([2.0 * np.pi - 0.01, 5.0]),
        covariance=np.diag([0.001**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
    )

    result = update_with_cylindrical_measurement(state, measurement)

    assert result.residual[0] == pytest.approx(-0.01)


def test_update_with_cylindrical_measurement_rejects_wrong_surface_type():
    state = make_state_on_layer()

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
