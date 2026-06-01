import numpy as np
import pytest

from openreco.diagnostics import (
    MomentumSummary,
    VectorSummary,
    chi2_values,
    covariance_diagonal,
    covariance_eigenvalues,
    covariance_has_nonnegative_diagonal,
    covariance_is_positive_semidefinite,
    covariance_is_symmetric,
    covariance_is_valid,
    covariance_standard_deviations,
    cylindrical_diagnostic_residuals,
    format_vector,
    kalman_pull_summary,
    kalman_pulls,
    kalman_residual_summary,
    kalman_residual_variances,
    kalman_residuals,
    measurement_covariance_pulls,
    momentum_from_state,
    momentum_summary,
    reduced_chi2,
    summarize_vectors,
    total_chi2,
)
from openreco.kalman import update_with_cylindrical_measurement
from openreco.measurements import Measurement
from openreco.particle_gun import Particle
from openreco.state import TrackState


def make_test_state():
    return TrackState(
        parameters=np.array([10.0, 0.0, 0.1, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )


def make_test_measurement():
    return Measurement(
        values=np.array([0.01, 5.2]),
        covariance=np.diag([0.001**2, 0.1**2]),
        layer_name="barrel_0",
        surface_type="cylinder",
    )


def make_test_result():
    state = make_test_state()
    measurement = make_test_measurement()

    return update_with_cylindrical_measurement(state, measurement)


def test_summarize_vectors():
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    summary = summarize_vectors(values)

    assert isinstance(summary, VectorSummary)
    np.testing.assert_allclose(summary.mean, np.array([2.0, 3.0]))
    np.testing.assert_allclose(summary.std, np.array([1.0, 1.0]))
    np.testing.assert_allclose(summary.rmse, np.sqrt(np.array([5.0, 10.0])))


def test_summarize_vectors_rejects_wrong_shape():
    with pytest.raises(ValueError):
        summarize_vectors(np.array([1.0, 2.0]))


def test_kalman_residuals():
    result = make_test_result()

    residuals = kalman_residuals([result])

    assert residuals.shape == (1, 1)
    np.testing.assert_allclose(residuals[0], result.residual)


def test_kalman_residuals_rejects_empty_results():
    with pytest.raises(ValueError):
        kalman_residuals([])


def test_kalman_residual_variances():
    result = make_test_result()

    variances = kalman_residual_variances([result])

    assert variances.shape == (1, 1)
    np.testing.assert_allclose(
        variances[0],
        np.diag(result.residual_covariance),
    )


def test_kalman_pulls_use_residual_covariance():
    result = make_test_result()

    pulls = kalman_pulls([result])

    expected = result.residual / np.sqrt(np.diag(result.residual_covariance))

    np.testing.assert_allclose(pulls[0], expected)


def test_kalman_residual_summary():
    result = make_test_result()

    summary = kalman_residual_summary([result])

    assert isinstance(summary, VectorSummary)
    assert summary.mean.shape == (1,)


def test_kalman_pull_summary():
    result = make_test_result()

    summary = kalman_pull_summary([result])

    assert isinstance(summary, VectorSummary)
    assert summary.mean.shape == (1,)


def test_chi2_values_and_total_chi2():
    result = make_test_result()

    values = chi2_values([result])

    assert values.shape == (1,)
    assert values[0] == pytest.approx(result.chi2)
    assert total_chi2([result]) == pytest.approx(result.chi2)


def test_chi2_values_rejects_empty_results():
    with pytest.raises(ValueError):
        chi2_values([])


def test_reduced_chi2():
    result = make_test_result()

    assert reduced_chi2([result], n_degrees_of_freedom=1) == pytest.approx(result.chi2)


def test_reduced_chi2_rejects_non_positive_ndof():
    result = make_test_result()

    with pytest.raises(ValueError):
        reduced_chi2([result], n_degrees_of_freedom=0)


def test_momentum_from_state():
    state = make_test_state()

    assert momentum_from_state(state) == pytest.approx(2.0)


def test_momentum_from_state_rejects_zero_q_over_p():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.1, 0.0, 0.0]),
        covariance=np.eye(5),
        z=5.0,
    )

    with pytest.raises(ValueError):
        momentum_from_state(state)


def test_momentum_summary():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([1.0, 0.0, 0.0]),
        charge=1.0,
    )

    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.1, 0.0, 0.5]),
        covariance=np.eye(5),
        z=5.0,
    )

    summary = momentum_summary(particle, state)

    assert isinstance(summary, MomentumSummary)
    assert summary.truth_p == pytest.approx(1.0)
    assert summary.fitted_p == pytest.approx(2.0)
    assert summary.absolute_error == pytest.approx(1.0)
    assert summary.relative_error == pytest.approx(1.0)


def test_covariance_diagonal():
    state = make_test_state()

    np.testing.assert_allclose(covariance_diagonal(state), np.ones(5))


def test_covariance_standard_deviations():
    state = TrackState(
        parameters=np.array([10.0, 0.0, 0.1, 0.0, 0.5]),
        covariance=np.diag([1.0, 4.0, 9.0, 16.0, 25.0]),
        z=5.0,
    )

    np.testing.assert_allclose(
        covariance_standard_deviations(state),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


def test_covariance_is_symmetric():
    state = make_test_state()

    assert covariance_is_symmetric(state)


def test_covariance_has_nonnegative_diagonal():
    state = make_test_state()

    assert covariance_has_nonnegative_diagonal(state)


def test_covariance_eigenvalues():
    state = make_test_state()

    np.testing.assert_allclose(covariance_eigenvalues(state), np.ones(5))


def test_covariance_is_positive_semidefinite():
    state = make_test_state()

    assert covariance_is_positive_semidefinite(state)


def test_covariance_is_positive_semidefinite_rejects_bad_tolerance():
    state = make_test_state()

    with pytest.raises(ValueError):
        covariance_is_positive_semidefinite(state, tolerance=-1.0)


def test_covariance_is_valid():
    state = make_test_state()

    assert covariance_is_valid(state)


def test_covariance_is_valid_rejects_non_psd_covariance():
    covariance = np.eye(5)
    covariance[0, 0] = -1.0

    with pytest.raises(ValueError):
        TrackState(
            parameters=np.array([10.0, 0.0, 0.1, 0.0, 0.5]),
            covariance=covariance,
            z=5.0,
        )


def test_cylindrical_diagnostic_residuals():
    result = make_test_result()
    measurement = make_test_measurement()

    residuals = cylindrical_diagnostic_residuals(
        results=[result],
        measurements=[measurement],
    )

    assert residuals.shape == (1, 2)


def test_cylindrical_diagnostic_residuals_rejects_empty_results():
    measurement = make_test_measurement()

    with pytest.raises(ValueError):
        cylindrical_diagnostic_residuals(
            results=[],
            measurements=[measurement],
        )


def test_cylindrical_diagnostic_residuals_rejects_length_mismatch():
    result = make_test_result()
    measurement = make_test_measurement()

    with pytest.raises(ValueError):
        cylindrical_diagnostic_residuals(
            results=[result],
            measurements=[measurement, measurement],
        )


def test_measurement_covariance_pulls():
    residuals = np.array([[0.001, 0.1]])
    measurement = make_test_measurement()

    pulls = measurement_covariance_pulls(
        residuals=residuals,
        measurements=[measurement],
    )

    np.testing.assert_allclose(pulls, np.array([[1.0, 1.0]]))


def test_measurement_covariance_pulls_rejects_wrong_shape():
    measurement = make_test_measurement()

    with pytest.raises(ValueError):
        measurement_covariance_pulls(
            residuals=np.array([0.001, 0.1]),
            measurements=[measurement],
        )


def test_measurement_covariance_pulls_rejects_length_mismatch():
    measurement = make_test_measurement()

    with pytest.raises(ValueError):
        measurement_covariance_pulls(
            residuals=np.array([[0.001, 0.1], [0.002, 0.2]]),
            measurements=[measurement],
        )


def test_format_vector():
    text = format_vector(np.array([1.23456, 2.34567]), precision=2)

    assert text == "[1.23, 2.35]"
