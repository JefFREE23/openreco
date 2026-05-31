import numpy as np
import pytest

from openreco.state import TrackState


def test_track_state_valid_creation():
    params = np.array([1.0, 2.0, 0.1, -0.2, 0.001])
    cov = np.eye(5)
    z = 10.0

    state = TrackState(parameters=params, covariance=cov, z=z)

    assert state.x == 1.0
    assert state.y == 2.0
    assert state.tx == 0.1
    assert state.ty == -0.2
    assert state.q_over_p == 0.001
    assert state.z == 10.0


def test_track_state_parameter_shape_check():
    params = np.array([1.0, 2.0, 0.1])
    cov = np.eye(5)

    with pytest.raises(ValueError):
        TrackState(parameters=params, covariance=cov, z=0.0)


def test_track_state_covariance_shape_check():
    params = np.zeros(5)
    cov = np.eye(4)

    with pytest.raises(ValueError):
        TrackState(parameters=params, covariance=cov, z=0.0)


def test_track_state_covariance_symmetry_check():
    params = np.zeros(5)
    cov = np.eye(5)
    cov[0, 1] = 0.5

    with pytest.raises(ValueError):
        TrackState(parameters=params, covariance=cov, z=0.0)


def test_track_state_copy_is_deep_copy():
    params = np.array([1.0, 2.0, 0.1, -0.2, 0.001])
    cov = np.eye(5)

    state = TrackState(parameters=params, covariance=cov, z=10.0)
    copied = state.copy()

    copied.parameters[0] = 99.0
    copied.covariance[0, 0] = 99.0

    assert state.parameters[0] == 1.0
    assert state.covariance[0, 0] == 1.0


def test_position_and_slopes():
    params = np.array([1.0, 2.0, 0.1, -0.2, 0.001])
    cov = np.eye(5)

    state = TrackState(parameters=params, covariance=cov, z=10.0)

    np.testing.assert_allclose(state.position(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(state.slopes(), np.array([0.1, -0.2]))
