import numpy as np
import pytest

from openreco.state import TrackState, make_cylindrical_state, make_planar_state


def test_cylindrical_state_creation():
    cov = np.eye(5)

    state = make_cylindrical_state(
        phi=0.5,
        z=10.0,
        dir0=0.1,
        dir1=0.2,
        q_over_p=0.3,
        covariance=cov,
        surface_radius=20.0,
        surface_name="barrel_0",
    )

    assert state.surface_type == "cylinder"
    assert state.surface_name == "barrel_0"
    assert state.surface_radius == 20.0
    assert state.phi == 0.5
    assert state.z == 10.0
    assert state.dir0 == 0.1
    assert state.dir1 == 0.2
    assert state.q_over_p == 0.3


def test_cylindrical_state_global_position():
    cov = np.eye(5)

    state = make_cylindrical_state(
        phi=np.pi / 2.0,
        z=10.0,
        dir0=0.1,
        dir1=0.2,
        q_over_p=0.3,
        covariance=cov,
        surface_radius=20.0,
    )

    np.testing.assert_allclose(
        state.global_position(),
        np.array([0.0, 20.0, 10.0]),
        atol=1e-12,
    )


def test_planar_state_creation():
    cov = np.eye(5)

    state = make_planar_state(
        x=1.0,
        y=2.0,
        tx=0.1,
        ty=-0.2,
        q_over_p=0.3,
        covariance=cov,
        surface_name="plane_0",
    )

    assert state.surface_type == "plane"
    assert state.surface_name == "plane_0"
    assert state.x == 1.0
    assert state.y == 2.0
    assert state.loc0 == 1.0
    assert state.loc1 == 2.0
    assert state.dir0 == 0.1
    assert state.dir1 == -0.2
    assert state.q_over_p == 0.3


def test_track_state_parameter_shape_check():
    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(4),
            covariance=np.eye(5),
            surface_type="cylinder",
            surface_radius=10.0,
        )


def test_track_state_covariance_shape_check():
    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(5),
            covariance=np.eye(4),
            surface_type="cylinder",
            surface_radius=10.0,
        )


def test_track_state_covariance_symmetry_check():
    cov = np.eye(5)
    cov[0, 1] = 0.5

    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(5),
            covariance=cov,
            surface_type="cylinder",
            surface_radius=10.0,
        )


def test_track_state_rejects_invalid_surface_type():
    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(5),
            covariance=np.eye(5),
            surface_type="sphere",
        )


def test_cylindrical_state_requires_radius():
    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(5),
            covariance=np.eye(5),
            surface_type="cylinder",
        )


def test_cylindrical_state_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(5),
            covariance=np.eye(5),
            surface_type="cylinder",
            surface_radius=0.0,
        )


def test_planar_state_rejects_radius():
    with pytest.raises(ValueError):
        TrackState(
            parameters=np.zeros(5),
            covariance=np.eye(5),
            surface_type="plane",
            surface_radius=10.0,
        )


def test_phi_only_for_cylinder():
    state = make_planar_state(
        x=1.0,
        y=2.0,
        tx=0.1,
        ty=0.2,
        q_over_p=0.3,
        covariance=np.eye(5),
    )

    with pytest.raises(AttributeError):
        _ = state.phi


def test_z_only_for_cylinder():
    state = make_planar_state(
        x=1.0,
        y=2.0,
        tx=0.1,
        ty=0.2,
        q_over_p=0.3,
        covariance=np.eye(5),
    )

    with pytest.raises(AttributeError):
        _ = state.z


def test_local_position_and_direction_parameters():
    state = make_cylindrical_state(
        phi=0.5,
        z=10.0,
        dir0=0.1,
        dir1=0.2,
        q_over_p=0.3,
        covariance=np.eye(5),
        surface_radius=20.0,
    )

    np.testing.assert_allclose(state.local_position(), np.array([0.5, 10.0]))
    np.testing.assert_allclose(state.direction_parameters(), np.array([0.1, 0.2]))


def test_track_state_copy_is_deep_copy():
    state = make_cylindrical_state(
        phi=0.5,
        z=10.0,
        dir0=0.1,
        dir1=0.2,
        q_over_p=0.3,
        covariance=np.eye(5),
        surface_radius=20.0,
        surface_name="barrel_0",
    )

    copied = state.copy()

    copied.parameters[0] = 99.0
    copied.covariance[0, 0] = 99.0

    assert state.parameters[0] == 0.5
    assert state.covariance[0, 0] == 1.0
    assert copied.surface_type == "cylinder"
    assert copied.surface_name == "barrel_0"
    assert copied.surface_radius == 20.0


def test_as_vector_returns_copy():
    state = make_cylindrical_state(
        phi=0.5,
        z=10.0,
        dir0=0.1,
        dir1=0.2,
        q_over_p=0.3,
        covariance=np.eye(5),
        surface_radius=20.0,
    )

    vector = state.as_vector()
    vector[0] = 99.0

    assert state.parameters[0] == 0.5
