import numpy as np
import pytest

from openreco.geometry import CylindricalLayer, DetectorLayer
from openreco.measurements import (
    Measurement,
    make_cylindrical_measurement,
    make_planar_measurement,
    smear_position,
)


def test_measurement_creation():
    measurement = Measurement(
        values=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        layer_name="layer_0",
        surface_type="plane",
    )

    assert measurement.dimension == 2
    np.testing.assert_allclose(measurement.values, np.array([1.0, 2.0]))
    np.testing.assert_allclose(measurement.covariance, np.eye(2))
    assert measurement.layer_name == "layer_0"
    assert measurement.surface_type == "plane"


def test_measurement_rejects_non_1d_values():
    with pytest.raises(ValueError):
        Measurement(
            values=np.zeros((2, 2)),
            covariance=np.eye(2),
            layer_name="layer_0",
            surface_type="plane",
        )


def test_measurement_rejects_wrong_covariance_shape():
    with pytest.raises(ValueError):
        Measurement(
            values=np.array([1.0, 2.0]),
            covariance=np.eye(3),
            layer_name="layer_0",
            surface_type="plane",
        )


def test_measurement_rejects_nonsymmetric_covariance():
    covariance = np.eye(2)
    covariance[0, 1] = 0.5

    with pytest.raises(ValueError):
        Measurement(
            values=np.array([1.0, 2.0]),
            covariance=covariance,
            layer_name="layer_0",
            surface_type="plane",
        )


def test_measurement_rejects_non_string_layer_name():
    with pytest.raises(TypeError):
        Measurement(
            values=np.array([1.0, 2.0]),
            covariance=np.eye(2),
            layer_name=123,
            surface_type="plane",
        )


def test_measurement_rejects_non_string_surface_type():
    with pytest.raises(TypeError):
        Measurement(
            values=np.array([1.0, 2.0]),
            covariance=np.eye(2),
            layer_name="layer_0",
            surface_type=123,
        )


def test_measurement_copy_is_deep_copy():
    measurement = Measurement(
        values=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        layer_name="layer_0",
        surface_type="plane",
    )

    copied = measurement.copy()

    copied.values[0] = 99.0
    copied.covariance[0, 0] = 99.0

    assert measurement.values[0] == 1.0
    assert measurement.covariance[0, 0] == 1.0


def test_smear_position_reproducible_with_rng():
    rng_1 = np.random.default_rng(123)
    rng_2 = np.random.default_rng(123)

    true_values = np.array([1.0, 2.0])
    covariance = np.diag([0.1**2, 0.2**2])

    smeared_1 = smear_position(true_values, covariance, rng=rng_1)
    smeared_2 = smear_position(true_values, covariance, rng=rng_2)

    np.testing.assert_allclose(smeared_1, smeared_2)


def test_smear_position_rejects_non_1d_true_values():
    with pytest.raises(ValueError):
        smear_position(np.zeros((2, 2)), np.eye(2))


def test_smear_position_rejects_wrong_covariance_shape():
    with pytest.raises(ValueError):
        smear_position(np.array([1.0, 2.0]), np.eye(3))


def test_smear_position_rejects_nonsymmetric_covariance():
    covariance = np.eye(2)
    covariance[0, 1] = 0.5

    with pytest.raises(ValueError):
        smear_position(np.array([1.0, 2.0]), covariance)


def test_make_planar_measurement():
    rng = np.random.default_rng(123)

    layer = DetectorLayer(name="layer_0", z=10.0)
    true_position = np.array([1.0, 2.0, 10.0])

    measurement = make_planar_measurement(
        layer=layer,
        true_position=true_position,
        sigma=0.1,
        rng=rng,
    )

    assert measurement.dimension == 2
    assert measurement.layer_name == "layer_0"
    assert measurement.surface_type == "plane"
    np.testing.assert_allclose(measurement.covariance, np.diag([0.01, 0.01]))


def test_make_planar_measurement_rejects_non_positive_sigma():
    layer = DetectorLayer(name="layer_0", z=10.0)
    true_position = np.array([1.0, 2.0, 10.0])

    with pytest.raises(ValueError):
        make_planar_measurement(layer, true_position, sigma=0.0)

    with pytest.raises(ValueError):
        make_planar_measurement(layer, true_position, sigma=-1.0)


def test_make_planar_measurement_rejects_wrong_true_position_shape():
    layer = DetectorLayer(name="layer_0", z=10.0)

    with pytest.raises(ValueError):
        make_planar_measurement(layer, np.array([1.0, 2.0]), sigma=0.1)


def test_make_cylindrical_measurement():
    rng = np.random.default_rng(123)

    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=100.0,
    )

    true_position = np.array([10.0, 0.0, 5.0])

    measurement = make_cylindrical_measurement(
        layer=layer,
        true_position=true_position,
        sigma_phi=0.001,
        sigma_z=0.1,
        rng=rng,
    )

    assert measurement.dimension == 2
    assert measurement.layer_name == "barrel_0"
    assert measurement.surface_type == "cylinder"
    np.testing.assert_allclose(
        measurement.covariance,
        np.diag([0.001**2, 0.1**2]),
    )


def test_make_cylindrical_measurement_phi_value_without_smearing():
    rng = np.random.default_rng(123)

    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=100.0,
    )

    true_position = np.array([0.0, 10.0, 5.0])

    measurement = make_cylindrical_measurement(
        layer=layer,
        true_position=true_position,
        sigma_phi=1e-12,
        sigma_z=1e-12,
        rng=rng,
    )

    np.testing.assert_allclose(
        measurement.values,
        np.array([np.pi / 2.0, 5.0]),
        atol=1e-9,
    )


def test_make_cylindrical_measurement_rejects_non_positive_sigmas():
    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=100.0,
    )
    true_position = np.array([10.0, 0.0, 5.0])

    with pytest.raises(ValueError):
        make_cylindrical_measurement(layer, true_position, sigma_phi=0.0, sigma_z=0.1)

    with pytest.raises(ValueError):
        make_cylindrical_measurement(layer, true_position, sigma_phi=0.001, sigma_z=0.0)

    with pytest.raises(ValueError):
        make_cylindrical_measurement(layer, true_position, sigma_phi=-1.0, sigma_z=0.1)

    with pytest.raises(ValueError):
        make_cylindrical_measurement(layer, true_position, sigma_phi=0.001, sigma_z=-1.0)


def test_make_cylindrical_measurement_rejects_wrong_true_position_shape():
    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=100.0,
    )

    with pytest.raises(ValueError):
        make_cylindrical_measurement(
            layer,
            np.array([10.0, 0.0]),
            sigma_phi=0.001,
            sigma_z=0.1,
        )


def test_make_cylindrical_measurement_rejects_position_not_on_radius():
    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=100.0,
    )

    true_position = np.array([9.0, 0.0, 5.0])

    with pytest.raises(ValueError):
        make_cylindrical_measurement(
            layer,
            true_position,
            sigma_phi=0.001,
            sigma_z=0.1,
        )


def test_make_cylindrical_measurement_rejects_position_outside_z_length():
    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=100.0,
    )

    true_position = np.array([10.0, 0.0, 101.0])

    with pytest.raises(ValueError):
        make_cylindrical_measurement(
            layer,
            true_position,
            sigma_phi=0.001,
            sigma_z=0.1,
        )
