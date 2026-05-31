import numpy as np
import pytest

from openreco.geometry import (
    BarrelDetector,
    CylindricalLayer,
    DetectorLayer,
    SimpleDetector,
    make_barrel_detector,
    make_uniform_detector,
)


def test_detector_layer_creation():
    layer = DetectorLayer(name="layer_0", z=10.0)

    assert layer.name == "layer_0"
    assert layer.z == 10.0
    assert layer.surface_type == "plane"


def test_detector_layer_converts_z_to_float():
    layer = DetectorLayer(name="layer_0", z=10)

    assert isinstance(layer.z, float)
    assert layer.z == 10.0


def test_detector_layer_rejects_non_string_name():
    with pytest.raises(TypeError):
        DetectorLayer(name=123, z=10.0)


def test_detector_sorts_layers_by_z():
    layers = [
        DetectorLayer(name="layer_2", z=20.0),
        DetectorLayer(name="layer_0", z=0.0),
        DetectorLayer(name="layer_1", z=10.0),
    ]

    detector = SimpleDetector(layers)

    np.testing.assert_allclose(detector.z_positions, np.array([0.0, 10.0, 20.0]))


def test_detector_rejects_empty_layers():
    with pytest.raises(ValueError):
        SimpleDetector([])


def test_detector_rejects_duplicate_names():
    layers = [
        DetectorLayer(name="layer", z=0.0),
        DetectorLayer(name="layer", z=10.0),
    ]

    with pytest.raises(ValueError):
        SimpleDetector(layers)


def test_detector_rejects_duplicate_z_positions():
    layers = [
        DetectorLayer(name="layer_0", z=0.0),
        DetectorLayer(name="layer_1", z=0.0),
    ]

    with pytest.raises(ValueError):
        SimpleDetector(layers)


def test_make_uniform_detector():
    detector = make_uniform_detector(n_layers=5, z_min=0.0, z_max=40.0)

    assert len(detector) == 5
    np.testing.assert_allclose(
        detector.z_positions,
        np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
    )


def test_make_uniform_detector_rejects_less_than_two_layers():
    with pytest.raises(ValueError):
        make_uniform_detector(n_layers=1, z_min=0.0, z_max=10.0)


def test_detector_getitem():
    detector = make_uniform_detector(n_layers=3, z_min=0.0, z_max=20.0)

    assert detector[0].name == "layer_0"
    assert detector[0].z == 0.0
    assert detector[2].name == "layer_2"
    assert detector[2].z == 20.0


def test_cylindrical_layer_creation():
    layer = CylindricalLayer(name="barrel_0", radius=20.0, half_length=100.0)

    assert layer.name == "barrel_0"
    assert layer.radius == 20.0
    assert layer.half_length == 100.0
    assert layer.surface_type == "cylinder"


def test_cylindrical_layer_converts_values_to_float():
    layer = CylindricalLayer(name="barrel_0", radius=20, half_length=100)

    assert isinstance(layer.radius, float)
    assert isinstance(layer.half_length, float)
    assert layer.radius == 20.0
    assert layer.half_length == 100.0


def test_cylindrical_layer_rejects_non_string_name():
    with pytest.raises(TypeError):
        CylindricalLayer(name=123, radius=20.0, half_length=100.0)


def test_cylindrical_layer_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        CylindricalLayer(name="bad", radius=0.0, half_length=100.0)

    with pytest.raises(ValueError):
        CylindricalLayer(name="bad", radius=-10.0, half_length=100.0)


def test_cylindrical_layer_rejects_non_positive_half_length():
    with pytest.raises(ValueError):
        CylindricalLayer(name="bad", radius=20.0, half_length=0.0)

    with pytest.raises(ValueError):
        CylindricalLayer(name="bad", radius=20.0, half_length=-100.0)


def test_cylindrical_layer_contains_z():
    layer = CylindricalLayer(name="barrel_0", radius=20.0, half_length=100.0)

    assert layer.contains_z(0.0)
    assert layer.contains_z(100.0)
    assert layer.contains_z(-100.0)
    assert not layer.contains_z(101.0)
    assert not layer.contains_z(-101.0)


def test_barrel_detector_sorts_layers_by_radius():
    layers = [
        CylindricalLayer(name="barrel_2", radius=60.0, half_length=100.0),
        CylindricalLayer(name="barrel_0", radius=20.0, half_length=100.0),
        CylindricalLayer(name="barrel_1", radius=40.0, half_length=100.0),
    ]

    detector = BarrelDetector(layers)

    np.testing.assert_allclose(detector.radii, np.array([20.0, 40.0, 60.0]))


def test_barrel_detector_rejects_empty_layers():
    with pytest.raises(ValueError):
        BarrelDetector([])


def test_barrel_detector_rejects_duplicate_names():
    layers = [
        CylindricalLayer(name="barrel", radius=20.0, half_length=100.0),
        CylindricalLayer(name="barrel", radius=40.0, half_length=100.0),
    ]

    with pytest.raises(ValueError):
        BarrelDetector(layers)


def test_barrel_detector_rejects_duplicate_radii():
    layers = [
        CylindricalLayer(name="barrel_0", radius=20.0, half_length=100.0),
        CylindricalLayer(name="barrel_1", radius=20.0, half_length=100.0),
    ]

    with pytest.raises(ValueError):
        BarrelDetector(layers)


def test_make_barrel_detector():
    detector = make_barrel_detector(
        radii=[20.0, 40.0, 60.0],
        half_length=100.0,
    )

    assert len(detector) == 3
    np.testing.assert_allclose(detector.radii, np.array([20.0, 40.0, 60.0]))

    assert detector[0].name == "barrel_0"
    assert detector[0].radius == 20.0
    assert detector[0].half_length == 100.0


def test_make_barrel_detector_sorts_radii():
    detector = make_barrel_detector(
        radii=[60.0, 20.0, 40.0],
        half_length=100.0,
    )

    np.testing.assert_allclose(detector.radii, np.array([20.0, 40.0, 60.0]))


def test_make_barrel_detector_rejects_empty_radii():
    with pytest.raises(ValueError):
        make_barrel_detector(radii=[], half_length=100.0)


def test_barrel_detector_getitem():
    detector = make_barrel_detector(
        radii=[20.0, 40.0, 60.0],
        half_length=100.0,
    )

    assert detector[0].name == "barrel_0"
    assert detector[0].radius == 20.0
    assert detector[2].name == "barrel_2"
    assert detector[2].radius == 60.0
