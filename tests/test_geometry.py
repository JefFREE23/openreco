import numpy as np
import pytest

from openreco.geometry import DetectorLayer, SimpleDetector, make_uniform_detector


def test_detector_layer_creation():
    layer = DetectorLayer(name="layer_0", z=10.0)

    assert layer.name == "layer_0"
    assert layer.z == 10.0


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
