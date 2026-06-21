from pathlib import Path

import numpy as np
import pytest

from openreco.external.acts_adapter import (
    ActsAdapterConfig,
    convert_acts_event_to_openreco,
    convert_acts_measurement_to_openreco,
    infer_cylindrical_layers,
)
from openreco.external.acts_loader import load_acts_dataset


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "acts_small"


def test_infer_cylindrical_layers_from_measurements():
    dataset = load_acts_dataset(DATASET_DIR)
    event = dataset.events[0]

    detector, layer_name_by_id = infer_cylindrical_layers(event.measurements)

    assert len(detector.layers) == 6
    assert sorted(layer_name_by_id) == [1, 2, 3, 4, 5, 6]

    radii = [layer.radius for layer in detector.layers]
    assert radii == sorted(radii)
    assert radii[0] == pytest.approx(10.0)
    assert radii[-1] == pytest.approx(60.0)


def test_convert_single_measurement_to_openreco_measurement():
    dataset = load_acts_dataset(DATASET_DIR)
    event = dataset.events[0]

    detector, layer_name_by_id = infer_cylindrical_layers(event.measurements)
    source = event.measurements[0]

    converted = convert_acts_measurement_to_openreco(source, layer_name_by_id)

    assert converted.layer_name == "acts_layer_1"
    assert converted.surface_type == "cylinder"
    assert converted.values.shape == (2,)
    assert converted.values[0] == pytest.approx(source.phi)
    assert converted.values[1] == pytest.approx(source.z)

    assert converted.covariance.shape == (2, 2)
    assert converted.covariance[0, 0] == pytest.approx(source.sigma_phi**2)
    assert converted.covariance[1, 1] == pytest.approx(source.sigma_z**2)
    assert converted.covariance[0, 1] == pytest.approx(0.0)
    assert converted.covariance[1, 0] == pytest.approx(0.0)


def test_convert_acts_event_to_openreco_event():
    dataset = load_acts_dataset(DATASET_DIR)
    event = dataset.events[0]

    adapted = convert_acts_event_to_openreco(event)

    assert adapted.event_id == 0
    assert len(adapted.truth_particles) == 2
    assert len(adapted.measurements) == 13
    assert len(adapted.detector.layers) == 6

    assert adapted.measurement_truth_ids[1] == 1
    assert adapted.measurement_truth_ids[7] == 2
    assert adapted.measurement_truth_ids[13] is None

    assert adapted.measurement_is_noise[13] is True


def test_adapter_preserves_measurement_order():
    dataset = load_acts_dataset(DATASET_DIR)
    event = dataset.events[0]

    adapted = convert_acts_event_to_openreco(event)

    source_ids = [m.measurement_id for m in adapted.source_measurements]
    expected_ids = [m.measurement_id for m in event.measurements]

    assert source_ids == expected_ids


def test_adapter_length_scale_changes_z_and_radius():
    dataset = load_acts_dataset(DATASET_DIR)
    event = dataset.events[0]

    config = ActsAdapterConfig(length_scale=10.0)
    adapted = convert_acts_event_to_openreco(event, config=config)

    assert adapted.detector.layers[0].radius == pytest.approx(100.0)
    assert adapted.measurements[0].values[1] == pytest.approx(10.0)
    assert adapted.measurements[0].covariance[1, 1] == pytest.approx(1.0)


def test_empty_event_raises():
    from openreco.external.acts_schema import ActsEvent

    event = ActsEvent(event_id=999, truth_particles=[], measurements=[])

    with pytest.raises(ValueError, match="no measurements"):
        convert_acts_event_to_openreco(event)


def test_unknown_layer_id_raises():
    dataset = load_acts_dataset(DATASET_DIR)
    event = dataset.events[0]
    source = event.measurements[0]

    with pytest.raises(KeyError, match="No OpenReco layer name"):
        convert_acts_measurement_to_openreco(source, layer_name_by_id={})
