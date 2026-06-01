import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from openreco.geometry import CylindricalLayer, make_barrel_detector
from openreco.visualization import (
    _validate_positions,
    draw_barrel_detector,
    draw_beamline_3d,
    draw_cylindrical_layer,
    draw_origin_3d,
    plot_cylindrical_track_event,
    plot_cylindrical_track_xy,
)


def test_validate_positions_accepts_valid_array():
    positions = np.zeros((3, 3))

    validated = _validate_positions("positions", positions)

    np.testing.assert_allclose(validated, positions)


def test_validate_positions_rejects_wrong_shape():
    with pytest.raises(ValueError):
        _validate_positions("positions", np.zeros(3))

    with pytest.raises(ValueError):
        _validate_positions("positions", np.zeros((3, 2)))


def test_draw_cylindrical_layer():
    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=50.0,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_cylindrical_layer(ax, layer)

    assert len(ax.collections) > 0

    plt.close(fig)


def test_draw_cylindrical_layer_rejects_bad_grid_settings():
    layer = CylindricalLayer(
        name="barrel_0",
        radius=10.0,
        half_length=50.0,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    with pytest.raises(ValueError):
        draw_cylindrical_layer(ax, layer, n_phi=3)

    with pytest.raises(ValueError):
        draw_cylindrical_layer(ax, layer, n_z=1)

    plt.close(fig)


def test_draw_barrel_detector():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_barrel_detector(ax, detector)

    assert len(ax.collections) >= 3

    plt.close(fig)


def test_plot_cylindrical_track_event():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.array(
        [
            [10.0, 0.0, 1.0],
            [20.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
        ]
    )

    measured_positions = truth_positions.copy()
    predicted_positions = truth_positions.copy()
    filtered_positions = truth_positions.copy()

    fig, ax = plot_cylindrical_track_event(
        detector=detector,
        truth_positions=truth_positions,
        measured_positions=measured_positions,
        predicted_positions=predicted_positions,
        filtered_positions=filtered_positions,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "OpenReco cylindrical track event"

    plt.close(fig)


def test_plot_cylindrical_track_event_without_optional_positions():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.array(
        [
            [10.0, 0.0, 1.0],
            [20.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
        ]
    )

    measured_positions = truth_positions.copy()

    fig, ax = plot_cylindrical_track_event(
        detector=detector,
        truth_positions=truth_positions,
        measured_positions=measured_positions,
        predicted_positions=None,
        filtered_positions=None,
        show_detector=False,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_plot_cylindrical_track_event_rejects_bad_positions():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.zeros((3, 3))
    measured_positions = np.zeros((3, 2))

    with pytest.raises(ValueError):
        plot_cylindrical_track_event(
            detector=detector,
            truth_positions=truth_positions,
            measured_positions=measured_positions,
        )
def test_plot_cylindrical_track_event_with_smoothed_positions():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.array(
        [
            [10.0, 0.0, 1.0],
            [20.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
        ]
    )

    measured_positions = truth_positions.copy()
    predicted_positions = truth_positions.copy()
    filtered_positions = truth_positions.copy()
    smoothed_positions = truth_positions.copy()

    fig, ax = plot_cylindrical_track_event(
        detector=detector,
        truth_positions=truth_positions,
        measured_positions=measured_positions,
        predicted_positions=predicted_positions,
        filtered_positions=filtered_positions,
        smoothed_positions=smoothed_positions,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)

def test_draw_beamline_3d_and_origin():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_beamline_3d(ax, z_min=0.0, z_max=10.0)
    draw_origin_3d(ax)

    assert len(ax.lines) >= 1
    assert len(ax.collections) >= 1

    plt.close(fig)


def test_plot_cylindrical_track_event_with_detector_z_range():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.array(
        [
            [10.0, 0.0, 1.0],
            [20.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
        ]
    )

    measured_positions = truth_positions.copy()

    fig, ax = plot_cylindrical_track_event(
        detector=detector,
        truth_positions=truth_positions,
        measured_positions=measured_positions,
        detector_z_range=(0.0, 10.0),
        show_beamline=True,
        show_origin=True,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_zlim()[0] == 0.0
    assert ax.get_zlim()[1] == 10.0

    plt.close(fig)


def test_plot_cylindrical_track_event_rejects_bad_detector_z_range():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.zeros((3, 3))
    measured_positions = np.zeros((3, 3))

    with pytest.raises(ValueError):
        plot_cylindrical_track_event(
            detector=detector,
            truth_positions=truth_positions,
            measured_positions=measured_positions,
            detector_z_range=(10.0, 0.0),
        )


def test_plot_cylindrical_track_xy():
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=50.0,
    )

    truth_positions = np.array(
        [
            [10.0, 0.0, 1.0],
            [20.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
        ]
    )

    measured_positions = truth_positions.copy()
    predicted_positions = truth_positions.copy()
    filtered_positions = truth_positions.copy()
    smoothed_positions = truth_positions.copy()

    fig, ax = plot_cylindrical_track_xy(
        detector=detector,
        truth_positions=truth_positions,
        measured_positions=measured_positions,
        predicted_positions=predicted_positions,
        filtered_positions=filtered_positions,
        smoothed_positions=smoothed_positions,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"

    plt.close(fig)
