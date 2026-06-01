"""
Visualization tools for OpenReco.

This module provides plotting helpers for the OpenReco v0 tracking chain.

Main v0 visualization:
    - cylindrical barrel detector wireframes
    - beamline and origin marker
    - truth trajectory
    - smeared measurements
    - predicted Kalman states
    - filtered Kalman states
    - smoothed Kalman states
    - optional x-y top view
"""

import numpy as np
import matplotlib.pyplot as plt

from openreco.geometry import BarrelDetector, CylindricalLayer


def draw_cylindrical_layer(
    ax,
    layer: CylindricalLayer,
    n_phi: int = 80,
    n_z: int = 5,
    alpha: float = 0.10,
    z_min: float | None = None,
    z_max: float | None = None,
):
    """
    Draw a cylindrical detector layer as a wireframe.
    """

    if n_phi < 4:
        raise ValueError("n_phi must be at least 4")

    if n_z < 2:
        raise ValueError("n_z must be at least 2")

    if z_min is None:
        z_min = -layer.half_length

    if z_max is None:
        z_max = layer.half_length

    z_min = float(z_min)
    z_max = float(z_max)

    if z_min >= z_max:
        raise ValueError("z_min must be smaller than z_max")

    z_min = max(z_min, -layer.half_length)
    z_max = min(z_max, layer.half_length)

    phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi)
    z_values = np.linspace(z_min, z_max, n_z)

    phi_grid, z_grid = np.meshgrid(phi_values, z_values)

    x_grid = layer.radius * np.cos(phi_grid)
    y_grid = layer.radius * np.sin(phi_grid)

    ax.plot_wireframe(
        x_grid,
        y_grid,
        z_grid,
        linewidth=0.35,
        alpha=alpha,
    )


def draw_barrel_detector(
    ax,
    detector: BarrelDetector,
    n_phi: int = 80,
    n_z: int = 5,
    alpha: float = 0.10,
    z_min: float | None = None,
    z_max: float | None = None,
):
    """
    Draw all cylindrical layers in a BarrelDetector.
    """

    for layer in detector.layers:
        draw_cylindrical_layer(
            ax=ax,
            layer=layer,
            n_phi=n_phi,
            n_z=n_z,
            alpha=alpha,
            z_min=z_min,
            z_max=z_max,
        )


def draw_beamline_3d(
    ax,
    z_min: float,
    z_max: float,
):
    """
    Draw the beamline along z at x=0, y=0.
    """

    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [z_min, z_max],
        linestyle="--",
        linewidth=1.0,
        label="beamline",
    )


def draw_origin_3d(ax):
    """
    Draw the origin point.
    """

    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=80,
        label="origin",
    )


def _validate_positions(name: str, positions: np.ndarray) -> np.ndarray:
    """
    Validate position array with shape (n, 3).
    """

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")

    return positions


def _resolve_detector_z_range(
    detector: BarrelDetector,
    detector_z_range: tuple[float, float] | None,
) -> tuple[float, float]:
    """
    Resolve detector display z-range.
    """

    if detector_z_range is None:
        max_half_length = max(layer.half_length for layer in detector.layers)
        return -max_half_length, max_half_length

    if len(detector_z_range) != 2:
        raise ValueError("detector_z_range must be a tuple/list of length 2")

    z_min, z_max = detector_z_range

    z_min = float(z_min)
    z_max = float(z_max)

    if z_min >= z_max:
        raise ValueError("detector_z_range must satisfy z_min < z_max")

    return z_min, z_max


def plot_cylindrical_track_event(
    detector: BarrelDetector,
    truth_positions: np.ndarray,
    measured_positions: np.ndarray,
    predicted_positions: np.ndarray | None = None,
    filtered_positions: np.ndarray | None = None,
    smoothed_positions: np.ndarray | None = None,
    title: str = "OpenReco cylindrical track event",
    show_detector: bool = True,
    detector_z_range: tuple[float, float] | None = None,
    show_beamline: bool = True,
    show_origin: bool = True,
):
    """
    Plot one cylindrical tracking event in 3D.
    """

    truth_positions = _validate_positions("truth_positions", truth_positions)
    measured_positions = _validate_positions("measured_positions", measured_positions)

    if predicted_positions is not None:
        predicted_positions = _validate_positions(
            "predicted_positions",
            predicted_positions,
        )

    if filtered_positions is not None:
        filtered_positions = _validate_positions(
            "filtered_positions",
            filtered_positions,
        )

    if smoothed_positions is not None:
        smoothed_positions = _validate_positions(
            "smoothed_positions",
            smoothed_positions,
        )

    z_min, z_max = _resolve_detector_z_range(detector, detector_z_range)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if show_detector:
        draw_barrel_detector(
            ax,
            detector,
            z_min=z_min,
            z_max=z_max,
            alpha=0.08,
        )

    if show_beamline:
        draw_beamline_3d(ax, z_min=z_min, z_max=z_max)

    if show_origin:
        draw_origin_3d(ax)

    ax.plot(
        truth_positions[:, 0],
        truth_positions[:, 1],
        truth_positions[:, 2],
        marker="o",
        label="truth trajectory",
    )

    ax.scatter(
        measured_positions[:, 0],
        measured_positions[:, 1],
        measured_positions[:, 2],
        label="smeared measurements",
    )

    if predicted_positions is not None:
        ax.plot(
            predicted_positions[:, 0],
            predicted_positions[:, 1],
            predicted_positions[:, 2],
            marker="x",
            label="predicted states",
        )

    if filtered_positions is not None:
        ax.plot(
            filtered_positions[:, 0],
            filtered_positions[:, 1],
            filtered_positions[:, 2],
            marker="s",
            label="filtered states",
        )

    if smoothed_positions is not None:
        ax.plot(
            smoothed_positions[:, 0],
            smoothed_positions[:, 1],
            smoothed_positions[:, 2],
            marker="^",
            label="smoothed states",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    max_radius = float(np.max(detector.radii))
    ax.set_xlim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_ylim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_zlim(z_min, z_max)

    ax.view_init(elev=22, azim=-55)

    plt.tight_layout()

    return fig, ax


def plot_cylindrical_track_xy(
    detector: BarrelDetector,
    truth_positions: np.ndarray,
    measured_positions: np.ndarray,
    predicted_positions: np.ndarray | None = None,
    filtered_positions: np.ndarray | None = None,
    smoothed_positions: np.ndarray | None = None,
    title: str = "OpenReco cylindrical track event: x-y view",
    show_detector: bool = True,
    show_origin: bool = True,
):
    """
    Plot one cylindrical tracking event in x-y top view.

    This view is useful for checking magnetic bending in a Bz field.
    """

    truth_positions = _validate_positions("truth_positions", truth_positions)
    measured_positions = _validate_positions("measured_positions", measured_positions)

    if predicted_positions is not None:
        predicted_positions = _validate_positions(
            "predicted_positions",
            predicted_positions,
        )

    if filtered_positions is not None:
        filtered_positions = _validate_positions(
            "filtered_positions",
            filtered_positions,
        )

    if smoothed_positions is not None:
        smoothed_positions = _validate_positions(
            "smoothed_positions",
            smoothed_positions,
        )

    fig, ax = plt.subplots(figsize=(7, 7))

    if show_detector:
        theta = np.linspace(0.0, 2.0 * np.pi, 200)

        for layer in detector.layers:
            ax.plot(
                layer.radius * np.cos(theta),
                layer.radius * np.sin(theta),
                linewidth=0.6,
                alpha=0.35,
            )

    if show_origin:
        ax.scatter([0.0], [0.0], marker="*", s=80, label="origin")

    ax.plot(
        truth_positions[:, 0],
        truth_positions[:, 1],
        marker="o",
        label="truth trajectory",
    )

    ax.scatter(
        measured_positions[:, 0],
        measured_positions[:, 1],
        label="smeared measurements",
    )

    if predicted_positions is not None:
        ax.plot(
            predicted_positions[:, 0],
            predicted_positions[:, 1],
            marker="x",
            label="predicted states",
        )

    if filtered_positions is not None:
        ax.plot(
            filtered_positions[:, 0],
            filtered_positions[:, 1],
            marker="s",
            label="filtered states",
        )

    if smoothed_positions is not None:
        ax.plot(
            smoothed_positions[:, 0],
            smoothed_positions[:, 1],
            marker="^",
            label="smoothed states",
        )

    max_radius = float(np.max(detector.radii))
    ax.set_xlim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_ylim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    return fig, ax
