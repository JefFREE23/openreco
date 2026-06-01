"""
Visualization tools for OpenReco.

This module provides plotting helpers for the OpenReco v0 tracking chain.

Main v0 visualization:
    - cylindrical barrel detector wireframes
    - truth trajectory
    - smeared measurements
    - predicted Kalman states
    - filtered Kalman states
    - smoothed Kalman states
"""

import numpy as np
import matplotlib.pyplot as plt

from openreco.geometry import BarrelDetector, CylindricalLayer


def draw_cylindrical_layer(
    ax,
    layer: CylindricalLayer,
    n_phi: int = 80,
    n_z: int = 5,
    alpha: float = 0.15,
):
    """
    Draw a cylindrical detector layer as a wireframe.

    Parameters
    ----------
    ax:
        Matplotlib 3D axis.
    layer:
        CylindricalLayer to draw.
    n_phi:
        Number of azimuthal points.
    n_z:
        Number of z grid points.
    alpha:
        Wireframe transparency.
    """

    if n_phi < 4:
        raise ValueError("n_phi must be at least 4")

    if n_z < 2:
        raise ValueError("n_z must be at least 2")

    phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi)
    z_values = np.linspace(-layer.half_length, layer.half_length, n_z)

    phi_grid, z_grid = np.meshgrid(phi_values, z_values)

    x_grid = layer.radius * np.cos(phi_grid)
    y_grid = layer.radius * np.sin(phi_grid)

    ax.plot_wireframe(
        x_grid,
        y_grid,
        z_grid,
        linewidth=0.4,
        alpha=alpha,
    )


def draw_barrel_detector(
    ax,
    detector: BarrelDetector,
    n_phi: int = 80,
    n_z: int = 5,
    alpha: float = 0.15,
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
        )


def _validate_positions(name: str, positions: np.ndarray) -> np.ndarray:
    """
    Validate position array with shape (n, 3).
    """

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")

    return positions


def plot_cylindrical_track_event(
    detector: BarrelDetector,
    truth_positions: np.ndarray,
    measured_positions: np.ndarray,
    predicted_positions: np.ndarray | None = None,
    filtered_positions: np.ndarray | None = None,
    smoothed_positions: np.ndarray | None = None,
    title: str = "OpenReco cylindrical track event",
    show_detector: bool = True,
):
    """
    Plot one cylindrical tracking event.

    Parameters
    ----------
    detector:
        BarrelDetector containing cylindrical layers.
    truth_positions:
        True positions with shape (n, 3).
    measured_positions:
        Measured hit positions with shape (n, 3).
    predicted_positions:
        Optional predicted Kalman positions with shape (n, 3).
    filtered_positions:
        Optional filtered Kalman positions with shape (n, 3).
    smoothed_positions:
        Optional smoothed Kalman positions with shape (n, 3).
    title:
        Plot title.
    show_detector:
        Whether to draw cylindrical detector wireframes.

    Returns
    -------
    fig, ax:
        Matplotlib figure and 3D axis.
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

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if show_detector:
        draw_barrel_detector(ax, detector)

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

    # Make x/y scaling visually comparable.
    max_radius = float(np.max(detector.radii))
    ax.set_xlim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_ylim(-1.1 * max_radius, 1.1 * max_radius)

    plt.tight_layout()

    return fig, ax
