print("SCRIPT STARTED")

"""
Straight-line single-track checkpoint example.

This is a small debugging checkpoint for OpenReco.

It uses:
    - planar detector layers
    - smeared planar measurements
    - simple least-squares straight-line fitting

Important:
    This is not the final OpenReco v0 demo.
    The final v0 demo will use cylindrical tracker layers and uniform B field.
"""

import numpy as np
import matplotlib.pyplot as plt

from openreco.geometry import make_uniform_detector
from openreco.measurements import make_planar_measurement


def true_straight_track(z_values, x0=1.0, y0=-2.0, tx=0.08, ty=-0.04, z0=0.0):
    """
    Generate true straight-line positions.

    x(z) = x0 + tx * (z - z0)
    y(z) = y0 + ty * (z - z0)
    """

    z_values = np.asarray(z_values, dtype=float)

    x_values = x0 + tx * (z_values - z0)
    y_values = y0 + ty * (z_values - z0)

    return np.column_stack([x_values, y_values, z_values])


def fit_straight_line(z_values, measured_xy):
    """
    Fit x(z) and y(z) independently with first-order polynomials.

    Returns
    -------
    fit_params:
        Dictionary containing x_intercept, x_slope, y_intercept, y_slope.
    fitted_xy:
        Fitted [x, y] values at the input z positions.
    """

    z_values = np.asarray(z_values, dtype=float)
    measured_xy = np.asarray(measured_xy, dtype=float)

    if measured_xy.shape != (len(z_values), 2):
        raise ValueError("measured_xy must have shape (n_layers, 2)")

    x_slope, x_intercept = np.polyfit(z_values, measured_xy[:, 0], deg=1)
    y_slope, y_intercept = np.polyfit(z_values, measured_xy[:, 1], deg=1)

    fitted_x = x_intercept + x_slope * z_values
    fitted_y = y_intercept + y_slope * z_values

    fit_params = {
        "x_intercept": x_intercept,
        "x_slope": x_slope,
        "y_intercept": y_intercept,
        "y_slope": y_slope,
    }

    fitted_xy = np.column_stack([fitted_x, fitted_y])

    return fit_params, fitted_xy


def compute_residuals(measured_xy, fitted_xy):
    """
    Compute measurement residuals.

    residual = measured - fitted
    """

    measured_xy = np.asarray(measured_xy, dtype=float)
    fitted_xy = np.asarray(fitted_xy, dtype=float)

    if measured_xy.shape != fitted_xy.shape:
        raise ValueError("measured_xy and fitted_xy must have the same shape")

    return measured_xy - fitted_xy


def main():
    rng = np.random.default_rng(123)

    detector = make_uniform_detector(
        n_layers=8,
        z_min=0.0,
        z_max=70.0,
    )

    true_positions = true_straight_track(
        detector.z_positions,
        x0=1.0,
        y0=-2.0,
        tx=0.08,
        ty=-0.04,
        z0=0.0,
    )

    sigma = 0.2

    measurements = [
        make_planar_measurement(
            layer=layer,
            true_position=true_position,
            sigma=sigma,
            rng=rng,
        )
        for layer, true_position in zip(detector.layers, true_positions)
    ]

    measured_xy = np.array([measurement.values for measurement in measurements])

    fit_params, fitted_xy = fit_straight_line(
        detector.z_positions,
        measured_xy,
    )

    residuals = compute_residuals(measured_xy, fitted_xy)
    rmse = np.sqrt(np.mean(residuals**2))

    print("Straight-line checkpoint")
    print("------------------------")
    print(f"Number of layers: {len(detector)}")
    print(f"Measurement sigma: {sigma:.3f}")
    print()
    print("Fitted parameters:")
    print(f"  x(z) = {fit_params['x_intercept']:.4f} + {fit_params['x_slope']:.4f} z")
    print(f"  y(z) = {fit_params['y_intercept']:.4f} + {fit_params['y_slope']:.4f} z")
    print()
    print(f"RMSE: {rmse:.4f}")
    print()
    print("Residuals [measured - fitted]:")
    for i, layer in enumerate(detector.layers):
        print(
            f"  {layer.name:8s} z={layer.z:6.2f} "
            f"dx={residuals[i, 0]: .4f} dy={residuals[i, 1]: .4f}"
        )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_positions[:, 0],
        true_positions[:, 1],
        true_positions[:, 2],
        label="true track",
    )

    ax.scatter(
        measured_xy[:, 0],
        measured_xy[:, 1],
        detector.z_positions,
        label="measured hits",
    )

    ax.plot(
        fitted_xy[:, 0],
        fitted_xy[:, 1],
        detector.z_positions,
        label="fitted track",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("OpenReco straight-line checkpoint")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
