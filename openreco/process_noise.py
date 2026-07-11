"""Process-noise helpers for OpenReco v3.0.

This module provides reconstruction-side covariance inflation terms for
material-induced multiple scattering.

The truth-side material effect is applied in event_generation.py. The helpers
here are for the reconstruction side: they build small process-noise matrices
that can later be added to the predicted bound-state covariance during EKF
fitting.

OpenReco's cylindrical bound-state parameter order is:

    [phi, z, alpha, tan_lambda, q_over_p]

For a compact first model, multiple scattering is represented as additional
uncertainty on the angular track parameters alpha and tan_lambda.
"""

from __future__ import annotations

import numpy as np

from openreco.detector_effects import (
    DetectorEffectsConfig,
    multiple_scattering_theta0,
)


BOUND_STATE_SIZE = 5
PHI_INDEX = 0
Z_INDEX = 1
ALPHA_INDEX = 2
TAN_LAMBDA_INDEX = 3
Q_OVER_P_INDEX = 4


def angular_process_noise_variance(
    *,
    p_gev: float,
    x_over_x0: float,
    process_noise_scale: float = 1.0,
    beta: float = 1.0,
    charge_abs: float = 1.0,
) -> float:
    """Return angular process-noise variance from material scattering.

    The variance is:

        (process_noise_scale * theta0)^2

    where theta0 is the Highland multiple-scattering RMS angle.
    """

    if process_noise_scale < 0.0:
        raise ValueError("process_noise_scale must be non-negative")

    theta0 = multiple_scattering_theta0(
        p_gev=p_gev,
        x_over_x0=x_over_x0,
        beta=beta,
        charge_abs=charge_abs,
    )

    return float((process_noise_scale * theta0) ** 2)


def material_process_noise_matrix(
    *,
    p_gev: float,
    x_over_x0: float,
    process_noise_scale: float = 1.0,
    beta: float = 1.0,
    charge_abs: float = 1.0,
    state_size: int = BOUND_STATE_SIZE,
) -> np.ndarray:
    """Return a bound-state process-noise covariance matrix.

    The returned matrix is diagonal and positive semidefinite. The simplified
    v3.0 model adds multiple-scattering uncertainty to the angular parameters:

        alpha
        tan_lambda

    It leaves phi, z, and q_over_p unchanged directly. The EKF propagation
    Jacobian can later transport angular uncertainty into measurement-space
    uncertainty.
    """

    if state_size < BOUND_STATE_SIZE:
        raise ValueError("state_size must be at least 5")

    variance = angular_process_noise_variance(
        p_gev=p_gev,
        x_over_x0=x_over_x0,
        process_noise_scale=process_noise_scale,
        beta=beta,
        charge_abs=charge_abs,
    )

    process_noise = np.zeros((state_size, state_size), dtype=float)
    process_noise[ALPHA_INDEX, ALPHA_INDEX] = variance
    process_noise[TAN_LAMBDA_INDEX, TAN_LAMBDA_INDEX] = variance

    return process_noise


def material_process_noise_for_layer(
    *,
    detector_effects: DetectorEffectsConfig,
    layer_id: int,
    p_gev: float,
    process_noise_scale: float = 1.0,
    beta: float = 1.0,
    charge_abs: float = 1.0,
    state_size: int = BOUND_STATE_SIZE,
) -> np.ndarray:
    """Return process-noise matrix for one detector layer."""

    material = detector_effects.material_for_layer(layer_id)

    return material_process_noise_matrix(
        p_gev=p_gev,
        x_over_x0=material.x_over_x0,
        process_noise_scale=process_noise_scale,
        beta=beta,
        charge_abs=charge_abs,
        state_size=state_size,
    )