"""
Kalman smoothing tools for OpenReco.

This module implements a Rauch-Tung-Striebel-style backward smoother for the
surface-bound OpenReco v0 Kalman filter.

The smoother uses:
    - filtered states
    - predicted states
    - transport Jacobians F_k
    - predicted covariances

For a filtered state k and next predicted state k+1:

    A_k = C_k^f F_{k+1}^T (C_{k+1}^-)^{-1}

    x_k^s = x_k^f + A_k (x_{k+1}^s - x_{k+1}^-)

    C_k^s = C_k^f + A_k (C_{k+1}^s - C_{k+1}^-) A_k^T
"""

from dataclasses import dataclass

import numpy as np

from openreco.kalman import KalmanUpdateResult, wrap_angle
from openreco.state import TrackState, make_cylindrical_state


@dataclass(frozen=True)
class SmoothingResult:
    """
    Result for one smoothed track state.
    """

    filtered_state: TrackState
    smoothed_state: TrackState
    smoothing_gain: np.ndarray
    layer_name: str

    def __post_init__(self):
        smoothing_gain = np.asarray(self.smoothing_gain, dtype=float)

        if smoothing_gain.shape != (5, 5):
            raise ValueError("smoothing_gain must have shape (5, 5)")

        if not isinstance(self.layer_name, str):
            raise TypeError("layer_name must be a string")

        object.__setattr__(self, "smoothing_gain", smoothing_gain)


def _bound_state_difference(
    first: TrackState,
    second: TrackState,
) -> np.ndarray:
    """
    Compute first.parameters - second.parameters with angular wrapping.

    For cylindrical bound states:
        parameter 0 = phi
        parameter 2 = alpha

    Both are angular.
    """

    if first.surface_type != second.surface_type:
        raise ValueError("states must have the same surface_type")

    difference = first.parameters - second.parameters

    if first.surface_type == "cylinder":
        difference = difference.copy()
        difference[0] = wrap_angle(difference[0])
        difference[2] = wrap_angle(difference[2])

    return difference


def _make_state_like(
    reference_state: TrackState,
    parameters: np.ndarray,
    covariance: np.ndarray,
) -> TrackState:
    """
    Create a TrackState on the same surface as reference_state.
    """

    parameters = np.asarray(parameters, dtype=float).copy()
    covariance = np.asarray(covariance, dtype=float)

    if reference_state.surface_type == "cylinder":
        parameters[0] = wrap_angle(parameters[0])
        parameters[2] = wrap_angle(parameters[2])

        return make_cylindrical_state(
            phi=parameters[0],
            z=parameters[1],
            dir0=parameters[2],
            dir1=parameters[3],
            q_over_p=parameters[4],
            covariance=covariance,
            surface_radius=reference_state.radius,
            surface_name=reference_state.surface_name,
        )

    return TrackState(
        parameters=parameters,
        covariance=covariance,
        surface_type=reference_state.surface_type,
        surface_name=reference_state.surface_name,
        surface_radius=reference_state.surface_radius,
    )


def compute_smoothing_gain(
    filtered_covariance: np.ndarray,
    transport_jacobian: np.ndarray,
    predicted_next_covariance: np.ndarray,
) -> np.ndarray:
    """
    Compute RTS smoothing gain.

    A_k = C_k^f F_{k+1}^T (C_{k+1}^-)^{-1}

    A pseudo-inverse is used for numerical robustness in this minimal v0 code.
    """

    filtered_covariance = np.asarray(filtered_covariance, dtype=float)
    transport_jacobian = np.asarray(transport_jacobian, dtype=float)
    predicted_next_covariance = np.asarray(predicted_next_covariance, dtype=float)

    if filtered_covariance.shape != (5, 5):
        raise ValueError("filtered_covariance must have shape (5, 5)")

    if transport_jacobian.shape != (5, 5):
        raise ValueError("transport_jacobian must have shape (5, 5)")

    if predicted_next_covariance.shape != (5, 5):
        raise ValueError("predicted_next_covariance must have shape (5, 5)")

    return (
        filtered_covariance
        @ transport_jacobian.T
        @ np.linalg.pinv(predicted_next_covariance)
    )


def smooth_track(
    kalman_results: list[KalmanUpdateResult],
) -> list[SmoothingResult]:
    """
    Smooth a sequence of Kalman filter results.

    The input must come from filter_cylindrical_track(), so each update result
    after prediction contains:
        - predicted_state
        - filtered_state
        - transport_jacobian
    """

    kalman_results = list(kalman_results)

    if len(kalman_results) == 0:
        raise ValueError("kalman_results must not be empty")

    n_results = len(kalman_results)

    smoothing_results: list[SmoothingResult | None] = [None] * n_results

    last_result = kalman_results[-1]

    smoothing_results[-1] = SmoothingResult(
        filtered_state=last_result.filtered_state,
        smoothed_state=last_result.filtered_state.copy(),
        smoothing_gain=np.zeros((5, 5)),
        layer_name=last_result.layer_name,
    )

    for index in range(n_results - 2, -1, -1):
        current_result = kalman_results[index]
        next_result = kalman_results[index + 1]

        if next_result.transport_jacobian is None:
            raise ValueError(
                "transport_jacobian is required for smoothing. "
                "Use filter_cylindrical_track() results."
            )

        current_filtered_state = current_result.filtered_state
        next_predicted_state = next_result.predicted_state
        next_smoothed_state = smoothing_results[index + 1].smoothed_state

        smoothing_gain = compute_smoothing_gain(
            filtered_covariance=current_filtered_state.covariance,
            transport_jacobian=next_result.transport_jacobian,
            predicted_next_covariance=next_predicted_state.covariance,
        )

        state_difference = _bound_state_difference(
            first=next_smoothed_state,
            second=next_predicted_state,
        )

        smoothed_parameters = (
            current_filtered_state.parameters
            + smoothing_gain @ state_difference
        )

        smoothed_covariance = (
            current_filtered_state.covariance
            + smoothing_gain
            @ (next_smoothed_state.covariance - next_predicted_state.covariance)
            @ smoothing_gain.T
        )

        smoothed_covariance = 0.5 * (
            smoothed_covariance + smoothed_covariance.T
        )

        smoothed_state = _make_state_like(
            reference_state=current_filtered_state,
            parameters=smoothed_parameters,
            covariance=smoothed_covariance,
        )

        smoothing_results[index] = SmoothingResult(
            filtered_state=current_filtered_state,
            smoothed_state=smoothed_state,
            smoothing_gain=smoothing_gain,
            layer_name=current_result.layer_name,
        )

    return smoothing_results


def smoothed_states(
    smoothing_results: list[SmoothingResult],
) -> list[TrackState]:
    """
    Extract smoothed states from smoothing results.
    """

    return [result.smoothed_state for result in smoothing_results]


def smoothed_positions(
    smoothing_results: list[SmoothingResult],
) -> np.ndarray:
    """
    Extract smoothed global positions with shape (n_states, 3).
    """

    return np.array(
        [result.smoothed_state.global_position() for result in smoothing_results],
        dtype=float,
    )
