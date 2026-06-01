"""
Kalman filter tools for OpenReco.

This module implements a minimal cylindrical EKF-style Kalman core for
OpenReco v0.

Roadmap intent:
    - state on a reference surface
    - covariance propagation
    - prediction through a magnetic field
    - measurement update
    - residuals and chi-square

Important:
    The final OpenReco v0 demo uses cylindrical tracker layers.

Current v0 simplification:
    TrackState stores [x, y, tx, ty, q_over_p] and keeps z separately.
    Therefore, cylindrical measurements [phi, z] are handled as:
        - phi: used in the Kalman update
        - z: available for diagnostics, but not used to update covariance

This avoids pretending that z is part of the 5D covariance when it is not.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from openreco.field import UniformMagneticField
from openreco.geometry import BarrelDetector, CylindricalLayer
from openreco.measurements import Measurement
from openreco.particle_gun import Particle
from openreco.propagation import (
    PropagationResult,
    propagate_to_cylindrical_layer,
    radial_distance,
)
from openreco.state import TrackState


@dataclass(frozen=True)
class KalmanUpdateResult:
    """
    Result from one Kalman prediction/update step.
    """

    predicted_state: TrackState
    filtered_state: TrackState
    residual: np.ndarray
    residual_covariance: np.ndarray
    kalman_gain: np.ndarray
    chi2: float
    layer_name: str

    def __post_init__(self):
        residual = np.asarray(self.residual, dtype=float)
        residual_covariance = np.asarray(self.residual_covariance, dtype=float)
        kalman_gain = np.asarray(self.kalman_gain, dtype=float)

        if residual.ndim != 1:
            raise ValueError("residual must be a 1D vector")

        measurement_dimension = residual.shape[0]

        if residual_covariance.shape != (measurement_dimension, measurement_dimension):
            raise ValueError("residual_covariance shape must match residual dimension")

        if kalman_gain.shape != (5, measurement_dimension):
            raise ValueError("kalman_gain must have shape (5, measurement_dimension)")

        if self.chi2 < 0.0:
            raise ValueError("chi2 must be non-negative")

        if not isinstance(self.layer_name, str):
            raise TypeError("layer_name must be a string")

        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "residual_covariance", residual_covariance)
        object.__setattr__(self, "kalman_gain", kalman_gain)
        object.__setattr__(self, "chi2", float(self.chi2))


@dataclass(frozen=True)
class KalmanPredictionResult:
    """
    Result of propagating a TrackState to a cylindrical layer.
    """

    predicted_state: TrackState
    transport_jacobian: np.ndarray
    propagation_result: PropagationResult

    def __post_init__(self):
        transport_jacobian = np.asarray(self.transport_jacobian, dtype=float)

        if transport_jacobian.shape != (5, 5):
            raise ValueError("transport_jacobian must have shape (5, 5)")

        object.__setattr__(self, "transport_jacobian", transport_jacobian)


def make_process_noise(diagonal: Iterable[float]) -> np.ndarray:
    """
    Create a diagonal 5x5 process-noise matrix.
    """

    diagonal = np.asarray(list(diagonal), dtype=float)

    if diagonal.shape != (5,):
        raise ValueError("process-noise diagonal must have shape (5,)")

    if np.any(diagonal < 0.0):
        raise ValueError("process-noise diagonal values must be non-negative")

    return np.diag(diagonal)


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to the range [-pi, pi).
    """

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def state_to_particle(state: TrackState) -> Particle:
    """
    Convert a TrackState into a truth-like Particle approximation.

    TrackState:
        [x, y, tx, ty, q_over_p]

    Slopes:
        tx = px / pz
        ty = py / pz

    The momentum magnitude is estimated from q_over_p:
        p = 1 / abs(q_over_p)

    The charge sign is taken from q_over_p.
    """

    if np.isclose(state.q_over_p, 0.0):
        raise ValueError("q_over_p must be nonzero for charged-particle propagation")

    charge = float(np.sign(state.q_over_p))
    momentum_magnitude = 1.0 / abs(state.q_over_p)

    direction = np.array([state.tx, state.ty, 1.0], dtype=float)
    direction = direction / np.linalg.norm(direction)

    momentum = momentum_magnitude * direction
    position = np.array([state.x, state.y, state.z], dtype=float)

    return Particle(
        position=position,
        momentum=momentum,
        charge=charge,
    )


def propagation_result_to_state(
    propagation_result: PropagationResult,
    q_over_p: float,
) -> TrackState:
    """
    Convert a PropagationResult back into a TrackState.
    """

    px, py, pz = propagation_result.momentum

    if np.isclose(pz, 0.0):
        raise ValueError("cannot create TrackState from momentum with pz close to zero")

    x, y, z = propagation_result.position

    parameters = np.array(
        [
            x,
            y,
            px / pz,
            py / pz,
            q_over_p,
        ],
        dtype=float,
    )

    covariance = np.eye(5)

    return TrackState(
        parameters=parameters,
        covariance=covariance,
        z=z,
    )


def _propagate_state_parameters_only(
    parameters: np.ndarray,
    z: float,
    field: UniformMagneticField,
    layer: CylindricalLayer,
    curvature_scale: float,
    max_s: float,
    n_scan: int,
) -> np.ndarray:
    """
    Internal helper for numerical transport Jacobian.
    """

    dummy_state = TrackState(
        parameters=np.asarray(parameters, dtype=float),
        covariance=np.eye(5),
        z=z,
    )

    particle = state_to_particle(dummy_state)

    propagation_result = propagate_to_cylindrical_layer(
        particle=particle,
        field=field,
        layer=layer,
        curvature_scale=curvature_scale,
        max_s=max_s,
        n_scan=n_scan,
    )

    propagated_state = propagation_result_to_state(
        propagation_result=propagation_result,
        q_over_p=dummy_state.q_over_p,
    )

    return propagated_state.parameters


def numerical_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    step: float = 1e-5,
) -> np.ndarray:
    """
    Compute a central-difference numerical Jacobian.
    """

    parameters = np.asarray(parameters, dtype=float)

    if parameters.ndim != 1:
        raise ValueError("parameters must be a 1D vector")

    if step <= 0.0:
        raise ValueError("step must be positive")

    base_output = np.asarray(function(parameters), dtype=float)

    if base_output.ndim != 1:
        raise ValueError("function output must be a 1D vector")

    jacobian = np.zeros((base_output.shape[0], parameters.shape[0]), dtype=float)

    for i in range(parameters.shape[0]):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[i] += step
        minus[i] -= step

        output_plus = np.asarray(function(plus), dtype=float)
        output_minus = np.asarray(function(minus), dtype=float)

        jacobian[:, i] = (output_plus - output_minus) / (2.0 * step)

    return jacobian


def predict_to_cylindrical_layer(
    state: TrackState,
    field: UniformMagneticField,
    layer: CylindricalLayer,
    process_noise: np.ndarray | None = None,
    curvature_scale: float = 0.003,
    max_s: float = 10000.0,
    n_scan: int = 1000,
    jacobian_step: float = 1e-5,
) -> KalmanPredictionResult:
    """
    Predict a TrackState to a cylindrical layer.

    This is the EKF prediction step:

        x_k^- = f_k(x_{k-1})
        C_k^- = F_k C_{k-1} F_k^T + Q_k

    where F_k is computed numerically.
    """

    particle = state_to_particle(state)

    propagation_result = propagate_to_cylindrical_layer(
        particle=particle,
        field=field,
        layer=layer,
        curvature_scale=curvature_scale,
        max_s=max_s,
        n_scan=n_scan,
    )

    predicted_state_without_covariance = propagation_result_to_state(
        propagation_result=propagation_result,
        q_over_p=state.q_over_p,
    )

    def propagation_function(parameters: np.ndarray) -> np.ndarray:
        return _propagate_state_parameters_only(
            parameters=parameters,
            z=state.z,
            field=field,
            layer=layer,
            curvature_scale=curvature_scale,
            max_s=max_s,
            n_scan=n_scan,
        )

    transport_jacobian = numerical_jacobian(
        function=propagation_function,
        parameters=state.parameters,
        step=jacobian_step,
    )

    predicted_covariance = (
        transport_jacobian @ state.covariance @ transport_jacobian.T
    )

    if process_noise is not None:
        process_noise = np.asarray(process_noise, dtype=float)

        if process_noise.shape != (5, 5):
            raise ValueError("process_noise must have shape (5, 5)")

        if not np.allclose(process_noise, process_noise.T):
            raise ValueError("process_noise must be symmetric")

        predicted_covariance = predicted_covariance + process_noise

    predicted_covariance = 0.5 * (
        predicted_covariance + predicted_covariance.T
    )

    predicted_state = TrackState(
        parameters=predicted_state_without_covariance.parameters,
        covariance=predicted_covariance,
        z=predicted_state_without_covariance.z,
    )

    return KalmanPredictionResult(
        predicted_state=predicted_state,
        transport_jacobian=transport_jacobian,
        propagation_result=propagation_result,
    )


def cylindrical_phi_prediction(state: TrackState) -> np.ndarray:
    """
    Predict cylindrical phi measurement.

    Current update measurement:
        [phi]

    The full cylindrical hit may contain [phi, z], but z is diagnostic only
    for this 5D TrackState representation.
    """

    phi = np.arctan2(state.y, state.x)

    return np.array([phi], dtype=float)


def cylindrical_phi_jacobian(state: TrackState) -> np.ndarray:
    """
    Measurement Jacobian for cylindrical phi.

    h(x) = atan2(y, x)

    State vector:
        [x, y, tx, ty, q_over_p]
    """

    radius_squared = state.x**2 + state.y**2

    if radius_squared <= 0.0:
        raise ValueError("cannot compute phi Jacobian at x=y=0")

    jacobian = np.zeros((1, 5), dtype=float)
    jacobian[0, 0] = -state.y / radius_squared
    jacobian[0, 1] = state.x / radius_squared

    return jacobian


def cylindrical_full_residual(
    state: TrackState,
    measurement: Measurement,
) -> np.ndarray:
    """
    Return full cylindrical residual [delta_phi, delta_z] for diagnostics.
    """

    if measurement.surface_type != "cylinder":
        raise ValueError("measurement must have surface_type='cylinder'")

    if measurement.dimension != 2:
        raise ValueError("cylindrical diagnostic residual expects [phi, z]")

    predicted_phi = np.arctan2(state.y, state.x)
    predicted_z = state.z

    residual_phi = wrap_angle(measurement.values[0] - predicted_phi)
    residual_z = measurement.values[1] - predicted_z

    return np.array([residual_phi, residual_z], dtype=float)


def update_state(
    predicted_state: TrackState,
    measurement_values: np.ndarray,
    measurement_covariance: np.ndarray,
    measurement_matrix: np.ndarray,
    predicted_measurement: np.ndarray,
    layer_name: str,
    wrap_first_residual: bool = False,
) -> KalmanUpdateResult:
    """
    Perform one Kalman update.

    residual = m - h(x)
    S = H C H^T + V
    K = C H^T S^-1
    x_new = x + K residual

    Joseph covariance form is used for numerical stability:
    C_new = (I - K H) C (I - K H)^T + K V K^T
    """

    measurement_values = np.asarray(measurement_values, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)
    measurement_matrix = np.asarray(measurement_matrix, dtype=float)
    predicted_measurement = np.asarray(predicted_measurement, dtype=float)

    if measurement_values.ndim != 1:
        raise ValueError("measurement_values must be a 1D vector")

    measurement_dimension = measurement_values.shape[0]

    if predicted_measurement.shape != measurement_values.shape:
        raise ValueError("predicted_measurement shape must match measurement_values")

    if measurement_covariance.shape != (measurement_dimension, measurement_dimension):
        raise ValueError("measurement_covariance shape must match measurement dimension")

    if measurement_matrix.shape != (measurement_dimension, 5):
        raise ValueError("measurement_matrix must have shape (measurement_dimension, 5)")

    if not isinstance(layer_name, str):
        raise TypeError("layer_name must be a string")

    residual = measurement_values - predicted_measurement

    if wrap_first_residual:
        residual = residual.copy()
        residual[0] = wrap_angle(residual[0])

    covariance = predicted_state.covariance
    h_matrix = measurement_matrix
    v_matrix = measurement_covariance

    residual_covariance = h_matrix @ covariance @ h_matrix.T + v_matrix
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)

    kalman_gain = covariance @ h_matrix.T @ np.linalg.inv(residual_covariance)

    filtered_parameters = predicted_state.parameters + kalman_gain @ residual

    identity = np.eye(5)
    filtered_covariance = (
        (identity - kalman_gain @ h_matrix)
        @ covariance
        @ (identity - kalman_gain @ h_matrix).T
        + kalman_gain @ v_matrix @ kalman_gain.T
    )

    filtered_covariance = 0.5 * (
        filtered_covariance + filtered_covariance.T
    )

    chi2 = float(residual.T @ np.linalg.inv(residual_covariance) @ residual)

    filtered_state = TrackState(
        parameters=filtered_parameters,
        covariance=filtered_covariance,
        z=predicted_state.z,
    )

    return KalmanUpdateResult(
        predicted_state=predicted_state,
        filtered_state=filtered_state,
        residual=residual,
        residual_covariance=residual_covariance,
        kalman_gain=kalman_gain,
        chi2=chi2,
        layer_name=layer_name,
    )


def update_with_cylindrical_measurement(
    predicted_state: TrackState,
    measurement: Measurement,
) -> KalmanUpdateResult:
    """
    Update a predicted TrackState using a cylindrical measurement.

    The stored cylindrical measurement is [phi, z].
    The Kalman update uses phi only because z is not part of the current
    5D TrackState covariance.

    z residual can still be inspected using cylindrical_full_residual().
    """

    if measurement.surface_type != "cylinder":
        raise ValueError("measurement must have surface_type='cylinder'")

    if measurement.dimension != 2:
        raise ValueError("cylindrical measurement must have dimension 2: [phi, z]")

    phi_value = np.array([measurement.values[0]], dtype=float)
    phi_covariance = np.array([[measurement.covariance[0, 0]]], dtype=float)

    return update_state(
        predicted_state=predicted_state,
        measurement_values=phi_value,
        measurement_covariance=phi_covariance,
        measurement_matrix=cylindrical_phi_jacobian(predicted_state),
        predicted_measurement=cylindrical_phi_prediction(predicted_state),
        layer_name=measurement.layer_name,
        wrap_first_residual=True,
    )


def filter_cylindrical_track(
    initial_state: TrackState,
    measurements: Iterable[Measurement],
    detector: BarrelDetector,
    field: UniformMagneticField,
    process_noise: np.ndarray | None = None,
    curvature_scale: float = 0.003,
    max_s: float = 10000.0,
    n_scan: int = 1000,
) -> list[KalmanUpdateResult]:
    """
    Run a cylindrical EKF over a BarrelDetector.

    For each layer:
        1. propagate current state to the cylindrical layer
        2. propagate covariance with numerical transport Jacobian
        3. update with the cylindrical phi measurement
    """

    measurements = list(measurements)

    if len(measurements) != len(detector):
        raise ValueError("number of measurements must match number of detector layers")

    results = []
    current_state = initial_state.copy()

    for measurement, layer in zip(measurements, detector.layers):
        prediction = predict_to_cylindrical_layer(
            state=current_state,
            field=field,
            layer=layer,
            process_noise=process_noise,
            curvature_scale=curvature_scale,
            max_s=max_s,
            n_scan=n_scan,
        )

        result = update_with_cylindrical_measurement(
            predicted_state=prediction.predicted_state,
            measurement=measurement,
        )

        results.append(result)
        current_state = result.filtered_state

    return results


def total_chi2(results: Iterable[KalmanUpdateResult]) -> float:
    """
    Return total chi-square from a list of Kalman update results.
    """

    return float(sum(result.chi2 for result in results))
