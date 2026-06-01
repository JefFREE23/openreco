"""
Kalman filter tools for OpenReco.

This module implements a minimal surface-bound cylindrical EKF for OpenReco v0.

Roadmap intent:
    - 5D state on a reference surface
    - 5x5 covariance
    - prediction through a homogeneous magnetic field
    - measurements attached to detector surfaces
    - Kalman update in local measurement space
    - residuals and chi-square

Cylindrical bound state convention:
    parameters = [phi, z, alpha, tan_lambda, q_over_p]

where:
    phi        = local angular coordinate on the cylinder
    z          = longitudinal coordinate on the cylinder
    alpha      = transverse momentum direction angle
    tan_lambda = pz / pt
    q_over_p   = charge / momentum

Cylindrical measurement convention:
    measurement = [phi, z]

This means the cylindrical Kalman update uses both local coordinates:
    [phi, z]
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from openreco.field import UniformMagneticField
from openreco.geometry import BarrelDetector, CylindricalLayer
from openreco.measurements import Measurement
from openreco.particle_gun import Particle
from openreco.propagation import PropagationResult, propagate_to_cylindrical_layer
from openreco.state import TrackState, make_cylindrical_state


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
    transport_jacobian: np.ndarray | None = None

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

        transport_jacobian = self.transport_jacobian

        if transport_jacobian is not None:
            transport_jacobian = np.asarray(transport_jacobian, dtype=float)

            if transport_jacobian.shape != (5, 5):
                raise ValueError("transport_jacobian must have shape (5, 5)")

            object.__setattr__(self, "transport_jacobian", transport_jacobian)

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
    Convert a cylindrical bound TrackState into a Particle-like free state.

    Bound state:
        [phi, z, alpha, tan_lambda, q_over_p]

    Momentum:
        p = 1 / abs(q_over_p)
        pt = p / sqrt(1 + tan_lambda^2)
        pz = pt * tan_lambda
        px = pt * cos(alpha)
        py = pt * sin(alpha)
    """

    if state.surface_type != "cylinder":
        raise ValueError("state_to_particle currently requires a cylindrical state")

    if np.isclose(state.q_over_p, 0.0):
        raise ValueError("q_over_p must be nonzero for charged-particle propagation")

    charge = float(np.sign(state.q_over_p))
    momentum_magnitude = 1.0 / abs(state.q_over_p)

    alpha = state.dir0
    tan_lambda = state.dir1

    pt = momentum_magnitude / np.sqrt(1.0 + tan_lambda**2)
    pz = pt * tan_lambda

    momentum = np.array(
        [
            pt * np.cos(alpha),
            pt * np.sin(alpha),
            pz,
        ],
        dtype=float,
    )

    position = state.global_position()

    return Particle(
        position=position,
        momentum=momentum,
        charge=charge,
    )


def propagation_result_to_cylindrical_state(
    propagation_result: PropagationResult,
    q_over_p: float,
    layer: CylindricalLayer,
    covariance: np.ndarray,
) -> TrackState:
    """
    Convert a propagation result to a cylindrical bound TrackState.

    Output state:
        [phi, z, alpha, tan_lambda, q_over_p]
    """

    x, y, z = propagation_result.position
    px, py, pz = propagation_result.momentum

    pt = np.sqrt(px**2 + py**2)

    if pt <= 0.0:
        raise ValueError("cannot create bound state with zero transverse momentum")

    phi = np.arctan2(y, x)
    alpha = np.arctan2(py, px)
    tan_lambda = pz / pt

    return make_cylindrical_state(
        phi=phi,
        z=z,
        dir0=alpha,
        dir1=tan_lambda,
        q_over_p=q_over_p,
        covariance=covariance,
        surface_radius=layer.radius,
        surface_name=layer.name,
    )


def _propagate_bound_parameters_only(
    parameters: np.ndarray,
    covariance: np.ndarray,
    surface_radius: float,
    surface_name: str,
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
        covariance=covariance,
        surface_type="cylinder",
        surface_name=surface_name,
        surface_radius=surface_radius,
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

    propagated_state = propagation_result_to_cylindrical_state(
        propagation_result=propagation_result,
        q_over_p=dummy_state.q_over_p,
        layer=layer,
        covariance=np.eye(5),
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

        difference = output_plus - output_minus

        # phi and alpha are angular coordinates.
        if difference.shape[0] >= 1:
            difference[0] = wrap_angle(difference[0])

        if difference.shape[0] >= 3:
            difference[2] = wrap_angle(difference[2])

        jacobian[:, i] = difference / (2.0 * step)

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
    Predict a cylindrical bound TrackState to another cylindrical layer.

    EKF prediction:
        x_k^- = f_k(x_{k-1})
        C_k^- = F_k C_{k-1} F_k^T + Q_k
    """

    if state.surface_type != "cylinder":
        raise ValueError("predict_to_cylindrical_layer requires a cylindrical state")

    particle = state_to_particle(state)

    propagation_result = propagate_to_cylindrical_layer(
        particle=particle,
        field=field,
        layer=layer,
        curvature_scale=curvature_scale,
        max_s=max_s,
        n_scan=n_scan,
    )

    predicted_state_without_covariance = propagation_result_to_cylindrical_state(
        propagation_result=propagation_result,
        q_over_p=state.q_over_p,
        layer=layer,
        covariance=np.eye(5),
    )

    def propagation_function(parameters: np.ndarray) -> np.ndarray:
        return _propagate_bound_parameters_only(
            parameters=parameters,
            covariance=np.eye(5),
            surface_radius=state.radius,
            surface_name=state.surface_name,
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

    predicted_covariance = transport_jacobian @ state.covariance @ transport_jacobian.T

    if process_noise is not None:
        process_noise = np.asarray(process_noise, dtype=float)

        if process_noise.shape != (5, 5):
            raise ValueError("process_noise must have shape (5, 5)")

        if not np.allclose(process_noise, process_noise.T):
            raise ValueError("process_noise must be symmetric")

        predicted_covariance = predicted_covariance + process_noise

    predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)

    predicted_state = make_cylindrical_state(
        phi=predicted_state_without_covariance.phi,
        z=predicted_state_without_covariance.z,
        dir0=predicted_state_without_covariance.dir0,
        dir1=predicted_state_without_covariance.dir1,
        q_over_p=predicted_state_without_covariance.q_over_p,
        covariance=predicted_covariance,
        surface_radius=layer.radius,
        surface_name=layer.name,
    )

    return KalmanPredictionResult(
        predicted_state=predicted_state,
        transport_jacobian=transport_jacobian,
        propagation_result=propagation_result,
    )


def cylindrical_measurement_prediction(state: TrackState) -> np.ndarray:
    """
    Predict cylindrical measurement [phi, z].
    """

    if state.surface_type != "cylinder":
        raise ValueError("cylindrical measurement prediction requires cylindrical state")

    return np.array([state.phi, state.z], dtype=float)


def cylindrical_measurement_matrix() -> np.ndarray:
    """
    Measurement matrix for cylindrical local measurement [phi, z].

    Bound state:
        [phi, z, alpha, tan_lambda, q_over_p]
    """

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def cylindrical_full_residual(
    state: TrackState,
    measurement: Measurement,
) -> np.ndarray:
    """
    Return full cylindrical residual [delta_phi, delta_z].
    """

    if measurement.surface_type != "cylinder":
        raise ValueError("measurement must have surface_type='cylinder'")

    if measurement.dimension != 2:
        raise ValueError("cylindrical residual expects [phi, z]")

    prediction = cylindrical_measurement_prediction(state)

    residual = measurement.values - prediction
    residual[0] = wrap_angle(residual[0])

    return residual


def update_state(
    predicted_state: TrackState,
    measurement_values: np.ndarray,
    measurement_covariance: np.ndarray,
    measurement_matrix: np.ndarray,
    predicted_measurement: np.ndarray,
    layer_name: str,
    wrap_first_residual: bool = False,
    transport_jacobian: np.ndarray | None = None,
) -> KalmanUpdateResult:
    """
    Perform one Kalman update.

    residual = m - h(x)
    S = H C H^T + V
    K = C H^T S^-1
    x_new = x + K residual

    Joseph covariance form:
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

    # Keep angular parameters wrapped.
    filtered_parameters = filtered_parameters.copy()
    filtered_parameters[0] = wrap_angle(filtered_parameters[0])
    filtered_parameters[2] = wrap_angle(filtered_parameters[2])

    identity = np.eye(5)

    filtered_covariance = (
        (identity - kalman_gain @ h_matrix)
        @ covariance
        @ (identity - kalman_gain @ h_matrix).T
        + kalman_gain @ v_matrix @ kalman_gain.T
    )

    filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)

    chi2 = float(residual.T @ np.linalg.inv(residual_covariance) @ residual)

    filtered_state = make_cylindrical_state(
        phi=filtered_parameters[0],
        z=filtered_parameters[1],
        dir0=filtered_parameters[2],
        dir1=filtered_parameters[3],
        q_over_p=filtered_parameters[4],
        covariance=filtered_covariance,
        surface_radius=predicted_state.radius,
        surface_name=predicted_state.surface_name,
    )

    return KalmanUpdateResult(
        predicted_state=predicted_state,
        filtered_state=filtered_state,
        residual=residual,
        residual_covariance=residual_covariance,
        kalman_gain=kalman_gain,
        chi2=chi2,
        layer_name=layer_name,
        transport_jacobian=transport_jacobian,
    )


def update_with_cylindrical_measurement(
    predicted_state: TrackState,
    measurement: Measurement,
    transport_jacobian: np.ndarray | None = None,
) -> KalmanUpdateResult:
    """
    Update a predicted cylindrical TrackState using a [phi, z] measurement.
    """

    if predicted_state.surface_type != "cylinder":
        raise ValueError("predicted_state must be cylindrical")

    if measurement.surface_type != "cylinder":
        raise ValueError("measurement must have surface_type='cylinder'")

    if measurement.dimension != 2:
        raise ValueError("cylindrical measurement must have dimension 2: [phi, z]")

    return update_state(
        predicted_state=predicted_state,
        measurement_values=measurement.values,
        measurement_covariance=measurement.covariance,
        measurement_matrix=cylindrical_measurement_matrix(),
        predicted_measurement=cylindrical_measurement_prediction(predicted_state),
        layer_name=measurement.layer_name,
        wrap_first_residual=True,
        transport_jacobian=transport_jacobian,
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
        1. propagate current bound state to the cylindrical layer
        2. propagate covariance with numerical transport Jacobian
        3. update with local measurement [phi, z]
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
            transport_jacobian=prediction.transport_jacobian,
        )

        results.append(result)
        current_state = result.filtered_state

    return results


def total_chi2(results: Iterable[KalmanUpdateResult]) -> float:
    """
    Return total chi-square from a list of Kalman update results.
    """

    return float(sum(result.chi2 for result in results))
