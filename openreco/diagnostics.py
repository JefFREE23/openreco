"""
Diagnostics tools for OpenReco.

This module provides roadmap-safe diagnostics for the OpenReco v0 tracking
chain.

Main diagnostics:
    - Kalman residuals
    - Kalman pulls using residual covariance S = H P H^T + R
    - chi-square summaries
    - momentum error
    - covariance checks

Important:
    For proper Kalman residual pulls, use the residual covariance stored in
    KalmanUpdateResult, not only the raw measurement covariance.

Current v0 note:
    The cylindrical Kalman update currently uses phi only.
    The full [dphi, dz] residual is still useful for diagnostics, but z is not
    part of the current 5D covariance update.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from openreco.kalman import KalmanUpdateResult, cylindrical_full_residual
from openreco.measurements import Measurement
from openreco.particle_gun import Particle
from openreco.state import TrackState


@dataclass(frozen=True)
class VectorSummary:
    """
    Summary statistics for a vector-valued quantity.
    """

    mean: np.ndarray
    std: np.ndarray
    rmse: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        std = np.asarray(self.std, dtype=float)
        rmse = np.asarray(self.rmse, dtype=float)

        if mean.ndim != 1:
            raise ValueError("mean must be a 1D vector")

        if std.shape != mean.shape:
            raise ValueError("std must have same shape as mean")

        if rmse.shape != mean.shape:
            raise ValueError("rmse must have same shape as mean")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "rmse", rmse)


@dataclass(frozen=True)
class MomentumSummary:
    """
    Momentum estimate summary.
    """

    truth_p: float
    fitted_p: float
    absolute_error: float
    relative_error: float

    def __post_init__(self):
        if self.truth_p <= 0.0:
            raise ValueError("truth_p must be positive")

        if self.fitted_p <= 0.0:
            raise ValueError("fitted_p must be positive")

        object.__setattr__(self, "truth_p", float(self.truth_p))
        object.__setattr__(self, "fitted_p", float(self.fitted_p))
        object.__setattr__(self, "absolute_error", float(self.absolute_error))
        object.__setattr__(self, "relative_error", float(self.relative_error))


def summarize_vectors(values: np.ndarray) -> VectorSummary:
    """
    Compute mean, standard deviation, and RMSE.

    Parameters
    ----------
    values:
        Array with shape (n_items, n_dimensions).
    """

    values = np.asarray(values, dtype=float)

    if values.ndim != 2:
        raise ValueError("values must have shape (n_items, n_dimensions)")

    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    rmse = np.sqrt(np.mean(values**2, axis=0))

    return VectorSummary(mean=mean, std=std, rmse=rmse)


def kalman_residuals(results: Iterable[KalmanUpdateResult]) -> np.ndarray:
    """
    Return Kalman residuals from update results.

    These residuals correspond to the measurement dimensions actually used
    in the Kalman update.

    For the current cylindrical v0 update, this is [dphi].
    """

    results = list(results)

    if len(results) == 0:
        raise ValueError("results must not be empty")

    return np.asarray([result.residual for result in results], dtype=float)


def kalman_residual_variances(results: Iterable[KalmanUpdateResult]) -> np.ndarray:
    """
    Return diagonal residual variances from Kalman residual covariance.

    This uses S = H P H^T + R from each KalmanUpdateResult.
    """

    results = list(results)

    if len(results) == 0:
        raise ValueError("results must not be empty")

    variances = []

    for result in results:
        diagonal = np.diag(result.residual_covariance)

        if np.any(diagonal <= 0.0):
            raise ValueError("residual covariance diagonal must be positive")

        variances.append(diagonal)

    return np.asarray(variances, dtype=float)


def kalman_pulls(results: Iterable[KalmanUpdateResult]) -> np.ndarray:
    """
    Compute Kalman residual pulls.

    Correct pull definition for Kalman residuals:

        pull = residual / sqrt(diag(S))

    where:

        S = H P H^T + R

    The residual covariance S is stored in KalmanUpdateResult.
    """

    results = list(results)

    residuals = kalman_residuals(results)
    variances = kalman_residual_variances(results)

    if residuals.shape != variances.shape:
        raise ValueError("residuals and variances must have the same shape")

    return residuals / np.sqrt(variances)


def kalman_residual_summary(results: Iterable[KalmanUpdateResult]) -> VectorSummary:
    """
    Summarize Kalman residuals.
    """

    return summarize_vectors(kalman_residuals(results))


def kalman_pull_summary(results: Iterable[KalmanUpdateResult]) -> VectorSummary:
    """
    Summarize Kalman pulls.

    Ideally, pulls should have:
        mean close to 0
        std close to 1
    """

    return summarize_vectors(kalman_pulls(results))


def chi2_values(results: Iterable[KalmanUpdateResult]) -> np.ndarray:
    """
    Return chi-square values for each Kalman update.
    """

    results = list(results)

    if len(results) == 0:
        raise ValueError("results must not be empty")

    return np.array([result.chi2 for result in results], dtype=float)


def total_chi2(results: Iterable[KalmanUpdateResult]) -> float:
    """
    Return total chi-square from Kalman results.
    """

    return float(np.sum(chi2_values(results)))


def reduced_chi2(
    results: Iterable[KalmanUpdateResult],
    n_degrees_of_freedom: int,
) -> float:
    """
    Return reduced chi-square.
    """

    if n_degrees_of_freedom <= 0:
        raise ValueError("n_degrees_of_freedom must be positive")

    return total_chi2(results) / float(n_degrees_of_freedom)


def momentum_from_state(state: TrackState) -> float:
    """
    Estimate momentum magnitude from q_over_p.

    p = 1 / abs(q_over_p)

    This assumes the current OpenReco v0 toy-unit convention is used
    consistently.
    """

    if np.isclose(state.q_over_p, 0.0):
        raise ValueError("cannot estimate momentum from q_over_p close to zero")

    return float(1.0 / abs(state.q_over_p))


def momentum_summary(
    truth_particle: Particle,
    fitted_state: TrackState,
) -> MomentumSummary:
    """
    Compare truth momentum with fitted momentum.
    """

    truth_p = truth_particle.p
    fitted_p = momentum_from_state(fitted_state)

    absolute_error = fitted_p - truth_p
    relative_error = absolute_error / truth_p

    return MomentumSummary(
        truth_p=truth_p,
        fitted_p=fitted_p,
        absolute_error=absolute_error,
        relative_error=relative_error,
    )


def covariance_diagonal(state: TrackState) -> np.ndarray:
    """
    Return covariance diagonal.
    """

    return np.diag(state.covariance).copy()


def covariance_standard_deviations(state: TrackState) -> np.ndarray:
    """
    Return standard deviations from covariance diagonal.
    """

    diagonal = covariance_diagonal(state)

    if np.any(diagonal < 0.0):
        raise ValueError("covariance diagonal contains negative values")

    return np.sqrt(diagonal)


def covariance_eigenvalues(state: TrackState) -> np.ndarray:
    """
    Return covariance eigenvalues for a symmetric covariance matrix.
    """

    if not covariance_is_symmetric(state):
        raise ValueError("covariance must be symmetric before eigenvalue check")

    return np.linalg.eigvalsh(state.covariance)


def covariance_is_symmetric(state: TrackState, atol: float = 1e-10) -> bool:
    """
    Check whether state covariance is symmetric.
    """

    return bool(np.allclose(state.covariance, state.covariance.T, atol=atol))


def covariance_has_nonnegative_diagonal(state: TrackState) -> bool:
    """
    Check whether covariance diagonal entries are non-negative.
    """

    return bool(np.all(np.diag(state.covariance) >= 0.0))


def covariance_is_positive_semidefinite(
    state: TrackState,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check whether covariance matrix is positive semi-definite.

    Tiny negative eigenvalues are allowed within tolerance because of
    floating-point arithmetic.
    """

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    eigenvalues = covariance_eigenvalues(state)

    return bool(np.all(eigenvalues >= -tolerance))


def covariance_is_valid(
    state: TrackState,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check basic covariance validity.

    A valid covariance should be:
        - symmetric
        - have non-negative diagonal
        - be positive semi-definite
    """

    return (
        covariance_is_symmetric(state)
        and covariance_has_nonnegative_diagonal(state)
        and covariance_is_positive_semidefinite(state, tolerance=tolerance)
    )


def cylindrical_diagnostic_residuals(
    results: Iterable[KalmanUpdateResult],
    measurements: Iterable[Measurement],
) -> np.ndarray:
    """
    Compute full cylindrical diagnostic residuals [dphi, dz].

    Important:
        In the current v0 EKF, only dphi is used in the Kalman update.
        dz is diagnostic only because z is not part of the current 5D
        covariance update.
    """

    results = list(results)
    measurements = list(measurements)

    if len(results) == 0:
        raise ValueError("results must not be empty")

    if len(results) != len(measurements):
        raise ValueError("results and measurements must have the same length")

    residuals = [
        cylindrical_full_residual(result.filtered_state, measurement)
        for result, measurement in zip(results, measurements)
    ]

    return np.asarray(residuals, dtype=float)


def measurement_covariance_pulls(
    residuals: np.ndarray,
    measurements: Iterable[Measurement],
) -> np.ndarray:
    """
    Compute residual pulls using raw measurement covariance only.

    This is useful for simple hit-level diagnostics, but it is NOT the main
    Kalman residual pull. For Kalman pulls, use kalman_pulls(results).
    """

    measurements = list(measurements)
    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (n_measurements, n_dimensions)")

    if len(measurements) != residuals.shape[0]:
        raise ValueError("number of measurements must match residual rows")

    variances = np.array(
        [np.diag(measurement.covariance) for measurement in measurements],
        dtype=float,
    )

    if residuals.shape != variances.shape:
        raise ValueError("residual dimension must match measurement covariance")

    if np.any(variances <= 0.0):
        raise ValueError("measurement covariance diagonal must be positive")

    return residuals / np.sqrt(variances)


def format_vector(values: np.ndarray, precision: int = 4) -> str:
    """
    Format a vector for compact printing.
    """

    values = np.asarray(values, dtype=float)

    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"
