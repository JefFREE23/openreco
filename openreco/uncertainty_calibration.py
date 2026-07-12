"""Uncertainty-calibration helpers for OpenReco v3.0.

This module provides small reusable tools for detector-effects studies.

The central calibration idea is:

    residual / predicted_uncertainty = pull

For a well-calibrated uncertainty model, pull distributions should have:

    mean  ~ 0
    width ~ 1

Similarly, chi2/ndof should be close to 1 for a statistically consistent
fit model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from statistics import mean, pstdev
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PullSummary:
    """Summary of a pull distribution."""

    n: int
    mean: float
    width: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class Chi2NdoFSummary:
    """Summary of chi2/ndof values."""

    n: int
    mean: float
    width: float
    minimum: float
    maximum: float
    target: float
    mean_distance_from_target: float


@dataclass(frozen=True)
class CalibrationChoice:
    """Best calibration point from a scan."""

    scale: float
    metric_value: float
    target_value: float
    absolute_distance: float
    index: int


def _finite_values(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if isfinite(float(value))]


def compute_pull_values(
    residuals: Iterable[float],
    uncertainties: Iterable[float],
) -> np.ndarray:
    """Compute residual / uncertainty pull values.

    Parameters
    ----------
    residuals:
        Residual values.
    uncertainties:
        One-sigma uncertainty values associated with the residuals.

    Returns
    -------
    numpy.ndarray
        Pull values with the same length as the input arrays.
    """

    residual_array = np.asarray(list(residuals), dtype=float)
    uncertainty_array = np.asarray(list(uncertainties), dtype=float)

    if residual_array.shape != uncertainty_array.shape:
        raise ValueError("residuals and uncertainties must have the same shape")

    if not np.all(np.isfinite(residual_array)):
        raise ValueError("residuals must be finite")

    if not np.all(np.isfinite(uncertainty_array)):
        raise ValueError("uncertainties must be finite")

    if np.any(uncertainty_array <= 0.0):
        raise ValueError("uncertainties must be positive")

    return residual_array / uncertainty_array


def compute_pull_mean(pulls: Iterable[float]) -> float:
    """Return the finite-value mean of a pull distribution."""

    finite_pulls = _finite_values(pulls)
    return mean(finite_pulls) if finite_pulls else float("nan")


def compute_pull_width(pulls: Iterable[float]) -> float:
    """Return the finite-value population width of a pull distribution."""

    finite_pulls = _finite_values(pulls)
    return pstdev(finite_pulls) if len(finite_pulls) >= 2 else float("nan")


def compute_pull_summary(pulls: Iterable[float]) -> PullSummary:
    """Return n, mean, width, min, and max for a pull distribution."""

    finite_pulls = _finite_values(pulls)

    if not finite_pulls:
        return PullSummary(
            n=0,
            mean=float("nan"),
            width=float("nan"),
            minimum=float("nan"),
            maximum=float("nan"),
        )

    return PullSummary(
        n=len(finite_pulls),
        mean=mean(finite_pulls),
        width=pstdev(finite_pulls) if len(finite_pulls) >= 2 else float("nan"),
        minimum=min(finite_pulls),
        maximum=max(finite_pulls),
    )


def compute_chi2_ndof_summary(
    chi2_ndof_values: Iterable[float],
    *,
    target: float = 1.0,
) -> Chi2NdoFSummary:
    """Summarize chi2/ndof values relative to a target value."""

    if target <= 0.0:
        raise ValueError("target must be positive")

    finite_values = _finite_values(chi2_ndof_values)

    if not finite_values:
        return Chi2NdoFSummary(
            n=0,
            mean=float("nan"),
            width=float("nan"),
            minimum=float("nan"),
            maximum=float("nan"),
            target=float(target),
            mean_distance_from_target=float("nan"),
        )

    mean_value = mean(finite_values)

    return Chi2NdoFSummary(
        n=len(finite_values),
        mean=mean_value,
        width=pstdev(finite_values) if len(finite_values) >= 2 else float("nan"),
        minimum=min(finite_values),
        maximum=max(finite_values),
        target=float(target),
        mean_distance_from_target=abs(mean_value - target),
    )


def _value_from_point(point: Any, key: str) -> float:
    if isinstance(point, dict):
        value = point[key]
    else:
        value = getattr(point, key)

    return float(value)


def find_best_calibration_scale(
    scan_points: Iterable[Any],
    *,
    scale_key: str = "process_noise_scale",
    metric_key: str = "mean_chi2_ndof",
    target_value: float = 1.0,
) -> CalibrationChoice:
    """Find the scan point whose metric is closest to the target value.

    This works with either dictionaries or dataclass-like objects.

    Example
    -------
    For a process-noise scan, choose the scale whose mean chi2/ndof is closest
    to 1.
    """

    if target_value <= 0.0:
        raise ValueError("target_value must be positive")

    best_choice: CalibrationChoice | None = None

    for index, point in enumerate(scan_points):
        scale = _value_from_point(point, scale_key)
        metric_value = _value_from_point(point, metric_key)

        if not isfinite(scale) or not isfinite(metric_value):
            continue

        absolute_distance = abs(metric_value - target_value)

        candidate = CalibrationChoice(
            scale=scale,
            metric_value=metric_value,
            target_value=float(target_value),
            absolute_distance=absolute_distance,
            index=index,
        )

        if (
            best_choice is None
            or candidate.absolute_distance < best_choice.absolute_distance
        ):
            best_choice = candidate

    if best_choice is None:
        raise ValueError("no finite calibration points were provided")

    return best_choice


def scan_process_noise_scale(
    process_noise_scales: Iterable[float],
    evaluator: Callable[[float], Any],
) -> list[Any]:
    """Evaluate a callable at each process-noise scale.

    This helper is intentionally generic. The evaluator can return a dict,
    dataclass, or any result object.
    """

    results: list[Any] = []

    for process_noise_scale in process_noise_scales:
        scale = float(process_noise_scale)

        if not isfinite(scale):
            raise ValueError("process_noise_scale values must be finite")

        if scale < 0.0:
            raise ValueError("process_noise_scale values must be non-negative")

        results.append(evaluator(scale))

    return results