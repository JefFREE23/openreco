from dataclasses import dataclass
from math import isnan

import numpy as np

from openreco.uncertainty_calibration import (
    compute_chi2_ndof_summary,
    compute_pull_mean,
    compute_pull_summary,
    compute_pull_values,
    compute_pull_width,
    find_best_calibration_scale,
    scan_process_noise_scale,
)


def test_compute_pull_values_and_summary():
    pulls = compute_pull_values(
        residuals=(-1.0, 0.0, 1.0),
        uncertainties=(1.0, 1.0, 1.0),
    )

    assert np.allclose(pulls, np.array([-1.0, 0.0, 1.0]))

    summary = compute_pull_summary(pulls)

    assert summary.n == 3
    assert summary.mean == 0.0
    assert np.isclose(summary.width, np.sqrt(2.0 / 3.0))
    assert summary.minimum == -1.0
    assert summary.maximum == 1.0


def test_compute_pull_mean_and_width_ignore_nonfinite_values():
    pulls = [-1.0, 0.0, 1.0, float("nan")]

    assert compute_pull_mean(pulls) == 0.0
    assert np.isclose(compute_pull_width(pulls), np.sqrt(2.0 / 3.0))


def test_compute_pull_values_rejects_invalid_inputs():
    try:
        compute_pull_values(
            residuals=(1.0, 2.0),
            uncertainties=(1.0,),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")

    try:
        compute_pull_values(
            residuals=(1.0,),
            uncertainties=(0.0,),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive uncertainty")


def test_compute_pull_summary_empty_returns_nan_values():
    summary = compute_pull_summary([])

    assert summary.n == 0
    assert isnan(summary.mean)
    assert isnan(summary.width)
    assert isnan(summary.minimum)
    assert isnan(summary.maximum)


def test_compute_chi2_ndof_summary():
    summary = compute_chi2_ndof_summary([0.8, 1.0, 1.2])

    assert summary.n == 3
    assert np.isclose(summary.mean, 1.0)
    assert np.isclose(summary.width, np.sqrt(0.08 / 3.0))
    assert summary.minimum == 0.8
    assert summary.maximum == 1.2
    assert summary.target == 1.0
    assert summary.mean_distance_from_target == 0.0


def test_find_best_calibration_scale_from_dicts():
    points = [
        {"process_noise_scale": 0.0, "mean_chi2_ndof": 1.40},
        {"process_noise_scale": 2.0, "mean_chi2_ndof": 1.12},
        {"process_noise_scale": 5.0, "mean_chi2_ndof": 0.98},
        {"process_noise_scale": 10.0, "mean_chi2_ndof": 0.70},
    ]

    choice = find_best_calibration_scale(points)

    assert choice.scale == 5.0
    assert choice.metric_value == 0.98
    assert choice.target_value == 1.0
    assert np.isclose(choice.absolute_distance, 0.02)
    assert choice.index == 2


def test_find_best_calibration_scale_from_objects():
    @dataclass(frozen=True)
    class Point:
        process_noise_scale: float
        mean_chi2_ndof: float

    points = [
        Point(process_noise_scale=0.0, mean_chi2_ndof=1.5),
        Point(process_noise_scale=1.0, mean_chi2_ndof=1.1),
        Point(process_noise_scale=2.0, mean_chi2_ndof=0.95),
    ]

    choice = find_best_calibration_scale(points)

    assert choice.scale == 2.0
    assert choice.metric_value == 0.95
    assert np.isclose(choice.absolute_distance, 0.05)


def test_scan_process_noise_scale_calls_evaluator():
    results = scan_process_noise_scale(
        process_noise_scales=(0.0, 2.0, 4.0),
        evaluator=lambda scale: {
            "process_noise_scale": scale,
            "mean_chi2_ndof": 1.0 / (1.0 + scale),
        },
    )

    assert len(results) == 3
    assert results[0]["process_noise_scale"] == 0.0
    assert results[1]["process_noise_scale"] == 2.0
    assert results[2]["process_noise_scale"] == 4.0