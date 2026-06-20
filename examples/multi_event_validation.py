"""
Internal multi-event validation for the OpenReco v0 single-track chain.

The script repeats a fixed single-particle toy event with randomized
measurement smearing, then runs the cylindrical detector, EKF fit,
RTS smoothing, pull calculation, and momentum-resolution checks.

This is a self-consistency test, not an external ACTS/GenericDetector validation.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from openreco.diagnostics import (
    covariance_is_valid,
    kalman_pulls,
    momentum_summary,
)
from openreco.field import UniformMagneticField
from openreco.geometry import make_barrel_detector
from openreco.kalman import filter_cylindrical_track, make_process_noise
from openreco.measurements import make_cylindrical_measurement
from openreco.particle_gun import make_fixed_particle
from openreco.propagation import propagate_to_barrel_detector
from openreco.smoothing import smooth_track
from openreco.state import make_cylindrical_state


def make_truth_assisted_seed_from_first_layer(
    truth_particle,
    first_truth_result,
    first_layer,
    position_sigma_phi=0.002,
    position_sigma_z=0.2,
    direction_sigma=0.03,
    q_over_p_sigma=0.05,
):
    """
    Create a truth-assisted cylindrical bound seed on the first detector layer.
    """

    x, y, z = first_truth_result.position
    px, py, pz = first_truth_result.momentum

    pt = np.sqrt(px**2 + py**2)

    if pt <= 0.0:
        raise ValueError("truth particle pt must be positive")

    phi = np.arctan2(y, x)
    alpha = np.arctan2(py, px)
    tan_lambda = pz / pt
    q_over_p = truth_particle.q_over_p

    covariance = np.diag(
        [
            position_sigma_phi**2,
            position_sigma_z**2,
            direction_sigma**2,
            direction_sigma**2,
            q_over_p_sigma**2,
        ]
    )

    return make_cylindrical_state(
        phi=phi,
        z=z,
        dir0=alpha,
        dir1=tan_lambda,
        q_over_p=q_over_p,
        covariance=covariance,
        surface_radius=first_layer.radius,
        surface_name=first_layer.name,
    )


def run_one_event(rng):
    """
    Run one single-particle OpenReco v0 event.
    """

    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        half_length=120.0,
    )

    field = UniformMagneticField(bx=0.0, by=0.0, bz=2.0)

    truth_particle = make_fixed_particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([2.0, 0.25, 2.0]),
        charge=1.0,
    )

    truth_results = propagate_to_barrel_detector(
        particle=truth_particle,
        field=field,
        detector=detector,
        curvature_scale=0.003,
    )

    sigma_phi = 0.001
    sigma_z = 0.15

    measurements = [
        make_cylindrical_measurement(
            layer=layer,
            true_position=truth_result.position,
            sigma_phi=sigma_phi,
            sigma_z=sigma_z,
            rng=rng,
        )
        for layer, truth_result in zip(detector.layers, truth_results)
    ]

    initial_state = make_truth_assisted_seed_from_first_layer(
        truth_particle=truth_particle,
        first_truth_result=truth_results[0],
        first_layer=detector.layers[0],
        position_sigma_phi=0.002,
        position_sigma_z=0.2,
        direction_sigma=0.03,
        q_over_p_sigma=0.05,
    )

    process_noise = make_process_noise(
        [1e-6, 1e-4, 1e-6, 1e-6, 1e-7]
    )

    kalman_results = filter_cylindrical_track(
        initial_state=initial_state,
        measurements=measurements,
        detector=detector,
        field=field,
        process_noise=process_noise,
        curvature_scale=0.003,
    )

    smoothing_results = smooth_track(kalman_results)
    final_smoothed_state = smoothing_results[-1].smoothed_state

    pulls = kalman_pulls(kalman_results)
    momentum = momentum_summary(truth_particle, final_smoothed_state)

    return {
        "pulls": pulls,
        "momentum_error": momentum.absolute_error,
        "relative_momentum_error": momentum.relative_error,
        "covariance_valid": covariance_is_valid(final_smoothed_state),
        "success": True,
    }


def run_validation(n_events=200, seed=12345):
    """
    Run many single-track events and collect validation summaries.
    """

    rng = np.random.default_rng(seed)

    all_pulls = []
    momentum_errors = []
    relative_momentum_errors = []
    covariance_valid_flags = []
    n_success = 0

    for _ in range(n_events):
        try:
            result = run_one_event(rng)

            all_pulls.append(result["pulls"])
            momentum_errors.append(result["momentum_error"])
            relative_momentum_errors.append(result["relative_momentum_error"])
            covariance_valid_flags.append(result["covariance_valid"])
            n_success += 1

        except Exception as error:
            print(f"Event failed: {error}")

    if n_success == 0:
        raise RuntimeError("No events were successfully reconstructed")

    all_pulls = np.vstack(all_pulls)
    momentum_errors = np.asarray(momentum_errors, dtype=float)
    relative_momentum_errors = np.asarray(relative_momentum_errors, dtype=float)
    covariance_valid_flags = np.asarray(covariance_valid_flags, dtype=bool)

    return {
        "n_events": n_events,
        "n_success": n_success,
        "success_rate": n_success / n_events,
        "pull_mean": np.mean(all_pulls, axis=0),
        "pull_std": np.std(all_pulls, axis=0, ddof=0),
        "momentum_error_mean": float(np.mean(momentum_errors)),
        "momentum_error_std": float(np.std(momentum_errors, ddof=0)),
        "relative_momentum_error_mean": float(np.mean(relative_momentum_errors)),
        "relative_momentum_error_std": float(np.std(relative_momentum_errors, ddof=0)),
        "covariance_valid_rate": float(np.mean(covariance_valid_flags)),
    }


def main():
    summary = run_validation(n_events=200, seed=12345)

    print("OpenReco v0 multi-event validation")
    print("----------------------------------")
    print(f"events requested:       {summary['n_events']}")
    print(f"events successful:      {summary['n_success']}")
    print(f"success rate:           {summary['success_rate']:.4f}")
    print(f"covariance valid rate:  {summary['covariance_valid_rate']:.4f}")
    print()
    print("Kalman pull summary:")
    print(f"  phi mean:             {summary['pull_mean'][0]: .4f}")
    print(f"  phi std:              {summary['pull_std'][0]: .4f}")
    print(f"  z mean:               {summary['pull_mean'][1]: .4f}")
    print(f"  z std:                {summary['pull_std'][1]: .4f}")
    print()
    print("Momentum error summary:")
    print(f"  abs error mean:       {summary['momentum_error_mean']: .4f}")
    print(f"  abs error std:        {summary['momentum_error_std']: .4f}")
    print(f"  rel error mean:       {summary['relative_momentum_error_mean']: .4f}")
    print(f"  rel error std:        {summary['relative_momentum_error_std']: .4f}")


if __name__ == "__main__":
    main()
