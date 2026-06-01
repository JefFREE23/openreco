"""
Single-track uniform-B cylindrical tracker demo.

This is the main OpenReco v0 demo.

It uses:
    - cylindrical barrel detector layers
    - uniform magnetic field along z
    - one truth particle
    - truth propagation through cylindrical layers
    - smeared cylindrical measurements [phi, z]
    - truth-assisted initial seed
    - surface-bound cylindrical EKF Kalman filter
    - cylindrical detector visualization

This is still a minimal v0 reconstruction chain, not ACTS-level tracking.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from openreco.diagnostics import (
    covariance_is_valid,
    format_vector,
    kalman_pull_summary,
    momentum_summary,
)
from openreco.field import UniformMagneticField
from openreco.geometry import make_barrel_detector
from openreco.kalman import (
    cylindrical_full_residual,
    filter_cylindrical_track,
    make_process_noise,
    total_chi2,
)
from openreco.measurements import make_cylindrical_measurement
from openreco.particle_gun import make_fixed_particle
from openreco.propagation import propagate_to_barrel_detector, radial_distance
from openreco.state import make_cylindrical_state
from openreco.visualization import plot_cylindrical_track_event


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

    This is allowed for OpenReco v0.
    Real triplet seeding comes later.
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


def extract_state_positions(results):
    """
    Extract predicted and filtered 3D positions from Kalman results.
    """

    predicted = []
    filtered = []

    for result in results:
        predicted.append(result.predicted_state.global_position())
        filtered.append(result.filtered_state.global_position())

    return np.array(predicted), np.array(filtered)


def measurement_positions_from_hits(measurements, detector):
    """
    Convert cylindrical [phi, z] measurements back to approximate xyz points.

    The radius comes from the corresponding detector layer.
    """

    positions = []

    for measurement, layer in zip(measurements, detector.layers):
        phi, z = measurement.values

        x = layer.radius * np.cos(phi)
        y = layer.radius * np.sin(phi)

        positions.append([x, y, z])

    return np.array(positions)


def print_summary(
    truth_particle,
    detector,
    truth_results,
    measurements,
    kalman_results,
):
    """
    Print reconstruction summary.
    """

    print("OpenReco v0 uniform-B cylindrical demo")
    print("--------------------------------------")
    print(f"Number of cylindrical layers: {len(detector)}")
    print(f"Layer radii: {detector.radii}")
    print()
    print("Truth particle:")
    print(f"  position = {truth_particle.position}")
    print(f"  momentum = {truth_particle.momentum}")
    print(f"  charge   = {truth_particle.charge}")
    print(f"  p        = {truth_particle.p:.4f}")
    print(f"  pt       = {truth_particle.pt:.4f}")
    print(f"  q/p      = {truth_particle.q_over_p:.6f}")
    print()

    final_state = kalman_results[-1].filtered_state
    mom = momentum_summary(truth_particle, final_state)
    pulls = kalman_pull_summary(kalman_results)

    print("Final filtered bound state:")
    print(f"  surface  = {final_state.surface_name}")
    print(f"  radius   = {final_state.radius:.4f}")
    print(f"  phi      = {final_state.phi:.6f}")
    print(f"  z        = {final_state.z:.4f}")
    print(f"  alpha    = {final_state.dir0:.6f}")
    print(f"  tanλ     = {final_state.dir1:.6f}")
    print(f"  q/p      = {final_state.q_over_p:.6f}")
    print(f"  global x = {final_state.x:.4f}")
    print(f"  global y = {final_state.y:.4f}")
    print()

    print("Momentum estimate:")
    print(f"  truth p  = {mom.truth_p:.4f}")
    print(f"  fitted p = {mom.fitted_p:.4f} ± {mom.fitted_sigma_p:.4f}")
    print(f"  abs err  = {mom.absolute_error:.4f}")
    print(f"  rel err  = {mom.relative_error:.4f}")
    print()

    print("Fit quality:")
    print(f"  total chi2       = {total_chi2(kalman_results):.4f}")
    print(f"  n updates        = {len(kalman_results)}")
    print(f"  pull mean [φ,z]  = {format_vector(pulls.mean, precision=4)}")
    print(f"  pull std  [φ,z]  = {format_vector(pulls.std, precision=4)}")
    print(f"  covariance valid = {covariance_is_valid(final_state)}")
    print()

    print("Layer residuals:")
    print("  Kalman update uses local cylindrical measurement [phi, z].")

    for truth_result, measurement, kalman_result in zip(
        truth_results,
        measurements,
        kalman_results,
    ):
        filtered_state = kalman_result.filtered_state
        full_residual = cylindrical_full_residual(filtered_state, measurement)

        truth_r = radial_distance(truth_result.position)
        filtered_r = radial_distance(filtered_state.global_position())

        print(
            f"  {kalman_result.layer_name:8s} "
            f"truth_r={truth_r:8.4f} "
            f"filt_r={filtered_r:8.4f} "
            f"dphi={full_residual[0]: .6f} "
            f"dz={full_residual[1]: .6f} "
            f"chi2={kalman_result.chi2: .4f}"
        )


def main():
    rng = np.random.default_rng(123)

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

    truth_positions = np.array([result.position for result in truth_results])

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

    predicted_positions, filtered_positions = extract_state_positions(kalman_results)
    measured_positions = measurement_positions_from_hits(measurements, detector)

    print_summary(
        truth_particle=truth_particle,
        detector=detector,
        truth_results=truth_results,
        measurements=measurements,
        kalman_results=kalman_results,
    )

    fig, ax = plot_cylindrical_track_event(
        detector=detector,
        truth_positions=truth_positions,
        measured_positions=measured_positions,
        predicted_positions=predicted_positions,
        filtered_positions=filtered_positions,
        title="OpenReco v0: bound-state EKF in uniform B with cylindrical layers",
        show_detector=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
