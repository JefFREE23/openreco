import numpy as np

from openreco.diagnostics import covariance_is_valid, kalman_pulls, momentum_summary
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
):
    x, y, z = first_truth_result.position
    px, py, pz = first_truth_result.momentum

    pt = np.sqrt(px**2 + py**2)

    phi = np.arctan2(y, x)
    alpha = np.arctan2(py, px)
    tan_lambda = pz / pt
    q_over_p = truth_particle.q_over_p

    covariance = np.diag(
        [
            0.002**2,
            0.2**2,
            0.03**2,
            0.03**2,
            0.05**2,
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


def run_one_test_event(seed=123):
    rng = np.random.default_rng(seed)

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

    measurements = [
        make_cylindrical_measurement(
            layer=layer,
            true_position=truth_result.position,
            sigma_phi=0.001,
            sigma_z=0.15,
            rng=rng,
        )
        for layer, truth_result in zip(detector.layers, truth_results)
    ]

    initial_state = make_truth_assisted_seed_from_first_layer(
        truth_particle=truth_particle,
        first_truth_result=truth_results[0],
        first_layer=detector.layers[0],
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

    return truth_particle, kalman_results, smoothing_results


def test_end_to_end_single_track_runs():
    truth_particle, kalman_results, smoothing_results = run_one_test_event()

    assert len(kalman_results) == 6
    assert len(smoothing_results) == 6

    final_state = smoothing_results[-1].smoothed_state

    assert final_state.surface_type == "cylinder"
    assert final_state.surface_name == "barrel_5"
    assert covariance_is_valid(final_state)


def test_end_to_end_pulls_are_finite():
    _, kalman_results, _ = run_one_test_event()

    pulls = kalman_pulls(kalman_results)

    assert pulls.shape == (6, 2)
    assert np.all(np.isfinite(pulls))


def test_end_to_end_momentum_estimate_is_reasonable():
    truth_particle, _, smoothing_results = run_one_test_event()

    final_state = smoothing_results[-1].smoothed_state
    momentum = momentum_summary(truth_particle, final_state)

    assert np.isfinite(momentum.fitted_p)
    assert np.isfinite(momentum.fitted_sigma_p)
    assert momentum.fitted_sigma_p >= 0.0

    # Loose sanity check for the current toy-unit v0 setup.
    assert abs(momentum.relative_error) < 0.25


def test_end_to_end_multiple_events_are_stable():
    relative_errors = []

    for seed in range(10):
        truth_particle, kalman_results, smoothing_results = run_one_test_event(seed=seed)

        pulls = kalman_pulls(kalman_results)
        final_state = smoothing_results[-1].smoothed_state
        momentum = momentum_summary(truth_particle, final_state)

        assert np.all(np.isfinite(pulls))
        assert covariance_is_valid(final_state)

        relative_errors.append(momentum.relative_error)

    relative_errors = np.asarray(relative_errors)

    assert np.all(np.isfinite(relative_errors))
    assert abs(np.mean(relative_errors)) < 0.25
