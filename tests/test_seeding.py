import numpy as np

from openreco.event_generation import generate_event
from openreco.seeding import build_triplet_seeds


def test_single_clean_particle_gives_at_least_one_triplet_seed():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-6,
        measurement_sigma_z=1.0e-6,
        pt_range=(2.0, 2.0),
        tan_lambda_range=(0.2, 0.2),
        charge_choices=(1,),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)

    assert len(seeds) >= 1


def test_seed_uses_three_hits_from_three_different_layers():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-6,
        measurement_sigma_z=1.0e-6,
        pt_range=(2.0, 2.0),
        charge_choices=(1,),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)
    seed = seeds[0]

    assert len(seed.hit_ids) == 3
    assert len(set(seed.hit_ids)) == 3
    assert seed.layer_names == ("barrel_0", "barrel_1", "barrel_2")
    assert len(set(seed.layer_names)) == 3


def test_seed_q_over_p_has_correct_positive_charge_sign():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-6,
        measurement_sigma_z=1.0e-6,
        pt_range=(2.0, 2.0),
        charge_choices=(1,),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)
    seed = seeds[0]

    assert seed.q_over_p > 0.0


def test_seed_q_over_p_has_correct_negative_charge_sign():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-6,
        measurement_sigma_z=1.0e-6,
        pt_range=(2.0, 2.0),
        charge_choices=(-1,),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)
    seed = seeds[0]

    assert seed.q_over_p < 0.0


def test_two_clean_particles_build_multiple_triplet_candidates():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-6,
        measurement_sigma_z=1.0e-6,
        pt_range=(2.0, 3.0),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)

    assert len(seeds) >= 2

    truth_matched_seeds = [seed for seed in seeds if seed.truth_particle_id is not None]
    matched_truth_ids = {seed.truth_particle_id for seed in truth_matched_seeds}

    assert matched_truth_ids == {0, 1}
