import numpy as np

from openreco.event_generation import count_noise_hits, count_real_hits, generate_event
from openreco.resonance import generate_two_body_resonance_decay


def test_generate_event_accepts_resonance_truth_particles():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    event = generate_event(
        event_id=7,
        n_particles=999,
        truth_particles=decay.truth_particles,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=np.random.default_rng(456),
    )

    assert event.event_id == 7
    assert len(event.truth_particles) == 2
    assert count_real_hits(event) == 12
    assert count_noise_hits(event) == 0

    truth_ids = {particle.truth_particle_id for particle in decay.truth_particles}
    hit_truth_ids = {
        hit.truth_particle_id
        for hit in event.measurements
        if not hit.is_noise
    }

    assert truth_ids == {0, 1}
    assert hit_truth_ids == truth_ids


def test_generate_event_custom_truth_particles_ignore_n_particles_argument():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    event = generate_event(
        event_id=0,
        n_particles=100,
        truth_particles=decay.truth_particles,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=np.random.default_rng(456),
    )

    assert len(event.truth_particles) == 2
    assert count_real_hits(event) == 12


def test_generate_event_custom_truth_particles_reject_wrong_type():
    try:
        generate_event(
            event_id=0,
            truth_particles=("not-a-truth-particle",),  # type: ignore[arg-type]
            rng=np.random.default_rng(123),
        )
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for invalid truth particle")