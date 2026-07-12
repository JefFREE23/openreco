from dataclasses import dataclass

import numpy as np

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.invariant_mass import JPSI_MASS_GEV, MUON_MASS_GEV
from openreco.resonance import (
    generate_two_body_resonance_decay,
    two_body_momentum_magnitude,
)
from openreco.resonance_reconstruction import (
    build_opposite_charge_mass_candidates,
    charge_from_track,
    momentum_vector_from_track,
    select_best_mass_candidate,
)


@dataclass(frozen=True)
class _MockSeed:
    alpha: float


@dataclass(frozen=True)
class _MockTrack:
    q_over_p: float
    pt_estimate: float
    p_estimate: float
    tan_lambda: float
    seed: _MockSeed
    smoothed_states: tuple = ()
    filtered_states: tuple = ()


def test_charge_from_track_uses_q_over_p_sign():
    positive = _MockTrack(
        q_over_p=0.5,
        pt_estimate=1.0,
        p_estimate=2.0,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=0.0),
    )
    negative = _MockTrack(
        q_over_p=-0.5,
        pt_estimate=1.0,
        p_estimate=2.0,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=0.0),
    )

    assert charge_from_track(positive) == 1
    assert charge_from_track(negative) == -1


def test_momentum_vector_from_track_uses_pt_direction_and_tan_lambda():
    track = _MockTrack(
        q_over_p=0.5,
        pt_estimate=2.0,
        p_estimate=np.sqrt(5.0),
        tan_lambda=0.5,
        seed=_MockSeed(alpha=0.0),
    )

    momentum = momentum_vector_from_track(track)

    assert np.allclose(momentum, np.array([2.0, 0.0, 1.0]))


def test_opposite_charge_tracks_reconstruct_jpsi_mass():
    daughter_p = two_body_momentum_magnitude(
        parent_mass=JPSI_MASS_GEV,
        daughter_mass_1=MUON_MASS_GEV,
        daughter_mass_2=MUON_MASS_GEV,
    )

    positive = _MockTrack(
        q_over_p=1.0 / daughter_p,
        pt_estimate=daughter_p,
        p_estimate=daughter_p,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=0.0),
    )
    negative = _MockTrack(
        q_over_p=-1.0 / daughter_p,
        pt_estimate=daughter_p,
        p_estimate=daughter_p,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=np.pi),
    )

    candidates = build_opposite_charge_mass_candidates([positive, negative])

    assert len(candidates) == 1
    assert np.isclose(candidates[0].mass, JPSI_MASS_GEV)
    assert np.isclose(candidates[0].mass_residual, 0.0)


def test_same_charge_tracks_do_not_form_candidate():
    first = _MockTrack(
        q_over_p=0.5,
        pt_estimate=1.0,
        p_estimate=1.0,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=0.0),
    )
    second = _MockTrack(
        q_over_p=0.4,
        pt_estimate=1.0,
        p_estimate=1.0,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=np.pi),
    )

    candidates = build_opposite_charge_mass_candidates([first, second])

    assert candidates == []


def test_select_best_mass_candidate_returns_none_when_no_pair():
    track = _MockTrack(
        q_over_p=0.5,
        pt_estimate=1.0,
        p_estimate=1.0,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=0.0),
    )

    assert select_best_mass_candidate([track]) is None


def test_select_best_mass_candidate_uses_closest_mass():
    daughter_p = two_body_momentum_magnitude(
        parent_mass=JPSI_MASS_GEV,
        daughter_mass_1=MUON_MASS_GEV,
        daughter_mass_2=MUON_MASS_GEV,
    )

    positive = _MockTrack(
        q_over_p=1.0 / daughter_p,
        pt_estimate=daughter_p,
        p_estimate=daughter_p,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=0.0),
    )
    good_negative = _MockTrack(
        q_over_p=-1.0 / daughter_p,
        pt_estimate=daughter_p,
        p_estimate=daughter_p,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=np.pi),
    )
    bad_negative = _MockTrack(
        q_over_p=-1.0 / (0.5 * daughter_p),
        pt_estimate=0.5 * daughter_p,
        p_estimate=0.5 * daughter_p,
        tan_lambda=0.0,
        seed=_MockSeed(alpha=np.pi),
    )

    candidate = select_best_mass_candidate(
        [positive, bad_negative, good_negative],
    )

    assert candidate is not None
    assert candidate.track_indices == (0, 2)
    assert abs(candidate.mass_residual) < 1.0e-12


def test_actual_resonance_reconstruction_produces_mass_candidate():
    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(123),
    )

    result = run_multi_track_reconstruction(
        event_id=21,
        n_particles=999,
        truth_particles=decay.truth_particles,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=456,
        chi2_threshold=100.0,
        min_hits=6,
        use_ekf_fit=False,
        make_plot=False,
    )

    candidate = select_best_mass_candidate(result.tracks)

    assert candidate is not None
    assert np.isfinite(candidate.mass)
    assert candidate.mass > 0.0
    assert abs(candidate.mass_residual) < 1.0