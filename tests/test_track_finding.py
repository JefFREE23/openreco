import numpy as np

from openreco.event_generation import generate_event
from openreco.seeding import build_triplet_seeds
from openreco.track_finding import find_tracks_from_seeds
from openreco.truth_matching import validate_reconstructed_tracks


def test_clean_single_particle_event_reconstructs_one_track():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-3,
        measurement_sigma_z=0.10,
        pt_range=(2.0, 2.0),
        tan_lambda_range=(0.2, 0.2),
        charge_choices=(1,),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)
    tracks = find_tracks_from_seeds(
        seeds,
        event.measurements_by_layer,
        chi2_threshold=25.0,
        min_hits=6,
    )

    assert len(tracks) == 1
    assert len(tracks[0].used_measurements) == 6
    assert tracks[0].fit_status == "accepted"

    summary = validate_reconstructed_tracks(
        tracks,
        n_truth_particles=len(event.truth_particles),
    )

    assert summary.tracking_efficiency == 1.0
    assert summary.fake_rate == 0.0
    assert summary.duplicate_rate == 0.0


def test_clean_two_particle_event_reconstructs_two_tracks():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-3,
        measurement_sigma_z=0.10,
        pt_range=(2.0, 3.0),
        tan_lambda_range=(-0.5, 0.5),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)
    tracks = find_tracks_from_seeds(
        seeds,
        event.measurements_by_layer,
        chi2_threshold=25.0,
        min_hits=6,
    )

    summary = validate_reconstructed_tracks(
        tracks,
        n_truth_particles=len(event.truth_particles),
    )

    matched_truth_ids = {
        match.matched_truth_particle_id
        for match in summary.matches
        if match.is_matched
    }

    assert len(tracks) == 2
    assert matched_truth_ids == {0, 1}
    assert summary.tracking_efficiency == 1.0
    assert summary.fake_rate == 0.0
    assert summary.duplicate_rate == 0.0


def test_noise_hits_do_not_dominate_when_chi2_cut_is_reasonable():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=3,
        measurement_sigma_phi=1.0e-3,
        measurement_sigma_z=0.10,
        pt_range=(2.0, 2.0),
        tan_lambda_range=(0.2, 0.2),
        charge_choices=(1,),
        rng=rng,
    )

    seeds = build_triplet_seeds(event.measurements_by_layer)
    tracks = find_tracks_from_seeds(
        seeds,
        event.measurements_by_layer,
        chi2_threshold=25.0,
        min_hits=6,
        max_tracks=1,
    )

    summary = validate_reconstructed_tracks(
        tracks,
        n_truth_particles=len(event.truth_particles),
    )

    assert len(tracks) == 1
    assert summary.tracking_efficiency == 1.0
    assert summary.fake_rate == 0.0
    assert summary.duplicate_rate == 0.0

    assert all(
        hit.truth_particle_id == 0
        for hit in tracks[0].used_measurements
    )
