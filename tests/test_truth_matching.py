from dataclasses import dataclass

from openreco.event_generation import generate_event
from openreco.truth_matching import (
    mark_duplicates,
    match_hits_to_truth,
    match_object_to_truth,
    validate_reconstructed_tracks,
)


@dataclass
class DummyTrack:
    track_id: int
    used_measurements: list


def test_track_with_five_of_six_hits_from_one_particle_is_matched():
    event = generate_event(
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
    )

    hits_particle_0 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 0
    ]

    hits_particle_1 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 1
    ]

    track_hits = hits_particle_0[:5] + hits_particle_1[:1]

    result = match_hits_to_truth(track_hits, object_id=7)

    assert result.object_id == 7
    assert result.n_hits == 6
    assert result.is_matched is True
    assert result.is_fake is False
    assert result.matched_truth_particle_id == 0
    assert result.matched_fraction == 5 / 6


def test_mixed_random_hits_below_threshold_are_fake():
    event = generate_event(
        n_particles=3,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
    )

    hits_particle_0 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 0
    ]

    hits_particle_1 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 1
    ]

    hits_particle_2 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 2
    ]

    mixed_hits = hits_particle_0[:2] + hits_particle_1[:2] + hits_particle_2[:2]

    result = match_hits_to_truth(mixed_hits, object_id=3, min_fraction=0.50)

    assert result.n_hits == 6
    assert result.is_matched is False
    assert result.is_fake is True
    assert result.matched_truth_particle_id is None


def test_two_tracks_matched_to_same_truth_particle_create_duplicate():
    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
    )

    hits = event.measurements

    track_a = DummyTrack(track_id=0, used_measurements=hits[:6])
    track_b = DummyTrack(track_id=1, used_measurements=hits[:6])

    match_a = match_object_to_truth(track_a)
    match_b = match_object_to_truth(track_b)

    marked = mark_duplicates([match_a, match_b])

    assert marked[0].is_matched is True
    assert marked[0].is_duplicate is False

    assert marked[1].is_matched is True
    assert marked[1].is_duplicate is True


def test_validate_reconstructed_tracks_reports_efficiency_fake_and_duplicate_rates():
    event = generate_event(
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
    )

    hits_particle_0 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 0
    ]

    hits_particle_1 = [
        hit for hit in event.measurements
        if hit.truth_particle_id == 1
    ]

    good_track_0 = DummyTrack(track_id=0, used_measurements=hits_particle_0[:6])
    duplicate_track_0 = DummyTrack(track_id=1, used_measurements=hits_particle_0[:6])
    good_track_1 = DummyTrack(track_id=2, used_measurements=hits_particle_1[:6])

    fake_track = DummyTrack(
        track_id=3,
        used_measurements=hits_particle_0[:2] + hits_particle_1[:2],
    )

    summary = validate_reconstructed_tracks(
        [good_track_0, duplicate_track_0, good_track_1, fake_track],
        n_truth_particles=2,
    )

    assert summary.n_truth_particles == 2
    assert summary.n_reconstructed_tracks == 4
    assert summary.n_matched_tracks == 3
    assert summary.n_fake_tracks == 1
    assert summary.n_duplicate_tracks == 1

    assert summary.tracking_efficiency == 1.0
    assert summary.fake_rate == 0.25
    assert summary.duplicate_rate == 0.25
