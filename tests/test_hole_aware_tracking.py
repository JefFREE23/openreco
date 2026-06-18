import numpy as np

from openreco.event_generation import generate_event
from openreco.seeding import build_triplet_seeds, build_triplet_seeds_for_layer_sets
from openreco.track_finding import find_tracks_from_seeds
from examples.multi_track_reconstruction import run_multi_track_reconstruction
from examples.v1_performance_scan import run_performance_scan, format_scan_results


def _event_with_missing_layer(layer_name: str = "barrel_1"):
    rng = np.random.default_rng(123)

    event = generate_event(
        event_id=0,
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        measurement_sigma_phi=1.0e-3,
        measurement_sigma_z=0.10,
        pt_range=(2.0, 5.0),
        tan_lambda_range=(-0.8, 0.8),
        rng=rng,
    )

    event.measurements_by_layer[layer_name] = []

    return event


def test_strict_first_three_layer_seeding_fails_when_inner_layer_is_missing():
    event = _event_with_missing_layer("barrel_1")

    seeds = build_triplet_seeds(event.measurements_by_layer)
    tracks = find_tracks_from_seeds(
        seeds,
        event.measurements_by_layer,
        min_hits=5,
        max_tracks=1,
    )

    assert seeds == []
    assert tracks == []


def test_hole_aware_seeding_recovers_track_with_one_missing_layer():
    event = _event_with_missing_layer("barrel_1")

    seeds = build_triplet_seeds_for_layer_sets(event.measurements_by_layer)
    tracks = find_tracks_from_seeds(
        seeds,
        event.measurements_by_layer,
        min_hits=5,
        max_tracks=1,
    )

    assert len(seeds) > 0
    assert len(tracks) == 1

    track = tracks[0]

    assert len(track.used_measurements) == 5
    assert track.n_holes == 1
    assert track.missing_layer_names == ("barrel_1",)


def test_multi_track_demo_supports_hole_aware_seed_mode():
    result = run_multi_track_reconstruction(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=123,
        min_hits=5,
        seed_mode="hole-aware",
        make_plot=False,
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].n_holes == 0


def test_performance_scan_reports_mean_holes_per_track():
    results = run_performance_scan(
        n_events=2,
        particle_counts=(1,),
        noise_hits_per_layer_values=(0,),
        hit_efficiencies=(1.0,),
        random_seed=123,
        min_hits=5,
        seed_mode="hole-aware",
    )

    assert len(results) == 1
    assert hasattr(results[0], "mean_holes_per_track")
    assert results[0].mean_holes_per_track >= 0.0

    summary = format_scan_results(results)

    assert "holes/track" in summary
