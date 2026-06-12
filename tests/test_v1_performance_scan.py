from pathlib import Path

from examples.v1_performance_scan import (
    format_scan_results,
    run_performance_scan,
    save_scan_csv,
)


def test_v1_performance_scan_runs_one_small_scan_point():
    results = run_performance_scan(
        n_events=3,
        particle_counts=(2,),
        noise_hits_per_layer_values=(0,),
        hit_efficiencies=(1.0,),
        random_seed=123,
    )

    assert len(results) == 1

    result = results[0]

    assert result.n_particles == 2
    assert result.noise_hits_per_layer == 0
    assert result.hit_efficiency == 1.0
    assert result.events_processed == 3
    assert result.truth_particles_generated == 6

    assert result.measurements_mean == 12.0
    assert result.seeds_mean > 0
    assert result.reconstructed_tracks_mean == 2.0

    assert result.tracking_efficiency_mean == 1.0
    assert result.fake_rate_mean == 0.0
    assert result.duplicate_rate_mean == 0.0

    assert result.mean_hits_per_track == 6.0
    assert result.runtime_per_event_s > 0.0


def test_v1_performance_scan_summary_contains_required_metrics():
    results = run_performance_scan(
        n_events=2,
        particle_counts=(1,),
        noise_hits_per_layer_values=(0,),
        hit_efficiencies=(1.0,),
        random_seed=123,
    )

    summary = format_scan_results(results)

    assert "OpenReco v1 performance scan" in summary
    assert "n_particles" in summary
    assert "noise/layer" in summary
    assert "hit_eff" in summary
    assert "events" in summary
    assert "eff" in summary
    assert "fake" in summary
    assert "dup" in summary
    assert "seeds/event" in summary
    assert "tracks/event" in summary
    assert "runtime/event" in summary


def test_v1_performance_scan_can_save_csv(tmp_path: Path):
    results = run_performance_scan(
        n_events=2,
        particle_counts=(1,),
        noise_hits_per_layer_values=(0,),
        hit_efficiencies=(1.0,),
        random_seed=123,
    )

    output_path = tmp_path / "scan.csv"
    saved_path = save_scan_csv(results, output_path)

    assert saved_path == output_path
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    csv_text = saved_path.read_text(encoding="utf-8")

    assert "n_particles" in csv_text
    assert "tracking_efficiency_mean" in csv_text
    assert "runtime_per_event_s" in csv_text
