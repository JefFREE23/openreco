from pathlib import Path

import pytest

from openreco.analysis.plotting import generate_standard_performance_plots

from openreco.analysis.performance import (
    PerformanceResult,
    PerformanceScanConfig,
    read_performance_results,
    write_performance_results,
)
from openreco.analysis.scans import default_v2_2_scan_grid, make_scan_grid


def test_performance_scan_config_validation_accepts_valid_config():
    config = PerformanceScanConfig(
        n_particles=5,
        noise_hits_per_layer=1,
        hit_efficiency=0.95,
        n_events=50,
        seed_mode="hole-aware",
    )

    assert config.n_particles == 5
    assert config.noise_hits_per_layer == 1
    assert config.hit_efficiency == 0.95
    assert config.seed_mode == "hole-aware"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_particles": 0},
        {"noise_hits_per_layer": -1},
        {"hit_efficiency": -0.1},
        {"hit_efficiency": 1.1},
        {"n_events": 0},
        {"seed_mode": "bad-mode"},
    ],
)
def test_performance_scan_config_rejects_invalid_values(kwargs):
    base = {
        "n_particles": 5,
        "noise_hits_per_layer": 1,
        "hit_efficiency": 0.95,
        "n_events": 50,
        "seed_mode": "hole-aware",
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        PerformanceScanConfig(**base)


def test_performance_result_csv_roundtrip(tmp_path: Path):
    result = PerformanceResult(
        n_particles=5,
        noise_hits_per_layer=1,
        hit_efficiency=1.0,
        events_processed=50,
        truth_particles_generated=250,
        measurements_mean=36.0,
        seeds_mean=216.0,
        reconstructed_tracks_mean=4.8,
        tracking_efficiency_mean=0.96,
        fake_rate_mean=0.0,
        duplicate_rate_mean=0.0,
        mean_hits_per_track=6.0,
        mean_holes_per_track=0.0,
        mean_chi2_ndof=0.955,
        covariance_valid_rate_mean=1.0,
        momentum_residual_mean=0.0033,
        momentum_residual_std=0.0659,
        runtime_total_s=106.14,
        runtime_per_event_s=2.1228,
    )

    output_path = tmp_path / "performance.csv"
    write_performance_results(output_path, [result])

    loaded = read_performance_results(output_path)

    assert loaded == [result]


def test_make_scan_grid_builds_expected_number_of_points():
    configs = make_scan_grid(
        n_particles=(1, 2),
        noise_hits_per_layer=(0, 1),
        hit_efficiencies=(1.0, 0.95),
        n_events=10,
    )

    assert len(configs) == 8
    assert all(config.n_events == 10 for config in configs)


def test_default_v2_2_scan_grid_matches_initial_scope():
    configs = default_v2_2_scan_grid()

    assert len(configs) == 12
    assert {config.n_particles for config in configs} == {1, 2, 5}
    assert {config.noise_hits_per_layer for config in configs} == {0, 1}
    assert {config.hit_efficiency for config in configs} == {1.0, 0.95}

from openreco.analysis.scans import run_and_write_tracking_performance_scan


def test_run_and_write_tracking_performance_scan_smoke(tmp_path: Path):
    output_path = tmp_path / "tracking_performance_summary.csv"

    configs = make_scan_grid(
        n_particles=(1,),
        noise_hits_per_layer=(0,),
        hit_efficiencies=(1.0,),
        n_events=2,
        seed=12345,
        seed_mode="hole-aware",
    )

    results = run_and_write_tracking_performance_scan(
        output_path=output_path,
        configs=configs,
    )

    assert output_path.exists()
    assert len(results) == 1

    result = results[0]
    assert result.n_particles == 1
    assert result.noise_hits_per_layer == 0
    assert result.hit_efficiency == 1.0
    assert result.events_processed == 2
    assert 0.0 <= result.tracking_efficiency_mean <= 1.0
    assert 0.0 <= result.fake_rate_mean <= 1.0
    assert 0.0 <= result.duplicate_rate_mean <= 1.0

def test_generate_standard_performance_plots(tmp_path: Path):
    results = [
        PerformanceResult(
            n_particles=1,
            noise_hits_per_layer=0,
            hit_efficiency=1.0,
            events_processed=2,
            truth_particles_generated=2,
            measurements_mean=6.0,
            seeds_mean=20.0,
            reconstructed_tracks_mean=1.0,
            tracking_efficiency_mean=1.0,
            fake_rate_mean=0.0,
            duplicate_rate_mean=0.0,
            mean_hits_per_track=6.0,
            mean_holes_per_track=0.0,
            mean_chi2_ndof=1.0,
            covariance_valid_rate_mean=1.0,
            momentum_residual_mean=0.0,
            momentum_residual_std=0.05,
            runtime_total_s=1.0,
            runtime_per_event_s=0.5,
        ),
        PerformanceResult(
            n_particles=1,
            noise_hits_per_layer=1,
            hit_efficiency=0.95,
            events_processed=2,
            truth_particles_generated=2,
            measurements_mean=12.0,
            seeds_mean=160.0,
            reconstructed_tracks_mean=0.9,
            tracking_efficiency_mean=0.9,
            fake_rate_mean=0.0,
            duplicate_rate_mean=0.0,
            mean_hits_per_track=5.8,
            mean_holes_per_track=0.2,
            mean_chi2_ndof=1.1,
            covariance_valid_rate_mean=1.0,
            momentum_residual_mean=0.0,
            momentum_residual_std=0.06,
            runtime_total_s=1.2,
            runtime_per_event_s=0.6,
        ),
    ]

    figure_dir = tmp_path / "figures"
    plots = generate_standard_performance_plots(results, figure_dir)

    expected_names = {
        "efficiency_vs_hit_efficiency",
        "fake_rate_vs_noise",
        "duplicate_rate_vs_noise",
        "chi2_vs_hit_efficiency",
        "momentum_resolution_vs_hit_efficiency",
        "runtime_vs_occupancy",
    }

    assert set(plots) == expected_names

    for path in plots.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0