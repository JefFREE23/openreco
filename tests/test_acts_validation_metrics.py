from pathlib import Path

from openreco.external.acts_export import generate_openreco_acts_dataset
from openreco.external.acts_loader import load_acts_dataset
from openreco.external.reconstruction import run_external_dataset_reconstruction
from openreco.validation.external_metrics import (
    compute_external_validation_metrics,
    event_summary_rows,
    track_summary_rows,
)
from openreco.validation.report import write_external_validation_report


def test_external_validation_metrics_are_bounded(tmp_path):
    generate_openreco_acts_dataset(
        tmp_path,
        n_events=1,
        n_particles=3,
        noise_hits_per_layer=1,
        random_seed=123,
    )

    dataset = load_acts_dataset(tmp_path)

    summary = run_external_dataset_reconstruction(
        dataset,
        seed_mode="hole-aware",
        min_hits=3,
        use_ekf_fit=False,
    )

    metrics = compute_external_validation_metrics(summary)

    assert metrics.n_events == 1
    assert metrics.n_truth_particles == 3
    assert 0.0 <= metrics.unique_tracking_efficiency <= 1.0
    assert metrics.raw_matched_track_efficiency >= 0.0
    assert 0.0 <= metrics.fake_rate <= 1.0
    assert 0.0 <= metrics.duplicate_rate <= 1.0


def test_event_and_track_summary_rows(tmp_path):
    generate_openreco_acts_dataset(
        tmp_path,
        n_events=1,
        n_particles=3,
        noise_hits_per_layer=0,
        random_seed=123,
    )

    dataset = load_acts_dataset(tmp_path)

    summary = run_external_dataset_reconstruction(
        dataset,
        seed_mode="hole-aware",
        min_hits=3,
        use_ekf_fit=False,
    )

    event_rows = event_summary_rows(summary)
    track_rows = track_summary_rows(summary)

    assert len(event_rows) == 1
    assert event_rows[0]["event_id"] == 0
    assert event_rows[0]["n_truth_particles"] == 3
    assert "unique_tracking_efficiency" in event_rows[0]

    assert len(track_rows) == summary.n_reco_tracks

    if track_rows:
        assert "momentum_relative_residual" in track_rows[0]
        assert "matched_truth_particle_id" in track_rows[0]


def test_write_external_validation_report(tmp_path):
    dataset_dir = tmp_path / "dataset"
    report_dir = tmp_path / "report"

    generate_openreco_acts_dataset(
        dataset_dir,
        n_events=1,
        n_particles=3,
        noise_hits_per_layer=0,
        random_seed=123,
    )

    dataset = load_acts_dataset(dataset_dir)

    summary = run_external_dataset_reconstruction(
        dataset,
        seed_mode="hole-aware",
        min_hits=3,
        use_ekf_fit=False,
    )

    paths = write_external_validation_report(
        summary,
        report_dir,
        make_plots=True,
    )

    assert paths.summary_csv.exists()
    assert paths.tracks_csv.exists()
    assert paths.efficiency_plot is not None
    assert paths.efficiency_plot.exists()
    assert paths.momentum_residual_plot is not None
    assert paths.momentum_residual_plot.exists()
