from pathlib import Path

from examples.multi_track_reconstruction import (
    format_demo_summary,
    run_multi_track_reconstruction,
)


def test_multi_track_reconstruction_runs_full_v1_chain(tmp_path: Path):
    plot_path = tmp_path / "v1_multi_track_event.png"

    result = run_multi_track_reconstruction(
        n_particles=3,
        hit_efficiency=1.0,
        noise_hits_per_layer=1,
        random_seed=123,
        make_plot=True,
        output_path=plot_path,
    )

    assert len(result.event.truth_particles) == 3
    assert len(result.event.measurements) == 24

    assert len(result.seeds) > 0
    assert len(result.tracks) == 3

    assert result.validation.tracking_efficiency == 1.0
    assert result.validation.fake_rate == 0.0
    assert result.validation.duplicate_rate == 0.0

    assert result.event.seeds == result.seeds
    assert result.event.reconstructed_tracks == result.tracks

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_multi_track_reconstruction_summary_contains_required_metrics(tmp_path: Path):
    result = run_multi_track_reconstruction(
        n_particles=3,
        hit_efficiency=1.0,
        noise_hits_per_layer=1,
        random_seed=123,
        make_plot=True,
        output_path=tmp_path / "event.png",
    )

    summary = format_demo_summary(result)

    assert "OpenReco v1 multi-track reconstruction" in summary
    assert "truth particles:" in summary
    assert "measurements:" in summary
    assert "seeds built:" in summary
    assert "reconstructed tracks:" in summary
    assert "matched tracks:" in summary
    assert "fake tracks:" in summary
    assert "duplicate tracks:" in summary
    assert "tracking efficiency:" in summary
    assert "fake rate:" in summary
    assert "duplicate rate:" in summary
    assert "mean chi2/ndof:" in summary
    assert "momentum rel error:" in summary
    assert "plot saved:" in summary
