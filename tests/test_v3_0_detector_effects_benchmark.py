import csv

from examples.v3_0_detector_effects_benchmark import (
    run_detector_effects_benchmark,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_v3_0_detector_effects_benchmark_writes_summary_files(tmp_path):
    input_dir = tmp_path / "v3_0_detector_effects"
    summary_csv = input_dir / "detector_effects_benchmark_summary.csv"
    report_md = input_dir / "detector_effects_benchmark_report.md"

    _write_csv(
        input_dir / "hit_resolution_scan.csv",
        [
            {
                "sigma_phi": "0.001",
                "sigma_z": "0.1",
                "tracking_efficiency_mean": "1.0",
                "fake_rate_mean": "0.0",
                "duplicate_rate_mean": "0.0",
                "mean_chi2_ndof": "1.0",
                "momentum_residual_mean": "0.0",
                "momentum_residual_std": "0.05",
                "momentum_uncertainty_mean": "0.2",
            },
            {
                "sigma_phi": "0.005",
                "sigma_z": "0.5",
                "tracking_efficiency_mean": "0.9",
                "fake_rate_mean": "0.0",
                "duplicate_rate_mean": "0.0",
                "mean_chi2_ndof": "1.2",
                "momentum_residual_mean": "0.01",
                "momentum_residual_std": "0.2",
                "momentum_uncertainty_mean": "0.3",
            },
        ],
    )

    _write_csv(
        input_dir / "process_noise_scan.csv",
        [
            {
                "process_noise_scale": "0.0",
                "x_over_x0_per_layer": "0.02",
                "tracking_efficiency_mean": "1.0",
                "fake_rate_mean": "0.0",
                "duplicate_rate_mean": "0.0",
                "mean_chi2_ndof": "1.4",
                "momentum_residual_mean": "0.0",
                "momentum_residual_std": "0.08",
                "momentum_uncertainty_mean": "0.2",
            },
            {
                "process_noise_scale": "5.0",
                "x_over_x0_per_layer": "0.02",
                "tracking_efficiency_mean": "1.0",
                "fake_rate_mean": "0.0",
                "duplicate_rate_mean": "0.0",
                "mean_chi2_ndof": "1.02",
                "momentum_residual_mean": "0.0",
                "momentum_residual_std": "0.08",
                "momentum_uncertainty_mean": "0.4",
            },
            {
                "process_noise_scale": "20.0",
                "x_over_x0_per_layer": "0.02",
                "tracking_efficiency_mean": "1.0",
                "fake_rate_mean": "0.0",
                "duplicate_rate_mean": "0.0",
                "mean_chi2_ndof": "0.7",
                "momentum_residual_mean": "0.0",
                "momentum_residual_std": "0.09",
                "momentum_uncertainty_mean": "1.0",
            },
        ],
    )

    rows = run_detector_effects_benchmark(
        input_dir=input_dir,
        summary_csv=summary_csv,
        report_md=report_md,
    )

    assert len(rows) == 2
    assert summary_csv.exists()
    assert report_md.exists()

    assert rows[0].study == "hit_resolution"
    assert rows[1].study == "process_noise_calibration"
    assert "process_noise_scale=5.0" in rows[1].comparison_parameters

    report_text = report_md.read_text(encoding="utf-8")
    assert "OpenReco v3.0 Detector-Effects Benchmark" in report_text
    assert "process_noise_calibration" in report_text