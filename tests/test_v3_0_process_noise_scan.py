from examples.v3_0_process_noise_scan import run_process_noise_scan


def test_v3_0_process_noise_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "process_noise_scan.csv"

    results = run_process_noise_scan(
        process_noise_scales=(0.0, 10.0),
        x_over_x0_per_layer=0.02,
        n_events=2,
        n_particles=2,
        min_hits=4,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 2
    assert output_csv.exists()

    assert results[0].process_noise_scale == 0.0
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4

    assert results[1].process_noise_scale == 10.0
    assert results[1].events_processed == 2
    assert results[1].truth_particles_generated == 4