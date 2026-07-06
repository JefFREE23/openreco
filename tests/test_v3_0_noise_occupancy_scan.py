from examples.v3_0_noise_occupancy_scan import run_noise_occupancy_scan


def test_v3_0_noise_occupancy_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "noise_occupancy_scan.csv"

    results = run_noise_occupancy_scan(
        mean_noise_hits_per_layer_values=(0.0, 1.0),
        n_events=2,
        n_particles=2,
        hit_efficiency=1.0,
        min_hits=4,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 2
    assert output_csv.exists()

    assert results[0].mean_noise_hits_per_layer == 0.0
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4

    assert results[1].mean_noise_hits_per_layer == 1.0
    assert results[1].noise_hits_mean >= 0.0