from examples.v3_0_hit_resolution_scan import run_hit_resolution_scan


def test_v3_0_hit_resolution_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "hit_resolution_scan.csv"

    results = run_hit_resolution_scan(
        sigma_phi_values=(0.001,),
        sigma_z_values=(0.10,),
        n_events=2,
        n_particles=2,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 1
    assert output_csv.exists()
    assert results[0].sigma_phi == 0.001
    assert results[0].sigma_z == 0.10
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4