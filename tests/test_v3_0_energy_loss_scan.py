from examples.v3_0_energy_loss_scan import run_energy_loss_scan


def test_v3_0_energy_loss_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "energy_loss_scan.csv"

    results = run_energy_loss_scan(
        energy_loss_mev_values=(0.0, 10.0),
        n_events=2,
        n_particles=2,
        min_hits=4,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 2
    assert output_csv.exists()

    assert results[0].energy_loss_mev_per_layer == 0.0
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4

    assert results[1].energy_loss_mev_per_layer == 10.0
    assert results[1].events_processed == 2
    assert results[1].truth_particles_generated == 4