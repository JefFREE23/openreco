from examples.v3_0_material_budget_scan import run_material_budget_scan


def test_v3_0_material_budget_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "material_budget_scan.csv"

    results = run_material_budget_scan(
        x_over_x0_values=(0.0, 0.01),
        n_events=2,
        n_particles=2,
        min_hits=4,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 2
    assert output_csv.exists()

    assert results[0].x_over_x0_per_layer == 0.0
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4

    assert results[1].x_over_x0_per_layer == 0.01
    assert results[1].events_processed == 2
    assert results[1].truth_particles_generated == 4