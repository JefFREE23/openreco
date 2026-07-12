from examples.v3_0_inefficiency_dead_layer_scan import (
    run_inefficiency_dead_layer_scan,
)


def test_v3_0_inefficiency_dead_layer_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "inefficiency_dead_layer_scan.csv"

    results = run_inefficiency_dead_layer_scan(
        hit_efficiency_values=(1.0,),
        dead_layer_scenarios=(
            ("none", ()),
            ("dead_2", (2,)),
        ),
        n_events=2,
        n_particles=2,
        noise_hits_per_layer=0,
        min_hits=4,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 2
    assert output_csv.exists()

    assert results[0].hit_efficiency == 1.0
    assert results[0].dead_layer_scenario == "none"
    assert results[0].dead_layers == "none"
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4

    assert results[1].dead_layer_scenario == "dead_2"
    assert results[1].dead_layers == "2"
    assert results[1].n_dead_layers == 1