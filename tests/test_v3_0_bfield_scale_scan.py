from examples.v3_0_bfield_scale_scan import run_bfield_scale_scan


def test_v3_0_bfield_scale_scan_writes_csv(tmp_path):
    output_csv = tmp_path / "bfield_scale_scan.csv"

    results = run_bfield_scale_scan(
        truth_scale=1.0,
        reco_scales=(1.0, 1.05),
        n_events=2,
        n_particles=2,
        min_hits=4,
        seed=123,
        output_path=output_csv,
    )

    assert len(results) == 2
    assert output_csv.exists()

    assert results[0].truth_scale == 1.0
    assert results[0].reco_scale == 1.0
    assert results[0].scale_mismatch == 1.0
    assert results[0].events_processed == 2
    assert results[0].truth_particles_generated == 4

    assert results[1].truth_scale == 1.0
    assert results[1].reco_scale == 1.05
    assert results[1].scale_mismatch == 1.05
    assert results[1].events_processed == 2
    assert results[1].truth_particles_generated == 4