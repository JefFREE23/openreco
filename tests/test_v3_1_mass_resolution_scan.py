import csv

import numpy as np

from examples.v3_1_mass_resolution_scan import (
    MassResolutionScanResult,
    detector_effects_for_scan_point,
    run_mass_resolution_scan,
    run_single_mass_resolution_point,
)


def test_detector_effects_for_mass_resolution_scan_point():
    baseline = detector_effects_for_scan_point(
        study_name="baseline",
        scan_value=0.0,
    )
    hit_resolution = detector_effects_for_scan_point(
        study_name="hit_resolution_scale",
        scan_value=2.0,
    )
    material = detector_effects_for_scan_point(
        study_name="material_budget",
        scan_value=0.01,
    )
    bfield = detector_effects_for_scan_point(
        study_name="bfield_reco_scale",
        scan_value=1.02,
    )

    assert baseline.hit_resolution.sigma_phi == 1.0e-3
    assert np.isclose(hit_resolution.hit_resolution.sigma_phi, 2.0e-3)
    assert np.isclose(hit_resolution.hit_resolution.sigma_z, 0.20)
    assert np.isclose(material.material_for_layer(0).x_over_x0, 0.01)
    assert np.isclose(bfield.b_field_scale.reco_scale, 1.02)


def test_run_single_mass_resolution_point_produces_candidate():
    result = run_single_mass_resolution_point(
        study_name="baseline",
        scan_value=0.0,
        n_events=3,
        seed=123,
        min_hits=6,
    )

    assert isinstance(result, MassResolutionScanResult)
    assert result.n_events == 3
    assert result.events_with_candidate >= 1
    assert result.candidate_efficiency > 0.0
    assert np.isfinite(result.mass_mean)
    assert np.isfinite(result.residual_mean)


def test_run_mass_resolution_scan_writes_csv(tmp_path):
    output_path = tmp_path / "mass_resolution_scan.csv"

    results = run_mass_resolution_scan(
        scan_points=(
            ("baseline", 0.0),
            ("hit_resolution_scale", 2.0),
            ("material_budget", 0.01),
            ("bfield_reco_scale", 1.02),
        ),
        n_events=3,
        seed=123,
        min_hits=6,
        output_path=output_path,
    )

    assert len(results) == 4
    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["study_name"] == "baseline"
    assert "mass_width" in rows[0]
    assert "residual_width" in rows[0]