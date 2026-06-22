from pathlib import Path

import pytest

from openreco.external.acts_loader import (
    group_by_event,
    load_acts_dataset,
    load_measurements,
    load_truth_particles,
)


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "acts_small"


def test_load_truth_particles():
    particles = load_truth_particles(DATASET_DIR / "truth_particles.csv")

    assert len(particles) == 3
    assert particles[0].event_id == 0
    assert particles[0].particle_id == 1
    assert particles[0].charge == 1
    assert particles[0].phi == pytest.approx(0.1)


def test_load_measurements():
    measurements = load_measurements(DATASET_DIR / "measurements.csv")

    assert len(measurements) == 19
    assert measurements[0].event_id == 0
    assert measurements[0].layer_id == 1
    assert measurements[0].r == pytest.approx(10.0)
    assert measurements[12].is_noise is True
    assert measurements[12].particle_id is None


def test_group_by_event():
    particles = load_truth_particles(DATASET_DIR / "truth_particles.csv")
    measurements = load_measurements(DATASET_DIR / "measurements.csv")

    events = group_by_event(particles, measurements)

    assert len(events) == 2
    assert events[0].event_id == 0
    assert len(events[0].truth_particles) == 2
    assert len(events[0].measurements) == 13

    assert events[1].event_id == 1
    assert len(events[1].truth_particles) == 1
    assert len(events[1].measurements) == 6


def test_load_acts_dataset():
    dataset = load_acts_dataset(DATASET_DIR)

    assert len(dataset.events) == 2
    assert dataset.metadata["n_truth_particles"] == 3
    assert dataset.metadata["n_measurements"] == 19


def test_missing_required_column_raises(tmp_path):
    bad_csv = tmp_path / "truth_particles.csv"
    bad_csv.write_text(
        "event_id,particle_id,charge\n"
        "0,1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_truth_particles(bad_csv)


def test_phi_outside_radian_range_raises(tmp_path):
    bad_csv = tmp_path / "truth_particles.csv"
    bad_csv.write_text(
        "event_id,particle_id,charge,x0,y0,z0,px,py,pz,pt,eta,phi\n"
        "0,1,1,0,0,0,1,0,0,1,0,4.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="phi must be in radians"):
        load_truth_particles(bad_csv)


def test_bad_radius_raises(tmp_path):
    bad_csv = tmp_path / "measurements.csv"
    bad_csv.write_text(
        "event_id,measurement_id,particle_id,layer_id,surface_id,"
        "x,y,z,r,phi,sigma_phi,sigma_z,is_noise\n"
        "0,1,1,1,101,10,0,0,99,0,0.001,0.1,False\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        load_measurements(bad_csv)


def test_negative_uncertainty_raises(tmp_path):
    bad_csv = tmp_path / "measurements.csv"
    bad_csv.write_text(
        "event_id,measurement_id,particle_id,layer_id,surface_id,"
        "x,y,z,r,phi,sigma_phi,sigma_z,is_noise\n"
        "0,1,1,1,101,10,0,0,10,0,-0.001,0.1,False\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sigma_phi must be positive"):
        load_measurements(bad_csv)
