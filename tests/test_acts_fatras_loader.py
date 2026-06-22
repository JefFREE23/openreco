from pathlib import Path

import pytest

from openreco.external.acts_fatras_loader import (
    find_fatras_event_ids,
    load_fatras_dataset,
    load_fatras_event,
)
from openreco.external.reconstruction import run_external_dataset_reconstruction


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "acts_fatras_sample"


def test_find_fatras_event_ids():
    event_ids = find_fatras_event_ids(DATASET_DIR)

    assert event_ids == [0]


def test_load_fatras_event():
    event = load_fatras_event(DATASET_DIR, 0)

    assert event.event_id == 0
    assert len(event.truth_particles) == 1
    assert len(event.measurements) == 11

    truth = event.truth_particles[0]
    assert truth.charge == -1
    assert truth.pt > 0
    assert -3.1416 <= truth.phi <= 3.1416

    first = event.measurements[0]
    assert first.event_id == 0
    assert first.measurement_id == 1
    assert first.particle_id == truth.particle_id
    assert first.layer_id == 1
    assert first.r > 0
    assert first.sigma_phi == pytest.approx(1.0e-3)
    assert first.sigma_z == pytest.approx(0.10)
    assert first.is_noise is False


def test_load_fatras_dataset():
    dataset = load_fatras_dataset(DATASET_DIR)

    assert len(dataset.events) == 1
    assert dataset.metadata["format"] == "acts_fatras"
    assert dataset.metadata["n_events"] == 1
    assert dataset.metadata["n_truth_particles"] == 1
    assert dataset.metadata["n_measurements"] == 11


def test_fatras_dataset_runs_external_reconstruction_without_crashing():
    dataset = load_fatras_dataset(DATASET_DIR)

    summary = run_external_dataset_reconstruction(
        dataset,
        seed_mode="hole-aware",
        min_hits=5,
        use_ekf_fit=False,
    )

    assert summary.n_events == 1
    assert summary.n_truth_particles == 1
    assert summary.n_measurements == 11
    assert summary.n_seeds >= 0
    assert summary.n_reco_tracks >= 0
