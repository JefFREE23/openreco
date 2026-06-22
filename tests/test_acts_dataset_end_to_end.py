from pathlib import Path

from openreco.external.acts_loader import load_acts_dataset
from openreco.external.reconstruction import (
    event_hits_from_adapted_event,
    run_external_dataset_reconstruction,
    run_external_event_reconstruction,
)
from openreco.external.acts_adapter import convert_acts_event_to_openreco


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "acts_small"


def test_external_event_converts_to_event_hits():
    dataset = load_acts_dataset(DATASET_DIR)
    adapted = convert_acts_event_to_openreco(dataset.events[0])

    hits_by_layer = event_hits_from_adapted_event(adapted)

    assert len(hits_by_layer) == 6
    assert sum(len(hits) for hits in hits_by_layer.values()) == 13

    first_layer_hits = hits_by_layer["acts_layer_1"]
    assert first_layer_hits[0].layer_name == "acts_layer_1"
    assert first_layer_hits[0].radius == 10.0
    assert first_layer_hits[0].truth_particle_id == 1


def test_external_event_reconstruction_runs_without_crashing():
    dataset = load_acts_dataset(DATASET_DIR)

    result = run_external_event_reconstruction(
        dataset.events[0],
        seed_mode="hole-aware",
        min_hits=3,
        use_ekf_fit=False,
    )

    assert result.event_id == 0
    assert len(result.adapted_event.measurements) == 13
    assert len(result.measurements_by_layer) == 6
    assert len(result.seeds) >= 0
    assert len(result.tracks) >= 0
    assert result.runtime_seconds >= 0.0


def test_external_dataset_reconstruction_summary_runs():
    dataset = load_acts_dataset(DATASET_DIR)

    summary = run_external_dataset_reconstruction(
        dataset,
        seed_mode="hole-aware",
        min_hits=3,
        use_ekf_fit=False,
    )

    assert summary.n_events == 2
    assert summary.n_truth_particles == 3
    assert summary.n_measurements == 19
    assert summary.n_seeds >= 0
    assert summary.n_reco_tracks >= 0
    assert summary.runtime_per_event >= 0.0
