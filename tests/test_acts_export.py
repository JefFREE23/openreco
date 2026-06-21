from pathlib import Path

from openreco.external.acts_export import generate_openreco_acts_dataset
from openreco.external.acts_loader import load_acts_dataset
from openreco.external.reconstruction import run_external_dataset_reconstruction


def test_generate_openreco_acts_dataset_writes_files(tmp_path):
    metadata = generate_openreco_acts_dataset(
        tmp_path,
        n_events=2,
        n_particles=3,
        noise_hits_per_layer=0,
        random_seed=123,
    )

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "truth_particles.csv").exists()
    assert (tmp_path / "measurements.csv").exists()

    assert metadata["n_events"] == 2
    assert metadata["n_truth_particles"] == 6
    assert metadata["n_measurements"] > 0


def test_generated_openreco_acts_dataset_loads_back(tmp_path):
    generate_openreco_acts_dataset(
        tmp_path,
        n_events=2,
        n_particles=3,
        noise_hits_per_layer=0,
        random_seed=123,
    )

    dataset = load_acts_dataset(tmp_path)

    assert len(dataset.events) == 2
    assert dataset.metadata["n_truth_particles"] == 6
    assert dataset.metadata["n_measurements"] > 0


def test_generated_openreco_acts_dataset_runs_external_reconstruction_with_ekf(tmp_path):
    generate_openreco_acts_dataset(
        tmp_path,
        n_events=1,
        n_particles=3,
        noise_hits_per_layer=0,
        random_seed=123,
    )

    dataset = load_acts_dataset(tmp_path)

    summary = run_external_dataset_reconstruction(
        dataset,
        seed_mode="hole-aware",
        min_hits=5,
        use_ekf_fit=True,
        max_fit_chi2_ndof=100.0,
    )

    assert summary.n_events == 1
    assert summary.n_truth_particles == 3
    assert summary.n_reco_tracks > 0
    assert summary.n_matched_tracks > 0
