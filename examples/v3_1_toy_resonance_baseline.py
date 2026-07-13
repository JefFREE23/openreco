"""OpenReco v3.1 toy resonance baseline example.

This example demonstrates the first downstream physics-observable workflow:

1. generate a toy J/psi -> mu+ mu- decay,
2. pass the two truth daughters into the OpenReco event generator,
3. reconstruct the two charged tracks,
4. select the best opposite-charge track pair,
5. compute the reconstructed invariant mass.

This is a controlled toy benchmark, not a realistic muon analysis.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.multi_track_reconstruction import run_multi_track_reconstruction
from openreco.invariant_mass import JPSI_MASS_GEV
from openreco.resonance import ToyResonanceDecay, generate_two_body_resonance_decay
from openreco.resonance_reconstruction import (
    ReconstructedResonanceCandidate,
    build_opposite_charge_mass_candidates,
    select_best_mass_candidate,
)


@dataclass(frozen=True)
class ToyResonanceBaselineResult:
    """Result container for the v3.1 toy resonance baseline example."""

    decay: ToyResonanceDecay
    reconstruction_result: object
    candidates: list[ReconstructedResonanceCandidate]
    best_candidate: ReconstructedResonanceCandidate | None

    @property
    def truth_mass(self) -> float:
        return self.decay.resonance_mass

    @property
    def reconstructed_mass(self) -> float:
        if self.best_candidate is None:
            return float("nan")

        return self.best_candidate.mass

    @property
    def mass_residual(self) -> float:
        if self.best_candidate is None:
            return float("nan")

        return self.best_candidate.mass_residual


def run_toy_resonance_baseline(
    *,
    random_seed: int = 123,
    event_id: int = 0,
    use_ekf_fit: bool = False,
) -> ToyResonanceBaselineResult:
    """Run a deterministic toy J/psi -> mu+ mu- reconstruction example."""

    decay = generate_two_body_resonance_decay(
        rng=np.random.default_rng(random_seed),
    )

    reconstruction_result = run_multi_track_reconstruction(
        event_id=event_id,
        n_particles=999,
        truth_particles=decay.truth_particles,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        random_seed=random_seed + 1,
        chi2_threshold=100.0,
        min_hits=6,
        use_ekf_fit=use_ekf_fit,
        make_plot=False,
    )

    candidates = build_opposite_charge_mass_candidates(
        reconstruction_result.tracks,
        truth_mass=JPSI_MASS_GEV,
    )

    best_candidate = select_best_mass_candidate(
        reconstruction_result.tracks,
        truth_mass=JPSI_MASS_GEV,
    )

    return ToyResonanceBaselineResult(
        decay=decay,
        reconstruction_result=reconstruction_result,
        candidates=candidates,
        best_candidate=best_candidate,
    )


def main() -> None:
    result = run_toy_resonance_baseline()

    validation = result.reconstruction_result.validation

    print("OpenReco v3.1 toy resonance baseline")
    print("--------------------------------------")
    print(f"Truth channel: J/psi -> mu+ mu-")
    print(f"Truth resonance mass: {result.truth_mass:.6f} GeV")
    print(f"Truth invariant mass from daughters: {result.decay.truth_invariant_mass:.6f} GeV")
    print(f"Truth daughters: {len(result.decay.truth_particles)}")
    print(f"Reconstructed tracks: {len(result.reconstruction_result.tracks)}")
    print(f"Matched tracks: {validation.n_matched_tracks}")
    print(f"Tracking efficiency: {validation.tracking_efficiency:.6f}")
    print(f"Opposite-charge candidates: {len(result.candidates)}")

    if result.best_candidate is None:
        print("Best reconstructed mass: nan")
        print("Mass residual: nan")
        return

    print(f"Best reconstructed mass: {result.reconstructed_mass:.6f} GeV")
    print(f"Mass residual: {result.mass_residual:+.6f} GeV")
    print(f"Best pair track indices: {result.best_candidate.track_indices}")
    print(f"Best pair charges: {result.best_candidate.charges}")


if __name__ == "__main__":
    main()