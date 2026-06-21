"""Export OpenReco-generated events into ACTS-style CSV files.

This is a Stage-B validation helper for OpenReco v2.

It lets us generate physically consistent OpenReco v1 events, write them
as external ACTS-style CSV files, reload them through the v2 loader, and
run the external reconstruction pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from openreco.event_generation import Event, EventHit, TruthParticle, generate_event


TRUTH_COLUMNS = [
    "event_id",
    "particle_id",
    "charge",
    "x0",
    "y0",
    "z0",
    "px",
    "py",
    "pz",
    "pt",
    "eta",
    "phi",
]

MEASUREMENT_COLUMNS = [
    "event_id",
    "measurement_id",
    "particle_id",
    "layer_id",
    "surface_id",
    "x",
    "y",
    "z",
    "r",
    "phi",
    "sigma_phi",
    "sigma_z",
    "is_noise",
]


def generate_openreco_acts_dataset(
    output_dir: str | Path,
    *,
    n_events: int = 5,
    n_particles: int = 5,
    hit_efficiency: float = 1.0,
    noise_hits_per_layer: int = 1,
    random_seed: int = 123,
    measurement_sigma_phi: float = 1.0e-3,
    measurement_sigma_z: float = 0.10,
    pt_range: tuple[float, float] = (2.0, 5.0),
    tan_lambda_range: tuple[float, float] = (-0.8, 0.8),
) -> dict[str, Any]:
    """Generate OpenReco events and export them as ACTS-style CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    events = [
        generate_event(
            event_id=event_id,
            n_particles=n_particles,
            hit_efficiency=hit_efficiency,
            noise_hits_per_layer=noise_hits_per_layer,
            measurement_sigma_phi=measurement_sigma_phi,
            measurement_sigma_z=measurement_sigma_z,
            pt_range=pt_range,
            tan_lambda_range=tan_lambda_range,
            rng=rng,
        )
        for event_id in range(n_events)
    ]

    truth_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []

    for event in events:
        truth_rows.extend(_truth_rows_from_event(event))
        measurement_rows.extend(_measurement_rows_from_event(event))

    _write_csv(output_path / "truth_particles.csv", TRUTH_COLUMNS, truth_rows)
    _write_csv(output_path / "measurements.csv", MEASUREMENT_COLUMNS, measurement_rows)
    _write_readme(output_path)

    return {
        "output_dir": str(output_path),
        "n_events": len(events),
        "n_truth_particles": len(truth_rows),
        "n_measurements": len(measurement_rows),
    }


def _truth_rows_from_event(event: Event) -> list[dict[str, Any]]:
    rows = []

    for particle in event.truth_particles:
        rows.append(_truth_particle_to_row(event.event_id, particle))

    return rows


def _truth_particle_to_row(event_id: int, particle: TruthParticle) -> dict[str, Any]:
    pt = float(particle.pt)
    phi = float(particle.phi)
    tan_lambda = float(particle.tan_lambda)

    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * tan_lambda

    eta = math.asinh(tan_lambda)

    return {
        "event_id": event_id,
        "particle_id": particle.truth_particle_id,
        "charge": particle.charge,
        "x0": _fmt(0.0),
        "y0": _fmt(0.0),
        "z0": _fmt(particle.z0),
        "px": _fmt(px),
        "py": _fmt(py),
        "pz": _fmt(pz),
        "pt": _fmt(pt),
        "eta": _fmt(eta),
        "phi": _fmt(phi),
    }


def _measurement_rows_from_event(event: Event) -> list[dict[str, Any]]:
    rows = []

    hits = _flatten_hits(event)

    for hit in hits:
        rows.append(_event_hit_to_measurement_row(event.event_id, hit))

    return rows


def _flatten_hits(event: Event) -> list[EventHit]:
    hits: list[EventHit] = []

    for layer_name in sorted(
        event.measurements_by_layer,
        key=lambda name: min(hit.layer_index for hit in event.measurements_by_layer[name]),
    ):
        hits.extend(sorted(event.measurements_by_layer[layer_name], key=lambda hit: hit.hit_id))

    return hits


def _event_hit_to_measurement_row(event_id: int, hit: EventHit) -> dict[str, Any]:
    layer_id = int(hit.layer_index) + 1
    surface_id = 1000 + layer_id

    radius = float(hit.radius)
    phi = float(hit.phi)
    z = float(hit.z)

    x = radius * math.cos(phi)
    y = radius * math.sin(phi)

    covariance = np.asarray(hit.covariance, dtype=float)

    if covariance.shape != (2, 2):
        raise ValueError(f"Hit {hit.hit_id} covariance must have shape (2, 2).")

    sigma_phi = math.sqrt(float(covariance[0, 0]))
    sigma_z = math.sqrt(float(covariance[1, 1]))

    particle_id = "" if hit.truth_particle_id is None else int(hit.truth_particle_id)

    return {
        "event_id": event_id,
        "measurement_id": hit.hit_id,
        "particle_id": particle_id,
        "layer_id": layer_id,
        "surface_id": surface_id,
        "x": _fmt(x),
        "y": _fmt(y),
        "z": _fmt(z),
        "r": _fmt(radius),
        "phi": _fmt(phi),
        "sigma_phi": _fmt(sigma_phi),
        "sigma_z": _fmt(sigma_z),
        "is_noise": "True" if hit.is_noise else "False",
    }


def _write_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_path: Path) -> None:
    readme = """# acts_openreco_generated

ACTS-style external validation dataset generated from OpenReco v1 events.

This dataset is used as Stage-B validation for OpenReco v2:

OpenReco v1 event generation
→ ACTS-style CSV export
→ OpenReco v2 external loader
→ OpenReco v2 adapter
→ OpenReco v1 reconstruction chain

This is not a real ACTS C++ export. It is a physically compatible external-format
dataset used to validate the v2 file interface and reconstruction path.
"""

    (output_path / "README.md").write_text(readme, encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export OpenReco-generated events as ACTS-style CSV files."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="datasets/acts_openreco_generated",
    )
    parser.add_argument("--n-events", type=int, default=5)
    parser.add_argument("--n-particles", type=int, default=5)
    parser.add_argument("--hit-efficiency", type=float, default=1.0)
    parser.add_argument("--noise-hits-per-layer", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=123)

    args = parser.parse_args()

    metadata = generate_openreco_acts_dataset(
        args.output,
        n_events=args.n_events,
        n_particles=args.n_particles,
        hit_efficiency=args.hit_efficiency,
        noise_hits_per_layer=args.noise_hits_per_layer,
        random_seed=args.random_seed,
    )

    print("Generated ACTS-style OpenReco dataset")
    print("")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
