"""CSV loader for ACTS-style external validation datasets."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from openreco.external.acts_schema import (
    ActsDataset,
    ActsEvent,
    ActsMeasurement,
    ActsTruthParticle,
)


TRUTH_REQUIRED_COLUMNS = {
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
}

MEASUREMENT_REQUIRED_COLUMNS = {
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
}


def load_truth_particles(path: str | Path) -> list[ActsTruthParticle]:
    """Load truth particles from truth_particles.csv."""

    rows = _read_csv_rows(path, TRUTH_REQUIRED_COLUMNS)
    particles: list[ActsTruthParticle] = []

    for row in rows:
        particle = ActsTruthParticle(
            event_id=_as_int(row["event_id"], "event_id"),
            particle_id=_as_int(row["particle_id"], "particle_id"),
            charge=_as_float(row["charge"], "charge"),
            x0=_as_float(row["x0"], "x0"),
            y0=_as_float(row["y0"], "y0"),
            z0=_as_float(row["z0"], "z0"),
            px=_as_float(row["px"], "px"),
            py=_as_float(row["py"], "py"),
            pz=_as_float(row["pz"], "pz"),
            pt=_as_float(row["pt"], "pt"),
            eta=_as_float(row["eta"], "eta"),
            phi=_as_float(row["phi"], "phi"),
        )
        _validate_phi(particle.phi, context=f"truth particle {particle.particle_id}")
        particles.append(particle)

    return particles


def load_measurements(path: str | Path) -> list[ActsMeasurement]:
    """Load measurements from measurements.csv."""

    rows = _read_csv_rows(path, MEASUREMENT_REQUIRED_COLUMNS)
    measurements: list[ActsMeasurement] = []

    for row in rows:
        is_noise = _as_bool(row["is_noise"], "is_noise")
        particle_id = _as_optional_int(row["particle_id"], "particle_id")

        measurement = ActsMeasurement(
            event_id=_as_int(row["event_id"], "event_id"),
            measurement_id=_as_int(row["measurement_id"], "measurement_id"),
            particle_id=particle_id,
            layer_id=_as_int(row["layer_id"], "layer_id"),
            surface_id=_as_int(row["surface_id"], "surface_id"),
            x=_as_float(row["x"], "x"),
            y=_as_float(row["y"], "y"),
            z=_as_float(row["z"], "z"),
            r=_as_float(row["r"], "r"),
            phi=_as_float(row["phi"], "phi"),
            sigma_phi=_as_float(row["sigma_phi"], "sigma_phi"),
            sigma_z=_as_float(row["sigma_z"], "sigma_z"),
            is_noise=is_noise,
        )

        _validate_phi(measurement.phi, context=f"measurement {measurement.measurement_id}")
        _validate_radius(measurement)
        _validate_uncertainty(measurement)

        if not measurement.is_noise and measurement.particle_id is None:
            raise ValueError(
                f"Measurement {measurement.measurement_id} is not noise but has no particle_id."
            )

        measurements.append(measurement)

    return measurements


def group_by_event(
    truth_particles: Iterable[ActsTruthParticle],
    measurements: Iterable[ActsMeasurement],
) -> list[ActsEvent]:
    """Group truth particles and measurements into ActsEvent objects by event_id."""

    truth_by_event: dict[int, list[ActsTruthParticle]] = defaultdict(list)
    measurements_by_event: dict[int, list[ActsMeasurement]] = defaultdict(list)

    for particle in truth_particles:
        truth_by_event[particle.event_id].append(particle)

    for measurement in measurements:
        measurements_by_event[measurement.event_id].append(measurement)

    event_ids = sorted(set(truth_by_event) | set(measurements_by_event))

    events = [
        ActsEvent(
            event_id=event_id,
            truth_particles=truth_by_event.get(event_id, []),
            measurements=measurements_by_event.get(event_id, []),
        )
        for event_id in event_ids
    ]

    return events


def load_acts_dataset(dataset_dir: str | Path) -> ActsDataset:
    """Load an ACTS-style dataset directory.

    Expected files:
    - truth_particles.csv
    - measurements.csv
    """

    dataset_path = Path(dataset_dir)

    truth_path = dataset_path / "truth_particles.csv"
    measurements_path = dataset_path / "measurements.csv"

    truth_particles = load_truth_particles(truth_path)
    measurements = load_measurements(measurements_path)

    events = group_by_event(truth_particles, measurements)

    return ActsDataset(
        events=events,
        metadata={
            "dataset_dir": str(dataset_path),
            "n_truth_particles": len(truth_particles),
            "n_measurements": len(measurements),
        },
    )


def _read_csv_rows(path: str | Path, required_columns: set[str]) -> list[dict[str, str]]:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        found_columns = set(reader.fieldnames)
        missing = required_columns - found_columns
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_str}")

        return list(reader)


def _as_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Column {name} must be an integer, got {value!r}") from exc


def _as_optional_int(value: str, name: str) -> int | None:
    clean = value.strip()
    if clean == "" or clean.lower() in {"none", "null", "nan"}:
        return None
    return _as_int(clean, name)


def _as_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Column {name} must be a float, got {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"Column {name} must be finite, got {value!r}")

    return parsed


def _as_bool(value: str, name: str) -> bool:
    clean = value.strip().lower()

    if clean in {"true", "1", "yes", "y"}:
        return True

    if clean in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Column {name} must be boolean-like, got {value!r}")


def _validate_phi(phi: float, context: str) -> None:
    if not (-math.pi <= phi <= math.pi):
        raise ValueError(f"{context}: phi must be in radians within [-pi, pi], got {phi}")


def _validate_radius(measurement: ActsMeasurement) -> None:
    expected = math.hypot(measurement.x, measurement.y)

    if not math.isclose(measurement.r, expected, rel_tol=1e-5, abs_tol=1e-4):
        raise ValueError(
            f"Measurement {measurement.measurement_id}: r={measurement.r} does not match "
            f"sqrt(x^2 + y^2)={expected}"
        )


def _validate_uncertainty(measurement: ActsMeasurement) -> None:
    if measurement.sigma_phi <= 0:
        raise ValueError(
            f"Measurement {measurement.measurement_id}: sigma_phi must be positive."
        )

    if measurement.sigma_z <= 0:
        raise ValueError(
            f"Measurement {measurement.measurement_id}: sigma_z must be positive."
        )
