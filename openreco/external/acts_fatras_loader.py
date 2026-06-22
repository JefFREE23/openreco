"""Loader for official ACTS GenericDetector/Fatras CSV output.

This module converts ACTS/Fatras event CSV files such as

    event000000000-particles_initial.csv
    event000000000-hits.csv

into the existing OpenReco v2 ActsDataset schema.

Scope:
- particles_initial.csv provides truth particles
- hits.csv provides simulated hit positions and truth particle labels
- cells.csv and particles_final.csv are not required for the first v2 importer

Important:
ACTS hit coordinates are typically mm-like detector coordinates. OpenReco's
toy barrel examples use smaller internal length scales. The default
length_scale=0.1 maps mm -> cm-like OpenReco units.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from openreco.external.acts_schema import (
    ActsDataset,
    ActsEvent,
    ActsMeasurement,
    ActsTruthParticle,
)


PARTICLES_INITIAL_REQUIRED_COLUMNS = {
    "particle_id",
    "particle_type",
    "process",
    "vx",
    "vy",
    "vz",
    "vt",
    "px",
    "py",
    "pz",
    "m",
    "q",
}

HITS_REQUIRED_COLUMNS = {
    "particle_id",
    "geometry_id",
    "tx",
    "ty",
    "tz",
    "tt",
    "tpx",
    "tpy",
    "tpz",
    "te",
    "deltapx",
    "deltapy",
    "deltapz",
    "deltae",
    "index",
}


@dataclass(frozen=True)
class ActsFatrasLoaderConfig:
    """Configuration for ACTS/Fatras CSV loading.

    length_scale:
        Multiplies ACTS positions. Default 0.1 treats ACTS mm as cm-like
        OpenReco units.

    radius_merge_tolerance:
        Nearby ACTS surfaces with median radii closer than this value after
        scaling are merged into one simplified OpenReco barrel layer.
    """

    sigma_phi: float = 1.0e-3
    sigma_z: float = 1.0
    length_scale: float = 0.1
    radius_merge_tolerance: float = 0.5


def load_fatras_dataset(
    dataset_dir: str | Path,
    *,
    max_events: int | None = None,
    config: ActsFatrasLoaderConfig | None = None,
) -> ActsDataset:
    """Load an official ACTS/Fatras CSV output directory."""

    if config is None:
        config = ActsFatrasLoaderConfig()

    dataset_path = Path(dataset_dir)

    event_ids = find_fatras_event_ids(dataset_path)

    if max_events is not None:
        event_ids = event_ids[:max_events]

    events = [
        load_fatras_event(dataset_path, event_id, config=config)
        for event_id in event_ids
    ]

    n_truth_particles = sum(len(event.truth_particles) for event in events)
    n_measurements = sum(len(event.measurements) for event in events)

    return ActsDataset(
        events=events,
        metadata={
            "dataset_dir": str(dataset_path),
            "format": "acts_fatras",
            "n_events": len(events),
            "n_truth_particles": n_truth_particles,
            "n_measurements": n_measurements,
            "length_scale": config.length_scale,
            "radius_merge_tolerance": config.radius_merge_tolerance,
        },
    )


def find_fatras_event_ids(dataset_dir: str | Path) -> list[int]:
    """Find event IDs from eventXXXXXXXXX-hits.csv files."""

    dataset_path = Path(dataset_dir)

    event_ids: list[int] = []
    pattern = re.compile(r"event(\d+)-hits\.csv$")

    for path in dataset_path.glob("event*-hits.csv"):
        match = pattern.match(path.name)
        if match is not None:
            event_ids.append(int(match.group(1)))

    if not event_ids:
        raise FileNotFoundError(
            f"No ACTS/Fatras event*-hits.csv files found in {dataset_path}"
        )

    return sorted(event_ids)


def load_fatras_event(
    dataset_dir: str | Path,
    event_id: int,
    *,
    config: ActsFatrasLoaderConfig | None = None,
) -> ActsEvent:
    """Load one ACTS/Fatras event into an ActsEvent."""

    if config is None:
        config = ActsFatrasLoaderConfig()

    dataset_path = Path(dataset_dir)
    event_prefix = f"event{event_id:09d}"

    particles_path = dataset_path / f"{event_prefix}-particles_initial.csv"
    hits_path = dataset_path / f"{event_prefix}-hits.csv"

    particle_rows = _read_csv_rows(
        particles_path,
        PARTICLES_INITIAL_REQUIRED_COLUMNS,
    )
    hit_rows = _read_csv_rows(
        hits_path,
        HITS_REQUIRED_COLUMNS,
    )

    truth_particles = [
        _particle_row_to_truth_particle(event_id, row, config=config)
        for row in particle_rows
    ]

    layer_id_by_geometry_id = _infer_layer_ids_from_hit_rows(
        hit_rows,
        config=config,
    )

    measurements = [
        _hit_row_to_measurement(
            event_id,
            row,
            layer_id_by_geometry_id=layer_id_by_geometry_id,
            config=config,
        )
        for row in hit_rows
    ]

    return ActsEvent(
        event_id=event_id,
        truth_particles=truth_particles,
        measurements=measurements,
    )


def _particle_row_to_truth_particle(
    event_id: int,
    row: dict[str, str],
    *,
    config: ActsFatrasLoaderConfig,
) -> ActsTruthParticle:
    particle_id = _as_int(row["particle_id"], "particle_id")

    px = _as_float(row["px"], "px")
    py = _as_float(row["py"], "py")
    pz = _as_float(row["pz"], "pz")

    pt = math.hypot(px, py)
    eta = math.asinh(pz / pt) if pt > 0 else 0.0
    phi = _wrap_phi_to_pi(math.atan2(py, px))

    return ActsTruthParticle(
        event_id=event_id,
        particle_id=particle_id,
        charge=_as_float(row["q"], "q"),
        x0=_as_float(row["vx"], "vx") * config.length_scale,
        y0=_as_float(row["vy"], "vy") * config.length_scale,
        z0=_as_float(row["vz"], "vz") * config.length_scale,
        px=px,
        py=py,
        pz=pz,
        pt=pt,
        eta=eta,
        phi=phi,
    )


def _hit_row_to_measurement(
    event_id: int,
    row: dict[str, str],
    *,
    layer_id_by_geometry_id: dict[int, int],
    config: ActsFatrasLoaderConfig,
) -> ActsMeasurement:
    geometry_id = _as_int(row["geometry_id"], "geometry_id")

    x = _as_float(row["tx"], "tx") * config.length_scale
    y = _as_float(row["ty"], "ty") * config.length_scale
    z = _as_float(row["tz"], "tz") * config.length_scale

    radius = math.hypot(x, y)
    phi = _wrap_phi_to_pi(math.atan2(y, x))

    particle_id = _as_optional_particle_id(row["particle_id"])

    hit_index = _as_int(row["index"], "index")
    measurement_id = hit_index + 1

    return ActsMeasurement(
        event_id=event_id,
        measurement_id=measurement_id,
        particle_id=particle_id,
        layer_id=layer_id_by_geometry_id[geometry_id],
        surface_id=geometry_id,
        x=x,
        y=y,
        z=z,
        r=radius,
        phi=phi,
        sigma_phi=config.sigma_phi,
        sigma_z=config.sigma_z * config.length_scale,
        is_noise=particle_id is None,
    )


def _infer_layer_ids_from_hit_rows(
    hit_rows: list[dict[str, str]],
    *,
    config: ActsFatrasLoaderConfig,
) -> dict[int, int]:
    """Infer simplified barrel layer IDs from ACTS geometry IDs.

    ACTS geometry IDs identify detailed surfaces/modules. OpenReco v2 maps
    those detailed surfaces to simplified cylindrical radius shells.
    """

    radii_by_geometry_id: dict[int, list[float]] = defaultdict(list)

    for row in hit_rows:
        geometry_id = _as_int(row["geometry_id"], "geometry_id")
        x = _as_float(row["tx"], "tx") * config.length_scale
        y = _as_float(row["ty"], "ty") * config.length_scale

        radii_by_geometry_id[geometry_id].append(math.hypot(x, y))

    geometry_radius_pairs = sorted(
        (
            geometry_id,
            float(median(radii)),
        )
        for geometry_id, radii in radii_by_geometry_id.items()
    )

    geometry_radius_pairs.sort(key=lambda pair: pair[1])

    clusters: list[list[tuple[int, float]]] = []

    for geometry_id, radius in geometry_radius_pairs:
        if not clusters:
            clusters.append([(geometry_id, radius)])
            continue

        current_cluster = clusters[-1]
        current_cluster_radius = median(radius_value for _, radius_value in current_cluster)

        if abs(radius - current_cluster_radius) <= config.radius_merge_tolerance:
            current_cluster.append((geometry_id, radius))
        else:
            clusters.append([(geometry_id, radius)])

    layer_id_by_geometry_id: dict[int, int] = {}

    for layer_index, cluster in enumerate(clusters, start=1):
        for geometry_id, _ in cluster:
            layer_id_by_geometry_id[geometry_id] = layer_index

    return layer_id_by_geometry_id


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


def _as_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Column {name} must be a float, got {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"Column {name} must be finite, got {value!r}")

    return parsed


def _as_optional_particle_id(value: str) -> int | None:
    clean = value.strip()

    if clean == "" or clean.lower() in {"none", "null", "nan"}:
        return None

    particle_id = int(clean)

    if particle_id == 0:
        return None

    return particle_id


def _wrap_phi_to_pi(phi: float) -> float:
    wrapped = (float(phi) + math.pi) % (2.0 * math.pi) - math.pi

    if wrapped == -math.pi:
        return math.pi

    return wrapped
