"""Adapter from ACTS-style external events into OpenReco objects.

This module does not run reconstruction. It only converts external
file-loaded ACTS-style event records into the OpenReco measurement
and barrel-detector conventions used by the v1 pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from openreco.external.acts_schema import ActsEvent, ActsMeasurement, ActsTruthParticle
from openreco.geometry import BarrelDetector, CylindricalLayer
from openreco.measurements import Measurement


@dataclass(frozen=True)
class ActsAdapterConfig:
    """Configuration for the ACTS-style to OpenReco adapter."""

    length_scale: float = 1.0
    layer_name_prefix: str = "acts_layer"
    default_half_length: float | None = None
    require_barrel_geometry: bool = True


@dataclass(frozen=True)
class AdaptedOpenRecoEvent:
    """OpenReco-compatible representation of an external ACTS-style event."""

    event_id: int
    detector: BarrelDetector
    measurements: list[Measurement]
    truth_particles: list[ActsTruthParticle]
    measurement_truth_ids: dict[int, int | None] = field(default_factory=dict)
    measurement_is_noise: dict[int, bool] = field(default_factory=dict)
    source_measurements: list[ActsMeasurement] = field(default_factory=list)


def infer_cylindrical_layers(
    measurements: list[ActsMeasurement],
    config: ActsAdapterConfig | None = None,
) -> tuple[BarrelDetector, dict[int, str]]:
    """Infer an OpenReco barrel detector from ACTS-style measurements.

    For v2.0, we intentionally support barrel/cylindrical geometry only.
    Each external layer_id becomes one OpenReco CylindricalLayer.
    The layer radius is the median r value for that layer.
    """

    if config is None:
        config = ActsAdapterConfig()

    if not measurements:
        raise ValueError("Cannot infer cylindrical layers from an empty measurement list.")

    radii_by_layer: dict[int, list[float]] = defaultdict(list)
    z_values: list[float] = []

    for measurement in measurements:
        if measurement.r <= 0:
            raise ValueError(
                f"Measurement {measurement.measurement_id} has non-positive radius {measurement.r}."
            )

        radii_by_layer[measurement.layer_id].append(measurement.r * config.length_scale)
        z_values.append(abs(measurement.z * config.length_scale))

    if config.default_half_length is not None:
        half_length = config.default_half_length * config.length_scale
    else:
        max_abs_z = max(z_values) if z_values else 0.0
        half_length = max(1.0, 1.2 * max_abs_z)

    layers: list[CylindricalLayer] = []
    layer_name_by_id: dict[int, str] = {}

    sorted_layer_ids = sorted(
        radii_by_layer,
        key=lambda layer_id: median(radii_by_layer[layer_id]),
    )

    for layer_id in sorted_layer_ids:
        radius = float(median(radii_by_layer[layer_id]))
        layer_name = f"{config.layer_name_prefix}_{layer_id}"

        layers.append(
            CylindricalLayer(
                name=layer_name,
                radius=radius,
                half_length=half_length,
            )
        )
        layer_name_by_id[layer_id] = layer_name

    return BarrelDetector(layers), layer_name_by_id


def convert_acts_measurement_to_openreco(
    measurement: ActsMeasurement,
    layer_name_by_id: dict[int, str],
    config: ActsAdapterConfig | None = None,
) -> Measurement:
    """Convert one ACTS-style measurement into an OpenReco Measurement."""

    if config is None:
        config = ActsAdapterConfig()

    if measurement.layer_id not in layer_name_by_id:
        raise KeyError(f"No OpenReco layer name found for layer_id={measurement.layer_id}.")

    values = np.array(
        [
            measurement.phi,
            measurement.z * config.length_scale,
        ],
        dtype=float,
    )

    covariance = np.diag(
        [
            measurement.sigma_phi**2,
            (measurement.sigma_z * config.length_scale) ** 2,
        ]
    )

    return Measurement(
        values=values,
        covariance=covariance,
        layer_name=layer_name_by_id[measurement.layer_id],
        surface_type="cylinder",
    )


def convert_acts_event_to_openreco(
    event: ActsEvent,
    config: ActsAdapterConfig | None = None,
) -> AdaptedOpenRecoEvent:
    """Convert an ActsEvent into an OpenReco-compatible external event."""

    if config is None:
        config = ActsAdapterConfig()

    if config.require_barrel_geometry and not event.measurements:
        raise ValueError("Cannot convert event with no measurements.")

    detector, layer_name_by_id = infer_cylindrical_layers(event.measurements, config)

    converted_measurements = [
        convert_acts_measurement_to_openreco(
            measurement=measurement,
            layer_name_by_id=layer_name_by_id,
            config=config,
        )
        for measurement in event.measurements
    ]

    measurement_truth_ids = {
        measurement.measurement_id: measurement.particle_id for measurement in event.measurements
    }

    measurement_is_noise = {
        measurement.measurement_id: measurement.is_noise for measurement in event.measurements
    }

    return AdaptedOpenRecoEvent(
        event_id=event.event_id,
        detector=detector,
        measurements=converted_measurements,
        truth_particles=list(event.truth_particles),
        measurement_truth_ids=measurement_truth_ids,
        measurement_is_noise=measurement_is_noise,
        source_measurements=list(event.measurements),
    )
