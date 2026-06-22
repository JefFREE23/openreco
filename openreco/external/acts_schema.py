"""Schema objects for ACTS-style external validation files.

These dataclasses are intentionally file-format based.
They do not depend on the ACTS C++ runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActsTruthParticle:
    event_id: int
    particle_id: int
    charge: float

    x0: float
    y0: float
    z0: float

    px: float
    py: float
    pz: float

    pt: float
    eta: float
    phi: float


@dataclass(frozen=True)
class ActsMeasurement:
    event_id: int
    measurement_id: int
    particle_id: int | None

    layer_id: int
    surface_id: int

    x: float
    y: float
    z: float

    r: float
    phi: float

    sigma_phi: float
    sigma_z: float

    is_noise: bool


@dataclass(frozen=True)
class ActsEvent:
    event_id: int
    truth_particles: list[ActsTruthParticle]
    measurements: list[ActsMeasurement]


@dataclass(frozen=True)
class ActsDataset:
    events: list[ActsEvent]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.events)
