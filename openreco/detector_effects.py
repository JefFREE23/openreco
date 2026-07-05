"""Detector-effects configuration models for OpenReco v3.0.

This module defines small, validated configuration objects for controlled
detector-effect studies. The objects here do not modify reconstruction by
themselves. They provide a clean interface for later chunks such as hit
resolution scans, material budgets, inefficiency, noise occupancy, process
noise, energy loss, and magnetic-field scale studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_probability(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")


def _normalize_layer_id(layer_id: int) -> int:
    if not isinstance(layer_id, int):
        raise TypeError(f"layer_id must be an int, got {type(layer_id).__name__}")
    if layer_id < 0:
        raise ValueError(f"layer_id must be non-negative, got {layer_id!r}")
    return layer_id


@dataclass(frozen=True)
class HitResolutionModel:
    """Measurement resolution model for cylindrical local measurements."""

    sigma_phi: float = 1.0e-3
    sigma_z: float = 0.1

    def __post_init__(self) -> None:
        _require_non_negative("sigma_phi", self.sigma_phi)
        _require_non_negative("sigma_z", self.sigma_z)

    @property
    def covariance_diagonal(self) -> tuple[float, float]:
        """Return the diagonal measurement covariance entries."""

        return self.sigma_phi**2, self.sigma_z**2


@dataclass(frozen=True)
class LayerMaterial:
    """Material assigned to one detector layer."""

    layer_id: int
    x_over_x0: float = 0.0
    energy_loss_mev: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_id", _normalize_layer_id(self.layer_id))
        _require_non_negative("x_over_x0", self.x_over_x0)
        _require_non_negative("energy_loss_mev", self.energy_loss_mev)


@dataclass(frozen=True)
class InefficiencyModel:
    """Simple hit-efficiency model."""

    hit_efficiency: float = 1.0

    def __post_init__(self) -> None:
        _require_probability("hit_efficiency", self.hit_efficiency)

    @property
    def drop_probability(self) -> float:
        return 1.0 - self.hit_efficiency


@dataclass(frozen=True)
class DeadLayerModel:
    """Set of detector layers that produce no measurements."""

    dead_layers: frozenset[int] = field(default_factory=frozenset)

    def __init__(self, dead_layers: Iterable[int] | None = None) -> None:
        normalized = frozenset(
            _normalize_layer_id(layer_id) for layer_id in (dead_layers or [])
        )
        object.__setattr__(self, "dead_layers", normalized)

    def is_dead(self, layer_id: int) -> bool:
        return _normalize_layer_id(layer_id) in self.dead_layers


@dataclass(frozen=True)
class NoiseOccupancyModel:
    """Random fake-hit occupancy model."""

    mean_noise_hits_per_layer: float = 0.0

    def __post_init__(self) -> None:
        _require_non_negative(
            "mean_noise_hits_per_layer", self.mean_noise_hits_per_layer
        )


@dataclass(frozen=True)
class MagneticFieldScale:
    """Truth/reconstruction magnetic-field scale configuration."""

    truth_scale: float = 1.0
    reco_scale: float = 1.0

    def __post_init__(self) -> None:
        _require_positive("truth_scale", self.truth_scale)
        _require_positive("reco_scale", self.reco_scale)

    @property
    def mismatch(self) -> float:
        """Return reco_scale / truth_scale."""

        return self.reco_scale / self.truth_scale


@dataclass(frozen=True)
class DetectorEffectsConfig:
    """Top-level detector-effects configuration for v3.0 studies."""

    hit_resolution: HitResolutionModel = field(default_factory=HitResolutionModel)
    layer_materials: tuple[LayerMaterial, ...] = field(default_factory=tuple)
    inefficiency: InefficiencyModel = field(default_factory=InefficiencyModel)
    dead_layers: DeadLayerModel = field(default_factory=DeadLayerModel)
    noise_occupancy: NoiseOccupancyModel = field(default_factory=NoiseOccupancyModel)
    b_field_scale: MagneticFieldScale = field(default_factory=MagneticFieldScale)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_materials", tuple(self.layer_materials))

        seen: set[int] = set()
        for material in self.layer_materials:
            if not isinstance(material, LayerMaterial):
                raise TypeError(
                    "layer_materials must contain LayerMaterial objects, "
                    f"got {type(material).__name__}"
                )
            if material.layer_id in seen:
                raise ValueError(f"duplicate material for layer {material.layer_id}")
            seen.add(material.layer_id)

    @classmethod
    def default(cls) -> "DetectorEffectsConfig":
        """Return the default clean-detector configuration."""

        return cls()

    @classmethod
    def with_uniform_material(
        cls,
        layer_ids: Iterable[int],
        x_over_x0: float,
        energy_loss_mev: float = 0.0,
        **kwargs,
    ) -> "DetectorEffectsConfig":
        """Build a config with the same material assigned to each layer."""

        materials = tuple(
            LayerMaterial(
                layer_id=layer_id,
                x_over_x0=x_over_x0,
                energy_loss_mev=energy_loss_mev,
            )
            for layer_id in layer_ids
        )
        return cls(layer_materials=materials, **kwargs)

    def material_for_layer(self, layer_id: int) -> LayerMaterial:
        """Return material for a layer, or vacuum material if unspecified."""

        normalized_layer_id = _normalize_layer_id(layer_id)
        for material in self.layer_materials:
            if material.layer_id == normalized_layer_id:
                return material
        return LayerMaterial(layer_id=normalized_layer_id)