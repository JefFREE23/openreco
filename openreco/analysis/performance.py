"""Performance-analysis data structures for OpenReco.

This module is intentionally lightweight. It provides stable containers for
tracking-performance scan configurations and results. The reconstruction scan
runner itself is added in the next v2.2 chunk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import csv
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PerformanceScanConfig:
    """Configuration for one controlled tracking-performance scan point."""

    n_particles: int
    noise_hits_per_layer: int
    hit_efficiency: float
    n_events: int = 50
    seed: int = 12345
    seed_mode: str = "hole-aware"

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.noise_hits_per_layer < 0:
            raise ValueError("noise_hits_per_layer must be non-negative")
        if not 0.0 <= self.hit_efficiency <= 1.0:
            raise ValueError("hit_efficiency must be between 0 and 1")
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if self.seed_mode not in {"strict", "hole-aware"}:
            raise ValueError("seed_mode must be either 'strict' or 'hole-aware'")


@dataclass(frozen=True)
class PerformanceResult:
    """Summary result for one tracking-performance scan point."""

    n_particles: int
    noise_hits_per_layer: int
    hit_efficiency: float
    events_processed: int
    truth_particles_generated: int
    measurements_mean: float
    seeds_mean: float
    reconstructed_tracks_mean: float
    tracking_efficiency_mean: float
    fake_rate_mean: float
    duplicate_rate_mean: float
    mean_hits_per_track: float
    mean_holes_per_track: float
    mean_chi2_ndof: float
    covariance_valid_rate_mean: float
    momentum_residual_mean: float
    momentum_residual_std: float
    runtime_total_s: float
    runtime_per_event_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-friendly dictionary representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "PerformanceResult":
        """Create a result from a CSV row or dictionary.

        CSV files store values as strings, so this method converts fields back
        to the expected numeric types.
        """

        int_fields = {
            "n_particles",
            "noise_hits_per_layer",
            "events_processed",
            "truth_particles_generated",
        }

        converted: dict[str, object] = {}
        for field in fields(cls):
            value = row[field.name]
            if field.name in int_fields:
                converted[field.name] = int(value)
            else:
                converted[field.name] = float(value)

        return cls(**converted)

    @classmethod
    def fieldnames(cls) -> list[str]:
        """Return the stable CSV column order."""

        return [field.name for field in fields(cls)]


def write_performance_results(
    path: str | Path,
    results: Iterable[PerformanceResult],
) -> None:
    """Write performance results to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(results)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PerformanceResult.fieldnames())
        writer.writeheader()
        for result in rows:
            writer.writerow(result.to_dict())


def read_performance_results(path: str | Path) -> list[PerformanceResult]:
    """Read performance results from CSV."""

    input_path = Path(path)

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [PerformanceResult.from_dict(row) for row in reader]