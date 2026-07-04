"""Reusable scan definitions and runners for OpenReco performance studies."""

from __future__ import annotations

from dataclasses import asdict, fields
from itertools import product
from pathlib import Path
from typing import Iterable

from examples.v1_performance_scan import (
    ScanConfig as V1ScanConfig,
    run_single_scan_point,
)

from openreco.analysis.performance import (
    PerformanceResult,
    PerformanceScanConfig,
    write_performance_results,
)


def make_scan_grid(
    n_particles: Iterable[int] = (1, 2, 5),
    noise_hits_per_layer: Iterable[int] = (0, 1),
    hit_efficiencies: Iterable[float] = (1.0, 0.95),
    n_events: int = 50,
    seed: int = 12345,
    seed_mode: str = "hole-aware",
) -> list[PerformanceScanConfig]:
    """Create a controlled grid of tracking-performance scan configurations."""

    configs: list[PerformanceScanConfig] = []

    for n, noise, hit_eff in product(
        n_particles,
        noise_hits_per_layer,
        hit_efficiencies,
    ):
        configs.append(
            PerformanceScanConfig(
                n_particles=int(n),
                noise_hits_per_layer=int(noise),
                hit_efficiency=float(hit_eff),
                n_events=int(n_events),
                seed=int(seed),
                seed_mode=seed_mode,
            )
        )

    return configs


def default_v2_2_scan_grid() -> list[PerformanceScanConfig]:
    """Default v2.2 scan grid."""

    return make_scan_grid(
        n_particles=(1, 2, 5),
        noise_hits_per_layer=(0, 1),
        hit_efficiencies=(1.0, 0.95),
        n_events=50,
        seed=12345,
        seed_mode="hole-aware",
    )


def _to_v1_config(config: PerformanceScanConfig) -> V1ScanConfig:
    """Convert a v2.2 analysis config into the existing v1 scan config."""

    valid_fields = {field.name for field in fields(V1ScanConfig)}
    data = asdict(config)

    if "random_seed" in valid_fields:
        data["random_seed"] = data.get("seed", 12345)

    if "min_hits" in valid_fields:
        data["min_hits"] = 5

    filtered_data = {
        key: value for key, value in data.items() if key in valid_fields
    }

    return V1ScanConfig(**filtered_data)


def _to_performance_result(v1_result: object) -> PerformanceResult:
    """Convert an existing v1 scan result into a v2.2 performance result."""

    data = asdict(v1_result)
    return PerformanceResult(**data)


def run_tracking_performance_scan(
    configs: Iterable[PerformanceScanConfig],
) -> list[PerformanceResult]:
    """Run a tracking-performance scan using OpenReco's reconstruction chain."""

    results: list[PerformanceResult] = []

    for config in configs:
        v1_config = _to_v1_config(config)
        v1_result = run_single_scan_point(v1_config)
        results.append(_to_performance_result(v1_result))

    return results


def run_and_write_tracking_performance_scan(
    output_path: str | Path,
    configs: Iterable[PerformanceScanConfig] | None = None,
) -> list[PerformanceResult]:
    """Run the tracking-performance scan and write the result CSV."""

    if configs is None:
        configs = default_v2_2_scan_grid()

    results = run_tracking_performance_scan(configs)
    write_performance_results(output_path, results)

    return results