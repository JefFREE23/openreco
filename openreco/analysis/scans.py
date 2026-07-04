"""Reusable scan definitions for OpenReco performance studies."""

from __future__ import annotations

from itertools import product
from typing import Iterable

from openreco.analysis.performance import PerformanceScanConfig


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
    """Default v2.2 scan grid.

    This mirrors the current v1 performance-scan structure and provides a
    stable starting point for the v2.2 analysis suite.
    """

    return make_scan_grid(
        n_particles=(1, 2, 5),
        noise_hits_per_layer=(0, 1),
        hit_efficiencies=(1.0, 0.95),
        n_events=50,
        seed=12345,
        seed_mode="hole-aware",
    )