"""Plotting helpers for OpenReco performance-analysis outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from openreco.analysis.performance import PerformanceResult


def ensure_figure_dir(path: str | Path) -> Path:
    """Create and return a figure-output directory."""

    figure_dir = Path(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def _as_results_list(
    results: Iterable[PerformanceResult],
) -> list[PerformanceResult]:
    rows = list(results)
    if not rows:
        raise ValueError("at least one performance result is required")
    return rows


def _result_value(result: PerformanceResult, field_name: str) -> float:
    return float(getattr(result, field_name))


def _group_results(
    results: Iterable[PerformanceResult],
    group_fields: tuple[str, ...],
) -> dict[tuple[object, ...], list[PerformanceResult]]:
    groups: dict[tuple[object, ...], list[PerformanceResult]] = defaultdict(list)

    for result in results:
        key = tuple(getattr(result, field_name) for field_name in group_fields)
        groups[key].append(result)

    return dict(groups)


def _format_group_label(
    group_fields: tuple[str, ...],
    group_values: tuple[object, ...],
) -> str:
    parts = []

    for field_name, value in zip(group_fields, group_values):
        clean_name = field_name.replace("_", " ")
        parts.append(f"{clean_name}={value}")

    return ", ".join(parts)


def _save_plot(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_metric_vs_hit_efficiency(
    results: Iterable[PerformanceResult],
    metric: str,
    output_path: str | Path,
    ylabel: str,
    title: str,
) -> Path:
    """Plot a performance metric against hit efficiency.

    Curves are grouped by particle multiplicity and noise occupancy.
    """

    rows = _as_results_list(results)
    group_fields = ("n_particles", "noise_hits_per_layer")
    groups = _group_results(rows, group_fields)

    plt.figure(figsize=(7.0, 4.5))

    for group_key, group_rows in sorted(groups.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row.hit_efficiency)
        x_values = [row.hit_efficiency for row in sorted_rows]
        y_values = [_result_value(row, metric) for row in sorted_rows]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=_format_group_label(group_fields, group_key),
        )

    plt.xlabel("Hit efficiency")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    return _save_plot(output_path)


def plot_metric_vs_noise(
    results: Iterable[PerformanceResult],
    metric: str,
    output_path: str | Path,
    ylabel: str,
    title: str,
) -> Path:
    """Plot a performance metric against noise hits per layer.

    Curves are grouped by particle multiplicity and hit efficiency.
    """

    rows = _as_results_list(results)
    group_fields = ("n_particles", "hit_efficiency")
    groups = _group_results(rows, group_fields)

    plt.figure(figsize=(7.0, 4.5))

    for group_key, group_rows in sorted(groups.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row.noise_hits_per_layer)
        x_values = [row.noise_hits_per_layer for row in sorted_rows]
        y_values = [_result_value(row, metric) for row in sorted_rows]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=_format_group_label(group_fields, group_key),
        )

    plt.xlabel("Noise hits per layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    return _save_plot(output_path)


def plot_runtime_vs_seed_occupancy(
    results: Iterable[PerformanceResult],
    output_path: str | Path,
) -> Path:
    """Plot runtime per event against mean seed occupancy."""

    rows = _as_results_list(results)
    group_fields = ("n_particles", "hit_efficiency")
    groups = _group_results(rows, group_fields)

    plt.figure(figsize=(7.0, 4.5))

    for group_key, group_rows in sorted(groups.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row.seeds_mean)
        x_values = [row.seeds_mean for row in sorted_rows]
        y_values = [row.runtime_per_event_s for row in sorted_rows]

        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=_format_group_label(group_fields, group_key),
        )

    plt.xlabel("Mean seeds per event")
    plt.ylabel("Runtime per event [s]")
    plt.title("Runtime scaling with seed occupancy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    return _save_plot(output_path)


def generate_standard_performance_plots(
    results: Iterable[PerformanceResult],
    figure_dir: str | Path,
) -> dict[str, Path]:
    """Generate the standard v2.2 performance-analysis figure set."""

    rows = _as_results_list(results)
    output_dir = ensure_figure_dir(figure_dir)

    plots = {
        "efficiency_vs_hit_efficiency": plot_metric_vs_hit_efficiency(
            results=rows,
            metric="tracking_efficiency_mean",
            output_path=output_dir / "efficiency_vs_hit_efficiency.png",
            ylabel="Tracking efficiency",
            title="Tracking efficiency vs hit efficiency",
        ),
        "fake_rate_vs_noise": plot_metric_vs_noise(
            results=rows,
            metric="fake_rate_mean",
            output_path=output_dir / "fake_rate_vs_noise.png",
            ylabel="Fake rate",
            title="Fake rate vs noise occupancy",
        ),
        "duplicate_rate_vs_noise": plot_metric_vs_noise(
            results=rows,
            metric="duplicate_rate_mean",
            output_path=output_dir / "duplicate_rate_vs_noise.png",
            ylabel="Duplicate rate",
            title="Duplicate rate vs noise occupancy",
        ),
        "chi2_vs_hit_efficiency": plot_metric_vs_hit_efficiency(
            results=rows,
            metric="mean_chi2_ndof",
            output_path=output_dir / "chi2_vs_hit_efficiency.png",
            ylabel="Mean chi2/ndof",
            title="Fit quality vs hit efficiency",
        ),
        "momentum_resolution_vs_hit_efficiency": plot_metric_vs_hit_efficiency(
            results=rows,
            metric="momentum_residual_std",
            output_path=output_dir / "momentum_resolution_vs_hit_efficiency.png",
            ylabel="Momentum residual width",
            title="Momentum resolution vs hit efficiency",
        ),
        "runtime_vs_occupancy": plot_runtime_vs_seed_occupancy(
            results=rows,
            output_path=output_dir / "runtime_vs_occupancy.png",
        ),
    }

    return plots