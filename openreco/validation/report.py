"""CSV and plot reports for OpenReco v2 external validation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openreco.external.reconstruction import ExternalDatasetRecoSummary
from openreco.validation.external_metrics import (
    compute_external_validation_metrics,
    compute_momentum_relative_residuals,
    event_summary_rows,
    track_summary_rows,
)


@dataclass(frozen=True)
class ExternalValidationReportPaths:
    """Paths produced by an external validation report."""

    summary_csv: Path
    tracks_csv: Path
    efficiency_plot: Path | None = None
    momentum_residual_plot: Path | None = None


def write_external_validation_report(
    summary: ExternalDatasetRecoSummary,
    output_dir: str | Path = "docs",
    *,
    make_plots: bool = True,
) -> ExternalValidationReportPaths:
    """Write CSV summaries and optional plots for external validation."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_csv = output_path / "v2_external_validation_summary.csv"
    tracks_csv = output_path / "v2_external_validation_tracks.csv"

    _write_csv(summary_csv, event_summary_rows(summary))
    _write_csv(tracks_csv, track_summary_rows(summary))

    efficiency_plot = None
    momentum_residual_plot = None

    if make_plots:
        image_dir = output_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        efficiency_plot = image_dir / "v2_efficiency_summary.png"
        momentum_residual_plot = image_dir / "v2_momentum_residuals.png"

        plot_efficiency_summary(summary, efficiency_plot)
        plot_momentum_residuals(summary, momentum_residual_plot)

    return ExternalValidationReportPaths(
        summary_csv=summary_csv,
        tracks_csv=tracks_csv,
        efficiency_plot=efficiency_plot,
        momentum_residual_plot=momentum_residual_plot,
    )


def plot_efficiency_summary(
    summary: ExternalDatasetRecoSummary,
    output_path: str | Path,
) -> None:
    """Save a compact validation metric bar plot."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    metrics = compute_external_validation_metrics(summary)

    labels = [
        "Unique efficiency",
        "Fake rate",
        "Duplicate rate",
    ]

    values = [
        metrics.unique_tracking_efficiency,
        metrics.fake_rate,
        metrics.duplicate_rate,
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(labels, values)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.set_ylabel("Rate")
    ax.set_title("OpenReco v2 external validation summary")
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_momentum_residuals(
    summary: ExternalDatasetRecoSummary,
    output_path: str | Path,
) -> None:
    """Save a histogram of matched-track momentum residuals."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    residuals = compute_momentum_relative_residuals(summary)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    if residuals:
        ax.hist(residuals, bins=min(20, max(5, len(residuals))))
        ax.set_xlabel("(p_reco - p_truth) / p_truth")
        ax.set_ylabel("Tracks")
    else:
        ax.text(
            0.5,
            0.5,
            "No matched non-duplicate tracks",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("OpenReco v2 momentum residuals")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            clean_row = {
                key: _clean_csv_value(value)
                for key, value in row.items()
            }
            writer.writerow(clean_row)


def _clean_csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""

    return value
