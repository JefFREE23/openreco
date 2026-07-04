"""Markdown report generation for OpenReco performance studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openreco.analysis.performance import PerformanceResult


def _as_results_list(results: Iterable[PerformanceResult]) -> list[PerformanceResult]:
    rows = list(results)
    if not rows:
        raise ValueError("at least one performance result is required")
    return rows


def _condition(row: PerformanceResult) -> str:
    return (
        f"n={row.n_particles}, "
        f"noise/layer={row.noise_hits_per_layer}, "
        f"hit_eff={row.hit_efficiency:.2f}"
    )


def build_tracking_performance_report(
    results: Iterable[PerformanceResult],
    figure_dir: str | Path | None = None,
) -> str:
    """Build the v2.2 tracking-performance Markdown report."""

    rows = _as_results_list(results)

    best_eff = max(rows, key=lambda row: row.tracking_efficiency_mean)
    worst_eff = min(rows, key=lambda row: row.tracking_efficiency_mean)
    slowest = max(rows, key=lambda row: row.runtime_per_event_s)
    widest_momentum = max(rows, key=lambda row: row.momentum_residual_std)

    lines = [
        "# OpenReco v2.2 Tracking Performance Analysis Report",
        "",
        "## Purpose",
        "",
        "OpenReco v2.2 uses the reconstruction chain as a controlled tracking-performance study tool.",
        "",
        "The scan varies particle multiplicity, hit efficiency, and noise occupancy. "
        "It measures tracking efficiency, fake rate, duplicate rate, holes, fit quality, "
        "momentum residuals, covariance validity, and runtime.",
        "",
        "## Main observations",
        "",
        f"- Best tracking efficiency: {best_eff.tracking_efficiency_mean:.4f} at `{_condition(best_eff)}`.",
        f"- Lowest tracking efficiency: {worst_eff.tracking_efficiency_mean:.4f} at `{_condition(worst_eff)}`.",
        f"- Widest momentum residual width: {widest_momentum.momentum_residual_std:.4f} at `{_condition(widest_momentum)}`.",
        f"- Slowest runtime per event: {slowest.runtime_per_event_s:.4f} s at `{_condition(slowest)}`.",
        "",
        "## Consolidated performance table",
        "",
        "| n particles | noise/layer | hit efficiency | efficiency | fake rate | duplicate rate | holes/track | chi2/ndof | momentum residual std | runtime/event [s] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(
        rows,
        key=lambda r: (r.n_particles, r.noise_hits_per_layer, r.hit_efficiency),
    ):
        lines.append(
            "| "
            f"{row.n_particles} | "
            f"{row.noise_hits_per_layer} | "
            f"{row.hit_efficiency:.2f} | "
            f"{row.tracking_efficiency_mean:.4f} | "
            f"{row.fake_rate_mean:.4f} | "
            f"{row.duplicate_rate_mean:.4f} | "
            f"{row.mean_holes_per_track:.4f} | "
            f"{row.mean_chi2_ndof:.4f} | "
            f"{row.momentum_residual_std:.4f} | "
            f"{row.runtime_per_event_s:.4f} |"
        )

    lines += [
        "",
        "## Figures",
        "",
    ]

    if figure_dir is not None:
        figure_path = Path(figure_dir)
        png_files = sorted(figure_path.glob("*.png"))

        if png_files:
            for png in png_files:
                title = png.stem.replace("_", " ")
                relative = Path("figures") / png.name
                lines += [
                    f"### {title}",
                    "",
                    f"![{title}]({relative.as_posix()})",
                    "",
                ]
        else:
            lines.append("No figure files were found.")
            lines.append("")
    else:
        lines.append("No figure directory was provided.")
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "This v2.2 study shows that OpenReco can now produce reproducible tracking-performance evidence: "
        "a CSV summary, standard plots, and a Markdown report from the same controlled reconstruction scan.",
        "",
        "The study is intentionally simplified. It is not detector-realistic yet. "
        "Its purpose is to isolate reconstruction behavior under controlled assumptions.",
        "",
    ]

    return "\n".join(lines)


def write_tracking_performance_report(
    results: Iterable[PerformanceResult],
    output_path: str | Path,
    figure_dir: str | Path | None = None,
) -> Path:
    """Write the v2.2 tracking-performance Markdown report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    text = build_tracking_performance_report(results, figure_dir=figure_dir)
    path.write_text(text, encoding="utf-8")

    return path