"""OpenReco v3.0 integrated detector-effects benchmark.

This script collects the v3.0 detector-effects scan CSV artifacts into one
compact benchmark summary.

Inputs are the existing scan outputs under:

    docs/reports/v3_0_detector_effects/

Outputs:

    detector_effects_benchmark_summary.csv
    detector_effects_benchmark_report.md

The goal is not to rerun all scans. The goal is to produce a single evidence
artifact that summarizes how each detector assumption affects reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_REPORT_DIR = Path("docs/reports/v3_0_detector_effects")


@dataclass(frozen=True)
class StudyDefinition:
    study: str
    filename: str
    parameter_columns: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    study: str
    source_csv: str
    n_points: int
    baseline_parameters: str
    comparison_parameters: str
    baseline_tracking_efficiency: float
    comparison_tracking_efficiency: float
    baseline_fake_rate: float
    comparison_fake_rate: float
    baseline_duplicate_rate: float
    comparison_duplicate_rate: float
    baseline_chi2_ndof: float
    comparison_chi2_ndof: float
    baseline_momentum_residual_mean: float
    comparison_momentum_residual_mean: float
    baseline_momentum_residual_std: float
    comparison_momentum_residual_std: float
    baseline_momentum_uncertainty: float
    comparison_momentum_uncertainty: float
    headline: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


DEFAULT_STUDIES: tuple[StudyDefinition, ...] = (
    StudyDefinition(
        study="hit_resolution",
        filename="hit_resolution_scan.csv",
        parameter_columns=("sigma_phi", "sigma_z"),
    ),
    StudyDefinition(
        study="inefficiency_dead_layers",
        filename="inefficiency_dead_layer_scan.csv",
        parameter_columns=("hit_efficiency", "dead_layer_scenario", "dead_layers"),
    ),
    StudyDefinition(
        study="noise_occupancy",
        filename="noise_occupancy_scan.csv",
        parameter_columns=("mean_noise_hits_per_layer",),
    ),
    StudyDefinition(
        study="material_budget",
        filename="material_budget_scan.csv",
        parameter_columns=("x_over_x0_per_layer",),
    ),
    StudyDefinition(
        study="process_noise_calibration",
        filename="process_noise_scan.csv",
        parameter_columns=("process_noise_scale", "x_over_x0_per_layer"),
    ),
    StudyDefinition(
        study="energy_loss",
        filename="energy_loss_scan.csv",
        parameter_columns=("energy_loss_mev_per_layer",),
    ),
    StudyDefinition(
        study="bfield_scale",
        filename="bfield_scale_scan.csv",
        parameter_columns=("truth_scale", "reco_scale", "scale_mismatch"),
    ),
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, str], key: str) -> float:
    raw_value = row.get(key, "")

    if raw_value in ("", None):
        return float("nan")

    try:
        return float(raw_value)
    except ValueError:
        return float("nan")


def _format_float(value: float) -> str:
    if not isfinite(value):
        return "nan"

    return f"{value:.4g}"


def _format_parameters(
    row: dict[str, str],
    parameter_columns: Iterable[str],
) -> str:
    parts: list[str] = []

    for column in parameter_columns:
        if column in row and row[column] not in ("", None):
            parts.append(f"{column}={row[column]}")

    return "; ".join(parts) if parts else "n/a"


def _choose_comparison_row(
    *,
    study: str,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    if study == "process_noise_calibration":
        finite_rows = [
            row
            for row in rows
            if isfinite(_as_float(row, "mean_chi2_ndof"))
        ]

        if finite_rows:
            return min(
                finite_rows,
                key=lambda row: abs(_as_float(row, "mean_chi2_ndof") - 1.0),
            )

    return rows[-1]


def _make_headline(
    *,
    study: str,
    baseline_parameters: str,
    comparison_parameters: str,
    baseline_chi2: float,
    comparison_chi2: float,
    baseline_mom_std: float,
    comparison_mom_std: float,
) -> str:
    return (
        f"{study}: {baseline_parameters} -> {comparison_parameters}; "
        f"chi2/ndof {_format_float(baseline_chi2)} -> "
        f"{_format_float(comparison_chi2)}; "
        f"momentum residual std {_format_float(baseline_mom_std)} -> "
        f"{_format_float(comparison_mom_std)}"
    )


def summarize_study(
    *,
    definition: StudyDefinition,
    csv_path: Path,
) -> BenchmarkSummaryRow:
    rows = _read_csv_rows(csv_path)

    if not rows:
        raise ValueError(f"empty scan CSV: {csv_path}")

    baseline = rows[0]
    comparison = _choose_comparison_row(
        study=definition.study,
        rows=rows,
    )

    baseline_parameters = _format_parameters(
        baseline,
        definition.parameter_columns,
    )
    comparison_parameters = _format_parameters(
        comparison,
        definition.parameter_columns,
    )

    baseline_chi2 = _as_float(baseline, "mean_chi2_ndof")
    comparison_chi2 = _as_float(comparison, "mean_chi2_ndof")
    baseline_mom_std = _as_float(baseline, "momentum_residual_std")
    comparison_mom_std = _as_float(comparison, "momentum_residual_std")

    return BenchmarkSummaryRow(
        study=definition.study,
        source_csv=str(csv_path),
        n_points=len(rows),
        baseline_parameters=baseline_parameters,
        comparison_parameters=comparison_parameters,
        baseline_tracking_efficiency=_as_float(
            baseline,
            "tracking_efficiency_mean",
        ),
        comparison_tracking_efficiency=_as_float(
            comparison,
            "tracking_efficiency_mean",
        ),
        baseline_fake_rate=_as_float(baseline, "fake_rate_mean"),
        comparison_fake_rate=_as_float(comparison, "fake_rate_mean"),
        baseline_duplicate_rate=_as_float(baseline, "duplicate_rate_mean"),
        comparison_duplicate_rate=_as_float(comparison, "duplicate_rate_mean"),
        baseline_chi2_ndof=baseline_chi2,
        comparison_chi2_ndof=comparison_chi2,
        baseline_momentum_residual_mean=_as_float(
            baseline,
            "momentum_residual_mean",
        ),
        comparison_momentum_residual_mean=_as_float(
            comparison,
            "momentum_residual_mean",
        ),
        baseline_momentum_residual_std=baseline_mom_std,
        comparison_momentum_residual_std=comparison_mom_std,
        baseline_momentum_uncertainty=_as_float(
            baseline,
            "momentum_uncertainty_mean",
        ),
        comparison_momentum_uncertainty=_as_float(
            comparison,
            "momentum_uncertainty_mean",
        ),
        headline=_make_headline(
            study=definition.study,
            baseline_parameters=baseline_parameters,
            comparison_parameters=comparison_parameters,
            baseline_chi2=baseline_chi2,
            comparison_chi2=comparison_chi2,
            baseline_mom_std=baseline_mom_std,
            comparison_mom_std=comparison_mom_std,
        ),
    )


def write_summary_csv(
    path: str | Path,
    rows: list[BenchmarkSummaryRow],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=BenchmarkSummaryRow.fieldnames(),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())

    return output_path


def write_markdown_report(
    path: str | Path,
    rows: list[BenchmarkSummaryRow],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# OpenReco v3.0 Detector-Effects Benchmark",
        "",
        "This report summarizes the v3.0 detector-effects scan CSV artifacts.",
        "",
        "The baseline point is the first row of each scan. The comparison point "
        "is the final row, except for process-noise calibration, where the "
        "comparison point is the scale with mean chi2/ndof closest to 1.",
        "",
        "## Summary",
        "",
        "| Study | Points | Baseline | Comparison | chi2/ndof | Momentum residual std |",
        "|---|---:|---|---|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row.study} | "
            f"{row.n_points} | "
            f"{row.baseline_parameters} | "
            f"{row.comparison_parameters} | "
            f"{_format_float(row.baseline_chi2_ndof)} -> "
            f"{_format_float(row.comparison_chi2_ndof)} | "
            f"{_format_float(row.baseline_momentum_residual_std)} -> "
            f"{_format_float(row.comparison_momentum_residual_std)} |"
        )

    lines.extend(
        [
            "",
            "## Headlines",
            "",
        ]
    )

    for row in rows:
        lines.append(f"- {row.headline}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_detector_effects_benchmark(
    *,
    input_dir: str | Path = DEFAULT_REPORT_DIR,
    summary_csv: str | Path = DEFAULT_REPORT_DIR
    / "detector_effects_benchmark_summary.csv",
    report_md: str | Path = DEFAULT_REPORT_DIR
    / "detector_effects_benchmark_report.md",
    studies: tuple[StudyDefinition, ...] = DEFAULT_STUDIES,
) -> list[BenchmarkSummaryRow]:
    input_path = Path(input_dir)

    summary_rows: list[BenchmarkSummaryRow] = []

    for definition in studies:
        csv_path = input_path / definition.filename

        if not csv_path.exists():
            continue

        summary_rows.append(
            summarize_study(
                definition=definition,
                csv_path=csv_path,
            )
        )

    if not summary_rows:
        raise ValueError(f"no v3.0 scan CSV files found in {input_path}")

    write_summary_csv(summary_csv, summary_rows)
    write_markdown_report(report_md, summary_rows)

    return summary_rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the OpenReco v3.0 detector-effects benchmark summary."
    )

    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory containing v3.0 detector-effects scan CSV files.",
    )

    parser.add_argument(
        "--summary-csv",
        default=str(DEFAULT_REPORT_DIR / "detector_effects_benchmark_summary.csv"),
        help="Output benchmark summary CSV path.",
    )

    parser.add_argument(
        "--report-md",
        default=str(DEFAULT_REPORT_DIR / "detector_effects_benchmark_report.md"),
        help="Output benchmark Markdown report path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    rows = run_detector_effects_benchmark(
        input_dir=args.input_dir,
        summary_csv=args.summary_csv,
        report_md=args.report_md,
    )

    print("OpenReco v3.0 detector-effects benchmark")
    print(f"studies summarized: {len(rows)}")
    print(f"summary CSV:        {args.summary_csv}")
    print(f"Markdown report:    {args.report_md}")
    print()

    for row in rows:
        print(row.headline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())