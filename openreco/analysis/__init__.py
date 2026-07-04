"""Analysis helpers for OpenReco tracking-performance studies."""

from openreco.analysis.performance import (
    PerformanceResult,
    PerformanceScanConfig,
    read_performance_results,
    write_performance_results,
)
from openreco.analysis.plotting import (
    ensure_figure_dir,
    generate_standard_performance_plots,
    plot_metric_vs_hit_efficiency,
    plot_metric_vs_noise,
    plot_runtime_vs_seed_occupancy,
)
from openreco.analysis.scans import (
    default_v2_2_scan_grid,
    make_scan_grid,
    run_and_write_tracking_performance_scan,
    run_tracking_performance_scan,
)

__all__ = [
    "PerformanceResult",
    "PerformanceScanConfig",
    "read_performance_results",
    "write_performance_results",
    "ensure_figure_dir",
    "generate_standard_performance_plots",
    "plot_metric_vs_hit_efficiency",
    "plot_metric_vs_noise",
    "plot_runtime_vs_seed_occupancy",
    "default_v2_2_scan_grid",
    "make_scan_grid",
    "run_tracking_performance_scan",
    "run_and_write_tracking_performance_scan",
]