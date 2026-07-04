"""Plotting helpers for OpenReco performance-analysis outputs.

The full plotting suite is added in a later v2.2 chunk. This module currently
contains small filesystem helpers used by the example script and tests.
"""

from __future__ import annotations

from pathlib import Path


def ensure_figure_dir(path: str | Path) -> Path:
    """Create and return a figure-output directory."""

    figure_dir = Path(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir