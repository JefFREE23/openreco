from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_path = repo_root / "examples" / "v2_2_tracking_performance_study.py"

    spec = importlib.util.spec_from_file_location("v2_2_study", example_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_v2_2_tracking_performance_study_dry_run(capsys):
    module = _load_example_module()

    exit_code = module.main(["--dry-run"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OpenReco v2.2 tracking performance study" in captured.out
    assert "Planned scan grid" in captured.out
    assert "scan points:" in captured.out