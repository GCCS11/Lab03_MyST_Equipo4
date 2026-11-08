"""Figures 2 and 6 are drawn from the artifacts of the single run without errors."""
import pytest

from config import ARTIFACTS_DIR
from strategy.figures import main

if not (ARTIFACTS_DIR / "senales.csv").exists() or not (ARTIFACTS_DIR / "theta.json").exists():
    pytest.skip("needs artifacts/senales.csv and artifacts/theta.json", allow_module_level=True)


def test_figures_2_and_6_are_written(tmp_path):
    main(out_dir=tmp_path)
    assert (tmp_path / "fig2_spread.png").stat().st_size > 10_000
    assert (tmp_path / "fig6_zscore.png").stat().st_size > 10_000
