"""Results by partition from main.py: the slices of the continuous backtest add up to the whole
run, and the trade log carries what the dashboard reads."""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from config import INITIAL_CAPITAL
from main import TRADE_COLUMNS, equity_artifact, split_metrics, trade_artifact
from strategy.signals import spread_direction

THETA = {"z_in": 1.0, "z_out": 0.25, "z_stop": 3.0, "h_max": 20}


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=600, name="date")
    prices = pd.DataFrame(
        50 * np.exp(np.cumsum(rng.normal(0, 0.01, (600, 3)), axis=0)), index=index, columns=["A", "B", "C"],
    )
    z = pd.Series(2.0 * np.sin(np.arange(600) / 9.0) + rng.normal(0, 0.3, 600), index=index)
    backtest = run_backtest(spread_direction(z, THETA), np.array([1.0, -0.6, -0.4]), prices)
    splits = pd.Series(np.repeat(["train", "test", "validation"], 200), index=index, name="split")
    return backtest, splits


def test_partitions_add_up_to_the_whole_run(run):
    backtest, splits = run
    metrics = split_metrics(backtest, splits)
    assert list(metrics) == ["train", "test", "validation"]
    assert sum(m.pnl_neto for m in metrics.values()) == pytest.approx(backtest.diario["pnl_neto"].sum())
    growth = np.prod([1 + m.retorno_total for m in metrics.values()])
    assert growth == pytest.approx(backtest.equity.iloc[-1] / INITIAL_CAPITAL)
    assert sum(m.n_operaciones for m in metrics.values()) == len(backtest.operaciones)


def test_trade_log_has_the_dashboard_columns_and_its_costs_add_up(run):
    backtest, splits = run
    trades = trade_artifact(backtest, splits)
    assert tuple(trades.columns) == TRADE_COLUMNS
    assert len(trades) > 5
    np.testing.assert_allclose(trades["pnl_bruto"] - trades["comisiones"] - trades["prestamo"], trades["pnl_neto"])
    assert (trades["split"].to_numpy() == splits.loc[trades["entrada"]].to_numpy()).all()


def test_equity_artifact(run):
    backtest, splits = run
    equity = equity_artifact(backtest, splits)
    assert equity.index.equals(backtest.diario.index)
    assert (equity["drawdown"] <= 0).all()
    assert equity["rolling_sharpe"].iloc[:251].isna().all() and equity["rolling_sharpe"].iloc[-1:].notna().all()
