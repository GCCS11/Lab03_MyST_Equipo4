"""Prueba 3: contabilidad del motor de backtest (backtest/engine.py).

Verifica la identidad de caja de cada paso (equity = cash + valor de mercado), que el motor
nunca mira precios futuros, que la banda rebalancea menos que el modo diario y, con un
golden-file, que la salida completa de un escenario fijo no cambia sin que alguien lo note.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestParams, run_backtest
from strategy.signals import SignalParams, spread_direction

GOLDEN_DIR = __file__.rsplit("/", 1)[0] + "/golden"
TICKERS = ["A", "B", "C"]
BETA = np.array([1.0, -0.5, -0.5])
BASE_PRICE = np.array([10.0, 20.0, 40.0])


def make_prices(n_periods, drift=(0.05, -0.03, 0.02), start="2021-01-04"):
    index = pd.bdate_range(start, periods=n_periods)
    drift = np.asarray(drift)
    data = [BASE_PRICE + drift * t for t in range(n_periods)]
    return pd.DataFrame(data, index=index, columns=TICKERS)


def make_rule(values, theta=None, start="2021-01-04"):
    """Regla a partir de una lista de z-scores, usando strategy.signals.spread_direction tal
    cual la usara main.py una vez exista theta.json."""
    theta = theta or SignalParams(z_in=2.0, z_out=0.5, z_stop=3.0, h_max=10)
    index = pd.bdate_range(start, periods=len(values))
    z = pd.Series(values, index=index, dtype=float)
    return spread_direction(z, theta)


Z_VALUES = [0.0, -2.5, -2.0, -1.5, -0.3, 0.0, 0.0, 2.6, 2.2, 0.3, 0.0, 0.0]


def test_equity_matches_cash_plus_market_value():
    prices = make_prices(len(Z_VALUES))
    rule = make_rule(Z_VALUES)
    result = run_backtest(rule, BETA, prices, BacktestParams())

    market_value = (result.posiciones * prices).sum(axis=1)
    np.testing.assert_allclose(result.diario["equity"], result.diario["cash"] + market_value)


def test_daily_pnl_decomposition_explains_the_equity_change():
    prices = make_prices(len(Z_VALUES))
    rule = make_rule(Z_VALUES)
    diario = run_backtest(rule, BETA, prices, BacktestParams()).diario

    pnl_neto = diario["pnl_bruto"] - diario["comisiones"] - diario["prestamo"]
    np.testing.assert_allclose(diario["pnl_neto"], pnl_neto)

    equity_change = diario["equity"].diff().fillna(diario["equity"].iloc[0] - 1_000_000.0)
    np.testing.assert_allclose(equity_change, diario["pnl_neto"])


def test_execution_is_shifted_one_period_after_the_signal():
    prices = make_prices(len(Z_VALUES))
    rule = make_rule(Z_VALUES)
    diario = run_backtest(rule, BETA, prices, BacktestParams()).diario

    np.testing.assert_array_equal(diario["direccion"].to_numpy()[1:], rule["d"].to_numpy()[:-1])
    assert diario["direccion"].iloc[0] == 0


def test_no_look_ahead_future_prices_do_not_change_past_equity():
    rule = make_rule(Z_VALUES)
    base = make_prices(len(Z_VALUES))
    changed = base.copy()
    cutoff = 7
    changed.iloc[cutoff:] = changed.iloc[cutoff:] * 5.0  # altera solo el futuro

    diario_base = run_backtest(rule, BETA, base, BacktestParams()).diario
    diario_changed = run_backtest(rule, BETA, changed, BacktestParams()).diario

    pd.testing.assert_frame_equal(diario_base.iloc[:cutoff], diario_changed.iloc[:cutoff])


def test_truncating_the_series_does_not_change_the_already_computed_steps():
    prices = make_prices(len(Z_VALUES))
    rule = make_rule(Z_VALUES)
    cutoff = 9

    full = run_backtest(rule, BETA, prices, BacktestParams()).diario
    partial = run_backtest(rule.iloc[:cutoff], BETA, prices.iloc[:cutoff], BacktestParams()).diario

    pd.testing.assert_frame_equal(full.iloc[:cutoff], partial)


def test_band_rebalances_less_than_the_daily_mode():
    index = pd.bdate_range("2021-01-04", periods=6)
    prices = pd.DataFrame(np.tile(BASE_PRICE, (6, 1)), index=index, columns=TICKERS)
    betas = pd.DataFrame(
        {"beta_A": 1.0, "beta_B": [-0.50, -0.51, -0.52, -0.53, -0.54, -0.55], "beta_C": -0.5},
        index=index,
    )
    rule = pd.DataFrame(
        {"d": [1, 1, 1, 1, 1, 0], "exit_reason": ["", "", "", "", "", "reversion"]}, index=index
    )

    daily = run_backtest(rule, betas, prices, BacktestParams(band=0.0))
    banded = run_backtest(rule, betas, prices, BacktestParams(band=0.05))

    assert banded.diario["comisiones"].sum() < daily.diario["comisiones"].sum()
    assert (banded.posiciones.diff().iloc[2:] == 0).all().all()


def test_operaciones_log_matches_the_rule_shifted_by_one_period():
    prices = make_prices(len(Z_VALUES))
    rule = make_rule(Z_VALUES)
    operaciones = run_backtest(rule, BETA, prices, BacktestParams()).operaciones

    assert operaciones["entrada"].tolist() == [pd.Timestamp("2021-01-06"), pd.Timestamp("2021-01-14")]
    assert operaciones["salida"].tolist() == [pd.Timestamp("2021-01-11"), pd.Timestamp("2021-01-18")]
    assert operaciones["direccion"].tolist() == [1, -1]
    assert operaciones["motivo"].tolist() == ["reversion", "reversion"]


def test_golden_file_contabilidad():
    prices = make_prices(len(Z_VALUES))
    rule = make_rule(Z_VALUES)
    result = run_backtest(rule, BETA, prices, BacktestParams())

    diario_esperado = pd.read_csv(f"{GOLDEN_DIR}/contabilidad_diario.csv", index_col="date", parse_dates=True)
    pd.testing.assert_frame_equal(result.diario, diario_esperado, check_names=False, check_freq=False)

    operaciones_esperadas = pd.read_csv(
        f"{GOLDEN_DIR}/contabilidad_operaciones.csv", parse_dates=["entrada", "salida"]
    )
    pd.testing.assert_frame_equal(result.operaciones, operaciones_esperadas)
