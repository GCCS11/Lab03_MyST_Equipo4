"""Unit tests for performance/metrics.py: Sharpe, Sortino, Calmar, max drawdown, win rate,
profit factor, atribucion de costos y el escritor de artifacts/metricas.json.

Los escenarios son sinteticos y los valores esperados se calculan a mano con formulas simples
(no se usa ningun precio ni division real de test o validacion).
"""
import json

import numpy as np
import pandas as pd
import pytest

from performance.metrics import compute_metrics, write_metricas_json

INITIAL_CAPITAL = 1_000_000.0


def make_diario(equity_values, pnl_bruto=0.0, comisiones=0.0, prestamo=0.0):
    index = pd.bdate_range("2021-01-04", periods=len(equity_values))
    equity = pd.Series(equity_values, index=index, dtype=float)
    n = len(equity)
    pnl_bruto = np.broadcast_to(pnl_bruto, n).astype(float)
    comisiones = np.broadcast_to(comisiones, n).astype(float)
    prestamo = np.broadcast_to(prestamo, n).astype(float)
    return pd.DataFrame({
        "direccion": 0, "pnl_bruto": pnl_bruto, "comisiones": comisiones, "prestamo": prestamo,
        "pnl_neto": pnl_bruto - comisiones - prestamo, "cash": equity, "equity": equity,
    }, index=index)


def make_operaciones(pnl_neto_values):
    index = pd.bdate_range("2021-01-04", periods=len(pnl_neto_values))
    return pd.DataFrame({
        "entrada": index, "salida": index, "direccion": 1,
        "pnl_neto": pnl_neto_values, "motivo": "reversion",
    })


def test_sharpe_and_sortino_match_the_hand_formula():
    returns = np.array([0.0, 0.02, -0.01, 0.03, -0.02])
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    diario = make_diario(equity)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    mean_r, std_r = returns.mean(), returns.std(ddof=0)
    sharpe_esperado = mean_r / std_r * np.sqrt(252)
    downside = np.where(returns < 0, returns, 0.0)
    sortino_esperado = mean_r / np.sqrt((downside ** 2).mean()) * np.sqrt(252)

    assert metrics.sharpe == pytest.approx(sharpe_esperado)
    assert metrics.sortino == pytest.approx(sortino_esperado)
    assert "sharpe" not in metrics.notas
    assert "sortino" not in metrics.notas


def test_max_drawdown_with_a_single_clean_peak_and_trough():
    equity = [1_000_000.0, 1_200_000.0, 900_000.0, 1_000_000.0, 1_100_000.0]
    diario = make_diario(equity)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    assert metrics.max_drawdown == pytest.approx(0.25)
    assert metrics.max_drawdown_inicio == pd.Timestamp("2021-01-05")
    assert metrics.max_drawdown_fin == pd.Timestamp("2021-01-06")


def test_calmar_is_annualized_return_over_max_drawdown():
    equity = [1_000_000.0, 1_200_000.0, 900_000.0, 1_000_000.0, 1_100_000.0]
    diario = make_diario(equity)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    esperado = metrics.retorno_anualizado / metrics.max_drawdown
    assert metrics.calmar == pytest.approx(esperado)


def test_calmar_is_zero_with_a_reason_when_there_is_no_drawdown():
    equity = [1_000_000.0, 1_010_000.0, 1_020_000.0, 1_030_000.0]
    diario = make_diario(equity)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    assert metrics.max_drawdown == 0.0
    assert metrics.calmar == 0.0
    assert metrics.notas["calmar"]


def test_win_rate_and_profit_factor_with_known_trades():
    operaciones = make_operaciones([100.0, -50.0, 200.0, -100.0])
    diario = make_diario([INITIAL_CAPITAL] * 4)

    metrics = compute_metrics(diario, operaciones, initial_capital=INITIAL_CAPITAL)

    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(300.0 / 150.0)


def test_win_rate_and_profit_factor_are_zero_with_a_reason_when_there_are_no_trades():
    diario = make_diario([INITIAL_CAPITAL] * 4)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    assert metrics.n_operaciones == 0
    assert metrics.win_rate == 0.0
    assert metrics.profit_factor == 0.0
    assert metrics.notas["win_rate"] == "sin operaciones cerradas"
    assert metrics.notas["profit_factor"] == "sin operaciones cerradas"


def test_profit_factor_is_zero_with_a_reason_when_there_are_no_losing_trades():
    operaciones = make_operaciones([100.0, 50.0])
    diario = make_diario([INITIAL_CAPITAL] * 2)

    metrics = compute_metrics(diario, operaciones, initial_capital=INITIAL_CAPITAL)

    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.profit_factor == 0.0
    assert metrics.notas["profit_factor"]


def test_cost_attribution_adds_up_with_known_numbers():
    diario = make_diario([INITIAL_CAPITAL] * 3, pnl_bruto=1000.0 / 3, comisiones=100.0 / 3, prestamo=50.0 / 3)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    assert metrics.pnl_bruto == pytest.approx(1000.0)
    assert metrics.comisiones == pytest.approx(100.0)
    assert metrics.prestamo == pytest.approx(50.0)
    assert metrics.pnl_neto == pytest.approx(850.0)
    assert metrics.pct_comisiones == pytest.approx(0.10)
    assert metrics.pct_prestamo == pytest.approx(0.05)


def test_cost_percentages_are_zero_with_a_reason_when_gross_pnl_is_zero():
    diario = make_diario([INITIAL_CAPITAL] * 3, pnl_bruto=0.0, comisiones=10.0, prestamo=5.0)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    assert metrics.pct_comisiones == 0.0
    assert metrics.pct_prestamo == 0.0
    assert metrics.notas["pct_comisiones"]
    assert metrics.notas["pct_prestamo"]


def test_rolling_sharpe_has_nan_during_burn_in_and_is_finite_after():
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.01])
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    diario = make_diario(equity)

    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL, trading_days=3)

    assert metrics.rolling_sharpe.iloc[:2].isna().all()
    assert metrics.rolling_sharpe.iloc[2:].notna().all()


def test_write_metricas_json_has_no_null_values_and_matches_dashboard_keys(tmp_path):
    equity = [1_000_000.0, 1_010_000.0, 990_000.0, 1_020_000.0]
    diario = make_diario(equity, pnl_bruto=10.0, comisiones=1.0, prestamo=0.5)
    metrics = compute_metrics(diario, make_operaciones([]), initial_capital=INITIAL_CAPITAL)

    path = tmp_path / "metricas.json"
    write_metricas_json(path, {"entrenamiento_sintetico": metrics})
    payload = json.loads(path.read_text(encoding="utf-8"))

    claves_dashboard = (
        "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "profit_factor",
        "retorno_total", "n_operaciones", "pnl_bruto", "comisiones", "prestamo", "pnl_neto",
    )
    bloque = payload["entrenamiento_sintetico"]
    for clave in claves_dashboard:
        assert clave in bloque
        assert bloque[clave] is not None
        assert np.isfinite(bloque[clave])
    assert bloque["notas"]["win_rate"] == "sin operaciones cerradas"
