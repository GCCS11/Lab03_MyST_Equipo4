"""Contabilidad aproximada y criterio de eleccion del walk-forward en entrenamiento."""
import numpy as np
import pandas as pd
import pytest

from config import BORROW_RATE_ANNUAL, COMMISSION, POSITION_FRACTION, TRADING_DAYS
from kalman.simulation import LEG, THETA0, simulate_weights_model
from kalman.tuning import choose_configuration, run_filters, strategy_returns


def test_position_earns_only_after_the_next_close():
    index = pd.bdate_range("2020-01-01", periods=5)
    log_prices = pd.DataFrame({"A": np.log([100, 100, 100, 110, 110.0]), "B": np.log([50.0] * 5)}, index=index)
    betas = pd.DataFrame({"A": [1.0] * 5, "B": [-1.0] * 5}, index=index)
    direction = pd.Series([0, 1, 1, 1, 1], index=index)
    net = strategy_returns(log_prices, betas, direction)

    weight = POSITION_FRACTION / 2
    daily_borrow = BORROW_RATE_ANNUAL / TRADING_DAYS * weight
    assert net.iloc[0] == 0.0 and net.iloc[1] == 0.0
    assert net.iloc[2] == pytest.approx(-COMMISSION * 2 * weight - daily_borrow)
    assert net.iloc[3] == pytest.approx(weight * 0.10 - daily_borrow)


def test_choice_respects_half_life_then_sharpe_then_lower_q():
    table = pd.DataFrame(
        {
            "weight_q_ratio": [1e-2, 1e-4, 1e-5, 1e-3],
            "median_half_life": [1.0, 10.0, 12.0, 80.0],
            "sharpe": [9.0, 0.5, 0.5, 8.0],
        }
    )
    assert choose_configuration(table)["weight_q_ratio"] == 1e-5


def test_run_filters_chains_both_filters_with_frozen_parameters():
    log_prices, _, truth = simulate_weights_model()
    params = {
        "beta0": list(np.concatenate([[1.0], THETA0])),
        "pata_normalizada": LEG,
        "filtro1": truth,
        "filtro2": {"Q": 1e-8, "R": 2.5e-5, "phi": 0.9, "lam": 0.99, "mu0": 0.1, "P0": 1e-6, "burn_in": 60, "entrada": "spread"},
    }
    first, second = run_filters(log_prices, params)
    assert first["spread"].iloc[0] == pytest.approx(log_prices.iloc[0].to_numpy() @ np.array(params["beta0"]))
    assert second["z"].iloc[:60].isna().all() and second["z"].iloc[60:].notna().all()
