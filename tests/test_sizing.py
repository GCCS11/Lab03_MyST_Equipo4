"""Unit tests for strategy/sizing.py: 80% split by |beta|, signs and the no-rebalance band."""
import numpy as np
import pandas as pd
import pytest

from config import SEED
from strategy.sizing import next_shares, target_position, target_shares

BETA = np.array([1.0, -0.5, -0.5])
PRICES = np.array([10.0, 20.0, 40.0])
EQUITY = 1_000_000.0


def test_target_splits_80_percent_by_abs_beta():
    shares = target_position(1, BETA, PRICES, EQUITY)
    np.testing.assert_allclose(shares, [40_000.0, -10_000.0, -5_000.0])  # 400k, 200k, 200k
    np.testing.assert_allclose(-target_position(-1, BETA, PRICES, EQUITY), shares)
    np.testing.assert_allclose(target_position(0, BETA, PRICES, EQUITY), 0.0)


def test_gross_exposure_is_80_percent_for_any_weights():
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        betas, prices = rng.normal(size=6), rng.uniform(5, 300, size=6)
        notional = target_position(1, betas, prices, EQUITY) * prices
        assert np.abs(notional).sum() == pytest.approx(0.8 * EQUITY)
        np.testing.assert_allclose(notional / np.abs(notional).sum(), betas / np.abs(betas).sum())


def test_open_position_rejects_missing_weights_or_prices():
    with pytest.raises(ValueError):
        target_position(1, [1.0, np.nan, -0.5], PRICES, EQUITY)
    with pytest.raises(ValueError):
        target_position(-1, BETA, [10.0, np.nan, 40.0], EQUITY)


def test_band_rebalances_only_the_legs_that_drifted():
    current = target_position(1, BETA, PRICES, EQUITY)
    drifted = np.array([1.0, -0.52, -0.5])  # gaps: A 0.40%, B 0.59%, C 0.20% of equity
    target = target_position(1, drifted, PRICES, EQUITY)

    np.testing.assert_allclose(next_shares(current, 1, 1, drifted, PRICES, EQUITY, band=0.0), target)
    np.testing.assert_allclose(next_shares(current, 1, 1, drifted, PRICES, EQUITY, band=np.inf), current)
    partial = next_shares(current, 1, 1, drifted, PRICES, EQUITY, band=0.003)
    np.testing.assert_allclose(partial, [target[0], target[1], current[2]])


def test_opening_closing_and_flipping_trade_the_whole_target():
    current = target_position(1, BETA, PRICES, EQUITY)
    np.testing.assert_allclose(
        next_shares(np.zeros(3), 0, 1, BETA, PRICES, EQUITY, band=np.inf), current
    )
    np.testing.assert_allclose(next_shares(current, 1, 0, BETA, PRICES, EQUITY, band=np.inf), 0.0)
    np.testing.assert_allclose(
        next_shares(current, 1, -1, BETA, PRICES, EQUITY, band=np.inf), -current
    )


def test_target_shares_follows_the_direction_path():
    index = pd.bdate_range("2021-01-04", periods=5)
    prices = pd.DataFrame(np.tile(PRICES, (5, 1)), index=index, columns=["A", "B", "C"])
    d = pd.Series([0, 1, 1, 0, -1], index=index)
    held = target_shares(d, BETA, prices, EQUITY)
    entry = target_position(1, BETA, PRICES, EQUITY)
    np.testing.assert_allclose(held.to_numpy(), [0 * entry, entry, entry, 0 * entry, -entry])


def test_band_keeps_entry_shares_while_daily_rebalancing_follows_beta():
    index = pd.bdate_range("2021-01-04", periods=4)
    prices = pd.DataFrame(np.tile(PRICES, (4, 1)), index=index, columns=["A", "B", "C"])
    betas = pd.DataFrame(
        {"beta_A": 1.0, "beta_B": [-0.50, -0.51, -0.52, -0.53], "beta_C": -0.5}, index=index
    )
    d = pd.Series([1, 1, 1, 1], index=index)
    daily = target_shares(d, betas, prices, EQUITY, band=0.0)
    banded = target_shares(d, betas, prices, EQUITY, band=0.05)
    assert daily.diff().abs().sum().sum() > 0
    assert (banded.diff().iloc[1:] == 0).all().all()


def test_equity_series_scales_the_position():
    index = pd.bdate_range("2021-01-04", periods=2)
    prices = pd.DataFrame(np.tile(PRICES, (2, 1)), index=index, columns=["A", "B", "C"])
    equity = pd.Series([EQUITY, 2 * EQUITY], index=index)
    held = target_shares(pd.Series([1, 1], index=index), BETA, prices, equity)
    np.testing.assert_allclose(held.iloc[1], 2 * held.iloc[0])
