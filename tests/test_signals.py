"""Unit tests for strategy/signals.py: band rule, lock after a stop, h_max and causality."""
import numpy as np
import pandas as pd
import pytest

from config import SEED
from strategy.signals import SignalParams, max_holding, spread_direction

THETA = SignalParams(z_in=2.0, z_out=0.5, z_stop=3.0, h_max=10)


def run(values, theta=THETA):
    """Apply the rule to a list of z-scores indexed by business days."""
    z = pd.Series(values, index=pd.bdate_range("2021-01-04", periods=len(values)), dtype=float)
    return spread_direction(z, theta)


def test_long_trade_opens_and_closes_on_reversion():
    out = run([0.0, -2.5, -1.5, -0.3])
    assert out["d"].tolist() == [0, 1, 1, 0]
    assert out["holding"].tolist() == [0, 0, 1, 2]
    assert out["exit_reason"].tolist() == ["", "", "", "reversion"]


def test_short_trade_mirrors_the_long():
    out = run([0.0, 2.5, 1.0, 0.2])
    assert out["d"].tolist() == [0, -1, -1, 0]
    assert out["exit_reason"].iloc[3] == "reversion"


def test_lock_after_stop_waits_until_the_exit_band():
    out = run([0.0, -2.5, -3.2, -2.6, -2.2, -0.3, -2.4])
    assert out["d"].tolist() == [0, 1, 0, 0, 0, 0, 1]
    assert out["exit_reason"].iloc[2] == "stop"


def test_h_max_closes_and_locks():
    out = run([-2.5, -2.2, -2.1, -2.3, -2.5], SignalParams(2.0, 0.5, 3.0, h_max=3))
    assert out["d"].tolist() == [1, 1, 1, 0, 0]
    assert out["exit_reason"].iloc[3] == "h_max"
    assert out["holding"].iloc[3] == 3


def test_overshoot_through_the_band_counts_as_reversion():
    out = run([-2.5, 2.5])
    assert out["d"].tolist() == [1, 0]
    assert out["exit_reason"].iloc[1] == "reversion"


def test_no_entry_beyond_the_stop_or_during_burn_in():
    assert run([-3.5, 3.5]).d.tolist() == [0, 0]
    assert run([np.nan, np.nan, -2.5]).d.tolist() == [0, 0, 1]


def test_never_holds_two_positions_or_flips_in_one_step():
    rng = np.random.default_rng(SEED)
    d = run(np.cumsum(rng.normal(0, 0.6, 2000)) % 8 - 4)["d"].to_numpy()
    assert set(np.unique(d)) <= {-1, 0, 1}
    assert not np.any(d[1:] * d[:-1] == -1)


def test_rule_is_causal_under_truncation():
    rng = np.random.default_rng(SEED)
    z = pd.Series(rng.normal(0, 1.5, 500), index=pd.bdate_range("2021-01-04", periods=500))
    full = spread_direction(z, THETA)
    for t in (50, 250, 499):
        partial = spread_direction(z.iloc[: t + 1], THETA)
        pd.testing.assert_frame_equal(partial, full.iloc[: t + 1])


def test_params_are_validated_and_accept_a_dict():
    with pytest.raises(ValueError):
        SignalParams(z_in=1.0, z_out=1.5, z_stop=3.0, h_max=5)
    with pytest.raises(ValueError):
        SignalParams(z_in=2.0, z_out=0.5, z_stop=3.0, h_max=0)
    out = run([-2.5], {"z_in": 2.0, "z_out": 0.5, "z_stop": 3.0, "h_max": 10})
    assert out["d"].tolist() == [1]


def test_max_holding_from_half_life():
    assert max_holding(8.9) == 18
    assert max_holding(0.2) == 1
    with pytest.raises(ValueError):
        max_holding(np.inf)
