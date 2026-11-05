"""Unit tests for strategy/vecm.py on simulated log prices (SEED = 42)."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.vecm import VECM

from config import SEED
from strategy.vecm import equilibrium_deviation, fit_vecm, model_half_life

BETA = np.array([1.0, -0.5, -0.5])
ALPHA = np.array([-0.05, 0.02, 0.03])  # beta'alpha = -0.075


def simulate_log_prices(alpha=ALPHA, beta=BETA, n_obs=3000, sigma=0.01, seed=SEED):
    """Log prices that follow dP_t = alpha * beta' P_{t-1} + e_t, starting at equilibrium."""
    rng = np.random.default_rng(seed)
    prices = np.empty((n_obs, len(beta)))
    prices[0] = 4.0
    for t in range(1, n_obs):
        prices[t] = prices[t - 1] + alpha * (beta @ prices[t - 1]) + sigma * rng.standard_normal(len(beta))
    index = pd.bdate_range("2014-01-02", periods=n_obs)
    return pd.DataFrame(prices, index=index, columns=["A", "B", "C"])


def test_recovers_known_adjustment_speeds():
    fit = fit_vecm(simulate_log_prices(), BETA)
    np.testing.assert_allclose(fit.alpha, ALPHA, atol=0.02)
    assert fit.is_error_correcting
    assert fit.beta_alpha == pytest.approx(BETA @ ALPHA, abs=0.02)
    assert fit.half_life == pytest.approx(model_half_life(BETA @ ALPHA), abs=3.0)


def test_matches_statsmodels_given_its_beta():
    log_prices = simulate_log_prices()
    reference = VECM(log_prices, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    fit = fit_vecm(log_prices, reference.beta[:, 0], k_ar_diff=1)
    np.testing.assert_allclose(fit.alpha, reference.alpha[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit.gamma[0], reference.gamma[:, :3], atol=1e-8)


def test_model_half_life():
    assert model_half_life(-0.1) == pytest.approx(-np.log(2) / np.log(0.9))
    assert model_half_life(-1.5) == pytest.approx(1.0)  # |1 - 1.5| = 0.5 halves every period
    assert model_half_life(0.05) == np.inf
    assert model_half_life(-2.5) == np.inf


def test_rejects_vector_of_wrong_length():
    with pytest.raises(ValueError):
        fit_vecm(simulate_log_prices(n_obs=50), [1.0, -1.0])


def test_equilibrium_deviation_with_fixed_and_dynamic_weights():
    index = pd.bdate_range("2020-01-01", periods=3)
    log_prices = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 1.0, 0.5]}, index=index)

    fixed = equilibrium_deviation(log_prices, [1.0, -2.0], c=0.5)
    assert fixed.tolist() == [-3.5, -0.5, 1.5]

    betas = pd.DataFrame({"beta_A": [1.0, 1.0, 1.0], "beta_B": [-2.0, -1.0, 0.0]}, index=index)
    assert equilibrium_deviation(log_prices, betas).tolist() == [-3.0, 1.0, 3.0]


def test_equilibrium_deviation_requires_every_weight():
    index = pd.bdate_range("2020-01-01", periods=2)
    log_prices = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}, index=index)
    with pytest.raises(ValueError):
        equilibrium_deviation(log_prices, pd.DataFrame({"beta_A": [1.0, 1.0]}, index=index))
