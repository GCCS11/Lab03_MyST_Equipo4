"""Prueba 4: innovaciones de ambos filtros, mas la causalidad y la mecanica de cada filtro."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import acorr_ljungbox

from kalman.calibration import calibrate_signal_filter, calibrate_weights_filter, half_life
from kalman.signal_filter import run_signal_filter
from kalman.simulation import LEG, THETA0, simulate_signal_model, simulate_weights_model
from kalman.weights_filter import run_weights_filter

BURN_IN = 60
BETA0 = np.concatenate([[1.0], THETA0])


def assert_white_with_unit_variance(innov_norm):
    """Varianza cercana a uno y Ljung-Box no significativo a 10 rezagos."""
    clean = innov_norm.iloc[BURN_IN:].dropna()
    assert clean.var() == pytest.approx(1.0, abs=0.15)
    assert acorr_ljungbox(clean, lags=[10])["lb_pvalue"].iloc[0] > 0.01


def signal_params(q=1e-7, r=1e-4, phi=0.0, lam=0.97, burn_in=BURN_IN):
    return {"Q": q, "phi": phi, "lam": lam, "mu0": 0.0, "R0": r, "burn_in": burn_in}


def test_filter_1_innovations_are_white_when_data_follow_the_model():
    log_prices, _, params = simulate_weights_model()
    frame = run_weights_filter(log_prices, BETA0, LEG, params)
    assert_white_with_unit_variance(frame["innov_norm"])


@pytest.mark.parametrize("phi", [0.0, 0.95])
def test_filter_2_innovations_are_white_when_data_follow_the_model(phi):
    spread = simulate_signal_model(phi=phi)
    frame = run_signal_filter(spread, signal_params(phi=phi))
    assert_white_with_unit_variance(frame["innov_norm"])


def test_ignoring_the_persistence_leaves_autocorrelated_innovations():
    log_prices, _, params = simulate_weights_model()
    frame = run_weights_filter(log_prices, BETA0, LEG, {**params, "phi": 0.0, "R": 1e-3})
    assert frame["innov_norm"].iloc[BURN_IN:].autocorr(1) > 0.5


def test_filter_1_tracks_weights_that_drift():
    log_prices, true_theta, params = simulate_weights_model(q=1e-6)
    frame = run_weights_filter(log_prices, BETA0, LEG, params)
    estimated = frame[[f"beta_{ticker}" for ticker in true_theta.columns]].to_numpy()
    assert np.median(np.abs(estimated - true_theta.to_numpy())[BURN_IN:]) < 0.02


def test_high_q_makes_the_filter_chase_the_spread():
    log_prices, _, params = simulate_weights_model()
    slow = run_weights_filter(log_prices, BETA0, LEG, {**params, "Q": 0.0})
    fast = run_weights_filter(log_prices, BETA0, LEG, {**params, "Q": 0.1 * params["R"]})
    assert half_life(fast["spread"]) < 0.5 * half_life(slow["spread"])


def test_filter_1_first_step_matches_hand_calculation():
    log_prices = pd.DataFrame({"A": [1.0, 1.2, 1.2], "B": [2.0, 2.0, 2.0]})
    params = {"Q": 0.01, "R": 0.04, "phi": 0.0, "c": 0.0}
    frame = run_weights_filter(log_prices, [1.0, -0.5], "A", params)
    assert frame["spread"].iloc[1] == pytest.approx(0.2)
    assert frame["innov"].iloc[1] == pytest.approx(0.2)
    assert frame["innov_var"].iloc[1] == pytest.approx(0.12)
    assert frame["gain_B"].iloc[1] == pytest.approx(-1 / 3)
    assert frame["pvar_B"].iloc[1] == pytest.approx(0.02 / 3)
    assert frame["beta_B"].iloc[2] == pytest.approx(-0.5 - 0.2 / 3)


def test_filter_2_first_step_matches_hand_calculation():
    spread = pd.Series([0.0, 2.0, 2.0])
    frame = run_signal_filter(spread, {"Q": 0.01, "phi": 0.0, "lam": 0.9, "mu0": 0.0, "R0": 1.0})
    assert frame["z"].iloc[1] == pytest.approx(2.0)
    assert frame["innov_var"].iloc[1] == pytest.approx(1.02)
    assert frame["gain"].iloc[1] == pytest.approx(0.02 / 1.02)
    assert frame["mu_prior"].iloc[2] == pytest.approx(2 * 0.02 / 1.02)
    assert frame["sigma_prior"].iloc[2] == pytest.approx(np.sqrt(1.3))


def test_posterior_z_shrinks_by_one_minus_the_gain_in_the_local_level():
    spread = simulate_signal_model()
    frame = run_signal_filter(spread, signal_params(q=1e-5, burn_in=0))
    posterior_mean = frame["mu_prior"].shift(-1)
    z_posterior = (spread - posterior_mean) / frame["sigma_prior"]
    expected = (1 - frame["gain"]) * frame["z"]
    np.testing.assert_allclose(z_posterior.iloc[1:-1], expected.iloc[1:-1])


@pytest.mark.parametrize("cut", [1200, 1800, 2400])
def test_filters_do_not_use_future_data(cut):
    log_prices, _, params = simulate_weights_model()
    shocked = log_prices.copy()
    shocked.iloc[cut + 1 :] += 0.5
    full = run_weights_filter(log_prices, BETA0, LEG, params)
    other = run_weights_filter(shocked, BETA0, LEG, params)
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], other.iloc[: cut + 1])

    z_full = run_signal_filter(full["spread"], signal_params())
    z_other = run_signal_filter(other["spread"], signal_params())
    pd.testing.assert_frame_equal(z_full.iloc[: cut + 1], z_other.iloc[: cut + 1])


def test_z_is_missing_during_the_burn_in():
    frame = run_signal_filter(simulate_signal_model(), signal_params())
    assert frame["z"].iloc[:BURN_IN].isna().all()
    assert frame["z"].iloc[BURN_IN:].notna().all()


@pytest.mark.parametrize("use_ar1", [True, False])
def test_calibration_recovers_static_filter_when_q_is_zero(use_ar1):
    log_prices, _, _ = simulate_weights_model(q=0.0)
    params = calibrate_weights_filter(log_prices, BETA0, LEG, q_ratio=0.0, use_ar1=use_ar1)
    frame = run_weights_filter(log_prices, BETA0, LEG, params)
    assert (frame["beta_A2"] == THETA0[0]).all()
    assert params["phi"] == (pytest.approx(0.95, abs=0.02) if use_ar1 else 0.0)


def test_signal_calibration_defaults_to_the_local_level():
    params = calibrate_signal_filter(simulate_signal_model())
    assert params["phi"] == 0.0
    assert params["burn_in"] == 60
