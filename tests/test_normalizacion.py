"""Prueba 2: el vector de pesos mantiene escala y signo entre reestimaciones consecutivas."""
import numpy as np
import pandas as pd
import pytest

from config import ARTIFACTS_DIR
from kalman.simulation import LEG, THETA0, simulate_weights_model
from kalman.weights_filter import run_weights_filter

BETA0 = np.concatenate([[1.0], THETA0])


@pytest.fixture
def filtered():
    log_prices, _, params = simulate_weights_model(q=1e-7)
    return log_prices, params, run_weights_filter(log_prices, BETA0, LEG, params)


def test_normalizing_leg_stays_at_plus_one(filtered):
    _, _, frame = filtered
    assert (frame[f"beta_{LEG}"] == 1.0).all()


def test_scale_and_sign_of_the_initial_vector_do_not_matter(filtered):
    log_prices, params, frame = filtered
    rescaled = run_weights_filter(log_prices, -3.0 * BETA0, LEG, params)
    pd.testing.assert_frame_equal(frame, rescaled)


def test_signs_match_the_initial_vector(filtered):
    _, _, frame = filtered
    weights = frame[[f"beta_{ticker}" for ticker in ["A2", "A3", "A4", "A5", "A6"]]].to_numpy()
    assert (np.sign(weights) == np.sign(THETA0)).all()


@pytest.mark.parametrize("t", [500, 1500, 2400])
def test_consecutive_reestimations_keep_past_weights(filtered, t):
    log_prices, params, frame = filtered
    previous = run_weights_filter(log_prices.iloc[: t + 1], BETA0, LEG, params)
    following = run_weights_filter(log_prices.iloc[: t + 2], BETA0, LEG, params)
    weights = frame.filter(like="beta_").iloc[: t + 1]
    pd.testing.assert_frame_equal(previous.filter(like="beta_"), weights)
    pd.testing.assert_frame_equal(following.filter(like="beta_").iloc[: t + 1], weights)


def test_rolling_johansen_has_no_sign_flips():
    path = ARTIFACTS_DIR / "eigvec_movil.csv"
    if not path.exists():
        pytest.skip("Falta artifacts/eigvec_movil.csv (tarea G4)")
    vectors = pd.read_csv(path, index_col="date").to_numpy()
    assert ((vectors[1:] * vectors[:-1]).sum(axis=1) > 0).all()
