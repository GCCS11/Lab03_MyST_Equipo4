"""Series sinteticas que siguen el modelo de cada filtro, para probar sin datos reales."""
import numpy as np
import pandas as pd

from config import SEED

TICKERS = ["A1", "A2", "A3", "A4", "A5", "A6"]
LEG = "A1"
THETA0 = np.array([-0.30, -0.25, -0.20, -0.15, -0.10])


def simulate_weights_model(n_rows=2500, phi=0.95, q=1e-8, r=2.5e-5, c=0.1, seed=SEED):
    """Seis log-precios cointegrados con pesos que siguen un paseo aleatorio de varianza q.

    Regresa los log-precios, la trayectoria real de los pesos y los parametros verdaderos.
    """
    rng = np.random.default_rng(seed)
    n_others = len(THETA0)
    factor = np.cumsum(rng.normal(0, 0.008, n_rows))
    others = 4.0 + factor[:, None] + np.cumsum(rng.normal(0, 0.006, (n_rows, n_others)), axis=0)
    theta = THETA0 + np.cumsum(rng.normal(0, np.sqrt(q), (n_rows, n_others)), axis=0)
    error = np.zeros(n_rows)
    for t in range(1, n_rows):
        error[t] = phi * error[t - 1] + rng.normal(0, np.sqrt(r))
    leg = c - (others * theta).sum(axis=1) + error

    index = pd.bdate_range("2014-01-02", periods=n_rows)
    log_prices = pd.DataFrame(np.column_stack([leg, others]), index=index, columns=TICKERS)
    true_theta = pd.DataFrame(theta, index=index, columns=TICKERS[1:])
    return log_prices, true_theta, {"Q": q, "R": r, "phi": phi, "c": c}


def simulate_signal_model(n_rows=2500, phi=0.0, q=1e-7, r=1e-4, mu0=0.0, seed=SEED):
    """Spread con media en paseo aleatorio y desviacion AR(1); phi = 0 es el nivel local."""
    rng = np.random.default_rng(seed)
    mean = mu0 + np.cumsum(rng.normal(0, np.sqrt(q), n_rows))
    deviation = np.zeros(n_rows)
    for t in range(1, n_rows):
        deviation[t] = phi * deviation[t - 1] + rng.normal(0, np.sqrt(r))
    index = pd.bdate_range("2014-01-02", periods=n_rows)
    return pd.Series(mean + deviation, index=index, name="spread")
