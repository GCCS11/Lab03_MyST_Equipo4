"""Parametros de ambos filtros, estimados unicamente con la particion de entrenamiento.

P0 sale de una pasada previa sobre los mismos datos: la varianza con la que termina. Si P0 = Q,
P_t tarda anios en asentarse y el burn-in no cumpliria su funcion (S13 p16).
"""
import numpy as np
import pandas as pd

from kalman.signal_filter import run_signal_filter
from kalman.weights_filter import normalize_beta, run_weights_filter

DEFAULT_LAMBDA = 0.97
DEFAULT_BURN_IN = 60
Q_RATIOS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def fit_ar1(series):
    """Ajusta S_t = a + phi S_{t-1} + e_t; regresa phi, la media de largo plazo y var(e_t)."""
    values = np.asarray(series, dtype=float)
    design = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    (intercept, phi), *_ = np.linalg.lstsq(design, values[1:], rcond=None)
    if not 0.0 < phi < 1.0:
        raise ValueError(f"La serie no revierte con un AR(1) positivo: phi = {phi:.4f}")
    residuals = values[1:] - intercept - phi * values[:-1]
    return float(phi), float(intercept / (1.0 - phi)), float(residuals.var(ddof=2))


def half_life(series):
    """Vida media del AR(1); infinita si no revierte y cero si no hay persistencia."""
    values = np.asarray(series, dtype=float)
    phi = np.polyfit(values[:-1], values[1:], 1)[0]
    if phi >= 1.0:
        return np.inf
    return float(-np.log(2) / np.log(phi)) if phi > 0.0 else 0.0


def log_likelihood(frame):
    """Log-verosimilitud gaussiana a partir de innov e innov_var de cualquiera de los filtros."""
    clean = frame[["innov", "innov_var"]].dropna()
    return float(-0.5 * (np.log(2 * np.pi * clean["innov_var"]) + clean["innov"] ** 2 / clean["innov_var"]).sum())


def calibrate_weights_filter(train_log_prices, beta0, leg, q_ratio=0.0, use_ar1=True, state_intercept=False):
    """Regresa Q, R, phi y c del filtro 1, con Q = q_ratio * R.

    Con use_ar1=False da el caso phi = 0: R y c salen de la varianza y la media del spread.
    Con state_intercept=True agrega Q_c = Q; junto con use_ar1=False es S14 p18.
    """
    position = list(train_log_prices.columns).index(leg)
    spread = train_log_prices.to_numpy(dtype=float) @ normalize_beta(beta0, position)
    if use_ar1:
        phi, c, r = fit_ar1(spread)
    else:
        phi, c, r = 0.0, float(spread.mean()), float(spread.var(ddof=1))
    params = {"Q": q_ratio * r, "R": r, "phi": phi, "c": c}
    if state_intercept:
        params["Q_c"] = params["Q"]
    frame = run_weights_filter(train_log_prices, beta0, leg, params)
    params["P0"] = float(frame.filter(like="pvar_").iloc[-1].mean())
    return params


def calibrate_signal_filter(train_spread, q_ratio=0.0, lam=DEFAULT_LAMBDA, burn_in=DEFAULT_BURN_IN, use_ar1=False):
    """Regresa Q, R, phi, lam, mu0 y burn_in del filtro 2, con Q = q_ratio * R.

    Por omision es el nivel local (phi = 0): mu0 y R son la media y la varianza del spread.
    Con lam=None la escala del z es la implicita en el modelo (S13 p21).
    """
    if use_ar1:
        phi, mu0, r = fit_ar1(train_spread)
    else:
        values = np.asarray(train_spread, dtype=float)
        phi, mu0, r = 0.0, float(values.mean()), float(values.var(ddof=1))
    params = {"Q": q_ratio * r, "R": r, "phi": phi, "lam": lam, "mu0": mu0, "burn_in": burn_in}
    params["P0"] = float(run_signal_filter(train_spread, params)["pvar"].iloc[-1])
    return params


def weights_q_table(train_log_prices, beta0, leg, ratios=Q_RATIOS, use_ar1=True, state_intercept=False, burn_in=DEFAULT_BURN_IN):
    """Diagnostico del filtro 1 por cada razon Q/R; no elige por si solo.

    La log-verosimilitud tiende a premiar una Q mayor porque los pesos persiguen el spread,
    asi que se lee junto con la vida media y el rango que recorren los pesos.
    """
    rows = []
    for ratio in ratios:
        params = calibrate_weights_filter(train_log_prices, beta0, leg, ratio, use_ar1, state_intercept)
        frame = run_weights_filter(train_log_prices, beta0, leg, params)
        weights = frame.filter(like="beta_").iloc[burn_in:]
        rows.append(
            {
                "q_ratio": ratio,
                "half_life": half_life(frame["spread"].iloc[burn_in:]),
                "log_likelihood": log_likelihood(frame),
                "max_weight_range": float((weights.max() - weights.min()).max()),
            }
        )
    return pd.DataFrame(rows)


def signal_q_table(train_spread, ratios=Q_RATIOS, lam=DEFAULT_LAMBDA, burn_in=DEFAULT_BURN_IN, use_ar1=False):
    """Diagnostico del filtro 2 por cada razon Q/R; no elige por si solo."""
    rows = []
    for ratio in ratios:
        params = calibrate_signal_filter(train_spread, ratio, lam, burn_in, use_ar1)
        frame = run_signal_filter(train_spread, params)
        z = frame["z"].dropna()
        rows.append(
            {
                "q_ratio": ratio,
                "log_likelihood": log_likelihood(frame),
                "z_std": float(z.std()),
                "z_autocorr": float(z.autocorr(1)),
                "share_abs_z_above_2": float((z.abs() > 2).mean()),
            }
        )
    return pd.DataFrame(rows)
