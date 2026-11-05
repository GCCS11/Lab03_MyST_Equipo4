"""Filtro 1: pesos dinamicos del vector cointegrante.

Convencion: la pata `leg` queda fija en +1 y el estado son los pesos de las demas patas,
de modo que S_t = p_leg,t + theta_t' x_t sobre log-precios.

    Transicion:  theta_t = theta_{t-1} + w_t,    w_t ~ N(0, Q I)
    Observacion: p_leg,t - c = -x_t' theta_t + e_t,    e_t ~ N(0, R)

El nivel c queda fijo: de la media del spread se encarga el filtro 2.
"""
import numpy as np
import pandas as pd


def normalize_beta(beta, leg_position):
    """Escala el vector para que la pata normalizada valga +1; fija escala y signo."""
    beta = np.asarray(beta, dtype=float)
    return beta / beta[leg_position]


def run_weights_filter(log_prices, beta0, leg, params):
    """Corre el filtro 1 con Q, R y c fijos en `params`.

    El spread de t usa los pesos previos beta_{t|t-1}, conocidos antes de ver los precios
    de t. Regresa una fila por fecha con beta_<ticker>, spread, gain_<comp>, pvar_<comp>
    (varianza posterior), innov, innov_var e innov_norm; <comp> son las patas no normalizadas.
    """
    if log_prices.isna().to_numpy().any():
        raise ValueError("log_prices tiene huecos; el filtro espera series completas")
    tickers = list(log_prices.columns)
    leg_position = tickers.index(leg)
    others = [i for i in range(len(tickers)) if i != leg_position]

    values = log_prices.to_numpy(dtype=float)
    y, x = values[:, leg_position], values[:, others]
    q, r, c = params["Q"], params["R"], params["c"]
    n_rows, n_states = len(values), len(others)

    theta = normalize_beta(beta0, leg_position)[others]
    state_noise = q * np.eye(n_states)
    cov = params.get("P0", q) * np.eye(n_states)

    betas = np.ones((n_rows, len(tickers)))
    gains, pvars = np.full((n_rows, n_states), np.nan), np.full((n_rows, n_states), np.nan)
    spread, innov, innov_var = (np.full(n_rows, np.nan) for _ in range(3))

    betas[0, others] = theta
    spread[0] = y[0] + x[0] @ theta
    pvars[0] = np.diag(cov)
    for t in range(1, n_rows):
        betas[t, others] = theta
        spread[t] = y[t] + x[t] @ theta

        cov = cov + state_noise
        design = -x[t]
        target = y[t] - c
        innov[t] = target - design @ theta
        innov_var[t] = design @ cov @ design + r
        gain = cov @ design / innov_var[t]

        theta = theta + gain * innov[t]
        cov = cov - np.outer(gain, design @ cov)
        gains[t], pvars[t] = gain, np.diag(cov)

    columns = {f"beta_{ticker}": betas[:, i] for i, ticker in enumerate(tickers)}
    columns["spread"] = spread
    for j, position in enumerate(others):
        columns[f"gain_{tickers[position]}"] = gains[:, j]
        columns[f"pvar_{tickers[position]}"] = pvars[:, j]
    columns.update(innov=innov, innov_var=innov_var, innov_norm=innov / np.sqrt(innov_var))
    return pd.DataFrame(columns, index=log_prices.index)
