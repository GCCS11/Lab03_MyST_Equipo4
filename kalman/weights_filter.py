"""Filtro 1: pesos dinamicos del vector cointegrante.

Convencion: la pata `leg` queda fija en +1 y el estado son los pesos de las demas patas,
de modo que S_t = p_leg,t + theta_t' x_t sobre log-precios.

    Transicion:  theta_t = theta_{t-1} + w_t,    w_t ~ N(0, Q I)
    Observacion: p_leg,t - c = -x_t' theta_t + u_t,    u_t = phi u_{t-1} + e_t,    e_t ~ N(0, R)

El error de equilibrio u_t es persistente: es justo lo que la estrategia quiere operar. Si se
trata como ruido blanco (phi = 0), las innovaciones heredan su autocorrelacion y el filtro
convierte la desviacion en cambio de pesos. Por eso la observacion se cuasi-diferencia:

    p_leg,t - phi p_leg,t-1 - (1 - phi) c = -(x_t - phi x_{t-1})' theta_t + e_t

Es una aproximacion: toma theta_{t-1} = theta_t dentro de u_{t-1}, lo que ignora un termino de
orden sqrt(Q) en la observacion.

El nivel c queda fijo y de la media se encarga el filtro 2, salvo que `params` traiga Q_c: en
ese caso c_t entra al estado como paseo aleatorio. Con phi = 0 y Q_c = Q es la ecuacion de S14 p18.
"""
import numpy as np
import pandas as pd


def normalize_beta(beta, leg_position):
    """Escala el vector para que la pata normalizada valga +1; fija escala y signo."""
    beta = np.asarray(beta, dtype=float)
    return beta / beta[leg_position]


def run_weights_filter(log_prices, beta0, leg, params):
    """Corre el filtro 1 con Q, R, phi y c fijos en `params`; P0 y Q_c son opcionales.

    El spread de t usa los pesos previos beta_{t|t-1}, conocidos antes de ver los precios
    de t. Regresa una fila por fecha con beta_<ticker>, spread, gain_<comp>, pvar_<comp>
    (varianza posterior), c_prior, innov, innov_var e innov_norm. <comp> son las patas no
    normalizadas, mas c si esta en el estado.
    """
    if log_prices.isna().to_numpy().any():
        raise ValueError("log_prices tiene huecos; el filtro espera series completas")
    tickers = list(log_prices.columns)
    leg_position = tickers.index(leg)
    others = [i for i in range(len(tickers)) if i != leg_position]

    values = log_prices.to_numpy(dtype=float)
    y, x = values[:, leg_position], values[:, others]
    q, r, phi, c = params["Q"], params["R"], params["phi"], params["c"]
    q_c = params.get("Q_c")
    n_rows, n_weights = len(values), len(others)
    offset = 0 if q_c is None else 1  # posicion del primer peso dentro del estado

    weights0 = normalize_beta(beta0, leg_position)[others]
    state = weights0 if q_c is None else np.concatenate([[c], weights0])
    state_noise = np.diag([q] * n_weights if q_c is None else [q_c] + [q] * n_weights)
    cov = params.get("P0", q) * np.eye(len(state))

    betas = np.ones((n_rows, len(tickers)))
    gains, pvars = np.full((n_rows, len(state)), np.nan), np.full((n_rows, len(state)), np.nan)
    spread, c_prior, innov, innov_var = (np.full(n_rows, np.nan) for _ in range(4))

    betas[0, others] = weights0
    spread[0] = y[0] + x[0] @ weights0
    c_prior[0] = c
    pvars[0] = np.diag(cov)
    for t in range(1, n_rows):
        theta = state[offset:]
        betas[t, others] = theta
        spread[t] = y[t] + x[t] @ theta
        c_prior[t] = c if q_c is None else state[0]

        cov = cov + state_noise
        design = -(x[t] - phi * x[t - 1])
        target = y[t] - phi * y[t - 1]
        if q_c is None:
            target = target - (1 - phi) * c
        else:
            design = np.concatenate([[1 - phi], design])
        innov[t] = target - design @ state
        innov_var[t] = design @ cov @ design + r
        gain = cov @ design / innov_var[t]

        state = state + gain * innov[t]
        cov = cov - np.outer(gain, design @ cov)
        gains[t], pvars[t] = gain, np.diag(cov)

    columns = {f"beta_{ticker}": betas[:, i] for i, ticker in enumerate(tickers)}
    columns["spread"] = spread
    components = ([] if q_c is None else ["c"]) + [tickers[position] for position in others]
    for j, component in enumerate(components):
        columns[f"gain_{component}"] = gains[:, j]
        columns[f"pvar_{component}"] = pvars[:, j]
    columns.update(c_prior=c_prior, innov=innov, innov_var=innov_var, innov_norm=innov / np.sqrt(innov_var))
    return pd.DataFrame(columns, index=log_prices.index)
