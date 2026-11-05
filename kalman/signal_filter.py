"""Filtro 2: media latente del spread y z-score en tiempo real (S13 p19-21).

    Transicion:  mu_t = mu_{t-1} + w_t,    w_t ~ N(0, Q)
    Observacion: S_t = mu_t + d_t,    d_t = phi d_{t-1} + e_t,    e_t ~ N(0, R)

Con phi = 0 es el modelo de nivel local de la clase. Con phi > 0 la observacion se
cuasi-diferencia, S_t - phi S_{t-1} = (1 - phi) mu_t + e_t, igual que en el filtro 1.

La ganancia siempre usa la R fija. La escala de la innovacion es la implicita en el modelo,
H^2 P_{t|t-1} + R, si lam es None, o el promedio exponencial de innovaciones al cuadrado si
lam tiene valor. El z-score usa los estadisticos previos a ver S_t:

    z_t = (S_t - mu_{t|t-1}) / sigma_{t|t-1},    sigma_{t|t-1}^2 = escala / (1 - phi^2)

Usar la media posterior juzgaria la observacion del dia con ella misma: con phi = 0,
z_posterior = (1 - K_t) z_previo, siempre mas chico en valor absoluto.
"""
import numpy as np
import pandas as pd


def run_signal_filter(spread, params):
    """Corre el filtro 2 con Q, R, phi, lam y mu0 fijos en `params`; P0 y burn_in son opcionales.

    Regresa una fila por fecha con mu_prior, sigma_prior, z, gain, pvar (varianza posterior
    de la media), innov, innov_var e innov_norm. La primera fila solo fija el punto de partida.
    z vale NaN ahi y durante el burn-in, para que ninguna regla opere antes de que termine.
    """
    values = np.asarray(spread, dtype=float)
    if np.isnan(values).any():
        raise ValueError("spread tiene huecos; el filtro espera una serie completa")
    q, r, phi, lam = params["Q"], params["R"], params["phi"], params["lam"]
    mu, cov = params["mu0"], params.get("P0", q)
    loading = 1.0 - phi
    stationary = 1.0 - phi**2
    n_rows = len(values)
    scale = r

    mu_prior, sigma_prior, z, gains, pvar, innov, innov_var = (
        np.full(n_rows, np.nan) for _ in range(7)
    )
    mu_prior[0], sigma_prior[0], pvar[0] = mu, np.sqrt(r / stationary), cov
    for t in range(1, n_rows):
        cov = cov + q
        model_variance = loading**2 * cov + r
        innov_var[t] = model_variance if lam is None else scale
        mu_prior[t] = mu
        sigma_prior[t] = np.sqrt(innov_var[t] / stationary)
        z[t] = (values[t] - mu) / sigma_prior[t]

        innov[t] = values[t] - phi * values[t - 1] - loading * mu
        gains[t] = cov * loading / model_variance

        mu = mu + gains[t] * innov[t]
        cov = (1.0 - gains[t] * loading) * cov
        pvar[t] = cov
        if lam is not None:
            scale = lam * scale + (1.0 - lam) * innov[t] ** 2

    z[: params.get("burn_in", 0)] = np.nan
    return pd.DataFrame(
        {
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "z": z,
            "gain": gains,
            "pvar": pvar,
            "innov": innov,
            "innov_var": innov_var,
            "innov_norm": innov / np.sqrt(innov_var),
        },
        index=getattr(spread, "index", None),
    )
