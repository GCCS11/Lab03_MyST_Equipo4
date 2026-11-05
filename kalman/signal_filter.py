"""Filtro 2: media y varianza latentes del spread y z-score en tiempo real.

    Transicion:  mu_t = mu_{t-1} + w_t,    w_t ~ N(0, Q)
    Observacion: S_t = mu_t + d_t,    d_t = phi d_{t-1} + e_t,    e_t ~ N(0, R_t)

Con phi = 0 es el modelo de nivel local. Con phi > 0 la observacion se cuasi-diferencia,
S_t - phi S_{t-1} = (1 - phi) mu_t + e_t, igual que en el filtro 1.

R_t se actualiza con un promedio exponencial de las innovaciones al cuadrado (factor `lam`).
El z-score usa los estadisticos previos a ver S_t:

    z_t = (S_t - mu_{t|t-1}) / sigma_{t|t-1},    sigma_{t|t-1}^2 = R_{t|t-1} / (1 - phi^2)

Usar la media posterior juzgaria la observacion del dia con ella misma: con phi = 0,
z_posterior = (1 - K_t) z_previo, siempre mas chico en valor absoluto.
"""
import numpy as np
import pandas as pd


def run_signal_filter(spread, params):
    """Corre el filtro 2 con Q, phi, lam, mu0, R0 y burn_in fijos en `params`.

    Regresa una fila por fecha con mu_prior, sigma_prior, z, gain, pvar (varianza posterior
    de la media), innov, innov_var e innov_norm. z vale NaN en la primera fila y durante el
    burn-in, para que ninguna regla pueda operar antes de que termine.
    """
    values = np.asarray(spread, dtype=float)
    if np.isnan(values).any():
        raise ValueError("spread tiene huecos; el filtro espera una serie completa")
    q, phi, lam = params["Q"], params["phi"], params["lam"]
    mu, r = params["mu0"], params["R0"]
    cov = params.get("P0", q)
    loading = 1.0 - phi
    stationary = 1.0 - phi**2
    n_rows = len(values)

    mu_prior, sigma_prior, z, gains, pvar, innov, innov_var = (
        np.full(n_rows, np.nan) for _ in range(7)
    )
    mu_prior[0], sigma_prior[0], pvar[0] = mu, np.sqrt(r / stationary), cov
    for t in range(1, n_rows):
        cov = cov + q
        mu_prior[t] = mu
        sigma_prior[t] = np.sqrt(r / stationary)
        z[t] = (values[t] - mu) / sigma_prior[t]

        innov[t] = values[t] - phi * values[t - 1] - loading * mu
        innov_var[t] = loading**2 * cov + r
        gains[t] = cov * loading / innov_var[t]

        mu = mu + gains[t] * innov[t]
        cov = (1.0 - gains[t] * loading) * cov
        r = lam * r + (1.0 - lam) * innov[t] ** 2
        pvar[t] = cov

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
