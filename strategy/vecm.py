"""VECM with the Johansen vector as the error-correction term, and the equilibrium deviation."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.weights import align_betas


@dataclass(frozen=True)
class VecmFit:
    """VECM estimated with a fixed cointegrating vector beta.

    Model: dP_t = intercept + alpha * ECT_{t-1} + sum_i Gamma_i dP_{t-i} + e_t,
    with ECT_t = beta' log P_t - c.
    """

    tickers: list
    beta: np.ndarray  # cointegrating vector, shape (n,)
    c: float  # equilibrium level: mean of beta' log P in the estimation sample
    alpha: np.ndarray  # adjustment speed of each asset, shape (n,)
    gamma: np.ndarray  # short-run coefficients Gamma_i, shape (k_ar_diff, n, n)
    intercept: np.ndarray  # shape (n,)
    residuals: pd.DataFrame
    beta_alpha: float  # beta' alpha: speed at which the portfolio corrects
    half_life: float  # model-implied half-life, in periods

    @property
    def is_error_correcting(self):
        """True when -2 < beta'alpha < 0, the condition for the portfolio to revert."""
        return bool(-2 < self.beta_alpha < 0)


def model_half_life(beta_alpha):
    """Half-life implied by S_t = (1 + beta'alpha) S_{t-1}: -ln 2 / ln|1 + beta'alpha|.

    Between -2 and -1 the correction overshoots (1 + beta'alpha < 0) but still decays, so the
    absolute value is used. Returns infinity when the portfolio does not error-correct.
    """
    rho = 1.0 + beta_alpha
    if not -1.0 < rho < 1.0:
        return np.inf
    if rho == 0.0:
        return 0.0
    return float(-np.log(2) / np.log(abs(rho)))


def fit_vecm(log_prices, beta, k_ar_diff=1):
    """Estimate the VECM by OLS, equation by equation, keeping beta fixed.

    Given beta, these OLS estimates of alpha, Gamma and the intercept coincide with Johansen's
    maximum likelihood (unrestricted constant). Pass only the training partition.
    """
    beta = np.asarray(beta, dtype=float)
    data = log_prices.dropna()
    if data.shape[1] != beta.size:
        raise ValueError(f"beta has {beta.size} weights but there are {data.shape[1]} assets")

    n_obs, n_assets = data.shape
    spread = data.to_numpy() @ beta
    c = float(spread.mean())
    ect = spread - c
    diffs = data.diff().to_numpy()

    first = k_ar_diff + 1  # first row with dP_t, ECT_{t-1} and every lag dP_{t-i} available
    target = diffs[first:]
    lags = [diffs[first - i : n_obs - i] for i in range(1, k_ar_diff + 1)]
    design = np.column_stack([np.ones(len(target)), ect[first - 1 : n_obs - 1], *lags])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)

    alpha = coef[1]
    gamma = np.array(
        [coef[2 + i * n_assets : 2 + (i + 1) * n_assets].T for i in range(k_ar_diff)]
    ).reshape(k_ar_diff, n_assets, n_assets)
    residuals = pd.DataFrame(
        target - design @ coef, index=data.index[first:], columns=data.columns
    )
    beta_alpha = float(beta @ alpha)
    return VecmFit(
        tickers=list(data.columns),
        beta=beta,
        c=c,
        alpha=alpha,
        gamma=gamma,
        intercept=coef[0],
        residuals=residuals,
        beta_alpha=beta_alpha,
        half_life=model_half_life(beta_alpha),
    )


def equilibrium_deviation(log_prices, betas, c=0.0):
    """Deviation from equilibrium S_t = beta_t' log P_t - c, the series that feeds Filter 2.

    `betas` is either a fixed vector (Johansen) or a DataFrame of time-varying weights with one
    column per ticker, named as the ticker or as `beta_<ticker>` (the Filter 1 output). For a
    causal signal pass the prior weights beta_{t|t-1}, known before seeing P_t.
    """
    weights = align_betas(betas, list(log_prices.columns), log_prices.index)
    values = (log_prices * weights).sum(axis=1, min_count=log_prices.shape[1])
    return (values - c).rename("equilibrium_deviation")
