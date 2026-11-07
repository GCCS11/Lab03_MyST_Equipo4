"""Spread construction and mean-reversion diagnostics."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def normalize_vector(vector):
    """Scale a vector so its largest absolute weight becomes +1.

    Returns the normalized vector and the index of the normalizing leg.
    """
    vector = np.asarray(vector, dtype=float)
    leg = int(np.argmax(np.abs(vector)))
    return vector / vector[leg], leg


def build_spread(log_prices, vector):
    """Return the spread S_t = beta' log P_t as a Series."""
    values = log_prices.to_numpy() @ np.asarray(vector, dtype=float)
    return pd.Series(values, index=log_prices.index, name="spread")


def adf_pvalue(series):
    """ADF p-value with a constant and AIC lag selection."""
    return adfuller(series.dropna(), regression="c", autolag="AIC", result_object=True).pvalue


def half_life(spread):
    """Half-life of mean reversion from dS_t = a + b S_{t-1} + e_t.

    Returns infinity when the spread does not revert (b outside (-1, 0)).
    """
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    design = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])
    slope = np.linalg.lstsq(design, delta.to_numpy(), rcond=None)[0][1]
    if not -1 < slope < 0:
        return np.inf
    return float(-np.log(2) / np.log(1 + slope))
