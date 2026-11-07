"""Unit-root diagnostics per asset: ADF on log prices, ADF on returns, KPSS on log prices."""
import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from config import ALPHA


def test_asset(log_series):
    """Run the three class tests on one log-price series."""
    series = log_series.dropna()
    returns = series.diff().dropna()
    adf_level = adfuller(series, regression="c", autolag="AIC", result_object=True).pvalue
    adf_return = adfuller(returns, regression="c", autolag="AIC", result_object=True).pvalue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_level = kpss(series, regression="c", nlags="auto")[1]
    return {
        "adf_level_p": adf_level,
        "adf_return_p": adf_return,
        "kpss_level_p": kpss_level,
        "is_i1": bool(adf_level >= ALPHA and adf_return < ALPHA and kpss_level < ALPHA),
    }


def stationarity_table(log_prices):
    """Return one row of test results per ticker."""
    rows = {ticker: test_asset(log_prices[ticker]) for ticker in log_prices.columns}
    return pd.DataFrame.from_dict(rows, orient="index")
