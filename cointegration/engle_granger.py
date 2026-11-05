"""Engle-Granger pre-filter using MacKinnon critical values (statsmodels coint)."""
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import coint


def engle_granger_direction(log_prices, dependent):
    """Regress one asset on the others and test the residuals for cointegration."""
    others = [col for col in log_prices.columns if col != dependent]
    y = log_prices[dependent]
    x = log_prices[others]
    p_value = coint(y, x, trend="c", autolag="aic")[1]
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    dw = durbin_watson(fit.resid)
    return {
        "dependent": dependent,
        "p_value": float(p_value),
        "r2": float(fit.rsquared),
        "durbin_watson": float(dw),
    }


def engle_granger_group(log_prices):
    """Test every asset as the dependent variable and keep the best direction."""
    results = [engle_granger_direction(log_prices, col) for col in log_prices.columns]
    best = min(results, key=lambda row: row["p_value"])
    return {
        "eg_min_p": best["p_value"],
        "eg_best_dependent": best["dependent"],
        "eg_r2": best["r2"],
        "eg_durbin_watson": best["durbin_watson"],
        "eg_r2_above_dw": bool(best["r2"] > best["durbin_watson"]),
        "eg_directions_tested": len(results),
    }
