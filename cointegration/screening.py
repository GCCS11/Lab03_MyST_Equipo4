"""Group screening on the training partition with a full registry of every group tested.

Run with: python -m cointegration.screening
"""
from itertools import combinations

import numpy as np
import pandas as pd

from config import (
    ALPHA,
    ARTIFACTS_DIR,
    CORR_THRESHOLD,
    CORR_WINDOW,
    MAX_GROUP_SIZE,
    MIN_GROUP_SIZE,
    MIN_OBSERVATIONS,
)
from cointegration.engle_granger import engle_granger_group
from cointegration.johansen import candidate_vectors, johansen_test, select_vector
from cointegration.stationarity import stationarity_table


def min_median_rolling_corr(log_prices):
    """Smallest pairwise median of the rolling correlation of log prices."""
    medians = [
        log_prices[a].rolling(CORR_WINDOW).corr(log_prices[b]).median()
        for a, b in combinations(log_prices.columns, 2)
    ]
    return float(np.nanmin(medians))


def screen_group(universe, tickers, log_prices, i1_tickers):
    """Run every stage on one group and record where it was discarded."""
    row = {"universe": universe, "group": "+".join(tickers), "n_assets": len(tickers)}
    data = log_prices[list(tickers)].dropna()
    row["n_obs"] = len(data)
    if len(data) < MIN_OBSERVATIONS:
        return {**row, "status": "descartado_datos"}
    if not all(t in i1_tickers for t in tickers):
        return {**row, "status": "descartado_activos_no_I1"}

    row["corr_min_pair_median"] = min_median_rolling_corr(data)
    if not row["corr_min_pair_median"] > CORR_THRESHOLD:
        return {**row, "status": "descartado_correlacion"}

    row.update(engle_granger_group(data))
    if not row["eg_min_p"] < ALPHA:
        return {**row, "status": "descartado_engle_granger"}

    result, summary = johansen_test(data)
    row.update(summary)
    if summary["johansen_rank_trace"] < 1:
        return {**row, "status": "descartado_johansen"}

    candidates = candidate_vectors(data, result, summary["johansen_rank_trace"])
    chosen = select_vector(candidates)
    if chosen is None:
        best = candidates.sort_values("adf_p").iloc[0]
        row.update(adf_spread_p=best["adf_p"], half_life=best["half_life"])
        return {**row, "status": "descartado_validacion_spread"}

    row.update(
        adf_spread_p=chosen["adf_p"],
        half_life=chosen["half_life"],
        normalizing_leg=chosen["leg"],
        vector=" ".join(f"{w:.4f}" for w in chosen["vector"]),
    )
    return {**row, "status": "finalista"}


def screen_universes(log_prices, universes, i1_table):
    """Screen every subset of every universe and return the registry."""
    i1_tickers = set(i1_table.index[i1_table["is_i1"]])
    rows = []
    for name, universe in universes.items():
        tickers = [t for t in universe["tickers"] if t in log_prices.columns]
        top = min(len(tickers), MAX_GROUP_SIZE)
        for size in range(MIN_GROUP_SIZE, top + 1):
            for group in combinations(tickers, size):
                rows.append(screen_group(name, group, log_prices, i1_tickers))
    return pd.DataFrame(rows)


def run_screening():
    """Load training prices, screen all universes and save the registries."""
    from data.loader import load_train_prices
    from data.universe import UNIVERSES

    log_prices = np.log(load_train_prices())
    i1_table = stationarity_table(log_prices.dropna(axis=1, thresh=MIN_OBSERVATIONS))
    registry = screen_universes(log_prices, UNIVERSES, i1_table)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    i1_table.to_csv(ARTIFACTS_DIR / "stationarity_tickers.csv")
    registry.to_csv(ARTIFACTS_DIR / "screening_groups.csv", index=False)

    print(f"Groups evaluated: {len(registry)}")
    print(registry["status"].value_counts().to_string())
    finalists = registry[registry["status"] == "finalista"]
    if not finalists.empty:
        cols = ["universe", "group", "adf_spread_p", "half_life", "normalizing_leg"]
        print(finalists.sort_values("adf_spread_p")[cols].head(15).to_string(index=False))
    return registry


if __name__ == "__main__":
    run_screening()
