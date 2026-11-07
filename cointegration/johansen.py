"""Johansen test and selection of the most mean-reverting cointegrating vector."""
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import ALPHA, HALF_LIFE_MAX, HALF_LIFE_MIN, JOHANSEN_DET_ORDER, JOHANSEN_K_AR_DIFF
from cointegration.spread import adf_pvalue, build_spread, half_life, normalize_vector


def count_rejections(statistics, critical_values):
    """Number of consecutive rejections of H0: rank <= r, starting at r = 0."""
    rank = 0
    for statistic, critical in zip(statistics, critical_values):
        if statistic <= critical:
            break
        rank += 1
    return rank


def johansen_test(log_prices):
    """Run Johansen and return the raw result and the summary used in the registry."""
    result = coint_johansen(
        log_prices, det_order=JOHANSEN_DET_ORDER, k_ar_diff=JOHANSEN_K_AR_DIFF
    )
    max_rank = log_prices.shape[1] - 1
    rank_trace = min(count_rejections(result.lr1, result.cvt[:, 1]), max_rank)
    rank_maxeig = min(count_rejections(result.lr2, result.cvm[:, 1]), max_rank)
    summary = {
        "johansen_rank_trace": rank_trace,
        "johansen_rank_maxeig": rank_maxeig,
        "trace_stat_r0": float(result.lr1[0]),
        "trace_crit95_r0": float(result.cvt[0, 1]),
        "maxeig_stat_r0": float(result.lr2[0]),
        "maxeig_crit95_r0": float(result.cvm[0, 1]),
        "top_eigenvalue": float(result.eig[0]),
    }
    return result, summary


def candidate_vectors(log_prices, result, rank):
    """Build the spread of each of the first `rank` vectors and score it."""
    rows = []
    for column in range(rank):
        vector, leg = normalize_vector(result.evec[:, column])
        spread = build_spread(log_prices, vector)
        hl = half_life(spread)
        p_value = adf_pvalue(spread)
        rows.append(
            {
                "column": column,
                "eigenvalue": float(result.eig[column]),
                "leg": log_prices.columns[leg],
                "vector": vector,
                "adf_p": float(p_value),
                "half_life": hl,
                "admissible": bool(p_value < ALPHA and HALF_LIFE_MIN <= hl <= HALF_LIFE_MAX),
            }
        )
    return pd.DataFrame(rows)


def select_vector(candidates):
    """Pick the admissible candidate with the lowest ADF p-value, or None."""
    admissible = candidates[candidates["admissible"]] if len(candidates) else candidates
    if admissible.empty:
        return None
    return admissible.sort_values("adf_p").iloc[0]
