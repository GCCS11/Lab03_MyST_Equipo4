"""Position sizing for the n-leg spread: 80% of equity split in proportion to |beta| (S14 p20),
with daily rebalancing or a no-rebalance band."""
import numpy as np
import pandas as pd

from config import POSITION_FRACTION
from strategy.weights import align_betas


def target_position(direction, betas_t, prices_t, equity_t, fraction=POSITION_FRACTION):
    """Shares per leg for one period: q_i = d * sgn(beta_i) * N_i / P_i,
    with N_i = fraction * V * |beta_i| / sum_j |beta_j|.

    Prices are real prices, not log prices. Shares are fractional.
    """
    betas_t = np.asarray(betas_t, dtype=float)
    prices_t = np.asarray(prices_t, dtype=float)
    if direction == 0:
        return np.zeros(betas_t.size)
    if not (np.all(np.isfinite(betas_t)) and np.all(prices_t > 0) and np.isfinite(equity_t)):
        raise ValueError("an open position needs finite weights, positive prices and finite equity")
    notional = fraction * equity_t * np.abs(betas_t) / np.abs(betas_t).sum()
    return direction * np.sign(betas_t) * notional / prices_t


def next_shares(
    current, prev_direction, direction, betas_t, prices_t, equity_t, band=0.0,
    fraction=POSITION_FRACTION,
):
    """Shares to hold after this period's decision.

    Opening, closing or flipping the spread trades the whole target. While the direction does not
    change, a leg moves to its target only when its gap exceeds `band`, measured as a fraction of
    equity: band = 0 rebalances every period as beta_t moves, band = inf keeps the entry shares.
    """
    target = target_position(direction, betas_t, prices_t, equity_t, fraction)
    if direction != prev_direction:
        return target
    current = np.asarray(current, dtype=float)
    gap = np.abs(current - target) * np.asarray(prices_t, dtype=float) / equity_t
    return np.where(gap > band, target, current)


def target_shares(d, betas, prices, equity, band=0.0, fraction=POSITION_FRACTION):
    """Shares held in each period for a path of directions, applying next_shares period by period.

    `d` is the direction Series (the `d` column of spread_direction), `betas` a fixed vector or
    the Filter 1 weights, `prices` real prices and `equity` a scalar or a Series aligned with `d`.
    The backtest, whose equity depends on its own trades, should call next_shares at each step.
    """
    tickers = list(prices.columns)
    weights = align_betas(betas, tickers, d.index).to_numpy()
    price_values = prices.reindex(d.index).to_numpy(dtype=float)
    if isinstance(equity, pd.Series):
        equity_values = equity.reindex(d.index).to_numpy(dtype=float)
    else:
        equity_values = np.full(len(d), float(equity))

    held = np.zeros((len(d), len(tickers)))
    current, prev_direction = np.zeros(len(tickers)), 0
    for t, direction in enumerate(d.to_numpy(dtype=int)):
        current = next_shares(
            current, prev_direction, direction, weights[t], price_values[t], equity_values[t], band,
            fraction,
        )
        held[t], prev_direction = current, direction
    return pd.DataFrame(held, index=d.index, columns=tickers)
