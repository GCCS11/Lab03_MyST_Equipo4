"""Daily n-leg backtest engine.

Takes the output of strategy.signals.spread_direction (the `d` and `exit_reason` columns) plus
betas and prices, and sizes each step with strategy.sizing.next_shares, because equity depends
on the engine's own trades and cannot be computed in advance like strategy.sizing.target_shares
does. The traded direction is the rule's `d` shifted one period: a decision made from z_t
executes at t+1's price.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BORROW_RATE_ANNUAL, COMMISSION, INITIAL_CAPITAL, POSITION_FRACTION, TRADING_DAYS
from strategy.sizing import next_shares
from strategy.weights import align_betas


@dataclass(frozen=True)
class BacktestParams:
    """Costs and sizing of the backtest. `band` is the no-rebalance band of next_shares:
    0.0 rebalances every period as beta_t moves, a positive value only when a leg drifts past it."""

    initial_capital: float = INITIAL_CAPITAL
    commission: float = COMMISSION
    borrow_rate_annual: float = BORROW_RATE_ANNUAL
    trading_days: int = TRADING_DAYS
    fraction: float = POSITION_FRACTION
    band: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Daily ledger, shares held per leg and the closed-trade log."""

    diario: pd.DataFrame      # direccion, pnl_bruto, comisiones, prestamo, pnl_neto, cash, equity
    posiciones: pd.DataFrame  # shares per leg, one column per ticker
    operaciones: pd.DataFrame  # entrada, salida, direccion, pnl_bruto, comisiones, prestamo,
    #                            pnl_neto, retorno, motivo (closed trades only)

    @property
    def equity(self):
        return self.diario["equity"]


def run_backtest(rule, betas, prices, params=BacktestParams()):
    """Run the n-leg backtest from a signal rule.

    `rule` is the DataFrame returned by strategy.signals.spread_direction, with a `d` column
    (target direction) and `exit_reason`. `betas` is a fixed vector or the Filter 1 weights, as
    accepted by strategy.weights.align_betas. `prices` are real prices for the same tickers and
    dates as `rule`. Only data up to each row's own date is used, so truncating `prices` or
    `rule` after any date t never changes the output up to t.

    Any position still open on the last row is force-closed that same day (with its closing
    commission), logged with motivo "fin_de_serie", so the backtest never ends with an
    unrealized position.
    """
    tickers = list(prices.columns)
    index = rule.index
    weights = align_betas(betas, tickers, index).to_numpy()
    price_values = prices.reindex(index).to_numpy(dtype=float)
    exec_dir = rule["d"].shift(1).fillna(0).to_numpy(dtype=int)
    exec_reason = rule["exit_reason"].shift(1).fillna("").to_numpy()
    if len(exec_dir) > 0 and exec_dir[-1] != 0:
        exec_dir[-1] = 0
        exec_reason[-1] = "fin_de_serie"

    n_steps, n_legs = len(index), len(tickers)
    shares = np.zeros((n_steps, n_legs))
    cash = np.empty(n_steps)
    equity = np.empty(n_steps)
    pnl_bruto = np.zeros(n_steps)
    comisiones = np.zeros(n_steps)
    prestamo = np.zeros(n_steps)

    current, prev_direction, prev_cash = np.zeros(n_legs), 0, params.initial_capital
    for t in range(n_steps):
        pnl_bruto[t] = float(current @ (price_values[t] - price_values[t - 1])) if t > 0 else 0.0
        pretrade_equity = (equity[t - 1] if t > 0 else params.initial_capital) + pnl_bruto[t]

        target = next_shares(
            current, prev_direction, exec_dir[t], weights[t], price_values[t], pretrade_equity,
            params.band, params.fraction,
        )
        delta = target - current
        comisiones[t] = params.commission * float(np.abs(delta) @ price_values[t])
        # el prestamo se cobra sobre la posicion con la que se empezo el dia (mantenida durante
        # la noche), no sobre la que se acaba de abrir hoy
        short_notional = float(np.abs(np.minimum(current, 0.0)) @ price_values[t])
        prestamo[t] = params.borrow_rate_annual / params.trading_days * short_notional

        cash[t] = prev_cash - float(delta @ price_values[t]) - comisiones[t] - prestamo[t]
        equity[t] = cash[t] + float(target @ price_values[t])
        shares[t] = target
        current, prev_direction, prev_cash = target, exec_dir[t], cash[t]

    pnl_neto = pnl_bruto - comisiones - prestamo
    diario = pd.DataFrame({
        "direccion": exec_dir, "pnl_bruto": pnl_bruto, "comisiones": comisiones,
        "prestamo": prestamo, "pnl_neto": pnl_neto, "cash": cash, "equity": equity,
    }, index=index)
    posiciones = pd.DataFrame(shares, index=index, columns=tickers)
    operaciones = _trade_log(diario, exec_reason, index, params.initial_capital)
    return BacktestResult(diario=diario, posiciones=posiciones, operaciones=operaciones)


OPERACIONES_COLUMNS = (
    "entrada", "salida", "direccion", "pnl_bruto", "comisiones", "prestamo", "pnl_neto",
    "retorno", "motivo",
)


def _trade_log(diario, exec_reason, index, initial_capital):
    """One row per closed trade: entrada and salida are the first and last day the position was
    held, including the day it closes (that day carries the closing commission). `retorno` is
    the trade's pnl_neto over the equity the day before it opened (or initial_capital, for a
    trade that opens on the very first row). Assumes the direction never flips sign in a single
    step without passing through 0, as strategy.signals.spread_direction guarantees.
    run_backtest forces direction to 0 on the last row when it would still be open, so every
    trade closes within the series.
    """
    direction = diario["direccion"].to_numpy()
    pnl_bruto = diario["pnl_bruto"].to_numpy()
    comisiones = diario["comisiones"].to_numpy()
    prestamo = diario["prestamo"].to_numpy()
    pnl_neto = diario["pnl_neto"].to_numpy()
    equity = diario["equity"].to_numpy()
    n_steps = len(direction)
    rows, t = [], 0
    while t < n_steps:
        if direction[t] == 0:
            t += 1
            continue
        side, start = direction[t], t
        end = start
        while end + 1 < n_steps and direction[end + 1] == side:
            end += 1
        close = end + 1
        if close >= n_steps:
            break
        equity_antes = equity[start - 1] if start > 0 else initial_capital
        neto = float(pnl_neto[start : close + 1].sum())
        rows.append({
            "entrada": index[start], "salida": index[close], "direccion": int(side),
            "pnl_bruto": float(pnl_bruto[start : close + 1].sum()),
            "comisiones": float(comisiones[start : close + 1].sum()),
            "prestamo": float(prestamo[start : close + 1].sum()),
            "pnl_neto": neto,
            "retorno": neto / equity_antes if equity_antes else 0.0,
            "motivo": exec_reason[close],
        })
        t = close + 1
    return pd.DataFrame(rows, columns=OPERACIONES_COLUMNS)
