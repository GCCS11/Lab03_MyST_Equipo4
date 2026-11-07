"""Metrics of a backtest.engine.BacktestResult: ratios, drawdown, cost attribution and the
rolling Sharpe, plus the writer of artifacts/metricas.json for dashboard/data.py.

No numeric field is ever null or infinite: when a metric cannot be computed (no trades, no
drawdown, no losing trades, etc.) it is reported as 0.0 and the reason goes in `notas`, keyed by
the metric's name, so a JSON reader never needs to special-case missing values.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, TRADING_DAYS

JSON_KEYS = (
    "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "profit_factor",
    "retorno_total", "n_operaciones", "pnl_bruto", "comisiones", "prestamo", "pnl_neto",
)


@dataclass(frozen=True)
class Metrics:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_inicio: "pd.Timestamp | None"
    max_drawdown_fin: "pd.Timestamp | None"
    retorno_anualizado: float
    volatilidad_anualizada: float
    win_rate: float
    profit_factor: float
    retorno_total: float
    n_operaciones: int
    pnl_bruto: float
    comisiones: float
    prestamo: float
    pnl_neto: float
    pct_comisiones: float
    pct_prestamo: float
    rolling_sharpe: pd.Series
    notas: dict = field(default_factory=dict)

    def to_json_dict(self):
        """Plain dict with exactly the keys dashboard/data.py and dashboard/figures.py read
        from metricas.json, plus the extra fields and `notas`. Dates become ISO strings and the
        rolling_sharpe series (not part of metricas.json) is left out."""
        data = {key: getattr(self, key) for key in JSON_KEYS}
        extra = {
            "max_drawdown_inicio": _iso_or_none(self.max_drawdown_inicio),
            "max_drawdown_fin": _iso_or_none(self.max_drawdown_fin),
            "retorno_anualizado": self.retorno_anualizado,
            "volatilidad_anualizada": self.volatilidad_anualizada,
            "pct_comisiones": self.pct_comisiones,
            "pct_prestamo": self.pct_prestamo,
            "notas": dict(self.notas),
        }
        return {**data, **extra}


def _iso_or_none(timestamp):
    return timestamp.isoformat() if timestamp is not None else None


def _safe_divide(numerator, denominator, reason, notas, key):
    """numerator / denominator, or 0.0 with `reason` recorded in notas[key] when the division
    is not defined (zero or non-finite denominator)."""
    if denominator == 0 or not np.isfinite(denominator):
        notas[key] = reason
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else _zero_with_reason(notas, key, reason)


def _zero_with_reason(notas, key, reason):
    notas[key] = reason
    return 0.0


def _max_drawdown(equity):
    """Largest peak-to-trough decline, as a positive fraction, with the peak and trough dates."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough_date = drawdown.idxmin()
    depth = -float(drawdown.loc[trough_date])
    if depth <= 0:
        return 0.0, None, None
    peak_date = equity.loc[:trough_date].idxmax()
    return depth, peak_date, trough_date


def compute_metrics(diario, operaciones, initial_capital=INITIAL_CAPITAL, trading_days=TRADING_DAYS):
    """Metrics of one backtest run. `diario` and `operaciones` are a BacktestResult's frames."""
    notas = {}
    equity = diario["equity"]
    returns = equity.pct_change().fillna(equity.iloc[0] / initial_capital - 1.0)

    mean_return, std_return = returns.mean(), returns.std(ddof=0)
    sharpe = _safe_divide(
        mean_return * np.sqrt(trading_days), std_return,
        "menos de dos retornos o retornos sin variacion", notas, "sharpe",
    )

    downside = returns.clip(upper=0.0)
    downside_std = np.sqrt((downside ** 2).mean())
    sortino = _safe_divide(
        mean_return * np.sqrt(trading_days), downside_std,
        "sin retornos negativos en la muestra", notas, "sortino",
    )

    max_drawdown, drawdown_inicio, drawdown_fin = _max_drawdown(equity)

    n_periods = len(equity)
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    retorno_anualizado = (
        float((1.0 + total_return) ** (trading_days / n_periods) - 1.0) if n_periods > 0 else 0.0
    )
    volatilidad_anualizada = float(std_return * np.sqrt(trading_days))

    calmar = _safe_divide(
        retorno_anualizado, max_drawdown, "sin drawdown en la muestra", notas, "calmar",
    )

    n_operaciones = len(operaciones)
    if n_operaciones == 0:
        win_rate = _zero_with_reason(notas, "win_rate", "sin operaciones cerradas")
        profit_factor = _zero_with_reason(notas, "profit_factor", "sin operaciones cerradas")
    else:
        win_rate = float((operaciones["pnl_neto"] > 0).mean())
        ganancias = float(operaciones.loc[operaciones["pnl_neto"] > 0, "pnl_neto"].sum())
        perdidas = float(-operaciones.loc[operaciones["pnl_neto"] < 0, "pnl_neto"].sum())
        profit_factor = _safe_divide(
            ganancias, perdidas, "sin operaciones perdedoras en la muestra", notas, "profit_factor",
        )

    pnl_bruto = float(diario["pnl_bruto"].sum())
    comisiones = float(diario["comisiones"].sum())
    prestamo = float(diario["prestamo"].sum())
    pnl_neto = float(diario["pnl_neto"].sum())
    pct_comisiones = _safe_divide(
        comisiones, abs(pnl_bruto), "pnl bruto es cero", notas, "pct_comisiones",
    )
    pct_prestamo = _safe_divide(
        prestamo, abs(pnl_bruto), "pnl bruto es cero", notas, "pct_prestamo",
    )

    rolling_mean = returns.rolling(trading_days).mean()
    rolling_std = returns.rolling(trading_days).std(ddof=0)
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(trading_days)).rename("rolling_sharpe")

    return Metrics(
        sharpe=sharpe, sortino=sortino, calmar=calmar, max_drawdown=max_drawdown,
        max_drawdown_inicio=drawdown_inicio, max_drawdown_fin=drawdown_fin,
        retorno_anualizado=retorno_anualizado, volatilidad_anualizada=volatilidad_anualizada,
        win_rate=win_rate, profit_factor=profit_factor, retorno_total=total_return,
        n_operaciones=n_operaciones, pnl_bruto=pnl_bruto, comisiones=comisiones,
        prestamo=prestamo, pnl_neto=pnl_neto, pct_comisiones=pct_comisiones,
        pct_prestamo=pct_prestamo, rolling_sharpe=rolling_sharpe, notas=notas,
    )


def write_metricas_json(path, metrics_by_split):
    """Write artifacts/metricas.json: {split: {...}} with the keys dashboard/data.py reads.

    `metrics_by_split` is a dict of split name to Metrics (e.g. {"train": metrics}).
    """
    payload = {split: metrics.to_json_dict() for split, metrics in metrics_by_split.items()}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
