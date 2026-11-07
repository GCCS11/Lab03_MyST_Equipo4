"""Loading of the backtest outputs in artifacts/. The dashboard never reruns the backtest.

Files written by main.py (dates in ISO format, one row per trading day unless stated):

- seleccion.json   tickers, beta_johansen, pata_normalizada, eigenvalor,
                   vida_media_entrenamiento, grupos_evaluados_total
- theta.json       z_in, z_out, z_stop, h_max, band
- metricas.json    {split: {sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor,
                   retorno_total, n_operaciones, pnl_bruto, comisiones, prestamo, pnl_neto}}
- precios.csv      date, <ticker>...                       (real prices)
- senales.csv      date, split, spread, mu_prior, sigma_prior, z, d, exit_reason
- pesos.csv        date, beta_<ticker>...                  (Filter 1 prior weights)
- eigvec_movil.csv date, <ticker>...                       (rolling Johansen first eigenvector)
- ganancias.csv    date, f1_gain_<component>..., f2_gain   (Kalman gains)
- varianzas.csv    date, f1_pvar_<component>..., f2_pvar   (posterior variances)
- equity.csv       date, split, equity, drawdown, rolling_sharpe
- operaciones.csv  one row per trade: entrada, salida, split, direccion (+1 long, -1 short),
                   pnl_bruto, comisiones, prestamo, pnl_neto, retorno, motivo

split takes the values train, test and validation.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
JSON_FILES = ("seleccion", "theta", "metricas")
SERIES_FILES = ("precios", "senales", "pesos", "eigvec_movil", "ganancias", "varianzas", "equity")


def artifacts_dir():
    """Folder to read: LAB03_ARTIFACTS if set, otherwise the repo's artifacts/."""
    return Path(os.environ.get("LAB03_ARTIFACTS", DEFAULT_DIR))


def load_artifacts(directory=None):
    """Read every artifact that exists. Missing files come back as None."""
    directory = Path(directory) if directory else artifacts_dir()
    artifacts = {}
    for name in JSON_FILES:
        path = directory / f"{name}.json"
        artifacts[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    for name in SERIES_FILES:
        path = directory / f"{name}.csv"
        artifacts[name] = pd.read_csv(path, index_col="date", parse_dates=True) if path.exists() else None
    trades = directory / "operaciones.csv"
    artifacts["operaciones"] = (
        pd.read_csv(trades, parse_dates=["entrada", "salida"]) if trades.exists() else None
    )
    return artifacts


def missing(artifacts):
    """Names of the artifacts that were not found."""
    return [name for name, value in artifacts.items() if value is None]


def split_bounds(frame):
    """First and last date of each split present in a frame with a split column."""
    grouped = frame.reset_index().groupby("split", sort=False)["date"]
    return {split: (dates.min(), dates.max()) for split, dates in grouped}


def select_splits(frame, splits, bounds=None):
    """Rows of the chosen splits: by the split column when the frame has one, otherwise by the
    date range of each split in `bounds`."""
    if "split" in frame.columns:
        return frame[frame["split"].isin(splits)]
    if not bounds:
        return frame
    mask = np.zeros(len(frame), dtype=bool)
    for start, end in bounds.values():
        mask |= (frame.index >= start) & (frame.index <= end)
    return frame[mask]


def filter_trades(trades, splits, directions, results):
    """Trade log restricted to the chosen splits, directions (+1, -1) and results
    ("Ganadora", "Perdedora")."""
    outcome = np.where(trades["pnl_neto"] > 0, "Ganadora", "Perdedora")
    keep = trades["split"].isin(splits) & trades["direccion"].isin(directions) & np.isin(outcome, results)
    return trades[keep]
