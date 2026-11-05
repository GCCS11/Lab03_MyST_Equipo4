"""Smoke tests of the dashboard on synthetic artifacts: loading, every figure, export and the app."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from config import SEED
from dashboard import figures
from dashboard.data import filter_trades, load_artifacts, missing, select_splits
from dashboard.export import export_html
from strategy.signals import SignalParams, spread_direction

TICKERS = ["DUK", "SO", "AEP", "XEL", "ED", "WEC"]
APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
THETA = {"z_in": 2.0, "z_out": 0.5, "z_stop": 3.5, "h_max": 30, "band": 0.0}


def write_fake_artifacts(directory, n_days=600, seed=SEED):
    """Write artifacts with the schema of dashboard/data.py, built from random series."""
    rng = np.random.default_rng(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2014-01-02", periods=n_days, name="date")
    position = np.arange(n_days)
    split = np.where(position < 0.6 * n_days, "train", np.where(position < 0.8 * n_days, "test", "validation"))

    prices = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(TICKERS))), axis=0))
    pd.DataFrame(prices, index=dates, columns=TICKERS).to_csv(directory / "precios.csv")

    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.01)
    spread = pd.Series(spread, index=dates)
    mu = spread.ewm(alpha=0.02).mean().shift(fill_value=0.0)
    sigma = (spread - mu).ewm(alpha=0.02).std().bfill()
    z = (spread - mu) / sigma
    rule = spread_direction(z, SignalParams(**{k: THETA[k] for k in ("z_in", "z_out", "z_stop", "h_max")}))
    pd.DataFrame({
        "split": split, "spread": spread, "mu_prior": mu, "sigma_prior": sigma, "z": z,
        "d": rule["d"], "exit_reason": rule["exit_reason"],
    }, index=dates).to_csv(directory / "senales.csv")

    walk = -0.2 + np.cumsum(rng.normal(0, 0.002, (n_days, len(TICKERS) - 1)), axis=0)
    weights = {f"beta_{TICKERS[0]}": 1.0} | {f"beta_{t}": walk[:, i] for i, t in enumerate(TICKERS[1:])}
    pd.DataFrame(weights, index=dates).to_csv(directory / "pesos.csv")
    eigvec = {TICKERS[0]: 1.0} | {t: walk[:, i] + rng.normal(0, 0.01, n_days) for i, t in enumerate(TICKERS[1:])}
    pd.DataFrame(eigvec, index=dates).to_csv(directory / "eigvec_movil.csv")

    decay = 0.05 + 0.5 * np.exp(-position / 40)
    components = ["c"] + TICKERS[1:]
    pd.DataFrame(
        {f"f1_gain_{c}": decay * rng.uniform(0.5, 1.5) for c in components} | {"f2_gain": decay},
        index=dates,
    ).to_csv(directory / "ganancias.csv")
    pd.DataFrame(
        {f"f1_pvar_{c}": 1e-4 * decay for c in components} | {"f2_pvar": 1e-3 * decay},
        index=dates,
    ).to_csv(directory / "varianzas.csv")

    equity = pd.Series(1e6 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n_days))), index=dates)
    returns = equity.pct_change()
    pd.DataFrame({
        "split": split, "equity": equity, "drawdown": equity / equity.cummax() - 1,
        "rolling_sharpe": returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252),
    }, index=dates).to_csv(directory / "equity.csv")

    entries = np.sort(rng.choice(n_days - 40, 30, replace=False))
    gross = rng.normal(800, 4000, len(entries))
    commissions, borrow = rng.uniform(1500, 2500, len(entries)), rng.uniform(50, 300, len(entries))
    trades = pd.DataFrame({
        "entrada": dates[entries], "salida": dates[entries + rng.integers(3, 30, len(entries))],
        "split": split[entries], "direccion": rng.choice([1, -1], len(entries)),
        "pnl_bruto": gross, "comisiones": commissions, "prestamo": borrow,
        "pnl_neto": gross - commissions - borrow, "motivo": rng.choice(["reversion", "stop", "h_max"], len(entries)),
    })
    trades["retorno"] = trades["pnl_neto"] / 800_000
    trades.to_csv(directory / "operaciones.csv", index=False)

    metrics = {}
    for name, group in trades.groupby("split"):
        wins, losses = group.pnl_neto[group.pnl_neto > 0].sum(), -group.pnl_neto[group.pnl_neto < 0].sum()
        metrics[name] = {
            "sharpe": 0.4, "sortino": 0.6, "calmar": 0.3, "max_drawdown": -0.08,
            "win_rate": float((group.pnl_neto > 0).mean()), "profit_factor": float(wins / losses),
            "retorno_total": float(group.pnl_neto.sum() / 1e6), "n_operaciones": len(group),
            "pnl_bruto": float(group.pnl_bruto.sum()), "comisiones": float(group.comisiones.sum()),
            "prestamo": float(group.prestamo.sum()), "pnl_neto": float(group.pnl_neto.sum()),
        }
    (directory / "metricas.json").write_text(json.dumps(metrics), encoding="utf-8")
    (directory / "theta.json").write_text(json.dumps(THETA), encoding="utf-8")
    (directory / "seleccion.json").write_text(json.dumps({
        "tickers": TICKERS, "beta_johansen": [1.0] + [-0.2] * 5, "pata_normalizada": TICKERS[0],
        "eigenvalor": 0.05, "vida_media_entrenamiento": 15.0, "grupos_evaluados_total": 1500,
    }), encoding="utf-8")
    return directory


def test_loads_every_artifact(tmp_path):
    artifacts = load_artifacts(write_fake_artifacts(tmp_path))
    assert missing(artifacts) == []
    assert isinstance(artifacts["senales"].index, pd.DatetimeIndex)
    assert artifacts["operaciones"]["entrada"].dtype.kind == "M"


def test_every_panel_builds_a_single_axis_figure(tmp_path):
    artifacts = load_artifacts(write_fake_artifacts(tmp_path))
    splits = ["train", "test", "validation"]
    panels = figures.executive_panels(artifacts, splits, "light") + figures.technical_panels(artifacts, splits, "dark")
    assert len(panels) == 12
    for panel in panels:
        assert panel.figure is not None, panel.key
        assert len(panel.figure.data) >= 1, panel.key
        overlays = [axis for name, axis in panel.figure.layout.to_plotly_json().items()
                    if name.startswith("yaxis") and isinstance(axis, dict) and axis.get("overlaying")]
        assert overlays == [], f"{panel.key} uses a second y-axis"


def test_missing_artifact_is_named(tmp_path):
    directory = write_fake_artifacts(tmp_path)
    (directory / "pesos.csv").unlink()
    panels = {p.key: p for p in figures.technical_panels(load_artifacts(directory), ["test"], "light")}
    assert panels["pesos"].figure is None and panels["pesos"].source == "pesos.csv"
    assert panels["spread"].figure is not None


def test_split_and_trade_filters(tmp_path):
    artifacts = load_artifacts(write_fake_artifacts(tmp_path))
    signals = select_splits(artifacts["senales"], ["test"])
    assert set(signals["split"]) == {"test"}
    bounds = {"test": (signals.index.min(), signals.index.max())}
    weights = select_splits(artifacts["pesos"], ["test"], bounds)
    assert weights.index.min() == signals.index.min() and weights.index.max() == signals.index.max()
    trades = filter_trades(artifacts["operaciones"], ["train", "test", "validation"], [1], ["Ganadora"])
    assert (trades["direccion"] == 1).all() and (trades["pnl_neto"] > 0).all()


def test_export_writes_one_html_with_both_views(tmp_path):
    artifacts = load_artifacts(write_fake_artifacts(tmp_path / "artifacts"))
    page = export_html(artifacts, tmp_path / "docs" / "dashboard.html").read_text(encoding="utf-8")
    assert "Vista ejecutiva" in page and "Vista técnica" in page
    assert page.count("plotly.js") >= 1


def test_app_runs_both_views(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB03_ARTIFACTS", str(write_fake_artifacts(tmp_path)))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    app.radio(key="vista").set_value("Técnica").run()
    assert not app.exception


def test_app_warns_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB03_ARTIFACTS", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("python main.py" in warning.value for warning in app.warning)
