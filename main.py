"""Full pipeline entry point. Run with: python main.py

Every parameter comes frozen from artifacts/ (chosen on the training partition only):
1. Prices of the selected group (seleccion.json) and the train / test / validation cut dates.
2. VECM on training with the Johansen vector: alpha, beta'alpha and the model half-life.
3. Filter 1: dynamic weights; the equilibrium deviation uses the prior weights beta_{t|t-1}.
4. Filter 2: latent mean and scale of the deviation, and the causal z-score.
5. Trading rule with the thresholds of theta.json.
6. One continuous backtest with costs from the first valid z-score to the end of the sample, and
   the metrics of each partition on its own slice of that run.
7. Artifacts for the dashboard (dashboard/data.py lists the files and columns).
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestParams, run_backtest
from cointegration.seleccion import PRICES_FILE
from config import ARTIFACTS_DIR, INITIAL_CAPITAL, SEED, set_seed
from kalman.signal_filter import run_signal_filter
from kalman.weights_filter import run_weights_filter
from performance.metrics import compute_metrics, write_metricas_json
from strategy.signals import spread_direction
from strategy.vecm import equilibrium_deviation, fit_vecm

SIGNAL_KEYS = ("Q", "R", "phi", "lam", "mu0", "P0", "burn_in")
RULE_KEYS = ("z_in", "z_out", "z_stop", "h_max")
TRADE_COLUMNS = (
    "entrada", "salida", "split", "direccion", "pnl_bruto", "comisiones", "prestamo", "pnl_neto",
    "retorno", "motivo",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Lab03 statistical arbitrage pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fast run that never overwrites artifacts/ or docs/",
    )
    return parser.parse_args()


def read_json(path):
    """Content of a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen(directory=ARTIFACTS_DIR):
    """Frozen inputs: the selection, the filter parameters and, once chosen, the thresholds."""
    directory = Path(directory)
    theta_file = directory / "theta.json"
    theta = read_json(theta_file) if theta_file.exists() else None
    return read_json(directory / "seleccion.json"), read_json(directory / "filtros.json"), theta


def load_log_prices(selection):
    """Log prices of the selected group from the frozen snapshot."""
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)[selection["tickers"]]
    return prices.apply(np.log).rename_axis("date")


def split_labels(index, selection):
    """train, test or validation for each date, with the cut dates frozen in seleccion.json."""
    train_end, test_end = pd.Timestamp(selection["train_end"]), pd.Timestamp(selection["test_end"])
    labels = np.where(index <= train_end, "train", np.where(index <= test_end, "test", "validation"))
    return pd.Series(labels, index=index, name="split")


def vecm_summary(log_prices, selection):
    """VECM on training with the Johansen vector as the error-correction term."""
    fit = fit_vecm(log_prices.loc[: selection["train_end"]], selection["beta_johansen"])
    return {
        "alpha": dict(zip(fit.tickers, fit.alpha.tolist())),
        "beta_alpha": fit.beta_alpha,
        "corrige_error": fit.is_error_correcting,
        "vida_media": fit.half_life,
        "c": fit.c,
    }


def run_pipeline(log_prices, selection, filters, theta=None, until=None):
    """Causal chain from log prices to signals, with every parameter frozen.

    With `until` it only sees data up to that date, so the z-score at t must not change when
    later data is added (test 1). Returns the Filter 1 and Filter 2 frames, the equilibrium
    deviation and, when `theta` is given, the output of the trading rule.
    """
    data = log_prices.loc[:until, selection["tickers"]]
    weights = run_weights_filter(data, filters["beta0"], filters["pata_normalizada"], filters["filtro1"])
    deviation = equilibrium_deviation(data, weights.filter(like="beta_"))
    signal = run_signal_filter(deviation, {key: filters["filtro2"][key] for key in SIGNAL_KEYS})
    rule = spread_direction(signal["z"], {key: theta[key] for key in RULE_KEYS}) if theta else None
    return {"weights": weights, "deviation": deviation, "signal": signal, "rule": rule}


def signal_artifacts(log_prices, splits, result):
    """Frames of the technical view, named as dashboard/data.py expects."""
    weights, signal, rule = result["weights"], result["signal"], result["rule"]
    signals = pd.DataFrame({
        "split": splits,
        "spread": result["deviation"],
        "mu_prior": signal["mu_prior"],
        "sigma_prior": signal["sigma_prior"],
        "z": signal["z"],
        "d": rule["d"] if rule is not None else 0,
        "exit_reason": rule["exit_reason"] if rule is not None else "",
    })
    return {
        "precios": np.exp(log_prices),
        "senales": signals,
        "pesos": weights.filter(like="beta_"),
        "ganancias": weights.filter(like="gain_").add_prefix("f1_").assign(f2_gain=signal["gain"]),
        "varianzas": weights.filter(like="pvar_").add_prefix("f1_").assign(f2_pvar=signal["pvar"]),
    }


def run_frozen_backtest(log_prices, result, theta):
    """Backtest of the frozen rule from the first valid z-score to the last date.

    It is one continuous run: the strategy trades through the cut dates as it would live, so a
    position open on a cut date carries over. The filters' burn-in (z is NaN) is left out
    because no decision was possible there, as in the training walk-forward.
    """
    start = result["signal"]["z"].first_valid_index()
    rule = result["rule"].loc[start:]
    betas = result["weights"].filter(like="beta_").loc[start:]
    prices = np.exp(log_prices).loc[start:]
    return run_backtest(rule, betas, prices, BacktestParams(band=theta["band"]))


def capital_before(backtest):
    """Equity at the close of the previous day, the base of each day's return."""
    return backtest.equity.shift(1, fill_value=INITIAL_CAPITAL)


def split_metrics(backtest, splits):
    """Metrics of each partition on its slice of the continuous run.

    Each partition starts from the equity it inherited, and a trade belongs to the partition
    where it was opened.
    """
    labels = splits.reindex(backtest.diario.index)
    trade_split = splits.reindex(backtest.operaciones["entrada"]).to_numpy()
    base = capital_before(backtest)
    metrics = {}
    for split in pd.unique(labels):
        days = (labels == split).to_numpy()
        metrics[split] = compute_metrics(
            backtest.diario[days], backtest.operaciones[trade_split == split],
            initial_capital=float(base[days].iloc[0]),
        )
    return metrics


def trade_artifact(backtest, splits):
    """Trade log as dashboard/data.py expects: partition of the entry day, costs of the days the
    position was held and the net return over the equity it started with."""
    diario, base = backtest.diario, capital_before(backtest)
    rows = []
    for trade in backtest.operaciones.itertuples(index=False):
        held = diario.loc[trade.entrada:trade.salida]
        rows.append({
            "entrada": trade.entrada, "salida": trade.salida, "split": splits.loc[trade.entrada],
            "direccion": trade.direccion, "pnl_bruto": held["pnl_bruto"].sum(),
            "comisiones": held["comisiones"].sum(), "prestamo": held["prestamo"].sum(),
            "pnl_neto": trade.pnl_neto, "retorno": trade.pnl_neto / base.loc[trade.entrada],
            "motivo": trade.motivo,
        })
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def equity_artifact(backtest, splits):
    """Daily equity with its drawdown from the running peak and the 252-day rolling Sharpe."""
    equity = backtest.equity
    return pd.DataFrame({
        "split": splits.reindex(equity.index),
        "equity": equity,
        "drawdown": equity / equity.cummax() - 1.0,
        "rolling_sharpe": compute_metrics(backtest.diario, backtest.operaciones).rolling_sharpe,
    })


def write_artifacts(directory, frames, vecm):
    """Write the CSV files and the VECM summary into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index_label="date")
    (directory / "vecm.json").write_text(json.dumps(vecm, indent=2) + "\n", encoding="utf-8")


def main(quick=False):
    set_seed()
    print(f"Seed fixed at {SEED}")
    print("Mode: quick" if quick else "Mode: full")

    selection, filters, theta = load_frozen()
    log_prices = load_log_prices(selection)
    splits = split_labels(log_prices.index, selection)
    print(f"Grupo: {', '.join(selection['tickers'])} · {log_prices.index[0].date()} a {log_prices.index[-1].date()}")
    print(f"Cortes: entrenamiento hasta {selection['train_end']}, prueba hasta {selection['test_end']}")

    vecm = vecm_summary(log_prices, selection)
    status = "corrige el error" if vecm["corrige_error"] else "NO corrige el error"
    print(f"VECM (entrenamiento): beta'alpha = {vecm['beta_alpha']:.4f}, {status}, vida media {vecm['vida_media']:.1f} días")

    result = run_pipeline(log_prices, selection, filters, theta)
    out_dir = Path(tempfile.mkdtemp(prefix="lab03_quick_")) if quick else ARTIFACTS_DIR
    frames = signal_artifacts(log_prices, splits, result)
    if theta is None:
        write_artifacts(out_dir, frames, vecm)
        print(f"Artifacts de señales escritos en {out_dir}")
        print("Pendiente: artifacts/theta.json (python -m strategy.theta). Sin theta no hay backtest.")
        return

    print(f"Theta congelado: z_in {theta['z_in']}, z_out {theta['z_out']}, z_stop {theta['z_stop']}, "
          f"h_max {theta['h_max']} días, banda {theta['band']:.2%}")
    backtest = run_frozen_backtest(log_prices, result, theta)
    metrics = split_metrics(backtest, splits)
    frames["equity"] = equity_artifact(backtest, splits)
    write_artifacts(out_dir, frames, vecm)
    trade_artifact(backtest, splits).to_csv(out_dir / "operaciones.csv", index=False)
    write_metricas_json(out_dir / "metricas.json", metrics)
    print(f"Artifacts escritos en {out_dir}")

    for split, values in metrics.items():
        print(f"{split:>10}: Sharpe {values.sharpe:5.2f} · retorno {values.retorno_total:7.1%} · "
              f"drawdown máx. {values.max_drawdown:6.1%} · {values.n_operaciones} operaciones · "
              f"comisiones {values.comisiones:,.0f} USD")


if __name__ == "__main__":
    args = parse_args()
    main(quick=args.quick)
