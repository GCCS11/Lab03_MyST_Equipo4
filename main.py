"""Full pipeline entry point. Run with: python main.py

Every parameter comes frozen from artifacts/ (chosen on the training partition only):
1. Prices of the selected group (seleccion.json) and the train / test / validation cut dates.
2. VECM on training with the Johansen vector: alpha, beta'alpha and the model half-life.
3. Filter 1: dynamic weights; the equilibrium deviation uses the prior weights beta_{t|t-1}.
4. Filter 2: latent mean and scale of the deviation, and the causal z-score.
5. Trading rule with the thresholds of theta.json.
6. Artifacts for the dashboard (dashboard/data.py lists the files and columns).
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cointegration.seleccion import PRICES_FILE
from config import ARTIFACTS_DIR, SEED, set_seed
from kalman.signal_filter import run_signal_filter
from kalman.weights_filter import run_weights_filter
from strategy.signals import spread_direction
from strategy.vecm import equilibrium_deviation, fit_vecm

SIGNAL_KEYS = ("Q", "R", "phi", "lam", "mu0", "P0", "burn_in")
RULE_KEYS = ("z_in", "z_out", "z_stop", "h_max")


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
    write_artifacts(out_dir, signal_artifacts(log_prices, splits, result), vecm)
    print(f"Artifacts de señales escritos en {out_dir}")

    if theta is None:
        print("Pendiente: artifacts/theta.json (búsqueda de umbrales, J6). Las señales quedan en 0.")
    else:
        trades = int((result["rule"]["exit_reason"] != "").sum())
        print(f"Operaciones cerradas con theta congelado: {trades}")
    print("Pendiente: motor de backtest y métricas (backtest/, performance/): equity, operaciones y métricas.")


if __name__ == "__main__":
    args = parse_args()
    main(quick=args.quick)
