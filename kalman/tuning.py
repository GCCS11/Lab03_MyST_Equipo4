"""Eleccion de Q, R, phi y lambda con walk-forward dentro de entrenamiento (S13 p23).

Criterio, fijado antes de correr:
1. Restriccion: en las ventanas de evaluacion, la vida media mediana del spread del filtro 1
   queda entre 5 y 60 dias (S14 p16). Asi se descartan las Q que persiguen el spread.
2. Objetivo: mayor Sharpe neto concatenando las ventanas de evaluacion, con los costos del
   enunciado, promediado entre dos juegos de umbrales provisionales. A igual Sharpe, menor Q.

En cada ventana, Johansen, phi, c y R se reestiman solo con los datos previos. La contabilidad
es una aproximacion en pesos para comparar configuraciones; no reemplaza al motor de backtest.

Correr con: python -m kalman.tuning
"""
import json
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import ARTIFACTS_DIR, BORROW_RATE_ANNUAL, COMMISSION, DATA_DIR, POSITION_FRACTION, TRADING_DAYS
from kalman.calibration import DEFAULT_BURN_IN, calibrate_signal_filter, calibrate_weights_filter, fit_ar1, half_life
from kalman.signal_filter import run_signal_filter
from kalman.weights_filter import run_weights_filter

MIN_FIT_ROWS = 1008
FOLD_ROWS = 231
HALF_LIFE_RANGE = (5, 60)
WEIGHT_RATIOS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
SIGNAL_RATIOS = (1e-4, 1e-3, 1e-2)
SIGNAL_LAMBDAS = (0.97, 0.99)
SIGNAL_USE_AR1 = (False, True)
PROVISIONAL_THETAS = ({"z_in": 2.0, "z_out": 0.5, "z_stop": 3.5}, {"z_in": 1.5, "z_out": 0.5, "z_stop": 3.0})
HOLDING_MULTIPLE = 2.0


def strategy_returns(log_prices, betas, direction):
    """Retornos netos en pesos: senal al cierre de t, operacion al cierre de t+1.

    El peso objetivo de cada activo es d * beta / suma(abs(beta)) * 80%, sobre log-precios.
    """
    gross = betas.to_numpy() / np.abs(betas.to_numpy()).sum(axis=1, keepdims=True)
    target = POSITION_FRACTION * direction.to_numpy()[:, None] * gross
    held = pd.DataFrame(target, index=log_prices.index, columns=log_prices.columns).shift(1).fillna(0.0)
    asset_returns = np.expm1(log_prices.diff().fillna(0.0))
    pnl = (held.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
    commissions = COMMISSION * held.diff().abs().fillna(held.abs()).sum(axis=1)
    borrow = BORROW_RATE_ANNUAL / TRADING_DAYS * held.clip(upper=0.0).abs().sum(axis=1)
    return pnl - commissions - borrow


def annual_sharpe(returns):
    """Sharpe anualizado sin tasa libre de riesgo; cero si no hubo variacion."""
    deviation = returns.std()
    return float(np.sqrt(TRADING_DAYS) * returns.mean() / deviation) if deviation > 0 else 0.0


def johansen_vector(fit_prices, leg):
    """Primer vector de Johansen con la pata fija en +1, estimado solo con `fit_prices`."""
    vector = coint_johansen(fit_prices, det_order=0, k_ar_diff=1).evec[:, 0]
    return vector / vector[list(fit_prices.columns).index(leg)]


def fold_filters(fit_end, eval_end, log_prices, leg, weight_ratio):
    """Calibra el filtro 1 hasta fit_end y lo corre hasta eval_end; regresa frame y spread de ajuste."""
    fit_prices = log_prices.iloc[:fit_end]
    beta = johansen_vector(fit_prices, leg)
    params = calibrate_weights_filter(fit_prices, beta, leg, weight_ratio)
    return run_weights_filter(log_prices.iloc[:eval_end], beta, leg, params)


def evaluate_configurations(train, leg, rule, holding_rule):
    """Regresa una fila por configuracion con Sharpe, operaciones y vida media fuera de ajuste."""
    folds = [(end, end + FOLD_ROWS) for end in range(MIN_FIT_ROWS, len(train) - FOLD_ROWS + 1, FOLD_ROWS)]
    rows = []
    for weight_ratio in WEIGHT_RATIOS:
        filtered = {fold: fold_filters(*fold, train, leg, weight_ratio) for fold in folds}
        half_lives = [half_life(filtered[fold]["spread"].iloc[fold[0] :]) for fold in folds]
        for use_ar1, lam, signal_ratio in product(SIGNAL_USE_AR1, SIGNAL_LAMBDAS, SIGNAL_RATIOS):
            sharpes, trades = [], []
            for base_theta in PROVISIONAL_THETAS:
                returns, entries = [], 0
                for fold, frame in filtered.items():
                    fit_end = fold[0]
                    spread = frame["spread"]
                    params = calibrate_signal_filter(spread.iloc[:fit_end], signal_ratio, lam, DEFAULT_BURN_IN, use_ar1)
                    z = run_signal_filter(spread, params)["z"]
                    h_max = holding_rule(half_life(spread.iloc[:fit_end]), HOLDING_MULTIPLE)
                    direction = rule(z, {**base_theta, "h_max": h_max})["d"]
                    betas = frame.filter(like="beta_").rename(columns=lambda name: name.removeprefix("beta_"))
                    net = strategy_returns(train.iloc[: fold[1]], betas[train.columns], direction)
                    returns.append(net.iloc[fit_end:])
                    entries += int((direction.iloc[fit_end:].diff().abs() > 0).sum() / 2)
                sharpes.append(annual_sharpe(pd.concat(returns)))
                trades.append(entries)
            rows.append(
                {
                    "weight_q_ratio": weight_ratio,
                    "signal_use_ar1": use_ar1,
                    "lam": lam,
                    "signal_q_ratio": signal_ratio,
                    "median_half_life": float(np.median(half_lives)),
                    "sharpe": float(np.mean(sharpes)),
                    "sharpe_theta_1": sharpes[0],
                    "sharpe_theta_2": sharpes[1],
                    "round_trips": float(np.mean(trades)),
                }
            )
    return pd.DataFrame(rows)


def choose_configuration(table):
    """Aplica el criterio del encabezado y regresa la fila elegida."""
    low, high = HALF_LIFE_RANGE
    admissible = table[table["median_half_life"].between(low, high)]
    if admissible.empty:
        raise ValueError("Ninguna configuracion cumple la vida media de 5 a 60 dias")
    return admissible.sort_values(["sharpe", "weight_q_ratio"], ascending=[False, True]).iloc[0]


def freeze_parameters(train, selection, chosen):
    """Calibra con todo entrenamiento la configuracion elegida y regresa filtros.json."""
    beta, leg = selection["beta_johansen"], selection["pata_normalizada"]
    weights = calibrate_weights_filter(train, beta, leg, float(chosen["weight_q_ratio"]))
    spread = run_weights_filter(train, beta, leg, weights)["spread"]
    signal = calibrate_signal_filter(
        spread, float(chosen["signal_q_ratio"]), float(chosen["lam"]), DEFAULT_BURN_IN, bool(chosen["signal_use_ar1"])
    )
    return {
        "tickers": selection["tickers"],
        "beta0": beta,
        "pata_normalizada": leg,
        "train_end": selection["train_end"],
        "filtro1": {**weights, "q_ratio": float(chosen["weight_q_ratio"])},
        "filtro2": {**signal, "q_ratio": float(chosen["signal_q_ratio"]), "entrada": "spread del filtro 1"},
        "burn_in": DEFAULT_BURN_IN,
        "walk_forward": {
            "sharpe": float(chosen["sharpe"]),
            "vueltas_completas": float(chosen["round_trips"]),
            "vida_media_mediana": float(chosen["median_half_life"]),
        },
    }


def run_filters(log_prices, params):
    """Corre ambos filtros con parametros congelados; regresa los dos DataFrames."""
    first = run_weights_filter(log_prices, params["beta0"], params["pata_normalizada"], params["filtro1"])
    keys = ("Q", "R", "phi", "lam", "mu0", "P0", "burn_in")
    second = run_signal_filter(first["spread"], {key: params["filtro2"][key] for key in keys})
    return first, second


def run_tuning():
    """Elige la configuracion y escribe artifacts/filtros.json y la malla completa."""
    from strategy.signals import max_holding, spread_direction

    selection = json.loads((ARTIFACTS_DIR / "seleccion.json").read_text())
    prices = pd.read_csv(DATA_DIR / "raw" / "prices_utilities.csv", index_col=0, parse_dates=True)
    train = np.log(prices.loc[: selection["train_end"], selection["tickers"]])

    table = evaluate_configurations(train, selection["pata_normalizada"], spread_direction, max_holding)
    chosen = choose_configuration(table)
    params = freeze_parameters(train, selection, chosen)
    table.to_csv(ARTIFACTS_DIR / "filtros_malla.csv", index=False)
    (ARTIFACTS_DIR / "filtros.json").write_text(json.dumps(params, indent=2) + "\n")
    return table, chosen, params


if __name__ == "__main__":
    _, _, frozen = run_tuning()
    print(json.dumps(frozen, indent=2))
