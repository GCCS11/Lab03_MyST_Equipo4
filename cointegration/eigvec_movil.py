"""Johansen movil sobre el grupo seleccionado (artifacts/seleccion.json).

Ventana de 252 dias, paso de 5, mismo det_order y k_ar_diff que cointegration/johansen.py
(fijados en config.py). Cada fila usa solo datos hasta su propia fecha: la ventana que termina
en t nunca ve precios posteriores a t.

El primer eigenvector de cada ventana se escala dividiendo por el peso de la pata de mayor
|peso| EN LA PRIMERA VENTANA, fija para todas las ventanas siguientes (no se recalcula la pata
por ventana). Dividir por el valor de esa misma pata en cada ventana tambien fija el signo:
v / v[pata] y (-v) / (-v[pata]) son el mismo vector, asi que las dos convenciones de signo del
eigenvector de Johansen (ambiguo hasta el signo) colapsan en una sola serie comparable entre
ventanas.

Si en alguna ventana esa pata de referencia pesa, en valor absoluto, menos del 5% del peso
maximo de esa misma ventana (antes de normalizar), dividir por ella es numericamente inestable:
la fila se conserva pero se marca con ref_inestable=True en vez de ocultarse.

Correr con: python -m cointegration.eigvec_movil
            python -m cointegration.eigvec_movil --hasta train   (unico modo permitido por ahora)
"""
import argparse
import json

import numpy as np
import pandas as pd

from cointegration.johansen import johansen_test
from cointegration.seleccion import PRICES_FILE
from config import ARTIFACTS_DIR
from data.loader import split_positions

WINDOW = 252
STEP = 5
REF_MIN_FRACTION = 0.05


def load_group_log_prices(hasta="train"):
    """Log precios del grupo de artifacts/seleccion.json, recortados al tramo pedido."""
    selection = json.loads((ARTIFACTS_DIR / "seleccion.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)[selection["tickers"]]
    if hasta == "train":
        train_end, _ = split_positions(len(prices))
        prices = prices.iloc[:train_end]
    return np.log(prices), selection["tickers"]


def rolling_eigenvector(log_prices, window=WINDOW, step=STEP):
    """Johansen por ventana: date, un peso por ticker, trace_stat y ref_inestable.

    `log_prices` debe tener solo las fechas permitidas (ya recortadas a entrenamiento si aplica):
    la ventana que termina en la fila t no usa ninguna fecha posterior a t.
    """
    tickers = list(log_prices.columns)
    starts = range(0, len(log_prices) - window + 1, step)
    leg = None
    rows = []
    for start in starts:
        window_data = log_prices.iloc[start : start + window]
        result, summary = johansen_test(window_data)
        vector = result.evec[:, 0]
        if leg is None:
            leg = int(np.argmax(np.abs(vector)))
        max_weight = float(np.abs(vector).max())
        ref_inestable = bool(abs(vector[leg]) < REF_MIN_FRACTION * max_weight)
        normalized = vector / vector[leg]
        row = {
            "date": log_prices.index[start + window - 1],
            "trace_stat": summary["trace_stat_r0"],
            "ref_inestable": ref_inestable,
        }
        row.update(zip(tickers, normalized))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("date")
    return table[tickers + ["trace_stat", "ref_inestable"]]


def run(hasta="train"):
    """Calcula la tabla y la escribe en artifacts/eigvec_movil.csv."""
    log_prices, _ = load_group_log_prices(hasta)
    table = rolling_eigenvector(log_prices)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ARTIFACTS_DIR / "eigvec_movil.csv", index_label="date")
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Johansen movil del grupo seleccionado")
    parser.add_argument("--hasta", choices=("train", "todo"), default="train")
    return parser.parse_args()


def main(hasta=None):
    """Corre el calculo y reporta cuantas ventanas quedaron con ref_inestable=True.

    Se niega a correr con hasta="todo": todavia no existe artifacts/theta.json congelado.
    """
    hasta = parse_args().hasta if hasta is None else hasta
    if hasta == "todo":
        raise SystemExit(
            "todavia no existe artifacts/theta.json congelado: no corras --hasta todo"
        )
    tabla = run(hasta)
    n_inestables = int(tabla["ref_inestable"].sum())
    print(f"Ventanas calculadas: {len(tabla)}")
    print(f"Ventanas con pata de referencia inestable (ref_inestable=True): {n_inestables}")
    return tabla


if __name__ == "__main__":
    main()
