"""Johansen movil sobre el grupo seleccionado (artifacts/seleccion.json).

Ventana de 252 dias, paso de 5, mismo det_order y k_ar_diff que cointegration/johansen.py
(fijados en config.py). Cada fila usa solo datos hasta su propia fecha: la ventana que termina
en t nunca ve precios posteriores a t.

El primer eigenvector de cada ventana se normaliza en dos pasos:
1. Escala: se divide por su norma L1 (la suma de los |pesos|), asi que ninguna pata concentra la
   normalizacion y no hay riesgo de dividir por un peso casi cero.
2. Signo: Johansen entrega el eigenvector ambiguo hasta el signo (v y -v son igual de validos),
   asi que cada ventana se alinea con la anterior por el signo de su producto punto. La primera
   ventana no tiene a quien alinearse: se orienta para que su pata de mayor |peso| quede positiva.

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


def load_group_log_prices(hasta="train"):
    """Log precios del grupo de artifacts/seleccion.json, recortados al tramo pedido."""
    selection = json.loads((ARTIFACTS_DIR / "seleccion.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)[selection["tickers"]]
    if hasta == "train":
        train_end, _ = split_positions(len(prices))
        prices = prices.iloc[:train_end]
    return np.log(prices), selection["tickers"]


def _normalize(vector, previous):
    """Escala por norma L1 y alinea el signo con `previous` (o con su propia pata mayor si
    `previous` es None, para la primera ventana)."""
    normalized = vector / np.abs(vector).sum()
    if previous is None:
        leg = int(np.argmax(np.abs(normalized)))
        return -normalized if normalized[leg] < 0 else normalized
    return -normalized if np.dot(normalized, previous) < 0 else normalized


def rolling_eigenvector(log_prices, window=WINDOW, step=STEP):
    """Johansen por ventana: date, un peso por ticker y trace_stat.

    `log_prices` debe tener solo las fechas permitidas (ya recortadas a entrenamiento si aplica):
    la ventana que termina en la fila t no usa ninguna fecha posterior a t.
    """
    tickers = list(log_prices.columns)
    starts = range(0, len(log_prices) - window + 1, step)
    previous = None
    rows = []
    for start in starts:
        window_data = log_prices.iloc[start : start + window]
        result, summary = johansen_test(window_data)
        normalized = _normalize(result.evec[:, 0], previous)
        previous = normalized
        row = {"date": log_prices.index[start + window - 1], "trace_stat": summary["trace_stat_r0"]}
        row.update(zip(tickers, normalized))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("date")
    return table[tickers + ["trace_stat"]]


def run(hasta="train"):
    """Calcula la tabla y la escribe en dos archivos: artifacts/eigvec_movil.csv con date y un
    peso por ticker (el formato que lee dashboard/data.py) y artifacts/eigvec_diagnostico.csv
    con date y trace_stat."""
    log_prices, tickers = load_group_log_prices(hasta)
    table = rolling_eigenvector(log_prices)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table[tickers].to_csv(ARTIFACTS_DIR / "eigvec_movil.csv", index_label="date")
    table[["trace_stat"]].to_csv(ARTIFACTS_DIR / "eigvec_diagnostico.csv", index_label="date")
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Johansen movil del grupo seleccionado")
    parser.add_argument("--hasta", choices=("train", "todo"), default="train")
    return parser.parse_args()


def main(hasta=None):
    """Corre el calculo y escribe los dos artifacts.

    Se niega a correr con hasta="todo": todavia no existe artifacts/theta.json congelado.
    """
    hasta = parse_args().hasta if hasta is None else hasta
    if hasta == "todo":
        raise SystemExit(
            "todavia no existe artifacts/theta.json congelado: no corras --hasta todo"
        )
    tabla = run(hasta)
    print(f"Ventanas calculadas: {len(tabla)}")
    return tabla


if __name__ == "__main__":
    main()
