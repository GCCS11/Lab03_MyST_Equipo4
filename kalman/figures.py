"""Figuras 3 y 5 del reporte: pesos dinamicos y ganancia de Kalman de ambos filtros.

Correr con: python -m kalman.figures            (solo entrenamiento)
            python -m kalman.figures --completo (despues de la corrida unica de prueba y validacion)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, DATA_DIR, DOCS_DIR
from kalman.tuning import run_filters

FIGURES_DIR = DOCS_DIR / "figures"


def mark_train_end(axis, train_end, last_date):
    """Marca el fin de entrenamiento si la serie continua despues de esa fecha."""
    if last_date > pd.Timestamp(train_end):
        axis.axvline(pd.Timestamp(train_end), color="0.3", linestyle="--", linewidth=1, label="Fin de entrenamiento")


def plot_weights(first, params):
    """Figura 3: una linea por activo; la pata normalizada queda fija en +1."""
    figure, axis = plt.subplots(figsize=(10, 4.8))
    burned = first.iloc[params["burn_in"] :]
    for ticker in params["tickers"]:
        axis.plot(burned.index, burned[f"beta_{ticker}"], linewidth=1.2, label=ticker)
    mark_train_end(axis, params["train_end"], burned.index[-1])
    axis.set_title("Figura 3. Evolución del vector de pesos del Filtro 1")
    axis.set_xlabel("Fecha")
    axis.set_ylabel("Peso previo sobre log-precios")
    axis.axhline(0, color="0.8", linewidth=0.8)
    axis.legend(ncol=4, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "fig3_pesos.png", dpi=160)
    plt.close(figure)


def plot_gains(first, second, params):
    """Figura 5: valor absoluto de la ganancia del filtro 1 por componente y ganancia del filtro 2."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    burned_first = first.iloc[params["burn_in"] :]
    for column in burned_first.filter(like="gain_").columns:
        smooth = burned_first[column].abs().rolling(21).mean()
        top.plot(smooth.index, smooth, linewidth=1.1, label=column.removeprefix("gain_"))
    top.set_title("Figura 5. Ganancia de Kalman de ambos filtros")
    top.set_ylabel("Filtro 1: |K| (media de 21 días)")
    top.legend(ncol=5, fontsize=9)
    burned_second = second.iloc[params["burn_in"] :]
    bottom.plot(burned_second.index, burned_second["gain"], color="tab:red", linewidth=1.3, label="Media del spread")
    bottom.set_ylabel("Filtro 2: K")
    bottom.set_xlabel("Fecha")
    bottom.legend(fontsize=9)
    for axis in (top, bottom):
        mark_train_end(axis, params["train_end"], burned_first.index[-1])
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "fig5_ganancias.png", dpi=160)
    plt.close(figure)


def main(complete=False):
    """Genera ambas figuras con los parametros congelados de artifacts/filtros.json."""
    params = json.loads((ARTIFACTS_DIR / "filtros.json").read_text())
    prices = pd.read_csv(DATA_DIR / "raw" / "prices_utilities.csv", index_col=0, parse_dates=True)
    prices = prices[params["tickers"]] if complete else prices.loc[: params["train_end"], params["tickers"]]
    first, second = run_filters(np.log(prices), params)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_weights(first, params)
    plot_gains(first, second, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figuras 3 y 5 del reporte")
    parser.add_argument("--completo", action="store_true", help="incluye prueba y validacion")
    main(complete=parser.parse_args().completo)
