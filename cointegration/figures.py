"""Figuras 1 y 4 del reporte: precios normalizados del grupo seleccionado y evolucion del
primer eigenvector de Johansen en ventana movil.

Correr con: python -m cointegration.figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cointegration.seleccion import PRICES_FILE
from config import ARTIFACTS_DIR, DOCS_DIR
from dashboard.theme import asset_colors, palette

FIGURES_DIR = DOCS_DIR / "figures"


def _unstable_spans(index, unstable):
    """Rangos [inicio, fin] de fechas consecutivas con ref_inestable=True, para sombrear."""
    spans = []
    start = None
    for position, flag in enumerate(unstable):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            spans.append((index[start], index[position - 1]))
            start = None
    if start is not None:
        spans.append((index[start], index[-1]))
    return spans


def plot_precios(selection, mode="light"):
    """Figura 1: precios del grupo normalizados a 100 al inicio, con los cortes de seleccion.json."""
    colors = palette(mode)
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)[selection["tickers"]]
    normalized = prices / prices.iloc[0] * 100.0

    figure, axis = plt.subplots(figsize=(10, 4.8))
    for ticker, color in asset_colors(selection["tickers"], mode).items():
        axis.plot(normalized.index, normalized[ticker], linewidth=1.2, label=ticker, color=color)
    axis.axvline(
        pd.Timestamp(selection["train_end"]), color=colors["muted"], linestyle="--",
        linewidth=1, label="Fin de entrenamiento",
    )
    axis.axvline(
        pd.Timestamp(selection["test_end"]), color=colors["muted"], linestyle=":",
        linewidth=1, label="Fin de prueba",
    )
    axis.set_title("Figura 1. Precios del grupo seleccionado (base 100)")
    axis.set_xlabel("Fecha")
    axis.set_ylabel("Precio normalizado (inicio = 100)")
    axis.legend(ncol=4, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "fig1_precios.png", dpi=160)
    plt.close(figure)


def plot_eigenvector(selection, mode="light"):
    """Figura 4: pesos normalizados del primer eigenvector por ventana, una linea por activo.

    Las ventanas con ref_inestable=True (pata de referencia con menos del 5% del peso maximo
    de su ventana) se marcan con una banda sombreada, porque dividir por esa pata ahi es
    numericamente inestable.
    """
    colors = palette(mode)
    table = pd.read_csv(ARTIFACTS_DIR / "eigvec_movil.csv", index_col="date", parse_dates=True)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    for ticker, color in asset_colors(selection["tickers"], mode).items():
        axis.plot(table.index, table[ticker], linewidth=1.2, label=ticker, color=color)
    axis.axhline(0, color=colors["baseline"], linewidth=0.8)

    spans = _unstable_spans(table.index, table["ref_inestable"].to_numpy())
    for i, (start, end) in enumerate(spans):
        axis.axvspan(
            start, end, color=colors["muted"], alpha=0.25,
            label="Pata de referencia inestable" if i == 0 else None,
        )

    axis.set_title("Figura 4. Evolución del primer eigenvector de Johansen (ventana móvil)")
    axis.set_xlabel("Fecha")
    axis.set_ylabel("Peso normalizado")
    axis.legend(ncol=4, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "fig4_eigenvector.png", dpi=160)
    plt.close(figure)


def main():
    selection = json.loads((ARTIFACTS_DIR / "seleccion.json").read_text(encoding="utf-8"))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_precios(selection)
    plot_eigenvector(selection)


if __name__ == "__main__":
    main()
