"""Unit tests for performance/figures.py: fig7_equity con una, dos y tres particiones, y
fig8_distribucion con operaciones y con cero operaciones. Datos sinteticos unicamente.
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from performance.figures import fig7_equity, fig8_distribucion


def make_equity(splits):
    """Curva de capital continua y creciente, una fila por dia, con las particiones de
    `splits` en el orden en que aparecen (cada una con 10 dias)."""
    n_per_split = 10
    index = pd.bdate_range("2021-01-04", periods=n_per_split * len(splits), name="date")
    equity = 1_000_000.0 + np.cumsum(np.full(len(index), 50.0))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    split_col = np.repeat(splits, n_per_split)
    return pd.DataFrame({"split": split_col, "equity": equity, "drawdown": drawdown}, index=index)


def make_operaciones(retornos):
    n = len(retornos)
    index = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({
        "entrada": index, "salida": index, "split": "train", "direccion": 1,
        "pnl_bruto": 0.0, "comisiones": 0.0, "prestamo": 0.0, "pnl_neto": 0.0,
        "retorno": retornos, "motivo": "reversion",
    })


@pytest.mark.parametrize("splits", [
    ["train"],
    ["train", "test"],
    ["train", "test", "validation"],
])
def test_fig7_equity_runs_with_any_number_of_splits(splits):
    equity = make_equity(splits)
    figure = fig7_equity(equity)

    axis = figure.axes[0]
    assert "Figura 7" in axis.get_title()
    # la linea de capital, mas una linea vertical de corte por cada particion despues de la
    # primera (las verticales tienen un solo valor distinto de x)
    expected_vlines = max(0, len(splits) - 1)
    assert len(axis.lines) == 1 + expected_vlines
    assert sum(1 for line in axis.lines if len(set(line.get_xdata())) == 1) == expected_vlines
    # el nombre de cada particion queda escrito sobre la grafica
    texts = {t.get_text().strip() for t in axis.texts}
    nombres = {"train": "Entrenamiento", "test": "Prueba", "validation": "Validación"}
    assert texts == {nombres[s] for s in splits}


def test_fig7_equity_has_a_secondary_axis_for_drawdown():
    equity = make_equity(["train", "test", "validation"])
    figure = fig7_equity(equity)
    assert len(figure.axes) == 2


def test_fig8_distribucion_with_trades_shows_mean_std_skew_and_n():
    rng = np.random.default_rng(0)
    operaciones = make_operaciones(rng.normal(0.0, 0.01, 40))
    figure = fig8_distribucion(operaciones)

    axis = figure.axes[0]
    assert "Figura 8" in axis.get_title()
    texto = " ".join(t.get_text() for t in axis.texts)
    assert "media" in texto and "desv" in texto and "asimetría" in texto and "n = 40" in texto
    assert len(axis.patches) > 0  # las barras del histograma


def test_fig8_distribucion_with_no_trades_shows_a_notice_and_no_histogram():
    operaciones = make_operaciones([])
    figure = fig8_distribucion(operaciones)

    axis = figure.axes[0]
    assert len(axis.patches) == 0
    texto = " ".join(t.get_text() for t in axis.texts)
    assert "No hay operaciones" in texto
