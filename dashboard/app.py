"""Lab 03 dashboard. Run from the repo root: streamlit run dashboard/app.py

It only reads the files in artifacts/ (written by python main.py) and never reruns the backtest.
Three views: the presentation script in 10 steps, an executive one for a portfolio manager and a
technical one for a quant researcher.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import figures  # noqa: E402
from dashboard.presentation import load_extras, presentation_view  # noqa: E402
from dashboard.data import artifacts_dir, filter_trades, load_artifacts, missing  # noqa: E402
from dashboard.theme import SPLIT_LABELS  # noqa: E402

TRADE_COLUMNS = {
    "entrada": "Entrada", "salida": "Salida", "split": "Partición", "direccion": "Dirección",
    "pnl_bruto": "P&L bruto", "comisiones": "Comisiones", "prestamo": "Préstamo",
    "pnl_neto": "P&L neto", "retorno": "Retorno", "motivo": "Motivo de salida",
}

st.set_page_config(page_title="Lab 03 · Arbitraje estadístico", layout="wide")


@st.cache_data
def cached_artifacts(directory):
    """Artifacts of the folder, read once per session."""
    return load_artifacts(directory)


@st.cache_data
def cached_extras(directory):
    """Artifacts that only the presentation view reads."""
    return load_extras(directory)


def show(panel):
    """A chart with its table view, or a note naming the missing artifact."""
    if panel.figure is None:
        st.info(f"Falta `{panel.source}` en artifacts/. Corre `python main.py`.")
        return
    st.plotly_chart(panel.figure, theme=None, key=panel.key)
    with st.expander("Ver datos"):
        st.dataframe(panel.data)


def executive_view(artifacts, splits, mode):
    """Performance card, equity and drawdown, cost attribution, trade log and returns."""
    metrics = artifacts["metricas"]
    if metrics:
        options = [s for s in SPLIT_LABELS if s in metrics]
        card_split = st.radio(
            "Tarjeta de desempeño", options, index=len(options) - 1,
            format_func=SPLIT_LABELS.get, horizontal=True, key="tarjeta",
        )
        for column, (label, value) in zip(st.columns(8), figures.kpis(metrics, card_split)):
            column.metric(label, value, border=True)
        with st.expander("Todas las particiones"):
            st.dataframe(figures.metrics_table(metrics))

    equity, costs, returns = figures.executive_panels(artifacts, splits, mode)
    show(equity)
    left, right = st.columns(2)
    with left:
        show(costs)
    with right:
        show(returns)

    st.subheader("Bitácora de operaciones")
    trades = artifacts["operaciones"]
    if trades is None:
        st.info("Falta `operaciones.csv` en artifacts/. Corre `python main.py`.")
        return
    direction_col, result_col = st.columns(2)
    directions = direction_col.multiselect(
        "Dirección", [1, -1], default=[1, -1], key="direccion",
        format_func=lambda d: "Spread largo" if d == 1 else "Spread corto",
    )
    results = result_col.multiselect("Resultado", ["Ganadora", "Perdedora"], default=["Ganadora", "Perdedora"], key="resultado")
    log = filter_trades(trades, splits, directions, results).rename(columns=TRADE_COLUMNS)
    log["Entrada"], log["Salida"] = log["Entrada"].dt.date, log["Salida"].dt.date
    log["Partición"] = log["Partición"].map(SPLIT_LABELS)
    log["Dirección"] = log["Dirección"].map({1: "Largo", -1: "Corto"})
    st.dataframe(
        log, hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="dollar")
            for name in ("P&L bruto", "Comisiones", "Préstamo", "P&L neto")
        } | {"Retorno": st.column_config.NumberColumn(format="percent")},
    )
    st.caption(f"{len(log)} de {len(trades)} operaciones con los filtros actuales.")


def technical_view(artifacts, splits, mode):
    """Spread and signals, weights, eigenvector, Kalman gains, convergence and rolling Sharpe."""
    for panel in figures.technical_panels(artifacts, splits, mode):
        show(panel)


def main():
    """Page layout: header, one filter row, then the chosen view."""
    artifacts = cached_artifacts(str(artifacts_dir()))
    mode = st.context.theme.type or "light"
    selection = artifacts["seleccion"] or {}

    st.title("Arbitraje estadístico con cointegración y filtros de Kalman")
    if selection:
        st.caption(
            f"Grupo: {', '.join(selection['tickers'])} · pata normalizada {selection['pata_normalizada']} · "
            f"{selection['grupos_evaluados_total']:,} grupos evaluados · Lab 03 MyST, Equipo 4"
        )
    absent = missing(artifacts)
    if len(absent) == len(artifacts):
        st.warning("No hay salidas del backtest en artifacts/. Corre `python main.py` primero.")
        st.stop()
    if absent:
        st.caption(f"Sin datos todavía: {', '.join(absent)}.")

    view_col, split_col = st.columns([1, 2])
    view = view_col.radio("Vista", ["Presentación", "Ejecutiva", "Técnica"], horizontal=True, key="vista")
    if view == "Presentación":
        presentation_view(artifacts, cached_extras(str(artifacts_dir())), mode)
        return
    splits = split_col.multiselect(
        "Particiones", list(SPLIT_LABELS), default=list(SPLIT_LABELS), format_func=SPLIT_LABELS.get, key="particiones",
    )
    if not splits:
        st.info("Elige al menos una partición.")
        st.stop()

    if view == "Ejecutiva":
        executive_view(artifacts, splits, mode)
    else:
        technical_view(artifacts, splits, mode)


main()
