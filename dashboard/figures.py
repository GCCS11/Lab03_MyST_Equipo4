"""Plotly figures of the dashboard. Pure functions: artifacts in, figures out, no Streamlit.

Rules: one y-axis per chart (the drawdown goes in its own panel), fixed color per asset, thin
lines, recessive grid, and every panel carries its data so the app can show a table view.
"""
from collections import namedtuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from dashboard.data import select_splits, split_bounds
from dashboard.theme import SPLIT_LABELS, asset_colors, palette

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
Panel = namedtuple("Panel", ["key", "figure", "data", "source"])


def money(value):
    """Dollar amount with the minus sign before the symbol: −$1,234."""
    return f"{'−' if value < 0 else ''}${abs(value):,.0f}"


def _rgba(hex_color, alpha):
    """Hex color with transparency, for area fills."""
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _style(fig, mode, title, y_title=None, height=380):
    """Shared chrome: title, surface, hairline grid, legend above the plot."""
    colors = palette(mode)
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": colors["ink"]}},
        height=height,
        margin={"l": 64, "r": 24, "t": 64, "b": 40},
        paper_bgcolor=colors["surface"],
        plot_bgcolor=colors["surface"],
        font={"family": FONT, "size": 12, "color": colors["ink_secondary"]},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=False, linecolor=colors["baseline"], ticks="outside", tickcolor=colors["baseline"])
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False, linecolor=colors["baseline"])
    if y_title:
        fig.update_layout(yaxis_title=y_title)
    return fig


def _mark_splits(fig, bounds, mode):
    """Hairline at the start of each split, labelled inside the top of the plot (the legend
    sits above it)."""
    colors = palette(mode)
    for split, (start, _) in bounds.items():
        fig.add_shape(
            type="line", xref="x", yref="paper", x0=start, x1=start, y0=0, y1=1,
            line={"color": colors["baseline"], "width": 1},
        )
        fig.add_annotation(
            x=start, y=1, xref="x", yref="paper", text=SPLIT_LABELS.get(split, split), xshift=4,
            showarrow=False, xanchor="left", yanchor="top", font={"color": colors["muted"], "size": 11},
        )


def lines_figure(frame, columns, names, colors, title, y_title, mode, bounds=None, log_y=False):
    """One thin line per column, with its fixed color."""
    fig = go.Figure()
    for column, name, color in zip(columns, names, colors):
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode="lines", name=name,
            line={"color": color, "width": 2}, hovertemplate="%{y:.4g}",
        ))
    _style(fig, mode, title, y_title)
    if log_y:
        fig.update_yaxes(type="log")
    if bounds:
        _mark_splits(fig, bounds, mode)
    return fig


def component_figure(frame, prefix, title, y_title, tickers, mode, bounds=None, log_y=False, skip=()):
    """Lines for every column that starts with `prefix`, except the names in `skip`.

    Assets keep their color; other components (the intercept c) are muted, and a lone
    series takes the first slot.
    """
    colors = palette(mode)
    by_asset = asset_colors(tickers, mode)
    columns = [col for col in frame.columns if col.startswith(prefix) and (col[len(prefix):] or col) not in skip]
    names = [col[len(prefix):] or col for col in columns]
    line_colors = [by_asset.get(name, colors["muted"]) for name in names]
    if len(columns) == 1 and names[0] not in by_asset:
        line_colors = [colors["series"][0]]
    return lines_figure(frame, columns, names, line_colors, title, y_title, mode, bounds, log_y)


def equity_drawdown(equity, mode):
    """Equity curve on top and its drawdown below, sharing the time axis."""
    colors = palette(mode)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["equity"], mode="lines", name="Capital",
        line={"color": colors["series"][0], "width": 2}, hovertemplate="%{y:$,.0f}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["drawdown"], mode="lines", name="Drawdown",
        line={"color": colors["series"][7], "width": 1.5}, fill="tozeroy",
        fillcolor=_rgba(colors["series"][7], 0.15), hovertemplate="%{y:.1%}",
    ), row=2, col=1)
    _style(fig, mode, "Curva de capital y drawdown", height=460)
    fig.update_yaxes(tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    _mark_splits(fig, split_bounds(equity), mode)
    return fig


def cost_waterfall(metrics, splits, mode):
    """Gross P&L broken down into commissions, borrow cost and net P&L."""
    colors = palette(mode)
    totals = {key: sum(metrics[s][key] for s in splits if s in metrics) for key in ("pnl_bruto", "comisiones", "prestamo")}
    net = totals["pnl_bruto"] - totals["comisiones"] - totals["prestamo"]
    values = [totals["pnl_bruto"], -totals["comisiones"], -totals["prestamo"], net]
    fig = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "relative", "total"],
        x=["P&L bruto", "Comisiones", "Préstamo", "P&L neto"],
        y=values, text=[money(v) for v in values], textposition="outside", cliponaxis=False,
        connector={"line": {"color": colors["baseline"], "width": 1}},
        totals={"marker": {"color": colors["series"][0]}},
        decreasing={"marker": {"color": colors["series"][1]}},
        increasing={"marker": {"color": colors["series"][2]}},
        hovertemplate="%{x}: %{y:$,.0f}<extra></extra>",
    ))
    _style(fig, mode, "Atribución de costos", "USD")
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(tickformat="$,.0f")
    return fig


def trade_returns(trades, mode):
    """Histogram of returns per trade with the fitted normal on the same count scale."""
    colors = palette(mode)
    returns = trades["retorno"].dropna().to_numpy() * 100
    fig = go.Figure()
    if len(returns) >= 2:
        edges = np.histogram_bin_edges(returns, bins=min(40, max(8, int(np.sqrt(len(returns))) * 2)))
        width = edges[1] - edges[0]
        fig.add_trace(go.Histogram(
            x=returns, xbins={"start": edges[0], "end": edges[-1], "size": width},
            name="Operaciones", marker={"color": colors["series"][0]},
            hovertemplate="%{x}: %{y} operaciones<extra></extra>",
        ))
        grid = np.linspace(edges[0], edges[-1], 200)
        mean, std = returns.mean(), returns.std(ddof=1)
        fig.add_trace(go.Scatter(
            x=grid, y=len(returns) * width * stats.norm.pdf(grid, mean, std), mode="lines",
            name="Normal ajustada", line={"color": colors["ink"], "width": 2}, hoverinfo="skip",
        ))
    else:
        fig.add_annotation(text="No hay operaciones suficientes", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    _style(fig, mode, "Retornos por operación", "Operaciones")
    fig.update_layout(hovermode="closest", bargap=0.06)
    fig.update_xaxes(title_text="Retorno neto (%)")
    return fig


def spread_signals(signals, mode):
    """Spread with its estimated mean and the entry and exit signals marked by shape."""
    colors = palette(mode)
    d = signals["d"].astype(int)
    previous = d.shift(fill_value=0)
    events = [
        ("Entrada larga", (d == 1) & (previous != 1), "triangle-up", colors["series"][2]),
        ("Entrada corta", (d == -1) & (previous != -1), "triangle-down", colors["series"][1]),
        ("Salida", signals["exit_reason"].fillna("").astype(str) != "", "x", colors["ink"]),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=signals.index, y=signals["spread"], mode="lines", name="Spread S_t",
        line={"color": colors["series"][0], "width": 1.5}, hovertemplate="%{y:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=signals.index, y=signals["mu_prior"], mode="lines", name="Media estimada (Filtro 2)",
        line={"color": colors["muted"], "width": 1.5}, hovertemplate="%{y:.4f}",
    ))
    for name, mask, symbol, color in events:
        fig.add_trace(go.Scatter(
            x=signals.index[mask], y=signals["spread"][mask], mode="markers", name=name,
            marker={"symbol": symbol, "size": 10, "color": color, "line": {"width": 1, "color": colors["surface"]}},
            hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
        ))
    _style(fig, mode, "Spread del portafolio con señales", "S_t", height=420)
    _mark_splits(fig, split_bounds(signals), mode)
    return fig


def zscore_thresholds(signals, theta, mode):
    """Causal z-score with the entry, exit and stop thresholds."""
    colors = palette(mode)
    fig = go.Figure(go.Scatter(
        x=signals.index, y=signals["z"], mode="lines", name="z_t",
        line={"color": colors["series"][0], "width": 1.5}, hovertemplate="%{y:.2f}",
    ))
    for label, level in (("z_stop", theta["z_stop"]), ("z_in", theta["z_in"]), ("z_out", theta["z_out"])):
        for sign in (1, -1):
            fig.add_hline(
                y=sign * level, line={"color": colors["muted"], "width": 1, "dash": "dot"},
                annotation_text=f"{'+' if sign > 0 else '−'}{label}", annotation_position="right",
                annotation_font={"color": colors["muted"], "size": 11},
            )
    _style(fig, mode, "z-score con umbrales", "z")
    fig.update_layout(showlegend=False, margin={"r": 72})
    _mark_splits(fig, split_bounds(signals), mode)
    return fig


def rolling_sharpe(equity, mode):
    """252-day rolling Sharpe over the whole sample."""
    colors = palette(mode)
    fig = lines_figure(
        equity, ["rolling_sharpe"], ["Sharpe móvil 252 días"], [colors["series"][0]],
        "Sharpe móvil de 252 días", "Sharpe", mode, bounds=split_bounds(equity),
    )
    fig.add_hline(y=0, line={"color": colors["baseline"], "width": 1})
    fig.update_layout(showlegend=False)
    return fig


def _fmt_ratio(value):
    return "∞" if np.isinf(value) else f"{value:.2f}"


METRICS = (
    ("sharpe", "Sharpe", _fmt_ratio),
    ("sortino", "Sortino", _fmt_ratio),
    ("calmar", "Calmar", _fmt_ratio),
    ("max_drawdown", "Drawdown máx.", lambda v: f"{v:.1%}"),
    ("win_rate", "Win rate", lambda v: f"{v:.1%}"),
    ("profit_factor", "Profit factor", _fmt_ratio),
    ("retorno_total", "Retorno total", lambda v: f"{v:.1%}"),
    ("n_operaciones", "Operaciones", lambda v: f"{int(v)}"),
)


def kpis(metrics, split):
    """(label, formatted value) of the performance card for one split."""
    values = metrics.get(split, {})
    return [(label, fmt(values[key]) if key in values else "—") for key, label, fmt in METRICS]


def metrics_table(metrics):
    """Every metric by split, formatted, as a table."""
    table = {
        SPLIT_LABELS.get(split, split): {label: fmt(values[key]) for key, label, fmt in METRICS if key in values}
        for split, values in metrics.items()
    }
    return pd.DataFrame(table)


def executive_panels(artifacts, splits, mode):
    """Figures of the executive view, in order. A panel with figure None names its missing file."""
    equity, trades, metrics = artifacts["equity"], artifacts["operaciones"], artifacts["metricas"]
    panels = []
    if equity is not None:
        data = select_splits(equity, splits)
        panels.append(Panel("equity", equity_drawdown(data, mode), data, "equity.csv"))
    else:
        panels.append(Panel("equity", None, None, "equity.csv"))
    panels.append(
        Panel("costos", cost_waterfall(metrics, splits, mode), metrics_table(metrics), "metricas.json")
        if metrics is not None else Panel("costos", None, None, "metricas.json")
    )
    if trades is not None:
        data = trades[trades["split"].isin(splits)]
        panels.append(Panel("retornos", trade_returns(data, mode), data, "operaciones.csv"))
    else:
        panels.append(Panel("retornos", None, None, "operaciones.csv"))
    return panels


def technical_panels(artifacts, splits, mode):
    """Figures of the technical view, in order. A panel with figure None names its missing file."""
    selection = artifacts["seleccion"] or {}
    tickers = selection.get("tickers", [])
    leg = selection.get("pata_normalizada")
    fixed = (leg,) if leg else ()
    note = f" · {leg} = 1, fuera de la gráfica" if leg else ""
    signals = artifacts["senales"]
    bounds = split_bounds(select_splits(signals, splits)) if signals is not None else None
    specs = [
        ("spread", "senales", lambda f: spread_signals(f, mode)),
        ("z", "senales", lambda f: zscore_thresholds(f, artifacts["theta"], mode)),
        ("pesos", "pesos", lambda f: component_figure(
            f, "beta_", f"Pesos β̂_t (Filtro 1){note}", "β", tickers, mode, bounds, skip=fixed)),
        ("eigvec", "eigvec_movil", lambda f: component_figure(
            f, "", f"Primer eigenvector de Johansen móvil{note}", "Peso", tickers, mode, bounds, skip=fixed)),
        ("ganancia_f1", "ganancias", lambda f: component_figure(f, "f1_gain_", "Ganancia de Kalman, Filtro 1", "K_t", tickers, mode, bounds)),
        ("ganancia_f2", "ganancias", lambda f: component_figure(f, "f2_gain", "Ganancia de Kalman, Filtro 2", "K_t", tickers, mode, bounds)),
        ("varianza_f1", "varianzas", lambda f: component_figure(f, "f1_pvar_", "Varianza posterior, Filtro 1", "P_t", tickers, mode, bounds, log_y=True)),
        ("varianza_f2", "varianzas", lambda f: component_figure(f, "f2_pvar", "Varianza posterior, Filtro 2", "P_t", tickers, mode, bounds, log_y=True)),
        ("sharpe", "equity", lambda f: rolling_sharpe(f, mode)),
    ]
    panels = []
    for key, name, build in specs:
        frame = artifacts[name]
        needs_theta = key == "z" and artifacts["theta"] is None
        if frame is None or needs_theta:
            panels.append(Panel(key, None, None, "theta.json" if needs_theta else f"{name}.csv"))
            continue
        data = select_splits(frame, splits, bounds)
        panels.append(Panel(key, build(data), data, f"{name}.csv"))
    return panels
