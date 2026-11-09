"""Vista de presentación: el guion de la exposición en 10 pasos.

Cada paso tiene un título que dice la conclusión, quién expone, tres o cuatro cifras y una sola
gráfica interactiva. Las notas del expositor van cerradas para no leerlas en pantalla. Igual que
las otras vistas, solo lee artifacts/ y nunca vuelve a correr el backtest.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard import figures
from dashboard.data import split_bounds
from dashboard.theme import SPLIT_LABELS, asset_colors, palette

EXTRA_JSON = ("vecm", "prueba_anidada", "filtros")
EXTRA_CSV = ("diagnostico_particiones", "seleccion_grupos", "theta_busqueda")
SPLITS = list(SPLIT_LABELS)
HEIGHT = 520

CSS = """
<style>
.step-chip{display:inline-flex;align-items:center;padding:4px 12px;border-radius:999px;font-size:.82rem;
  border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);margin-right:6px;color:rgba(232,237,248,.8)}
.step-chip.who{color:#fff;background:linear-gradient(180deg,#5b86ff,#3a5cf0);border-color:transparent;
  box-shadow:0 0 16px rgba(79,125,255,.45)}
.step-title{font-size:2.15rem;font-weight:600;letter-spacing:-.015em;line-height:1.15;margin:.5rem 0 1rem}
.kpi{position:relative;overflow:hidden;border-radius:16px;padding:14px 16px 16px;height:100%;
  background:linear-gradient(160deg,rgba(66,110,255,.22),rgba(20,38,110,.18) 60%,rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.1);box-shadow:0 12px 36px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.06)}
.kpi::before{content:"";position:absolute;left:16px;right:16px;top:0;height:2px;border-radius:2px;
  background:var(--tone);box-shadow:0 0 14px var(--tone)}
.kpi-label{font-size:.85rem;color:rgba(232,237,248,.7);display:flex;align-items:center;gap:7px}
.kpi-label::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--tone);box-shadow:0 0 8px var(--tone)}
.kpi-value{font-size:1.9rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.2;margin-top:6px;color:#f3f6fd}
.kpi-pos{--tone:#2fd6a0}.kpi-neg{--tone:#ff5d6c}.kpi-mid{--tone:#ffc247}.kpi-info{--tone:#5b8cff}
@media (prefers-reduced-motion:no-preference){
  .step-title{animation:rise 1s cubic-bezier(.16,1,.3,1) both}
  .kpi{animation:rise .95s cubic-bezier(.16,1,.3,1) both;animation-delay:calc(120ms + var(--i,0)*90ms)}
  .kpi::before{transform-origin:left;animation:grow 1.2s cubic-bezier(.16,1,.3,1) both;
    animation-delay:calc(320ms + var(--i,0)*90ms)}
  @keyframes rise{from{opacity:0;transform:translateY(14px);filter:blur(5px)}to{opacity:1;transform:none;filter:none}}
  @keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}}
</style>
"""


def load_extras(directory):
    """Artifacts that only this view uses. Missing files come back as None."""
    directory = Path(directory)
    extras = {}
    for name in EXTRA_JSON:
        path = directory / f"{name}.json"
        extras[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    for name in EXTRA_CSV:
        path = directory / f"{name}.csv"
        extras[name] = pd.read_csv(path) if path.exists() else None
    return extras


def split_color(split, mode):
    """Color con significado: entrenamiento verde, prueba ámbar, validación rojo."""
    series = palette(mode)["series"]
    return {"train": series[2], "test": series[3], "validation": series[7]}.get(split, series[0])


def _big(fig):
    """Tamaño para proyectar: más alto y con letra más grande."""
    if fig is not None:
        fig.update_layout(height=HEIGHT, font={"size": 14})
    return fig


def pct(value, digits=1):
    return "—" if value is None or pd.isna(value) else f"{value:+.{digits}%}".replace("-", "−")


# ---------- gráficas propias de esta vista (funciones puras) ----------

def prices_figure(prices, bounds, mode):
    """Precios ajustados en base 100, una línea por activo con su color fijo."""
    base = prices / prices.iloc[0] * 100
    colors = asset_colors(list(base.columns), mode)
    return figures.lines_figure(base, list(base.columns), list(base.columns), [colors[c] for c in base.columns],
                                "Precios ajustados (base 100)", "Índice", mode, bounds)


def funnel_counts(registry):
    """Grupos que sobreviven cada etapa del embudo de selección."""
    johansen = int((registry["johansen_rank_trace"] >= 1).sum())
    candidates = registry[registry["status"] == "candidato"]
    eg = int((candidates["eg_min_p"] < 0.05).sum())
    return [("Grupos de 6", len(registry)), ("Pasan Johansen", johansen),
            ("Spread admisible", len(candidates)), ("Pasan Engle-Granger", eg), ("Elegido", 1)]


def funnel_figure(registry, mode):
    colors = palette(mode)
    stages = funnel_counts(registry)
    fig = go.Figure(go.Funnel(
        y=[name for name, _ in stages], x=[count for _, count in stages], textinfo="value",
        marker={"color": [colors["series"][0]] * (len(stages) - 1) + [colors["series"][2]]},
        connector={"line": {"color": colors["baseline"]}},
    ))
    figures._style(fig, mode, "Embudo de selección, solo con entrenamiento (número de grupos)")
    fig.update_layout(margin={"l": 170})
    return fig


def theta_figure(search, theta, mode):
    """Sharpe medio del walk-forward de cada θ de la malla, con el elegido marcado."""
    colors = palette(mode)
    summary = search.groupby("theta_id")["sharpe"].agg(["mean", "std"])
    fig = go.Figure(go.Histogram(x=summary["mean"], nbinsx=30, name="θ de la malla",
                                 marker={"color": figures._rgba(colors["series"][0], 0.55)}))
    chosen = theta.get("theta_id") if theta else None
    if chosen in summary.index:
        x = float(summary.loc[chosen, "mean"])
        fig.add_vline(x=x, line={"color": colors["series"][2], "width": 3})
        fig.add_annotation(x=x, y=1, yref="paper", text=f"θ elegido ({x:.2f})", showarrow=False,
                           xanchor="right", xshift=-8, yanchor="top", font={"color": colors["series"][2], "size": 14})
    figures._style(fig, mode, f"Sharpe medio en 5 pliegues de entrenamiento ({len(summary)} combinaciones de θ)",
                   "Número de θ")
    fig.update_xaxes(title="Sharpe medio del walk-forward")
    fig.update_layout(margin={"b": 70})
    return fig


def exits_figure(trades, mode):
    """P&L neto por partición y motivo de salida; el texto es el número de operaciones."""
    colors = palette(mode)
    tone = {"reversion": colors["series"][2], "stop": colors["series"][7], "h_max": colors["series"][3],
            "fin_de_serie": colors["muted"]}
    names = {"reversion": "Reversión", "stop": "Stop", "h_max": "Tenencia máxima", "fin_de_serie": "Fin de serie"}
    table = trades.groupby(["split", "motivo"])["pnl_neto"].agg(["sum", "count"]).reset_index()
    fig = go.Figure()
    for motive in [m for m in names if m in set(table["motivo"])]:
        rows = table[table["motivo"] == motive].set_index("split").reindex(SPLITS)
        fig.add_trace(go.Bar(
            x=[SPLIT_LABELS[s] for s in SPLITS], y=rows["sum"], name=names[motive], marker={"color": tone[motive]},
            text=[f"{int(c)} op." if pd.notna(c) else "" for c in rows["count"]], textposition="outside",
            hovertemplate="%{y:$,.0f}",
        ))
    figures._style(fig, mode, "P&L neto por motivo de salida (el texto es el número de operaciones)", "P&L neto")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat="$,.0f", zeroline=True, zerolinecolor=colors["baseline"])
    return fig


def diagnostics_figure(diag, mode):
    """Vida media, ventanas con relación y ADF del spread, con el vector de entrenamiento fijo."""
    colors = palette(mode)
    diag = diag.set_index("split").reindex(SPLITS)
    labels = [SPLIT_LABELS[s] for s in SPLITS]
    bar_colors = [split_color(s, mode) for s in SPLITS]
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Vida media del spread (días)", "Ventanas de 2 años con relación", "ADF del spread (p)"))
    fig.add_trace(go.Bar(x=labels, y=diag["vida_media"], marker={"color": bar_colors},
                         text=[f"{v:.0f}" for v in diag["vida_media"]], textposition="outside"), 1, 1)
    fig.add_trace(go.Bar(x=labels, y=diag["ventanas_con_relacion"], marker={"color": bar_colors},
                         text=[f"{v:.0%}" for v in diag["ventanas_con_relacion"]], textposition="outside"), 1, 2)
    fig.add_trace(go.Bar(x=labels, y=diag["adf_p"], marker={"color": bar_colors},
                         text=[f"{v:.2f}" for v in diag["adf_p"]], textposition="outside"), 1, 3)
    fig.add_hline(y=0.05, line={"color": colors["ink_secondary"], "dash": "dot"}, row=1, col=3)
    figures._style(fig, mode, "La relación se deshace después de 2021")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    return fig


def nested_figure(nested, mode):
    """Prueba anidada: fracción de candidatos internos que siguen estacionarios en el tramo siguiente."""
    total = nested["candidatos_train_interno"]
    counts = nested["candidatos_estacionarios"]
    labels = ["Test interno (2018-2020)", "Validación interna (2020-2021)"]
    values = [counts["test"] / total, counts["validation"] / total]
    colors = palette(mode)
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker={"color": [colors["series"][7], colors["series"][3]]},
        text=[f"{counts['test']} de {total}", f"{counts['validation']} de {total}"], textposition="outside",
        hovertemplate="%{y:.0%}",
    ))
    figures._style(fig, mode, f"Prueba anidada dentro de entrenamiento: {total} candidatos internos que siguen estacionarios",
                   "Fracción de candidatos")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


# ---------- contenido de los pasos ----------

def _metric(metrics, split, key):
    return (metrics or {}).get(split, {}).get(key)


def build_steps(artifacts, extras, mode):
    """Lista de pasos: dict con quién, tiempo, título, cifras (etiqueta, valor, tono), figura, fuente y notas."""
    selection = artifacts["seleccion"] or {}
    tickers = selection.get("tickers", [])
    theta, metrics, trades = artifacts["theta"] or {}, artifacts["metricas"], artifacts["operaciones"]
    filters, vecm, nested = extras["filtros"] or {}, extras["vecm"] or {}, extras["prueba_anidada"]
    registry, diag, search = extras["seleccion_grupos"], extras["diagnostico_particiones"], extras["theta_busqueda"]
    signals = artifacts["senales"]
    bounds = split_bounds(signals) if signals is not None else None
    technical = {p.key: p for p in figures.technical_panels(artifacts, SPLITS, mode)}
    executive = {p.key: p for p in figures.executive_panels(artifacts, SPLITS, mode)}
    steps = []

    prices = artifacts["precios"]
    n_days = len(prices) if prices is not None else None
    steps.append({
        "quien": "Gian", "tiempo": "1:20",
        "titulo": "Seis utilities de EE. UU.: ¿relación estructural o artefacto de probar muchos grupos?",
        "cifras": [("Grupo", " · ".join(tickers) or "—", "info"),
                   ("Grupos evaluados en total", f"{selection.get('grupos_evaluados_total', 0):,}", "mid"),
                   ("Días de datos (2014 a 2026)", f"{n_days:,}" if n_days else "—", "info"),
                   ("Partición fija", "60 / 20 / 20", "info")],
        "figura": prices_figure(prices, bounds, mode) if prices is not None else None, "fuente": "precios.csv",
        "notas": ["Universo: 10 utilities reguladas, con tarifas reguladas y la misma sensibilidad a tasas.",
                  "Log-precios ajustados y cortes con fechas fijas: entrenamiento hasta sep-2021, prueba hasta abr-2024.",
                  "Todo se decide con entrenamiento; prueba y validación se corren una sola vez."],
    })

    chosen = None
    if registry is not None and tickers:
        match = registry[registry["group"] == "+".join(tickers)]
        chosen = match.iloc[0] if len(match) else None
    stages = dict(funnel_counts(registry)) if registry is not None else {}
    steps.append({
        "quien": "Gian", "tiempo": "1:20",
        "titulo": f"De {stages.get('Grupos de 6', 210)} grupos de 6, {stages.get('Pasan Engle-Granger', '—')} pasan el embudo; elegimos el más estable",
        "cifras": [("Pasan Johansen (traza, 95%)", f"{stages.get('Pasan Johansen', '—')}", "info"),
                   ("Pasan Engle-Granger", f"{stages.get('Pasan Engle-Granger', '—')}", "info"),
                   ("Estabilidad del elegido", f"{chosen['stability']:.2f}" if chosen is not None else "—", "pos"),
                   ("Vida media en entrenamiento", f"{selection.get('vida_media', np.nan):.1f} días", "pos")],
        "figura": funnel_figure(registry, mode) if registry is not None else None, "fuente": "seleccion_grupos.csv",
        "notas": ["Candidato: Johansen por traza al 95%, spread con ADF p < 0.05 y vida media de 5 a 60 días.",
                  "Engle-Granger prioriza sin eliminar (S14 p16).",
                  "Gana la mayor estabilidad: fracción de subventanas de 2, 3 y 4 años que repiten la prueba."],
    })

    steps.append({
        "quien": "Gian", "tiempo": "0:40",
        "titulo": "El vector de Johansen se mueve incluso dentro de entrenamiento: por eso los pesos van con Kalman",
        "cifras": [("Pata normalizada", selection.get("pata_normalizada", "—"), "info"),
                   ("Eigenvalor en entrenamiento", f"{selection.get('eigenvalor', np.nan):.3f}", "info")],
        "figura": technical.get("eigvec").figure if "eigvec" in technical else None, "fuente": "eigvec_movil.csv",
        "notas": ["Johansen móvil: primer eigenvector en ventanas que avanzan por toda la muestra.",
                  "Si el vector cambia, un vector fijo deja de ser el portafolio que revierte.",
                  "Transición a Ian: el Filtro 1 estima esos pesos en el tiempo."],
    })

    f1, f2 = filters.get("filtro1", {}), filters.get("filtro2", {})
    steps.append({
        "quien": "Ian", "tiempo": "1:40",
        "titulo": "Filtro 1: pesos dinámicos con una Q chica para no absorber la reversión que queremos operar",
        "cifras": [("Q / R", f"{f1.get('q_ratio', np.nan):.0e}", "info"), ("φ (cuasi-diferencia)", f"{f1.get('phi', np.nan):.2f}", "info"),
                   ("Burn-in", f"{filters.get('burn_in', '—')} días", "info")],
        "figura": technical.get("pesos").figure if "pesos" in technical else None, "fuente": "pesos.csv",
        "notas": ["Estado (c, β₂…β₆) con AEP fija en 1; observación cuasi-diferenciada (con φ = 0 es la de la diapositiva).",
                  "El spread usa los pesos previos a la actualización: es causal.",
                  "Q y R se eligieron con walk-forward dentro de entrenamiento (filtros_malla.csv)."],
    })

    steps.append({
        "quien": "Ian", "tiempo": "1:40",
        "titulo": "Filtro 2: z-score causal con la media y la varianza previas a ver el dato del día",
        "cifras": [("λ de la varianza", f"{f2.get('lam', np.nan):.2f}", "info"),
                   ("Entrada / salida", f"{theta.get('z_in', np.nan):.1f} / {theta.get('z_out', np.nan):.1f}", "info"),
                   ("Stop / tenencia máxima", f"{theta.get('z_stop', np.nan):.1f} / {theta.get('h_max', '—')} días", "mid")],
        "figura": technical.get("z").figure if "z" in technical else None, "fuente": "senales.csv",
        "notas": ["Nivel local para μ y varianza recursiva; z_t = (S_t − μ previa) / σ previa.",
                  "Usar la media posterior juzgaría el dato con él mismo: la prueba 1 lo verifica.",
                  "Las ganancias de Kalman y las varianzas posteriores están en la vista técnica."],
    })

    steps.append({
        "quien": "JP", "tiempo": "0:50",
        "titulo": "El VECM confirma la corrección de error, y θ se congeló antes de ver prueba y validación",
        "cifras": [("β'α (debe estar entre −2 y 0)", f"{vecm.get('beta_alpha', np.nan):.3f}", "pos"),
                   ("Vida media del VECM", f"{vecm.get('vida_media', np.nan):.1f} días", "pos"),
                   ("Combinaciones de θ", f"{theta.get('combinaciones_evaluadas', '—')}", "info"),
                   ("Commit de θ", "4557c17", "info")],
        "figura": theta_figure(search, theta, mode) if search is not None else None, "fuente": "theta_busqueda.csv",
        "notas": ["Regla de θ: mayor Sharpe medio menos una desviación, con al menos 3 operaciones en cada pliegue.",
                  "Candado: tras un stop no se reabre hasta que |z| < z_out.",
                  "Posición: 80% del capital proporcional a |β|, con banda de no rebalanceo."],
    })

    def sharpe_card(split):
        value = _metric(metrics, split, "sharpe")
        ret = _metric(metrics, split, "retorno_total")
        text = "—" if value is None else f"{value:.2f}  ·  {pct(ret)}"
        return (f"Sharpe · retorno, {SPLIT_LABELS[split].lower()}", text, "pos" if (ret or 0) > 0 else "neg")

    steps.append({
        "quien": "JP", "tiempo": "0:50",
        "titulo": "Gana en entrenamiento y en prueba, pierde en validación",
        "cifras": [sharpe_card(s) for s in SPLITS] + [
            ("Comisiones / P&L bruto (entr.)", f"{_metric(metrics, 'train', 'pct_comisiones') or 0:.0%}", "mid")],
        "figura": executive.get("equity").figure if "equity" in executive else None, "fuente": "equity.csv",
        "notas": ["Backtest continuo con costos: 0.125% por pata y 0.25% anual de préstamo.",
                  "Cada operación pertenece a la partición donde se abrió.",
                  "El error estándar del Sharpe ronda 0.6: la diferencia entre tramos no es significativa por sí sola."],
    })

    stops = trades[(trades["split"] == "validation") & (trades["motivo"] == "stop")] if trades is not None else None
    n_val = int((trades["split"] == "validation").sum()) if trades is not None else 0
    steps.append({
        "quien": "JP", "tiempo": "0:40",
        "titulo": f"La pérdida viene de posiciones que dejaron de revertir: {len(stops) if stops is not None else '—'} de {n_val} cierres por stop en validación",
        "cifras": [("Stops en validación", f"{len(stops)} de {n_val}" if stops is not None else "—", "neg"),
                   ("P&L de esos stops", figures.money(stops["pnl_neto"].sum()) if stops is not None else "—", "neg"),
                   ("Dirección", "todas en corto" if stops is not None and (stops["direccion"] == -1).all() else "mixta", "mid")],
        "figura": exits_figure(trades, mode) if trades is not None else None, "fuente": "operaciones.csv",
        "notas": ["El enunciado lo anticipa (3.6): esta estrategia pierde cuando una posición deja de revertir.",
                  "El stop y la tenencia máxima limitaron la pérdida.",
                  "El spread subió de forma persistente, sobre todo por AEP."],
    })

    rows = diag.set_index("split").reindex(SPLITS) if diag is not None else None

    def trio(column, fmt):
        return " → ".join(fmt(v) for v in rows[column]) if rows is not None else "—"

    steps.append({
        "quien": "JP", "tiempo": "1:10",
        "titulo": "La relación existió entre 2014 y 2021, pero se deshace después",
        "cifras": [("Vida media (días)", trio("vida_media", lambda v: f"{v:.0f}"), "neg"),
                   ("Ventanas con relación", trio("ventanas_con_relacion", lambda v: f"{v:.0%}"), "neg"),
                   ("ADF del spread (p)", trio("adf_p", lambda v: f"{v:.2f}"), "neg")],
        "figura": diagnostics_figure(diag, mode) if diag is not None else None, "fuente": "diagnostico_particiones.csv",
        "notas": ["Johansen re-estimado dentro de prueba y de validación tampoco encuentra relación.",
                  "AEP explica casi todo el desplazamiento del spread y es la pata con la α más chica del VECM.",
                  "Hipótesis: el choque de tasas de 2022 y el spin-off de Constellation (EXC). El ADF en 2.5 años tiene poco poder."],
    })

    steps.append({
        "quien": "JP", "tiempo": "0:30",
        "titulo": "¿Estructural o artefacto? Ninguna de las dos en forma pura",
        "cifras": [("Candidatos internos", f"{nested['candidatos_train_interno']}" if nested else "—", "info"),
                   ("Siguen estacionarios en el tramo siguiente",
                    f"{nested['candidatos_estacionarios']['test']} de {nested['candidatos_train_interno']}" if nested else "—", "neg"),
                   ("Defensa propuesta", "prueba causal que apague", "pos")],
        "figura": nested_figure(nested, mode) if nested else None, "fuente": "prueba_anidada.json",
        "notas": ["Real pero dependiente del periodo: existió de 2014 a 2021.",
                  "Inflada por pruebas múltiples: pasar el embudo casi no predice que la relación siga.",
                  "Mejora: volver a probar la relación de forma causal y apagar la estrategia cuando se cae."],
    })
    for step in steps:
        _big(step["figura"])
    return steps


# ---------- vista de Streamlit ----------

def _go(delta, total):
    import streamlit as st
    st.session_state["paso"] = min(max((st.session_state.get("paso") or 1) + delta, 1), total)


def presentation_view(artifacts, extras, mode):
    import streamlit as st
    steps = build_steps(artifacts, extras, mode)
    total = len(steps)
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("paso", 1)
    nav_prev, nav_pick, nav_next = st.columns([1.4, 7.2, 1.4])
    nav_prev.button("◀ Anterior", on_click=_go, args=(-1, total), use_container_width=True, key="anterior")
    nav_next.button("Siguiente ▶", on_click=_go, args=(1, total), use_container_width=True, key="siguiente")
    nav_pick.segmented_control("Paso", list(range(1, total + 1)), key="paso", label_visibility="collapsed",
                               format_func=lambda i: f"{i}. {steps[i - 1]['quien']}")
    index = st.session_state.get("paso") or 1
    step = steps[index - 1]
    st.markdown(
        f"<span class='step-chip who'>Expone {step['quien']}</span><span class='step-chip'>Paso {index} de {total}</span>"
        f"<span class='step-chip'>≈ {step['tiempo']} min</span><div class='step-title'>{step['titulo']}</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(len(step["cifras"]))
    for position, (column, (label, value, tone)) in enumerate(zip(columns, step["cifras"])):
        column.markdown(f"<div class='kpi kpi-{tone}' style='--i:{position}'><div class='kpi-label'>{label}</div>"
                        f"<div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)
    st.write("")
    if step["figura"] is None:
        st.info(f"Falta `{step['fuente']}` en artifacts/. Corre `python main.py`.")
    else:
        st.plotly_chart(step["figura"], theme=None, key=f"paso_{index}", config={"scrollZoom": False})
    with st.expander("Notas del expositor"):
        for note in step["notas"]:
            st.markdown(f"- {note}")
    st.progress(index / total)
