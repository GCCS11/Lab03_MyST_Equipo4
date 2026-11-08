"""Walk-forward dentro de entrenamiento (G6): busca umbrales sobre K pliegues contiguos del
tramo de entrenamiento, sin tocar nunca prueba ni validacion.

NO elige el theta ganador ni escribe artifacts/theta.json: eso lo decide y congela Juan Pablo
con su propio criterio. Este modulo solo deja la tabla por theta y pliegue, y una funcion que
la resume y ordena.

Cada theta se evalua K veces, una por pliegue, de forma independiente: strategy.signals.
spread_direction se corre sobre el z de cada pliegue por separado (cada pliegue arranca plano,
sin arrastrar una posicion abierta del pliegue anterior), y backtest.engine.run_backtest y
performance.metrics.compute_metrics se calculan sobre ese mismo pliegue. El z-score en si viene
de una sola corrida causal de main.run_pipeline sobre todo el entrenamiento (test 1 ya prueba
que no cambia si se le agrega informacion futura), asi que el filtro de Kalman ve toda la
historia que le corresponde aunque el pliegue sea mas corto.

El primer pliegue incluye el burn-in de los filtros (z vale NaN ahi, ver kalman/signal_filter.
py). Esos dias se recortan del pliegue 0 antes de calcular sus metricas, para no contarlos como
dias de mercado plano: no hubo decision posible, no hubo mercado "plano".

Correr con: python -m backtest.walk_forward
"""
import itertools
import json

import numpy as np
import pandas as pd

from backtest.engine import BacktestParams, run_backtest
from config import ARTIFACTS_DIR
from data.loader import split_positions
from main import load_frozen, load_log_prices, run_pipeline
from performance.metrics import compute_metrics
from strategy.signals import max_holding, spread_direction

RULE_KEYS = ("z_in", "z_out", "z_stop", "h_max")

# Malla moderada: 3 x 2 x 3 x 3 x 3 = 162 combinaciones, todas validas con estos rangos
# (z_out <= 0.5 < 1.0 <= z_in <= 2.0 < 2.5 <= z_stop). Cada combinacion es una prueba mas que
# hay que declarar en el reporte: por eso la malla se mantiene chica a proposito.
DEFAULT_GRID = {
    "z_in": (1.0, 1.5, 2.0),
    "z_out": (0.25, 0.5),
    "z_stop": (2.5, 3.0, 3.5),
    "h_max_multiplo": (1.0, 2.0, 3.0),
    "band": (0.0, 0.0075, 0.015),
}


def build_theta_grid(half_life, grid=None):
    """Combinaciones (z_in, z_out, z_stop, h_max, band) validas de la malla.

    h_max sale de multiplicar `half_life` por cada h_max_multiplo (strategy.signals.
    max_holding). Se descartan las combinaciones que no cumplen 0 <= z_out < z_in < z_stop; si
    ninguna sobrevive, la malla esta mal definida y se falla con un error claro en vez de
    correr una busqueda vacia.
    """
    grid = grid or DEFAULT_GRID
    combos = []
    for z_in, z_out, z_stop, h_mult, band in itertools.product(
        grid["z_in"], grid["z_out"], grid["z_stop"], grid["h_max_multiplo"], grid["band"]
    ):
        if 0 <= z_out < z_in < z_stop:
            combos.append({
                "z_in": z_in, "z_out": z_out, "z_stop": z_stop,
                "h_max": max_holding(half_life, h_mult), "band": band,
            })
    if not combos:
        raise ValueError(
            "la malla no tiene ninguna combinacion valida: revisa 0 <= z_out < z_in < z_stop"
        )
    return combos


def make_folds(index, n_folds=5):
    """Posiciones [inicio, fin) de n_folds pliegues contiguos, sin traslape, que cubren todo
    `index`."""
    if n_folds < 1:
        raise ValueError("n_folds debe ser al menos 1")
    edges = np.linspace(0, len(index), n_folds + 1, dtype=int)
    return list(zip(edges[:-1], edges[1:]))


def assert_train_only(index, train_end):
    """Falla con un error claro si `index` tiene alguna fecha posterior a `train_end`."""
    train_end = pd.Timestamp(train_end)
    beyond = index[index > train_end]
    if len(beyond):
        raise ValueError(
            f"el walk-forward solo puede usar entrenamiento (hasta {train_end.date()}); "
            f"se recibieron {len(beyond)} fechas posteriores a eso, la primera es "
            f"{pd.Timestamp(beyond[0]).date()}"
        )


def trim_burn_in(folds, z):
    """Avanza el inicio del primer pliegue hasta el primer z valido.

    Los dias de burn-in de los filtros (z en NaN, por diseno de kalman.signal_filter) no son
    dias de mercado plano: no hubo senal con la que decidir nada. Se excluyen del pliegue 0 en
    vez de contarlos como observaciones sin movimiento. Regresa los pliegues ajustados y
    cuantos dias se recortaron.
    """
    first_valid = z.first_valid_index()
    first_valid_pos = 0 if first_valid is None else z.index.get_loc(first_valid)
    start0, end0 = folds[0]
    burn_in_days = max(0, first_valid_pos - start0)
    adjusted = [(max(start0, first_valid_pos), end0), *folds[1:]]
    return adjusted, burn_in_days


def evaluate_theta(theta, z, betas, prices, folds):
    """Una fila por pliegue con las metricas de ese pliegue para este theta.

    Cada pliegue corre spread_direction y run_backtest de forma independiente, sobre su propio
    tramo de z, betas y precios: arranca plano, sin la posicion que pudiera seguir abierta al
    final del pliegue anterior.
    """
    rule_params = {key: theta[key] for key in RULE_KEYS}
    rows = []
    for fold_id, (start, end) in enumerate(folds):
        z_fold = z.iloc[start:end]
        rule_fold = spread_direction(z_fold, rule_params)
        betas_fold = betas.iloc[start:end]
        prices_fold = prices.iloc[start:end]
        result = run_backtest(rule_fold, betas_fold, prices_fold, BacktestParams(band=theta["band"]))
        metrics = compute_metrics(result.diario, result.operaciones)
        rows.append({
            "pliegue": fold_id,
            "sharpe": metrics.sharpe,
            "retorno_total": metrics.retorno_total,
            "max_drawdown": metrics.max_drawdown,
            "n_operaciones": metrics.n_operaciones,
            "comisiones": metrics.comisiones,
            "prestamo": metrics.prestamo,
        })
    return rows


def run_walk_forward(n_folds=5, grid=None, directory=ARTIFACTS_DIR):
    """Corre la malla sobre n_folds pliegues de entrenamiento y escribe
    artifacts/theta_busqueda.csv (una fila por theta y pliegue) y
    artifacts/theta_busqueda_meta.json (cuantas combinaciones se evaluaron). No elige ganador ni
    escribe theta.json."""
    selection, filters, _ = load_frozen()
    log_prices = load_log_prices(selection)
    train_end_row, _ = split_positions(len(log_prices))
    train_end = log_prices.index[train_end_row - 1]

    result = run_pipeline(log_prices, selection, filters, theta=None, until=train_end)
    z = result["signal"]["z"]
    betas = result["weights"].filter(like="beta_")
    prices = np.exp(log_prices[selection["tickers"]]).reindex(z.index)
    assert_train_only(z.index, train_end)

    folds = make_folds(z.index, n_folds)
    folds, burn_in_days = trim_burn_in(folds, z)
    combos = build_theta_grid(selection["vida_media"], grid)

    rows = []
    for theta_id, theta in enumerate(combos):
        for row in evaluate_theta(theta, z, betas, prices, folds):
            rows.append({"theta_id": theta_id, **theta, **row})
    table = pd.DataFrame(rows)

    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / "theta_busqueda.csv", index=False)
    meta = {
        "combinaciones_evaluadas": len(combos),
        "pliegues": n_folds,
        "train_end": str(pd.Timestamp(train_end).date()),
        "dias_burn_in_pliegue_0": int(burn_in_days),
    }
    (directory / "theta_busqueda_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return table, meta


def rank_by_sharpe_minus_std(table, min_trades=3):
    """Resumen por theta: media y desviacion estandar del Sharpe entre pliegues, y cuantos
    pliegues tuvieron al menos `min_trades` operaciones. Ordena de mayor a menor por media
    menos una desviacion estandar del Sharpe. Solo ordena: no elige ni congela ningun theta,
    eso lo decide quien use esta tabla con theta.json.
    """
    grouped = table.groupby("theta_id")
    summary = grouped[["z_in", "z_out", "z_stop", "h_max", "band"]].first()
    summary["sharpe_media"] = grouped["sharpe"].mean()
    summary["sharpe_std"] = grouped["sharpe"].std(ddof=0).fillna(0.0)
    summary["pliegues_con_operaciones"] = grouped["n_operaciones"].apply(
        lambda s: int((s >= min_trades).sum())
    )
    summary["sharpe_media_menos_std"] = summary["sharpe_media"] - summary["sharpe_std"]
    return summary.sort_values("sharpe_media_menos_std", ascending=False).reset_index()


if __name__ == "__main__":
    tabla, meta = run_walk_forward()
    print(f"Combinaciones de theta evaluadas: {meta['combinaciones_evaluadas']}")
    print(f"Pliegues: {meta['pliegues']}")
    print(f"Dias de burn-in excluidos del pliegue 0: {meta['dias_burn_in_pliegue_0']}")
    resumen = rank_by_sharpe_minus_std(tabla)
    print(resumen.head(10).to_string(index=False))
