"""Unit tests for backtest/walk_forward.py: pliegues sin traslape que cubren todo el
entrenamiento, la malla invalida se rechaza, el candado de fechas de entrenamiento, el recorte
del burn-in del pliegue 0 y el resumen que ordena por media menos una desviacion del Sharpe.

Datos sinteticos unicamente: nada de precios ni particiones reales de test o validacion.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import (
    assert_train_only,
    build_theta_grid,
    evaluate_theta,
    make_folds,
    rank_by_sharpe_minus_std,
    trim_burn_in,
)
from strategy.signals import max_holding

TICKERS = ["A", "B", "C"]
BETA = np.array([1.0, -0.5, -0.5])
BASE_PRICE = np.array([10.0, 20.0, 40.0])


def test_folds_do_not_overlap_and_cover_the_whole_index():
    index = pd.bdate_range("2021-01-04", periods=23)
    folds = make_folds(index, n_folds=5)

    assert len(folds) == 5
    covered = [position for start, end in folds for position in range(start, end)]
    assert covered == list(range(len(index)))


def test_invalid_grid_is_rejected():
    grid = {
        "z_in": (1.0,), "z_out": (2.0,), "z_stop": (3.0,),
        "h_max_multiplo": (1.0,), "band": (0.0,),
    }
    with pytest.raises(ValueError):
        build_theta_grid(half_life=10.0, grid=grid)


def test_valid_grid_keeps_only_combinations_with_z_out_below_z_in_below_z_stop():
    grid = {
        "z_in": (1.0, 2.0), "z_out": (0.5, 1.5), "z_stop": (3.0,),
        "h_max_multiplo": (1.0, 2.0), "band": (0.0,),
    }
    combos = build_theta_grid(half_life=10.0, grid=grid)

    assert len(combos) == 6
    assert all(0 <= c["z_out"] < c["z_in"] < c["z_stop"] for c in combos)
    h_max_values = {c["h_max"] for c in combos}
    assert h_max_values == {max_holding(10.0, 1.0), max_holding(10.0, 2.0)}


def test_train_only_lock_accepts_dates_up_to_train_end():
    index = pd.bdate_range("2021-01-04", periods=5)
    train_end = index[2]
    assert_train_only(index[:3], train_end)  # no debe lanzar nada


def test_train_only_lock_rejects_dates_after_train_end():
    index = pd.bdate_range("2021-01-04", periods=5)
    train_end = index[2]
    with pytest.raises(ValueError):
        assert_train_only(index, train_end)


def test_trim_burn_in_advances_only_the_first_fold():
    index = pd.bdate_range("2021-01-04", periods=8)
    z = pd.Series([np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    folds = make_folds(index, n_folds=2)

    adjusted, burn_in_days = trim_burn_in(folds, z)

    assert burn_in_days == 3
    assert adjusted[0] == (3, 4)
    assert adjusted[1] == folds[1]


def test_trim_burn_in_is_a_no_op_when_z_has_no_leading_nan():
    index = pd.bdate_range("2021-01-04", periods=8)
    z = pd.Series(np.arange(8.0), index=index)
    folds = make_folds(index, n_folds=2)

    adjusted, burn_in_days = trim_burn_in(folds, z)

    assert burn_in_days == 0
    assert adjusted == folds


def test_evaluate_theta_returns_one_row_per_fold():
    index = pd.bdate_range("2021-01-04", periods=12)
    prices = pd.DataFrame(
        [BASE_PRICE + np.array([0.05, -0.03, 0.02]) * t for t in range(12)],
        index=index, columns=TICKERS,
    )
    z = pd.Series(
        [0.0, -2.5, -2.0, -1.5, -0.3, 0.0, 0.0, 2.6, 2.2, 0.3, 0.0, 0.0], index=index,
    )
    betas = pd.DataFrame(np.tile(BETA, (12, 1)), index=index, columns=[f"beta_{t}" for t in TICKERS])
    theta = {"z_in": 2.0, "z_out": 0.5, "z_stop": 3.0, "h_max": 10, "band": 0.0}
    folds = make_folds(index, n_folds=2)

    rows = evaluate_theta(theta, z, betas, prices, folds)

    assert len(rows) == 2
    assert [row["pliegue"] for row in rows] == [0, 1]
    for row in rows:
        assert set(row) == {
            "pliegue", "sharpe", "retorno_total", "max_drawdown",
            "n_operaciones", "comisiones", "prestamo",
        }
        assert isinstance(row["n_operaciones"], (int, np.integer))


def test_rank_by_sharpe_minus_std_orders_by_mean_minus_std():
    table = pd.DataFrame([
        {"theta_id": 0, "z_in": 1.0, "z_out": 0.5, "z_stop": 3.0, "h_max": 10, "band": 0.0,
         "pliegue": 0, "sharpe": 1.0, "n_operaciones": 5},
        {"theta_id": 0, "z_in": 1.0, "z_out": 0.5, "z_stop": 3.0, "h_max": 10, "band": 0.0,
         "pliegue": 1, "sharpe": 2.0, "n_operaciones": 1},
        {"theta_id": 0, "z_in": 1.0, "z_out": 0.5, "z_stop": 3.0, "h_max": 10, "band": 0.0,
         "pliegue": 2, "sharpe": 3.0, "n_operaciones": 4},
        {"theta_id": 1, "z_in": 1.5, "z_out": 0.25, "z_stop": 3.5, "h_max": 8, "band": 0.01,
         "pliegue": 0, "sharpe": 0.5, "n_operaciones": 3},
        {"theta_id": 1, "z_in": 1.5, "z_out": 0.25, "z_stop": 3.5, "h_max": 8, "band": 0.01,
         "pliegue": 1, "sharpe": 0.5, "n_operaciones": 3},
        {"theta_id": 1, "z_in": 1.5, "z_out": 0.25, "z_stop": 3.5, "h_max": 8, "band": 0.01,
         "pliegue": 2, "sharpe": 0.5, "n_operaciones": 3},
    ])

    resumen = rank_by_sharpe_minus_std(table, min_trades=3)

    assert resumen["theta_id"].tolist() == [0, 1]
    assert resumen["sharpe_media"].tolist() == pytest.approx([2.0, 0.5])
    assert resumen["sharpe_std"].tolist() == pytest.approx([np.sqrt(2.0 / 3.0), 0.0])
    assert resumen["pliegues_con_operaciones"].tolist() == [2, 3]
    assert resumen["sharpe_media_menos_std"].iloc[0] > resumen["sharpe_media_menos_std"].iloc[1]
