"""Freezing of theta (J6): the declared rule picks the expected row and theta.json is written
only once."""
import json

import pandas as pd
import pytest

from strategy.theta import choose_theta, freeze_theta


def walk_forward_table():
    """Three thetas over three folds. Theta 0 has the best mean Sharpe but one fold with a
    single trade; theta 1 is steady; theta 2 has a higher mean than theta 1 but much more
    dispersion."""
    rows = []
    sharpes = {0: (3.0, 2.5, 2.8), 1: (0.6, 0.5, 0.55), 2: (1.6, -0.4, 0.6)}
    trades = {0: (5, 1, 6), 1: (4, 4, 5), 2: (6, 5, 7)}
    for theta_id in sharpes:
        for fold in range(3):
            rows.append({
                "theta_id": theta_id, "z_in": 1.5 + 0.5 * theta_id, "z_out": 0.5, "z_stop": 3.0,
                "h_max": 18, "band": 0.0075, "pliegue": fold, "sharpe": sharpes[theta_id][fold],
                "retorno_total": 0.0, "max_drawdown": 0.0, "n_operaciones": trades[theta_id][fold],
                "comisiones": 0.0, "prestamo": 0.0,
            })
    return pd.DataFrame(rows)


def test_choose_theta_needs_trades_in_every_fold_and_penalizes_dispersion():
    theta = choose_theta(walk_forward_table(), min_trades=3)
    assert theta["theta_id"] == 1
    assert theta["elegibles"] == 2
    assert isinstance(theta["h_max"], int)
    assert set(theta) >= {"z_in", "z_out", "z_stop", "h_max", "band"}


def test_choose_theta_fails_when_no_theta_trades_enough():
    with pytest.raises(ValueError):
        choose_theta(walk_forward_table(), min_trades=10)


def test_freeze_theta_writes_once(tmp_path):
    walk_forward_table().to_csv(tmp_path / "theta_busqueda.csv", index=False)
    meta = {"combinaciones_evaluadas": 3, "pliegues": 3, "train_end": "2021-09-08"}
    (tmp_path / "theta_busqueda_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    frozen = freeze_theta(tmp_path)
    assert json.loads((tmp_path / "theta.json").read_text(encoding="utf-8")) == frozen
    assert frozen["combinaciones_evaluadas"] == 3
    with pytest.raises(FileExistsError):
        freeze_theta(tmp_path)
