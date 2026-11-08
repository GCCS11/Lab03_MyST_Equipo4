"""Frozen trading thresholds (J6): picks theta from the training walk-forward and writes
artifacts/theta.json, which is committed before test and validation are ever run.

Rule, fixed before looking at test or validation: among the thetas with at least MIN_TRADES
closed trades in every fold, the one with the highest mean Sharpe minus one standard deviation
across folds (backtest.walk_forward.rank_by_sharpe_minus_std). Penalizing the dispersion favors
a theta that works in every fold over one that shines in a single fold.

Run once with: python -m strategy.theta
"""
import json
from pathlib import Path

import pandas as pd

from backtest.walk_forward import rank_by_sharpe_minus_std
from config import ARTIFACTS_DIR

MIN_TRADES = 3
CRITERION = (
    "Mayor Sharpe medio menos una desviación estándar entre pliegues del walk-forward en "
    f"entrenamiento, entre los theta con al menos {MIN_TRADES} operaciones en cada pliegue"
)


def choose_theta(table, min_trades=MIN_TRADES):
    """Winning row of the walk-forward table as a dict ready for theta.json."""
    ranking = rank_by_sharpe_minus_std(table, min_trades)
    n_folds = table["pliegue"].nunique()
    eligible = ranking[ranking["pliegues_con_operaciones"] == n_folds]
    if eligible.empty:
        raise ValueError(f"ningún theta tiene al menos {min_trades} operaciones en los {n_folds} pliegues")
    best = eligible.iloc[0]
    return {
        "z_in": float(best["z_in"]),
        "z_out": float(best["z_out"]),
        "z_stop": float(best["z_stop"]),
        "h_max": int(best["h_max"]),
        "band": float(best["band"]),
        "theta_id": int(best["theta_id"]),
        "sharpe_media": float(best["sharpe_media"]),
        "sharpe_std": float(best["sharpe_std"]),
        "criterio": CRITERION,
        "elegibles": int(len(eligible)),
    }


def freeze_theta(directory=ARTIFACTS_DIR, overwrite=False):
    """Choose theta from theta_busqueda.csv and write theta.json next to it.

    Refuses to overwrite an existing theta.json: once test and validation have been run, a new
    theta would be chosen with information from outside training.
    """
    directory = Path(directory)
    target = directory / "theta.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} ya existe: el theta se congela una sola vez")
    meta = json.loads((directory / "theta_busqueda_meta.json").read_text(encoding="utf-8"))
    theta = choose_theta(pd.read_csv(directory / "theta_busqueda.csv"))
    theta.update({
        "combinaciones_evaluadas": meta["combinaciones_evaluadas"],
        "pliegues": meta["pliegues"],
        "train_end": meta["train_end"],
    })
    target.write_text(json.dumps(theta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return theta


if __name__ == "__main__":
    frozen = freeze_theta()
    print(json.dumps(frozen, indent=2, ensure_ascii=False))
