"""Seleccion del grupo de 6 utilities con el embudo de S14 p16, solo con entrenamiento.

Regla, fijada antes de correr:
1. Candidatos: pasan Johansen por traza al 95% y tienen un vector con ADF p < 0.05 y vida
   media de 5 a 60 dias.
2. Engle-Granger prioriza (S14 p16) sin eliminar: primero los grupos con p < 0.05, que es lo
   que pide el enunciado.
3. Gana la mayor estabilidad: fraccion de subventanas de entrenamiento que repiten el paso 1,
   promediada entre ventanas de 2, 3 y 4 anios. El eigenvalor solo desempata.

Correr con: python -m cointegration.seleccion
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from cointegration.engle_granger import engle_granger_group
from cointegration.johansen import candidate_vectors, johansen_test
from config import ALPHA, ARTIFACTS_DIR, DATA_DIR, TEST_FRACTION, TRAIN_FRACTION

PRICES_FILE = DATA_DIR / "raw" / "prices_utilities.csv"
GROUP_SIZE = 6
HALF_LIFE_RANGE = (5, 60)
WINDOWS = (504, 756, 1008)
WINDOW_STEP = 126
PRIOR_GROUPS_EVALUATED = 1366


def best_vector(log_prices):
    """Etapas 4 a 6: regresa el resumen de Johansen y el mejor vector admisible, o None."""
    result, summary = johansen_test(log_prices)
    rank = summary["johansen_rank_trace"]
    if rank < 1:
        return summary, None
    candidates = candidate_vectors(log_prices, result, rank)
    low, high = HALF_LIFE_RANGE
    admissible = candidates[(candidates["adf_p"] < ALPHA) & candidates["half_life"].between(low, high)]
    return summary, None if admissible.empty else admissible.sort_values("adf_p").iloc[0]


def stability(log_prices, window):
    """Fraccion de subventanas donde el grupo vuelve a pasar las etapas 4 a 6."""
    starts = range(0, len(log_prices) - window + 1, WINDOW_STEP)
    return float(np.mean([best_vector(log_prices.iloc[s : s + window])[1] is not None for s in starts]))


def screen_group(train, tickers):
    """Registra un grupo con todas sus etapas; las subventanas solo se calculan si es candidato."""
    data = train[list(tickers)]
    pairs = np.triu_indices(len(tickers), 1)
    row = {"group": "+".join(tickers), "mean_return_corr": float(data.diff().corr().to_numpy()[pairs].mean())}
    summary, chosen = best_vector(data)
    row.update(summary)
    if chosen is None:
        return {**row, "status": "descartado_johansen_o_spread"}
    row.update(
        status="candidato",
        column=int(chosen["column"]),
        adf_spread_p=float(chosen["adf_p"]),
        half_life=float(chosen["half_life"]),
        normalizing_leg=chosen["leg"],
        vector=" ".join(f"{weight:.6f}" for weight in chosen["vector"]),
        eg_min_p=engle_granger_group(data)["eg_min_p"],
    )
    for window in WINDOWS:
        row[f"stability_{window}"] = stability(data, window)
    row["stability"] = float(np.mean([row[f"stability_{window}"] for window in WINDOWS]))
    return row


def select_group(registry):
    """Aplica la regla del encabezado y regresa la fila ganadora."""
    candidates = registry[registry["status"] == "candidato"].copy()
    candidates["eg_passes"] = candidates["eg_min_p"] < ALPHA
    return candidates.sort_values(["eg_passes", "stability", "top_eigenvalue"], ascending=False).iloc[0]


def run_selection():
    """Corre el embudo y escribe el registro y artifacts/seleccion.json."""
    log_prices = np.log(pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True))
    n_rows = len(log_prices)
    train_rows = int(n_rows * TRAIN_FRACTION)
    test_rows = int(n_rows * (TRAIN_FRACTION + TEST_FRACTION))
    train = log_prices.iloc[:train_rows]

    registry = pd.DataFrame(
        [screen_group(train, tickers) for tickers in combinations(log_prices.columns, GROUP_SIZE)]
    )
    winner = select_group(registry)
    tickers = winner["group"].split("+")
    selection = {
        "tickers": tickers,
        "beta_johansen": [float(weight) for weight in winner["vector"].split()],
        "pata_normalizada": winner["normalizing_leg"],
        "eigenvalor": float(winner["top_eigenvalue"]),
        "vida_media": float(winner["half_life"]),
        "train_end": str(log_prices.index[train_rows - 1].date()),
        "test_end": str(log_prices.index[test_rows - 1].date()),
        "grupos_evaluados_total": PRIOR_GROUPS_EVALUATED + len(registry),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_csv(ARTIFACTS_DIR / "seleccion_grupos.csv", index=False)
    (ARTIFACTS_DIR / "seleccion.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))
    print(registry["status"].value_counts().to_string())
    return selection


if __name__ == "__main__":
    run_selection()
