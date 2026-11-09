"""Diagnostico de la relacion por particion y prueba anidada 60/20/20 dentro de entrenamiento.

Responde "estructural o artefacto" con cifras propias. Es solo diagnostico: ninguna decision
del pipeline sale de aqui, y la parte por particion lee prueba y validacion, asi que se niega a
correr hasta que artifacts/theta.json este congelado.

1. Por particion, con el vector de Johansen de entrenamiento fijo (artifacts/seleccion.json):
   ADF del spread, vida media, volatilidad del spread, rango de Johansen re-estimado dentro del
   tramo y fraccion de ventanas moviles de 2 anios (que terminan en el tramo) con relacion.
2. Prueba anidada (solo entrenamiento): se parte entrenamiento otra vez en 60/20/20, se repite la
   seleccion completa en el train interno y se mide si los candidatos siguen estacionarios con
   su vector fijo en el test y la validacion internos.

Correr con: python -m cointegration.diagnostico
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from cointegration.johansen import johansen_test
from cointegration.seleccion import GROUP_SIZE, PRICES_FILE, screen_group, select_group
from cointegration.spread import adf_pvalue, build_spread, half_life
from config import ALPHA, ARTIFACTS_DIR
from data.loader import split_positions

SPLITS = ("train", "test", "validation")
ROLLING_WINDOW = 504
ROLLING_STEP = 21


def split_labels(index):
    """Etiqueta de particion de cada fecha, con los mismos cortes que el resto del pipeline."""
    train_end, test_end = split_positions(len(index))
    labels = np.array(["validation"] * len(index), dtype=object)
    labels[:train_end] = "train"
    labels[train_end:test_end] = "test"
    return pd.Series(labels, index=index, name="split")


def spread_stats(log_prices, vector):
    """ADF, vida media y volatilidad del spread con un vector fijo."""
    spread = build_spread(log_prices, vector)
    return {"adf_p": float(adf_pvalue(spread)), "vida_media": half_life(spread), "volatilidad": float(spread.std())}


def has_relation(log_prices):
    return johansen_test(log_prices)[1]["johansen_rank_trace"] >= 1


def rolling_relation(log_prices, labels, window=ROLLING_WINDOW, step=ROLLING_STEP):
    """Fraccion de ventanas moviles con relacion de Johansen, agrupadas por la fecha en que terminan."""
    ends = range(window, len(log_prices) + 1, step)
    rows = [{"split": labels.iloc[end - 1], "relacion": has_relation(log_prices.iloc[end - window : end])} for end in ends]
    return pd.DataFrame(rows).groupby("split")["relacion"].mean()


def partition_diagnostics(log_prices, vector):
    """Una fila por particion con el vector de entrenamiento fijo."""
    labels = split_labels(log_prices.index)
    rolling = rolling_relation(log_prices, labels)
    rows = []
    for split in SPLITS:
        segment = log_prices[labels == split]
        rows.append({
            "split": split,
            "inicio": str(segment.index[0].date()),
            "fin": str(segment.index[-1].date()),
            **spread_stats(segment, vector),
            "johansen_rango_en_tramo": int(johansen_test(segment)[1]["johansen_rank_trace"]),
            "ventanas_con_relacion": float(rolling.get(split, np.nan)),
        })
    return pd.DataFrame(rows)


def nested_test(train, chosen_tickers):
    """Seleccion repetida en el train interno y su desempeno en el test y la validacion internos."""
    labels = split_labels(train.index)
    inner = {split: train[labels == split] for split in SPLITS}
    registry = pd.DataFrame([screen_group(inner["train"], tickers) for tickers in combinations(train.columns, GROUP_SIZE)])
    candidates = registry[registry["status"] == "candidato"]

    def out_of_sample(row):
        vector = [float(weight) for weight in row["vector"].split()]
        tickers = row["group"].split("+")
        return {split: spread_stats(inner[split][tickers], vector) for split in ("test", "validation")}

    stats = {row["group"]: out_of_sample(row) for _, row in candidates.iterrows()}
    winner = select_group(registry)
    chosen = "+".join(chosen_tickers)
    still_stationary = {split: sum(s[split]["adf_p"] < ALPHA for s in stats.values()) for split in ("test", "validation")}
    return {
        "cortes_internos": {split: [str(inner[split].index[0].date()), str(inner[split].index[-1].date())] for split in SPLITS},
        "grupos": len(registry),
        "candidatos_train_interno": len(candidates),
        "ganador_interno": winner["group"],
        "ganador_interno_estabilidad": float(winner["stability"]),
        "ganador_interno_fuera_de_muestra": stats[winner["group"]],
        "candidatos_estacionarios": still_stationary,
        "grupo_elegido_es_candidato_interno": chosen in stats,
        "grupo_elegido_fuera_de_muestra": stats.get(chosen),
    }


def main():
    if not (ARTIFACTS_DIR / "theta.json").exists():
        raise SystemExit("El diagnostico lee prueba y validacion: primero hay que congelar artifacts/theta.json.")
    selection = json.loads((ARTIFACTS_DIR / "seleccion.json").read_text(encoding="utf-8"))
    all_prices = np.log(pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True))
    log_prices = all_prices[selection["tickers"]]

    table = partition_diagnostics(log_prices, selection["beta_johansen"])
    table.to_csv(ARTIFACTS_DIR / "diagnostico_particiones.csv", index=False)
    print(table.to_string(index=False))

    train = all_prices[split_labels(all_prices.index) == "train"]
    nested = nested_test(train, selection["tickers"])
    (ARTIFACTS_DIR / "prueba_anidada.json").write_text(json.dumps(nested, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(nested, indent=2))


if __name__ == "__main__":
    main()
