"""Contrato de artifacts/seleccion.json y regla de seleccion entre grupos."""
import json

import numpy as np
import pandas as pd

from cointegration.seleccion import select_group
from config import ARTIFACTS_DIR


def test_selection_file_follows_the_contract():
    selection = json.loads((ARTIFACTS_DIR / "seleccion.json").read_text())
    assert len(selection["tickers"]) == 6
    assert len(selection["beta_johansen"]) == 6
    position = selection["tickers"].index(selection["pata_normalizada"])
    assert selection["beta_johansen"][position] == 1.0
    assert np.abs(selection["beta_johansen"]).max() == 1.0
    assert selection["train_end"] < selection["test_end"]
    assert 5 <= selection["vida_media"] <= 60


def test_rule_prefers_engle_granger_then_stability_then_eigenvalue():
    registry = pd.DataFrame(
        {
            "group": ["sin_eg", "menos_estable", "gana", "empata_menor", "descartado"],
            "status": ["candidato"] * 4 + ["descartado_johansen_o_spread"],
            "eg_min_p": [0.20, 0.01, 0.04, 0.03, 0.01],
            "stability": [0.90, 0.40, 0.70, 0.70, 1.00],
            "top_eigenvalue": [0.09, 0.09, 0.03, 0.02, 0.09],
        }
    )
    assert select_group(registry)["group"] == "gana"
