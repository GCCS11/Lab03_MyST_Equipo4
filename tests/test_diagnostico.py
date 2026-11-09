"""Diagnostico por particion: con datos sinteticos cointegrados en los tres tramos, el spread con
el vector verdadero es estacionario en cada uno y Johansen encuentra la relacion."""
import numpy as np
import pandas as pd

from cointegration.diagnostico import SPLITS, partition_diagnostics, split_labels


def cointegrated_prices(n=1500, seed=42):
    """Tres log-precios: dos paseos aleatorios y una tercera serie que revierte a su combinacion."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 0.005)
    y = 0.6 * x[:, 0] + 0.4 * x[:, 1] + noise
    index = pd.bdate_range("2015-01-01", periods=n, name="date")
    return pd.DataFrame({"Y": y, "A": x[:, 0], "B": x[:, 1]}, index=index)


def test_split_labels_follow_the_60_20_20_cut():
    labels = split_labels(pd.bdate_range("2020-01-01", periods=100))
    assert labels.value_counts().to_dict() == {"train": 60, "test": 20, "validation": 20}
    assert labels.iloc[59] == "train" and labels.iloc[60] == "test" and labels.iloc[80] == "validation"


def test_partition_diagnostics_finds_the_relation_in_every_split():
    table = partition_diagnostics(cointegrated_prices(), [1.0, -0.6, -0.4])
    assert table["split"].tolist() == list(SPLITS)
    assert (table["adf_p"] < 0.05).all()
    assert (table["johansen_rango_en_tramo"] >= 1).all()
    assert table["vida_media"].between(1, 10).all()
