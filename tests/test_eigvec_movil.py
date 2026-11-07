"""Unit tests for cointegration/eigvec_movil.py: causalidad, consistencia de signo y escala,
ausencia de NaN, y la bandera ref_inestable cuando la pata de referencia colapsa.

Los datos son sinteticos y cointegrados por construccion (una tendencia comun I(1) mas ruido
idiosincratico estacionario). No se usa ningun precio ni particion real de test o validacion.
"""
import json

import numpy as np
import pandas as pd
import pytest

from cointegration import eigvec_movil
from cointegration.eigvec_movil import load_group_log_prices, main, rolling_eigenvector
from data.loader import split_positions

SEED = 42


def make_cointegrated_log_prices(n=150, seed=SEED):
    """Dos series I(1) que comparten la misma tendencia comun, con ruido idiosincratico chico:
    su diferencia es estacionaria, asi que el par esta cointegrado por construccion."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0, 1.0, n))
    noise_a = rng.normal(0.0, 0.05, n)
    noise_b = rng.normal(0.0, 0.05, n)
    index = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"A": common + noise_a, "B": common + noise_b}, index=index)


def test_causality_truncating_gives_the_same_rows_up_to_t():
    log_prices = make_cointegrated_log_prices()
    full = rolling_eigenvector(log_prices, window=50, step=10)
    partial = rolling_eigenvector(log_prices.iloc[:70], window=50, step=10)

    pd.testing.assert_frame_equal(full.iloc[: len(partial)], partial)


def test_scale_is_fixed_at_one_on_the_reference_leg():
    log_prices = make_cointegrated_log_prices()
    table = rolling_eigenvector(log_prices, window=50, step=10)

    ref = next(t for t in ("A", "B") if np.isclose(table[t].iloc[0], 1.0))
    assert (table[ref] == 1.0).all()


def test_sign_is_consistent_across_windows():
    log_prices = make_cointegrated_log_prices()
    table = rolling_eigenvector(log_prices, window=50, step=10)

    ref = next(t for t in ("A", "B") if np.isclose(table[t].iloc[0], 1.0))
    other = "B" if ref == "A" else "A"
    stable = table.loc[~table["ref_inestable"], other]

    signo_inicial = np.sign(stable.iloc[0])
    assert signo_inicial != 0
    assert (np.sign(stable) == signo_inicial).all()


def test_csv_has_no_nan():
    log_prices = make_cointegrated_log_prices()
    table = rolling_eigenvector(log_prices, window=50, step=10)
    assert not table.isna().any().any()


def test_ref_inestable_flags_windows_where_the_reference_leg_collapses(monkeypatch):
    vectors = [
        np.array([1.0, -1.0]),   # ventana 0: fija la pata de referencia (A)
        np.array([0.02, -1.0]),  # ventana 1: A cae a 2% del maximo de esa ventana -> inestable
        np.array([0.8, -1.0]),   # ventana 2: vuelve a ser estable
    ]
    calls = iter(vectors)

    class FakeResult:
        def __init__(self, vector):
            self.evec = np.column_stack([vector, vector])  # solo se usa la columna 0

    def fake_johansen_test(window_data):
        return FakeResult(next(calls)), {"trace_stat_r0": 0.0}

    monkeypatch.setattr(eigvec_movil, "johansen_test", fake_johansen_test)

    index = pd.bdate_range("2021-01-04", periods=20)
    log_prices = pd.DataFrame({"A": 0.0, "B": 0.0}, index=index)
    table = rolling_eigenvector(log_prices, window=10, step=5)

    assert table["ref_inestable"].tolist() == [False, True, False]
    assert table["A"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert table["B"].tolist() == pytest.approx([-1.0, -50.0, -1.25])


def test_load_group_log_prices_restricts_to_train_when_asked(tmp_path, monkeypatch):
    index = pd.bdate_range("2021-01-04", periods=20)
    prices = pd.DataFrame(
        {"A": np.linspace(10.0, 20.0, 20), "B": np.linspace(5.0, 15.0, 20)}, index=index
    )
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "seleccion.json").write_text(json.dumps({"tickers": ["A", "B"]}))

    monkeypatch.setattr(eigvec_movil, "PRICES_FILE", prices_path)
    monkeypatch.setattr(eigvec_movil, "ARTIFACTS_DIR", artifacts_dir)

    log_prices_train, tickers = load_group_log_prices("train")
    log_prices_todo, _ = load_group_log_prices("todo")

    train_end, _ = split_positions(len(prices))
    assert len(log_prices_train) == train_end
    assert len(log_prices_todo) == len(prices)
    assert tickers == ["A", "B"]


def test_main_refuses_to_run_hasta_todo():
    with pytest.raises(SystemExit):
        main(hasta="todo")
