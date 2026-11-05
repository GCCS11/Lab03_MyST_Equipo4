"""Test 1 (causality): the z-score at t does not change when the whole pipeline is rerun with
data only up to t, for several t in test and validation."""
import numpy as np
import pandas as pd
import pytest

from config import ARTIFACTS_DIR

pytest.importorskip("kalman.signal_filter", reason="needs the Kalman filters")
pytest.importorskip("cointegration.seleccion", reason="needs the group selection")
if not (ARTIFACTS_DIR / "seleccion.json").exists() or not (ARTIFACTS_DIR / "filtros.json").exists():
    pytest.skip("needs artifacts/seleccion.json and artifacts/filtros.json", allow_module_level=True)

from main import load_frozen, load_log_prices, run_pipeline, split_labels  # noqa: E402

THETA = {"z_in": 2.0, "z_out": 0.5, "z_stop": 3.5, "h_max": 18}  # causality does not depend on it


@pytest.fixture(scope="module")
def frozen():
    selection, filters, _ = load_frozen(ARTIFACTS_DIR)
    return load_log_prices(selection), selection, filters


@pytest.fixture(scope="module")
def full_run(frozen):
    log_prices, selection, filters = frozen
    return run_pipeline(log_prices, selection, filters, THETA)


def checkpoints(log_prices, selection):
    """One date in test and two in validation, away from the cut dates."""
    test = log_prices.loc[selection["train_end"]:].index
    validation = log_prices.loc[selection["test_end"]:].index
    return [test[100], validation[100], validation[-2]]


def test_z_score_at_t_ignores_later_data(frozen, full_run):
    log_prices, selection, filters = frozen
    for t in checkpoints(log_prices, selection):
        partial = run_pipeline(log_prices, selection, filters, THETA, until=t)
        assert partial["signal"]["z"].index[-1] == t
        assert partial["signal"]["z"].loc[t] == pytest.approx(full_run["signal"]["z"].loc[t], abs=1e-12)
        pd.testing.assert_series_equal(partial["weights"].loc[t], full_run["weights"].loc[t])
        assert partial["rule"]["d"].loc[t] == full_run["rule"]["d"].loc[t]


def test_equilibrium_deviation_is_the_filter_1_spread(full_run):
    np.testing.assert_allclose(full_run["deviation"], full_run["weights"]["spread"], rtol=0, atol=1e-12)


def test_splits_use_the_frozen_cut_dates(frozen):
    log_prices, selection, _ = frozen
    splits = split_labels(log_prices.index, selection)
    assert splits.loc[: selection["train_end"]].eq("train").all()
    assert splits.loc[selection["test_end"]:].iloc[1:].eq("validation").all()
    assert set(splits) == {"train", "test", "validation"}
