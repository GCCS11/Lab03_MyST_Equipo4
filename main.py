"""Full pipeline entry point. Run with: python main.py

Every parameter comes frozen from artifacts/ (chosen on the training partition only):
1. Prices of the selected group (seleccion.json) and the train / test / validation cut dates.
2. VECM on training with the Johansen vector: alpha, beta'alpha and the model half-life.
3. Filter 1: dynamic weights; the equilibrium deviation uses the prior weights beta_{t|t-1}.
4. Filter 2: latent mean and scale of the deviation, and the causal z-score.
5. Trading rule with the thresholds of theta.json.
6. backtest.engine and performance.metrics, run separately on train, test and validation (each
   partition starts flat with the initial capital). Artifacts for the dashboard
   (dashboard/data.py lists the files and columns).

Candado de seguridad (ver resolve_theta): sin artifacts/theta.json solo se corre entrenamiento,
con un theta provisional marcado como NO congelado. Con theta.json se exige el tag de git
THETA_TAG sobre un commit que incluya su ultima version, y cada corrida sobre prueba o
validacion queda registrada en artifacts/corridas_test.log.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestParams, run_backtest
from cointegration.seleccion import PRICES_FILE
from config import ARTIFACTS_DIR, ROOT_DIR, SEED, set_seed
from data.loader import split_positions
from kalman.signal_filter import run_signal_filter
from kalman.weights_filter import run_weights_filter
from performance.metrics import compute_metrics, write_metricas_json
from strategy.signals import spread_direction
from strategy.vecm import equilibrium_deviation, fit_vecm

SIGNAL_KEYS = ("Q", "R", "phi", "lam", "mu0", "P0", "burn_in")
RULE_KEYS = ("z_in", "z_out", "z_stop", "h_max")
SPLITS = ("train", "test", "validation")
THETA_TAG = "theta-congelado"


def parse_args():
    parser = argparse.ArgumentParser(description="Lab03 statistical arbitrage pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="corre todo el pipeline sobre datos sinteticos chicos, en un directorio temporal, "
        "sin tocar artifacts/ ni datos reales de ningun split",
    )
    return parser.parse_args()


def read_json(path):
    """Content of a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen(directory=ARTIFACTS_DIR):
    """Frozen inputs: the selection, the filter parameters and, once chosen, the thresholds."""
    directory = Path(directory)
    theta_file = directory / "theta.json"
    theta = read_json(theta_file) if theta_file.exists() else None
    return read_json(directory / "seleccion.json"), read_json(directory / "filtros.json"), theta


def load_log_prices(selection):
    """Log prices of the selected group from the frozen snapshot."""
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)[selection["tickers"]]
    return prices.apply(np.log).rename_axis("date")


def split_labels(index, selection):
    """train, test or validation for each date, with the cut dates frozen in seleccion.json."""
    train_end, test_end = pd.Timestamp(selection["train_end"]), pd.Timestamp(selection["test_end"])
    labels = np.where(index <= train_end, "train", np.where(index <= test_end, "test", "validation"))
    return pd.Series(labels, index=index, name="split")


def vecm_summary(log_prices, selection):
    """VECM on training with the Johansen vector as the error-correction term."""
    fit = fit_vecm(log_prices.loc[: selection["train_end"]], selection["beta_johansen"])
    return {
        "alpha": dict(zip(fit.tickers, fit.alpha.tolist())),
        "beta_alpha": fit.beta_alpha,
        "corrige_error": fit.is_error_correcting,
        "vida_media": fit.half_life,
        "c": fit.c,
    }


def run_pipeline(log_prices, selection, filters, theta=None, until=None):
    """Causal chain from log prices to signals, with every parameter frozen.

    With `until` it only sees data up to that date, so the z-score at t must not change when
    later data is added (test 1). Returns the Filter 1 and Filter 2 frames, the equilibrium
    deviation and, when `theta` is given, the output of the trading rule.
    """
    data = log_prices.loc[:until, selection["tickers"]]
    weights = run_weights_filter(data, filters["beta0"], filters["pata_normalizada"], filters["filtro1"])
    deviation = equilibrium_deviation(data, weights.filter(like="beta_"))
    signal = run_signal_filter(deviation, {key: filters["filtro2"][key] for key in SIGNAL_KEYS})
    rule = spread_direction(signal["z"], {key: theta[key] for key in RULE_KEYS}) if theta else None
    return {"weights": weights, "deviation": deviation, "signal": signal, "rule": rule}


def signal_artifacts(log_prices, splits, result):
    """Frames of the technical view, named as dashboard/data.py expects."""
    weights, signal, rule = result["weights"], result["signal"], result["rule"]
    signals = pd.DataFrame({
        "split": splits,
        "spread": result["deviation"],
        "mu_prior": signal["mu_prior"],
        "sigma_prior": signal["sigma_prior"],
        "z": signal["z"],
        "d": rule["d"] if rule is not None else 0,
        "exit_reason": rule["exit_reason"] if rule is not None else "",
    })
    return {
        "precios": np.exp(log_prices),
        "senales": signals,
        "pesos": weights.filter(like="beta_"),
        "ganancias": weights.filter(like="gain_").add_prefix("f1_").assign(f2_gain=signal["gain"]),
        "varianzas": weights.filter(like="pvar_").add_prefix("f1_").assign(f2_pvar=signal["pvar"]),
    }


def write_artifacts(directory, frames, vecm):
    """Write the CSV files and the VECM summary into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index_label="date")
    (directory / "vecm.json").write_text(json.dumps(vecm, indent=2) + "\n", encoding="utf-8")


# --- Candado de seguridad sobre theta.json --------------------------------------------------

def _run_git(args, repo_dir):
    """Resultado crudo de `git <args>` en `repo_dir`, sin lanzar si el comando falla."""
    return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True)


def git_last_commit(path, repo_dir=ROOT_DIR):
    """Hash del ultimo commit que toco `path`, o None si no tiene historial de git."""
    result = _run_git(["log", "-1", "--format=%H", "--", str(path)], repo_dir)
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_tag_commit(tag, repo_dir=ROOT_DIR):
    """Hash al que apunta `tag`, o None si el tag no existe."""
    result = _run_git(["rev-list", "-n", "1", tag], repo_dir)
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_is_ancestor(commit, tag_commit, repo_dir=ROOT_DIR):
    """True si `commit` es `tag_commit` o uno de sus ancestros."""
    return _run_git(["merge-base", "--is-ancestor", commit, tag_commit], repo_dir).returncode == 0


def git_head(repo_dir=ROOT_DIR):
    """Hash del commit actual (HEAD), o None si no se puede leer."""
    result = _run_git(["rev-parse", "HEAD"], repo_dir)
    return result.stdout.strip() if result.returncode == 0 else None


def assert_theta_is_frozen(theta_path, tag=THETA_TAG, repo_dir=ROOT_DIR):
    """Exige que exista el tag de git `tag` y que el ultimo commit que toco `theta_path` sea
    ese commit o uno de sus ancestros. Falla con un mensaje claro si no."""
    commit = git_last_commit(theta_path, repo_dir)
    if commit is None:
        raise RuntimeError(
            f"{theta_path} no tiene historial de git: congelalo con un commit antes de usarlo"
        )
    tag_commit = git_tag_commit(tag, repo_dir)
    if tag_commit is None:
        raise RuntimeError(
            f"falta el tag de git '{tag}': theta.json no esta declarado como congelado"
        )
    if commit != tag_commit and not git_is_ancestor(commit, tag_commit, repo_dir):
        raise RuntimeError(
            f"el commit de {theta_path} ({commit[:8]}) es posterior al tag '{tag}' "
            f"({tag_commit[:8]}): theta.json cambio despues de congelarse"
        )


def log_test_run(directory, theta_path, repo_dir=ROOT_DIR):
    """Agrega una linea a artifacts/corridas_test.log: fecha y hora, hash de theta.json y hash
    del commit actual. No bloquea repeticiones, solo deja registro de cuantas veces se corrio
    sobre prueba o validacion, para declararlo en el reporte."""
    theta_hash = hashlib.sha256(Path(theta_path).read_bytes()).hexdigest()[:12]
    commit = (git_head(repo_dir) or "sin_commit")[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "corridas_test.log", "a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\ttheta_sha256={theta_hash}\tcommit={commit}\n")


def provisional_theta(directory=ARTIFACTS_DIR):
    """Theta NO congelado: la combinacion mejor rankeada (media menos una desviacion estandar
    del Sharpe) de la busqueda walk-forward, para poder correr entrenamiento mientras nadie
    congela artifacts/theta.json."""
    from backtest.walk_forward import rank_by_sharpe_minus_std  # import tardio: evita el ciclo

    path = Path(directory) / "theta_busqueda.csv"
    if not path.exists():
        raise RuntimeError(
            f"no existe artifacts/theta.json ni {path}: corre primero backtest.walk_forward"
        )
    mejor = rank_by_sharpe_minus_std(pd.read_csv(path)).iloc[0]
    return {
        "z_in": float(mejor["z_in"]), "z_out": float(mejor["z_out"]),
        "z_stop": float(mejor["z_stop"]), "h_max": int(mejor["h_max"]),
        "band": float(mejor["band"]),
    }


def resolve_theta(directory=ARTIFACTS_DIR, repo_dir=ROOT_DIR):
    """Decide que theta usar y que particiones procesar.

    Sin artifacts/theta.json: theta provisional y solo entrenamiento. Con theta.json: exige el
    candado de git y habilita las tres particiones.
    """
    theta_path = Path(directory) / "theta.json"
    if not theta_path.exists():
        return provisional_theta(directory), False, ("train",)
    assert_theta_is_frozen(theta_path, repo_dir=repo_dir)
    return read_json(theta_path), True, SPLITS


# --- Motor de backtest y metricas, por particion ---------------------------------------------

def backtest_by_split(result, log_prices, selection, splits, splits_to_run, params=None):
    """Corre backtest.engine.run_backtest y performance.metrics.compute_metrics por separado en
    cada particion de `splits_to_run`, cada una arrancando plano con el capital inicial."""
    params = params or BacktestParams()
    rule = result["rule"]
    betas = result["weights"].filter(like="beta_").reindex(rule.index)
    prices = np.exp(log_prices[selection["tickers"]]).reindex(rule.index)
    splits_aligned = splits.reindex(rule.index)

    resultados, metricas = {}, {}
    for split in splits_to_run:
        mask = (splits_aligned == split).to_numpy()
        if not mask.any():
            continue
        bt = run_backtest(rule.loc[mask], betas.loc[mask], prices.loc[mask], params)
        resultados[split] = bt
        metricas[split] = compute_metrics(bt.diario, bt.operaciones, params.initial_capital)
    return resultados, metricas


def backtest_artifacts(resultados, metricas):
    """equity.csv (date, split, equity, drawdown, rolling_sharpe), metricas_rolling_sharpe.csv
    y operaciones.csv (con split), listos para escribir con to_csv."""
    equity_frames, operaciones_frames = [], []
    for split, bt in resultados.items():
        equity = bt.diario["equity"]
        drawdown = equity / equity.cummax() - 1.0
        rolling = metricas[split].rolling_sharpe.reindex(equity.index)
        equity_frames.append(pd.DataFrame({
            "split": split, "equity": equity, "drawdown": drawdown, "rolling_sharpe": rolling,
        }))
        trades = bt.operaciones.copy()
        trades.insert(2, "split", split)
        operaciones_frames.append(trades)

    if equity_frames:
        equity_csv = pd.concat(equity_frames).rename_axis("date")
    else:
        equity_csv = pd.DataFrame(columns=["split", "equity", "drawdown", "rolling_sharpe"]).rename_axis("date")
    rolling_csv = equity_csv[["split", "rolling_sharpe"]]

    operaciones_columns = ["entrada", "salida", "split", "direccion", "pnl_bruto", "comisiones",
                            "prestamo", "pnl_neto", "retorno", "motivo"]
    operaciones_csv = (
        pd.concat(operaciones_frames, ignore_index=True) if operaciones_frames
        else pd.DataFrame(columns=operaciones_columns)
    )
    return equity_csv, rolling_csv, operaciones_csv


def write_backtest_artifacts(directory, resultados, metricas):
    """Escribe equity.csv, metricas_rolling_sharpe.csv, operaciones.csv y metricas.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    equity_csv, rolling_csv, operaciones_csv = backtest_artifacts(resultados, metricas)
    equity_csv.to_csv(directory / "equity.csv", index_label="date")
    rolling_csv.to_csv(directory / "metricas_rolling_sharpe.csv", index_label="date")
    operaciones_csv.to_csv(directory / "operaciones.csv", index=False)
    write_metricas_json(directory / "metricas.json", metricas)


# --- Datos sinteticos para --quick --------------------------------------------------------

def synthetic_inputs(n_rows=150, seed=SEED):
    """selection, filters y theta fabricados en memoria: una pareja I(1) cointegrada por
    construccion mas una tercera pata de ruido, suficiente para correr el pipeline completo
    sin tocar ningun dato real."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n_rows)
    common = np.cumsum(rng.normal(0.0, 0.01, n_rows))
    log_prices = pd.DataFrame({
        "X1": 4.0 + common + rng.normal(0.0, 0.002, n_rows),
        "X2": 3.5 + 0.8 * common + rng.normal(0.0, 0.002, n_rows),
        "X3": 3.0 + rng.normal(0.0, 0.01, n_rows),
    }, index=index)

    train_end, test_end = split_positions(n_rows)
    selection = {
        "tickers": ["X1", "X2", "X3"],
        "beta_johansen": [1.0, -0.8, 0.0],
        "pata_normalizada": "X1",
        "eigenvalor": 0.05,
        "vida_media": 10.0,
        "train_end": str(index[train_end - 1].date()),
        "test_end": str(index[test_end - 1].date()),
        "grupos_evaluados_total": 1,
    }
    filters = {
        "tickers": selection["tickers"],
        "beta0": selection["beta_johansen"],
        "pata_normalizada": "X1",
        "train_end": selection["train_end"],
        "filtro1": {"Q": 1e-6, "R": 1e-3, "phi": 0.0, "c": 0.0, "P0": 1e-4},
        "filtro2": {"Q": 1e-6, "R": 1e-3, "phi": 0.0, "lam": 0.94, "mu0": 0.0, "burn_in": 5, "P0": 1e-4},
        "burn_in": 5,
    }
    theta = {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 15, "band": 0.0}
    return log_prices, selection, filters, theta


def main(quick=False):
    set_seed()
    print(f"Seed fixed at {SEED}")
    print("Mode: quick" if quick else "Mode: full")

    if quick:
        log_prices, selection, filters, theta = synthetic_inputs()
        theta_congelado, splits_to_run = True, SPLITS
        out_dir = Path(tempfile.mkdtemp(prefix="lab03_quick_"))
    else:
        selection, filters, _ = load_frozen()
        log_prices = load_log_prices(selection)
        theta, theta_congelado, splits_to_run = resolve_theta()
        out_dir = ARTIFACTS_DIR
        if not theta_congelado:
            train_end_row, _ = split_positions(len(log_prices))
            log_prices = log_prices.iloc[:train_end_row]
            print("artifacts/theta.json no existe: theta provisional (NO congelado), de la busqueda walk-forward.")
            print("No se procesan prueba ni validacion.")

    splits = split_labels(log_prices.index, selection)
    print(f"Grupo: {', '.join(selection['tickers'])} · {log_prices.index[0].date()} a {log_prices.index[-1].date()}")
    print(f"Cortes: entrenamiento hasta {selection['train_end']}, prueba hasta {selection['test_end']}")

    vecm = vecm_summary(log_prices, selection)
    status = "corrige el error" if vecm["corrige_error"] else "NO corrige el error"
    print(f"VECM (entrenamiento): beta'alpha = {vecm['beta_alpha']:.4f}, {status}, vida media {vecm['vida_media']:.1f} días")

    until = None if theta_congelado else pd.Timestamp(selection["train_end"])
    result = run_pipeline(log_prices, selection, filters, theta, until=until)
    write_artifacts(out_dir, signal_artifacts(log_prices, splits, result), vecm)
    print(f"Artifacts de señales escritos en {out_dir}")

    resultados, metricas = backtest_by_split(result, log_prices, selection, splits, splits_to_run)
    write_backtest_artifacts(out_dir, resultados, metricas)
    print(f"Artifacts de backtest y metricas escritos en {out_dir}: {', '.join(resultados)}")

    if not quick and theta_congelado and ("test" in resultados or "validation" in resultados):
        log_test_run(ARTIFACTS_DIR, ARTIFACTS_DIR / "theta.json")
        print("Corrida sobre prueba o validacion registrada en artifacts/corridas_test.log")

    for split, m in metricas.items():
        print(f"{split}: sharpe={m.sharpe:.2f}, n_operaciones={m.n_operaciones}, retorno_total={m.retorno_total:.2%}")
    return out_dir


if __name__ == "__main__":
    args = parse_args()
    main(quick=args.quick)
