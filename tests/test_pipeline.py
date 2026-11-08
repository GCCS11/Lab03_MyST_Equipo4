"""Unit tests for main.py: el pipeline completo con --quick, y el candado de seguridad sobre
artifacts/theta.json (el tag de git theta-congelado y el registro de corridas sobre prueba y
validacion).

El candado se prueba contra un repositorio de git desechable en tmp_path, nunca contra este
repositorio: no se crea ningun tag real aqui.
"""
import json
import subprocess

import pandas as pd
import pytest

import main as pipeline


def run_git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def make_repo(tmp_path):
    """Repo de git chico, con un commit inicial, listo para agregar theta.json."""
    run_git(["init"], tmp_path)
    run_git(["config", "user.email", "test@example.com"], tmp_path)
    run_git(["config", "user.name", "Test"], tmp_path)
    (tmp_path / "README.md").write_text("repo de prueba\n", encoding="utf-8")
    run_git(["add", "README.md"], tmp_path)
    run_git(["commit", "-m", "inicial"], tmp_path)
    return tmp_path


def commit_theta(repo_dir, theta, message="agrega theta.json"):
    theta_path = repo_dir / "theta.json"
    theta_path.write_text(json.dumps(theta), encoding="utf-8")
    run_git(["add", "theta.json"], repo_dir)
    run_git(["commit", "-m", message], repo_dir)
    return theta_path


def test_quick_generates_every_artifact_with_the_expected_columns():
    out_dir = pipeline.main(quick=True)

    equity = pd.read_csv(out_dir / "equity.csv")
    assert list(equity.columns) == ["date", "split", "equity", "drawdown", "rolling_sharpe"]
    assert set(equity["split"]) <= {"train", "test", "validation"}

    rolling = pd.read_csv(out_dir / "metricas_rolling_sharpe.csv")
    assert list(rolling.columns) == ["date", "split", "rolling_sharpe"]

    operaciones = pd.read_csv(out_dir / "operaciones.csv")
    assert list(operaciones.columns) == [
        "entrada", "salida", "split", "direccion", "pnl_bruto", "comisiones",
        "prestamo", "pnl_neto", "retorno", "motivo",
    ]

    metricas = json.loads((out_dir / "metricas.json").read_text(encoding="utf-8"))
    assert set(metricas) == {"train", "test", "validation"}
    campos_esperados = {
        "sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "profit_factor",
        "retorno_total", "n_operaciones", "pnl_bruto", "comisiones", "prestamo", "pnl_neto",
    }
    for valores in metricas.values():
        assert campos_esperados <= set(valores)
        for campo in campos_esperados:
            assert valores[campo] is not None

    for nombre in ("precios", "senales", "pesos", "ganancias", "varianzas"):
        assert (out_dir / f"{nombre}.csv").exists()
    assert (out_dir / "vecm.json").exists()


def test_quick_never_writes_corridas_test_log():
    out_dir = pipeline.main(quick=True)
    assert not (out_dir / "corridas_test.log").exists()


def test_resolve_theta_is_provisional_and_train_only_without_theta_json(tmp_path):
    tabla = pd.DataFrame([
        {"theta_id": 0, "z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0,
         "pliegue": 0, "sharpe": 1.0, "n_operaciones": 5, "comisiones": 1.0, "prestamo": 0.0,
         "retorno_total": 0.01, "max_drawdown": 0.01},
    ])
    tabla.to_csv(tmp_path / "theta_busqueda.csv", index=False)

    theta, congelado, splits = pipeline.resolve_theta(directory=tmp_path)

    assert congelado is False
    assert splits == ("train",)
    assert theta == {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0}


def test_resolve_theta_fails_clearly_without_grid_or_theta(tmp_path):
    with pytest.raises(RuntimeError):
        pipeline.resolve_theta(directory=tmp_path)


def test_lock_rejects_a_missing_tag(tmp_path):
    repo = make_repo(tmp_path)
    theta_path = commit_theta(repo, {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0})

    with pytest.raises(RuntimeError, match="falta el tag"):
        pipeline.assert_theta_is_frozen(theta_path, repo_dir=repo)


def test_lock_accepts_theta_committed_before_the_tag(tmp_path):
    repo = make_repo(tmp_path)
    theta_path = commit_theta(repo, {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0})
    run_git(["tag", "theta-congelado"], repo)

    pipeline.assert_theta_is_frozen(theta_path, repo_dir=repo)  # no debe lanzar nada


def test_lock_rejects_theta_edited_after_the_tag(tmp_path):
    repo = make_repo(tmp_path)
    theta_path = commit_theta(repo, {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0})
    run_git(["tag", "theta-congelado"], repo)
    commit_theta(repo, {"z_in": 2.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0}, "cambia theta despues del tag")

    with pytest.raises(RuntimeError, match="posterior al tag"):
        pipeline.assert_theta_is_frozen(theta_path, repo_dir=repo)


def test_resolve_theta_uses_the_lock_when_theta_json_exists(tmp_path):
    repo = make_repo(tmp_path)
    commit_theta(repo, {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0})
    # sin el tag: resolve_theta debe negarse, no solo el candado de bajo nivel
    with pytest.raises(RuntimeError, match="falta el tag"):
        pipeline.resolve_theta(directory=repo, repo_dir=repo)

    run_git(["tag", "theta-congelado"], repo)
    theta, congelado, splits = pipeline.resolve_theta(directory=repo, repo_dir=repo)
    assert congelado is True
    assert splits == pipeline.SPLITS
    assert theta == {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0}


def test_log_test_run_appends_without_blocking_repetitions(tmp_path):
    repo = make_repo(tmp_path)
    theta_path = commit_theta(repo, {"z_in": 1.0, "z_out": 0.3, "z_stop": 2.5, "h_max": 10, "band": 0.0})

    pipeline.log_test_run(repo, theta_path, repo_dir=repo)
    pipeline.log_test_run(repo, theta_path, repo_dir=repo)

    lines = (repo / "corridas_test.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        assert "theta_sha256=" in line
        assert "commit=" in line
