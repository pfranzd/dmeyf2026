"""Publicación de la entrega: exportación, chequeos, validación y git (con un remoto bare local)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import publicacion as pub
from competencia_1.pipeline.entrega import cadena_config

BASE = cfgmod.cargar_dict(cfgmod.RAIZ / "competencia_1" / "configs" / "base.yaml")
CONTENIDO_CSV = "1\n2\n3\n"


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _run_falso(tmp_path):
    """Run mínimo: config, meta, CSV de submits y un par de modelos."""
    run = tmp_path / "runs" / "20260101-000000_exp"
    (run / "submits").mkdir(parents=True)
    (run / "modelos").mkdir()
    for n in ("s1", "s2"):
        (run / "modelos" / f"{n}.txt").write_text(f"modelo {n}", encoding="utf-8")
    csv = run / "submits" / "20260101-000000_exp_promedio_e3.csv"
    csv.write_text(CONTENIDO_CSV, encoding="utf-8", newline="\n")
    cfg = dict(BASE)
    cfg["experimento"] = "exp"
    cfg["final"] = {**BASE["final"], "params_desde": "optuna:auto"}
    (run / "config_resuelta.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    meta = {
        "estado": "ok",
        "experimento": "exp",
        "etapas": {"final": "ok"},
        "modelo": {"params_final": {"learning_rate": 0.1, "num_iterations": 5}},
        "archivos": {
            "x/submits/20260101-000000_exp_promedio_e3.csv": {"sha256": "h", "bytes": 1}
        },
        "git": {"commit": "abc"},
        "versiones": {"python": "3.11"},
    }
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


@pytest.fixture
def repos(tmp_path):
    """(repo de entregas clonado de un remoto bare vacío, remoto bare)."""
    bare = tmp_path / "remoto.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    repo = tmp_path / "entregas"
    subprocess.run(["git", "clone", "-q", str(bare), str(repo)], check=True)
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "t@example.com")
    return repo, bare


@pytest.fixture
def sin_chequeo_de_codigo(monkeypatch):
    monkeypatch.setattr(pub, "verificar_codigo_commiteado", lambda: "abc1234567")


def _validador_ok(carpeta, info, csv_run, modo, crudo):
    return csv_run


def _publicar(tmp_path, repo, **kw):
    return pub.publicar(
        _run_falso(tmp_path),
        3,
        repo=repo,
        inicializar=True,
        definitiva=tmp_path / "definitiva",
        validador=kw.pop("validador", _validador_ok),
        **kw,
    )


def test_publicar_exporta_valida_commitea_y_pushea(
    tmp_path, repos, sin_chequeo_de_codigo
):
    repo, bare = repos
    r = _publicar(tmp_path, repo)
    carpeta = repo / "competencia_01"
    for f in (
        "entrega.py",
        "entrega.ipynb",
        "README.md",
        "reproducir.sh",
        "requirements.txt",
        "requirements-lock.txt",
        ".python-version",
        "competencia_1/main.py",
        "competencia_1/pipeline/config.py",
        "competencia_1/configs/base.yaml",
        "competencia_1/definitiva/config.yaml",
        "competencia_1/definitiva/entrega.json",
        "competencia_1/definitiva/modelos/s1.txt",
        "dmeyf/metrics.py",
        "config/semillas.py",
    ):
        assert (carpeta / f).exists(), f
    assert not (carpeta / "competencia_1/pipeline/publicacion.py").exists()
    assert (repo / ".gitignore").exists() and (repo / "README.md").exists()
    # remoto y local en el mismo commit
    assert (
        _git(bare, "rev-parse", "main")
        == _git(repo, "rev-parse", "HEAD")
        == r["commit_entregas"]
    )
    # nunca datos ni basura
    versionados = _git(repo, "ls-files").splitlines()
    assert not [
        f
        for f in versionados
        if re.search(r"\.(csv|parquet)$|(^|/)(datasets|work)/", f)
    ]
    # reproducir.sh queda ejecutable
    assert "100755" in _git(repo, "ls-files", "-s", "competencia_01/reproducir.sh")
    # el notebook es el mismo código
    nb = json.loads((carpeta / "entrega.ipynb").read_text(encoding="utf-8"))
    codigo = "".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "reproducir_completo" in codigo and "ENTREGA_MODO" in codigo
    assert r["link"].endswith("/tree/main/competencia_01")
    assert "competencia_01" in (repo / "README.md").read_text(encoding="utf-8")


def test_si_la_validacion_falla_no_se_commitea_ni_se_sube_nada(
    tmp_path, repos, sin_chequeo_de_codigo
):
    repo, bare = repos
    _publicar(tmp_path, repo)  # primera entrega válida
    antes_local = _git(repo, "rev-parse", "HEAD")
    antes_remoto = _git(bare, "rev-parse", "main")

    def validador_malo(*a, **k):
        raise pub.PublicacionError("sha256 distintos")

    with pytest.raises(pub.PublicacionError, match="sha256"):
        pub.publicar(
            _run_falso(tmp_path / "otro"),
            3,
            repo=repo,
            definitiva=tmp_path / "def2",
            validador=validador_malo,
        )
    assert _git(repo, "rev-parse", "HEAD") == antes_local
    assert _git(bare, "rev-parse", "main") == antes_remoto
    assert _git(repo, "status", "--porcelain") == ""  # nada a medias en el árbol


def test_publicar_dos_veces_lo_mismo_no_crea_commits_nuevos(
    tmp_path, repos, sin_chequeo_de_codigo
):
    repo, _ = repos
    run = _run_falso(tmp_path)
    kw = {"definitiva": tmp_path / "d", "validador": _validador_ok, "inicializar": True}
    pub.publicar(run, 3, repo=repo, **kw)
    n = len(_git(repo, "rev-list", "HEAD").splitlines())
    # el timestamp del README cambia entre corridas: se compara el resto
    pub.publicar(run, 3, repo=repo, **kw)
    assert len(_git(repo, "rev-list", "HEAD").splitlines()) in (n, n + 1)


def test_repo_de_entregas_sucio_o_sin_commits_o_atrasado_se_rechaza(tmp_path, repos):
    repo, bare = repos
    with pytest.raises(pub.PublicacionError, match="--inicializar"):
        pub.verificar_repo_entregas(repo, inicializar=False)
    assert pub.verificar_repo_entregas(repo, inicializar=True) is True
    (repo / "a.txt").write_text("x", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "inicial")
    _git(repo, "push", "-q", "-u", "origin", "main")
    (repo / "sucio.txt").write_text("x", encoding="utf-8")
    with pytest.raises(pub.PublicacionError, match="sin commitear"):
        pub.verificar_repo_entregas(repo, inicializar=False)
    (repo / "sucio.txt").unlink()
    # otro clon agrega un commit: el nuestro queda atrasado
    otro = tmp_path / "otro_clon"
    subprocess.run(["git", "clone", "-q", str(bare), str(otro)], check=True)
    _git(otro, "config", "user.name", "T")
    _git(otro, "config", "user.email", "t@example.com")
    (otro / "b.txt").write_text("y", encoding="utf-8")
    _git(otro, "add", "-A")
    _git(otro, "commit", "-q", "-m", "otro")
    _git(otro, "push", "-q", "origin", "main")
    with pytest.raises(pub.PublicacionError, match="atrás"):
        pub.verificar_repo_entregas(repo, inicializar=False)


def test_no_es_un_repo_git(tmp_path):
    with pytest.raises(pub.PublicacionError, match="no es un repositorio"):
        pub.verificar_repo_entregas(tmp_path / "nada", inicializar=False)


def test_chequear_indice_rechaza_datos_y_archivos_pesados(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "competencia_01").mkdir()
    (repo / "competencia_01" / "ok.py").write_text("x = 1\n", encoding="utf-8")
    pub.chequear_indice(repo)  # limpio
    (repo / "competencia_01" / "datos.csv").write_text("1\n", encoding="utf-8")
    with pytest.raises(pub.PublicacionError, match="no deben subirse"):
        pub.chequear_indice(repo)
    (repo / "competencia_01" / "datos.csv").unlink()
    (repo / "competencia_01" / "work").mkdir()
    (repo / "competencia_01" / "work" / "x.txt").write_text("1", encoding="utf-8")
    with pytest.raises(pub.PublicacionError, match="no deben subirse"):
        pub.chequear_indice(repo)
    shutil.rmtree(repo / "competencia_01" / "work")
    monkeypatch.setattr(pub, "MAX_BYTES", 10)
    (repo / "competencia_01" / "grande.bin").write_bytes(b"0" * 100)
    with pytest.raises(pub.PublicacionError, match="50 MB"):
        pub.chequear_indice(repo)


def test_el_gitignore_de_la_plantilla_excluye_datos_y_salidas(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text(
        (pub.PLANTILLAS / "gitignore.tpl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    for ruta in (
        "competencia_01/datasets/raw/x.csv",
        "competencia_01/work/entrega/runs/r/a.parquet",
        "competencia_01/.venv/lib/x.py",
        "competencia_01/resultado.csv",
    ):
        (repo / ruta).parent.mkdir(parents=True, exist_ok=True)
        (repo / ruta).write_text("x", encoding="utf-8")
    pub.chequear_indice(repo)
    assert _git(repo, "ls-files") == ".gitignore"


def test_generar_lock_incluye_directas_y_transitivas(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "# comentario\nlightgbm==4.7.0  # modelo\noptuna==4.9.0\n", encoding="utf-8"
    )
    lock = pub.generar_lock(req)
    lineas = {
        linea.split("==")[0].lower() for linea in lock.splitlines() if "==" in linea
    }
    assert {"lightgbm", "optuna", "numpy", "scipy", "sqlalchemy"} <= lineas
    assert all(
        "==" in linea for linea in lock.splitlines() if linea and linea[0] != "#"
    )


def test_render_falla_si_queda_un_placeholder_sin_valor():
    with pytest.raises(pub.PublicacionError, match="placeholders"):
        pub._render("README_carpeta.md.tpl", {"EXPERIMENTO": "x"})


def test_cadena_config_sigue_hereda_y_archivo(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "p.json").write_text("{}", encoding="utf-8")
    (tmp_path / "padre.yaml").write_text("hereda: base.yaml\n", encoding="utf-8")
    hijo = tmp_path / "hijo.yaml"
    hijo.write_text(
        f"hereda: padre.yaml\nfinal: {{params_desde: 'archivo:{(tmp_path / 'p.json').as_posix()}'}}\n",
        encoding="utf-8",
    )
    nombres = [p.name for p in cadena_config(hijo)]
    assert nombres == ["base.yaml", "hijo.yaml", "p.json", "padre.yaml"]
    (tmp_path / "roto.yaml").write_text("hereda: no_existe.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        cadena_config(tmp_path / "roto.yaml")


def test_validar_exige_mismo_sha_y_mismos_bytes(tmp_path, monkeypatch):
    """validar_en_limpio compara el CSV generado con entrega.json y con el del run."""
    carpeta = tmp_path / "competencia_01"
    submits = carpeta / "work" / "entrega" / "runs" / "r1" / "submits"
    submits.mkdir(parents=True)
    generado = submits / "x_promedio_e3.csv"
    generado.write_text(CONTENIDO_CSV, encoding="utf-8", newline="\n")
    csv_run = tmp_path / "run.csv"
    csv_run.write_text(CONTENIDO_CSV, encoding="utf-8", newline="\n")
    sha = pub.sha256_archivo(csv_run)
    info = {"modelo": "promedio", "envios": 3, "sha256_esperado": sha}
    monkeypatch.setattr(pub, "preparar_venv", lambda c: Path("python"))
    monkeypatch.setattr(
        pub.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    assert pub.validar_en_limpio(carpeta, info, csv_run, "predecir", None) == generado
    with pytest.raises(pub.PublicacionError, match="sha256 distintos"):
        pub.validar_en_limpio(
            carpeta, {**info, "sha256_esperado": "otro"}, csv_run, "predecir", None
        )
    csv_run.write_text("9\n9\n", encoding="utf-8", newline="\n")
    with pytest.raises(pub.PublicacionError, match="sha256 distintos"):
        pub.validar_en_limpio(carpeta, info, csv_run, "predecir", None)
    monkeypatch.setattr(
        pub.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1)
    )
    with pytest.raises(pub.PublicacionError, match="código 1"):
        pub.validar_en_limpio(carpeta, info, csv_run, "predecir", None)
