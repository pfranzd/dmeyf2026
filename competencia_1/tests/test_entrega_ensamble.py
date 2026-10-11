"""Entrega de un ensamble de runs: promover_ensamble y su publicación."""

import json
import subprocess

import pytest
import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import publicacion as pub
from competencia_1.pipeline.entrega import _cortes_del_run, promover, promover_ensamble
from competencia_1.tests.test_entrega import _run_falso

CSV_ENS = b"5" + bytes([10]) + b"7" + bytes([10])


def _ensamble_falso(tmp_path, envios=(11000,)):
    runs = []
    for sub in ("a", "b"):
        run = _run_falso(tmp_path / sub, envios=envios)
        (run / "modelos").mkdir()
        for n in ("s1", "s2"):
            (run / "modelos" / f"{n}.txt").write_text(f"{sub} {n}", encoding="utf-8")
        runs.append(run)
    ens = tmp_path / "ensambles" / "20260102-000000_ens99"
    (ens / "submits").mkdir(parents=True)
    for n in envios:
        (ens / "submits" / f"20260102-000000_ens99_e{n}.csv").write_bytes(CSV_ENS)
    meta = {
        "ensamble_id": "20260102-000000_ens99",
        "metodo": "rank",
        "entradas": [
            {"run": str(r), "parquet": "x", "sha256": f"p{i}", "peso": 1.0}
            for i, r in enumerate(runs)
        ],
        "git": {"commit": "def"},
    }
    (ens / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ens, runs


def test_promover_ensamble_arma_componentes_y_entrega(tmp_path):
    ens, runs = _ensamble_falso(tmp_path)
    destino = tmp_path / "def"
    # una entrega previa de un run suelto queda reemplazada por completo
    promover(runs[0], 11000, destino=destino, con_modelos=True)
    promover_ensamble(ens, 11000, destino=destino)
    assert not (destino / "config.yaml").exists() and not (destino / "modelos").exists()
    info = json.loads((destino / "entrega.json").read_text(encoding="utf-8"))
    assert info["tipo"] == "ensamble" and info["modelo"] == "ensamble"
    assert info["experimento"] == "ens99" and info["metodo"] == "rank"
    assert info["csv_original"] == "20260102-000000_ens99_e11000.csv"
    import hashlib

    assert info["sha256_esperado"] == hashlib.sha256(CSV_ENS).hexdigest()
    assert info["modelos_incluidos"] == 4 and info["pesos"] == [1.0, 1.0]
    assert [c["sha256_csv"] for c in info["componentes"]] == ["h11000", "h11000"]
    for i, c in enumerate(info["componentes"]):
        sub = destino / "componentes" / f"{i}_exp"
        assert (sub / "config.yaml").exists() and (sub / "params.json").exists()
        assert sorted(p.name for p in (sub / "modelos").glob("*.txt")) == [
            "s1.txt",
            "s2.txt",
        ]
        cfg = yaml.safe_load((sub / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["final"]["modelos_desde"].endswith(f"componentes/{i}_exp/modelos")
        assert cfg["final"]["params_desde"].endswith(f"componentes/{i}_exp/params.json")
        cfgmod.desde_dict(cfg)
    # y volver a un run suelto borra los componentes
    promover(runs[0], 11000, destino=destino)
    assert not (destino / "componentes").exists()


def test_componente_sin_el_corte_se_verifica_con_otro(tmp_path):
    ens, runs = _ensamble_falso(tmp_path, envios=(10500,))
    meta = json.loads((runs[0] / "meta.json").read_text())
    assert _cortes_del_run(meta, runs[0]) == [10500]
    # los runs solo registran 10500; el ensamble se entrega con 10500 también
    info = json.loads(
        (
            promover_ensamble(ens, 10500, destino=tmp_path / "d") / "entrega.json"
        ).read_text()
    )
    assert {c["envios_verificacion"] for c in info["componentes"]} == {10500}


def test_promover_ensamble_conserva_publico_solo_mismo_corte(tmp_path):
    ens, _ = _ensamble_falso(tmp_path, envios=(10000, 11000))
    d = tmp_path / "d"
    promover_ensamble(ens, 10000, destino=d, resultado_publico=110.19)
    promover_ensamble(ens, 10000, destino=d)
    assert json.loads((d / "entrega.json").read_text())["resultado_publico"] == 110.19
    promover_ensamble(ens, 11000, destino=d)
    assert json.loads((d / "entrega.json").read_text())["resultado_publico"] is None


def test_promover_ensamble_exige_el_csv_del_corte(tmp_path):
    ens, _ = _ensamble_falso(tmp_path)
    with pytest.raises(ValueError, match="10000 envíos"):
        promover_ensamble(ens, 10000, destino=tmp_path / "d")


def test_publicar_un_ensamble(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "verificar_codigo_commiteado", lambda: "abc1234567")
    bare = tmp_path / "remoto.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    repo = tmp_path / "entregas"
    subprocess.run(["git", "clone", "-q", str(bare), str(repo)], check=True)
    for k, v in (("user.name", "Test"), ("user.email", "t@example.com")):
        subprocess.run(["git", "-C", str(repo), "config", k, v], check=True)

    ens, _ = _ensamble_falso(tmp_path)
    vistos = {}

    def validador(carpeta, info, csv_run, modo, crudo):
        vistos.update(info=info, csv_run=csv_run)
        return csv_run

    r = pub.publicar(
        ens,
        11000,
        repo=repo,
        inicializar=True,
        definitiva=tmp_path / "definitiva",
        validador=validador,
        resultado_publico=110.19,
    )
    carpeta = repo / "competencia_01"
    assert vistos["info"]["tipo"] == "ensamble"
    assert vistos["csv_run"].name == "20260102-000000_ens99_e11000.csv"
    for f in (
        "competencia_1/definitiva/entrega.json",
        "competencia_1/definitiva/componentes/0_exp/modelos/s1.txt",
        "competencia_1/definitiva/componentes/1_exp/config.yaml",
        "README.md",
        "entrega.py",
        "reproducir.sh",
    ):
        assert (carpeta / f).exists(), f
    readme = (carpeta / "README.md").read_text(encoding="utf-8")
    assert "ensamble ens99" in readme and "| 1 | `exp` | 2 | 1 |" in readme
    assert "@@" not in readme and "@@" not in (carpeta / "entrega.py").read_text(
        "utf-8"
    )
    assert r["sha256"] == vistos["info"]["sha256_esperado"]
    remoto = subprocess.run(
        ["git", "-C", str(bare), "rev-parse", "main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert remoto == r["commit_entregas"]
    raiz = (repo / "README.md").read_text(encoding="utf-8")
    assert "ens99" in raiz and "110.19 M" in raiz
