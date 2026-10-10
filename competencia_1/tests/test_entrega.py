import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.entrega import promover

BASE = cfgmod.cargar_dict(cfgmod.RAIZ / "competencia_1" / "configs" / "base.yaml")
PARAMS = {"learning_rate": 0.01, "num_iterations": 100, "min_data_in_leaf": 870}


def _run_falso(tmp_path, estado="ok", final="ok", envios=(11000,)):
    run = tmp_path / "runs" / "20260101-000000_exp"
    run.mkdir(parents=True)
    cfg = dict(BASE)
    cfg["experimento"] = "exp"
    cfg["final"] = {
        **BASE["final"],
        "params_desde": "estable:x",
        "reescalar_min_data": True,
    }
    cfg["etapas"] = {**BASE["etapas"], "optuna": True, "final": True, "salida": True}
    (run / "config_resuelta.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    archivos = {
        f"x/submits/20260101-000000_exp_promedio_e{n}.csv": {
            "sha256": f"h{n}",
            "bytes": 1,
        }
        for n in envios
    }
    meta = {
        "estado": estado,
        "experimento": "exp",
        "etapas": {"final": final},
        "modelo": {"params_final": PARAMS},
        "archivos": archivos,
        "git": {"commit": "abc"},
        "versiones": {"python": "3.11"},
    }
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


def test_promover_escribe_los_tres_archivos(tmp_path):
    run = _run_falso(tmp_path)
    destino = promover(run, 11000, destino=tmp_path / "def")
    cfg = yaml.safe_load((destino / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["final"]["reescalar_min_data"] is False
    assert cfg["final"]["params_ignorar_fe_hash"] is True
    assert cfg["salida"]["envios"] == [11000]
    assert cfg["datos"]["reconstruir_target"] is True
    activas = {k for k, v in cfg["etapas"].items() if v}
    assert activas == {"datos", "features", "final", "salida"}
    assert json.loads((destino / "params.json").read_text())["params"] == PARAMS
    info = json.loads((destino / "entrega.json").read_text())
    assert info["sha256_esperado"] == "h11000" and info["envios"] == 11000
    cfgmod.desde_dict(cfg)  # el config escrito es válido


def test_promover_csv_procesado(tmp_path):
    destino = promover(
        _run_falso(tmp_path), 11000, destino=tmp_path / "d", desde_crudo=False
    )
    cfg = yaml.safe_load((destino / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["datos"]["reconstruir_target"] is False


def test_promover_conserva_resultado_publico(tmp_path):
    run = _run_falso(tmp_path)
    destino = promover(run, 11000, destino=tmp_path / "d")
    p = destino / "entrega.json"
    info = json.loads(p.read_text())
    info["resultado_publico"] = 105.6
    p.write_text(json.dumps(info), encoding="utf-8")
    promover(run, 11000, destino=destino)
    assert json.loads(p.read_text())["resultado_publico"] == 105.6


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"estado": "failed"}, "estado"),
        ({"final": "pendiente"}, "etapa 'final'"),
        ({"envios": (10000,)}, "11000 envíos"),
    ],
)
def test_promover_rechaza_runs_invalidos(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        promover(_run_falso(tmp_path, **kw), 11000, destino=tmp_path / "d")
    assert not (tmp_path / "d").exists()


def test_dmeyf_work_cambia_la_carpeta_de_salidas(tmp_path):
    cod = "from competencia_1.pipeline.tracking import WORK; print(WORK)"
    env = {**os.environ, "DMEYF_WORK": str(tmp_path / "otro")}
    out = subprocess.run(
        [sys.executable, "-c", cod],
        capture_output=True,
        text=True,
        env=env,
        cwd=cfgmod.RAIZ,
        check=True,
    ).stdout.strip()
    assert Path(out) == tmp_path / "otro"
    sin = {k: v for k, v in os.environ.items() if k != "DMEYF_WORK"}
    out = subprocess.run(
        [sys.executable, "-c", cod],
        capture_output=True,
        text=True,
        env=sin,
        cwd=cfgmod.RAIZ,
        check=True,
    ).stdout.strip()
    assert Path(out) == cfgmod.RAIZ / "work" / "competencia_1"


def test_promover_un_ensamble_y_resolverlo_de_vuelta(tmp_path):
    from competencia_1.pipeline.final import resolver_conjunto

    run = _run_falso(tmp_path)
    miembros = [
        {"etiqueta": "t3", "params": {**PARAMS, "num_leaves": 9}},
        {"etiqueta": "t7", "params": {**PARAMS, "num_leaves": 5}},
    ]
    meta_p = run / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta["modelo"]["params_final"] = miembros
    meta_p.write_text(json.dumps(meta), encoding="utf-8")

    destino = promover(run, 11000, destino=tmp_path / "d")
    guardado = json.loads((destino / "params.json").read_text())
    assert guardado["ensamble"] == miembros and "params" not in guardado
    cfg = cfgmod.desde_dict(
        yaml.safe_load((destino / "config.yaml").read_text(encoding="utf-8"))
    )
    # las features se fijan por el mismo config: no se exige el fe_hash del json
    assert resolver_conjunto(cfg, "cualquier_hash") == [
        ("t3", miembros[0]["params"]),
        ("t7", miembros[1]["params"]),
    ]


def test_promover_toma_un_corte_no_registrado_desde_submits(tmp_path):
    run = _run_falso(tmp_path, envios=(11000,))
    (run / "submits").mkdir()
    csv = run / "submits" / "20260101-000000_exp_promedio_e10500.csv"
    csv.write_bytes(b"1" + bytes([10]) + b"2" + bytes([10]))
    destino = promover(run, 10500, destino=tmp_path / "d")
    info = json.loads((destino / "entrega.json").read_text())
    import hashlib

    assert info["sha256_esperado"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert info["csv_original"] == csv.name and info["envios"] == 10500


def test_promover_con_modelos_copia_y_configura_solo_predecir(tmp_path):
    run = _run_falso(tmp_path)
    (run / "modelos").mkdir()
    for n in ("s1", "s2"):
        (run / "modelos" / f"{n}.txt").write_text(f"modelo {n}", encoding="utf-8")
    destino = promover(run, 11000, destino=tmp_path / "d", con_modelos=True)
    assert sorted(p.name for p in (destino / "modelos").glob("*.txt")) == [
        "s1.txt",
        "s2.txt",
    ]
    cfg = yaml.safe_load((destino / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["final"]["modelos_desde"].endswith("/modelos")
    assert json.loads((destino / "entrega.json").read_text())["modelos_incluidos"] == 2
    cfgmod.desde_dict(cfg)
    # volver a promover sin modelos limpia los anteriores y vuelve al modo entrenar
    promover(run, 11000, destino=destino)
    assert not (destino / "modelos").exists()
    assert (
        yaml.safe_load((destino / "config.yaml").read_text())["final"]["modelos_desde"]
        is None
    )


def test_promover_con_modelos_exige_que_el_run_los_tenga(tmp_path):
    with pytest.raises(ValueError, match="modelos guardados"):
        promover(_run_falso(tmp_path), 11000, destino=tmp_path / "d", con_modelos=True)
    assert not (tmp_path / "d" / "config.yaml").exists()


def test_cargar_modelo_valida_existencia_y_cantidad_de_features(tmp_path):
    import lightgbm as lgb
    import numpy as np

    from competencia_1.pipeline.final import _cargar_modelo

    with pytest.raises(FileNotFoundError, match="falta el modelo"):
        _cargar_modelo(tmp_path / "no.txt", 3)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    b = lgb.train({"objective": "binary", "verbosity": -1}, lgb.Dataset(X, y), 5)
    ruta = tmp_path / "m.txt"
    b.save_model(str(ruta))
    assert _cargar_modelo(ruta, 3).num_feature() == 3
    with pytest.raises(ValueError, match="features"):
        _cargar_modelo(ruta, 4)
    # un modelo guardado y vuelto a cargar predice exactamente igual
    assert np.array_equal(_cargar_modelo(ruta, 3).predict(X), b.predict(X))
