import json

import numpy as np
import polars as pl
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import estabilidad as est
from competencia_1.pipeline import modelo as md
from competencia_1.pipeline import optimizacion as op
from competencia_1.pipeline import tracking
from competencia_1.pipeline.simulacion import mascara_top, simular_public_private
from dmeyf.metrics import GANANCIA_ACIERTO

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"
FE_HASH = "abcdef012345"
ESPACIO = (
    "optuna.espacio={num_leaves: {tipo: int, low: 4, high: 16, log: false}, "
    "num_iterations: {tipo: int, low: 5, high: 20, log: false}, "
    "learning_rate: {tipo: float, low: 0.05, high: 0.3, log: true}}"
)


def _cfg(*ov):
    base = [
        ESPACIO,
        "optuna.n_trials=3",
        "lgbm.fijos.num_threads=2",
        "dataset.undersampling=0.5",
        "estabilidad.top_k=2",
        "estabilidad.n_semillas=3",
        "estabilidad.n_simulaciones=20",
    ]
    return cfgmod.cargar_config(BASE_YAML, [*base, *ov])[0]


# ------------------------------------------------------- selección
def test_estadisticos_y_criterios():
    e = est.estadisticos([10.0, 20.0, 30.0, 100.0])
    assert e["mediana"] == 25.0 and e["media"] == 40.0
    assert e["min"] == 10.0 and e["max"] == 100.0
    assert est.puntaje(e, "mediana") == 25.0
    assert est.puntaje(e, "media_menos_desvio") == pytest.approx(40.0 - e["std"])
    with pytest.raises(ValueError):
        est.puntaje(e, "maximo")


def test_elegir_no_premia_el_valor_afortunado():
    # trial 0: un pico afortunado (max alto) pero mediana baja; trial 1: estable
    res = {
        0: {"estadisticos": est.estadisticos([100, 100, 100, 100, 400])},
        1: {"estadisticos": est.estadisticos([180, 190, 200, 190, 185])},
    }
    assert est.elegir(res, "mediana") == 1
    assert est.elegir(res, "media_menos_desvio") == 1
    # empate exacto -> el trial más antiguo
    igual = {
        5: {"estadisticos": est.estadisticos([1, 2, 3])},
        2: {"estadisticos": est.estadisticos([1, 2, 3])},
    }
    assert est.elegir(igual, "mediana") == 2


# ------------------------------------------------------ simulación
def _mes(n=20_000, pos=300, seed=0):
    rng = np.random.default_rng(seed)
    es = np.zeros(n, dtype=bool)
    es[rng.choice(n, pos, replace=False)] = True
    return es, rng


def test_mascara_top():
    m = mascara_top(np.array([0.1, 0.9, 0.5, 0.9]), 2)
    assert m.tolist() == [
        False,
        True,
        False,
        True,
    ]  # empate: respeta el orden de entrada
    with pytest.raises(ValueError):
        mascara_top(np.array([0.1]), 2)


def test_modelo_perfecto_nunca_pierde_en_el_private():
    es, rng = _mes()
    r = simular_public_private(
        es,
        {"perfecto": es.astype(float), "azar": rng.random(len(es))},
        int(es.sum()),
        50,
        7,
    )
    assert r["prob_ganador_public_pierde_private"] == 0.0
    assert r["arrepentimiento_private_medio"] == 0.0
    perf = r["candidatos"]["perfecto"]
    esperado = es.sum() * GANANCIA_ACIERTO  # escalado a un mes completo
    assert perf["public_media"] == pytest.approx(esperado, rel=0.02)
    assert perf["private_media"] == pytest.approx(esperado, rel=0.02)


def test_inversion_public_private_segun_la_distancia_entre_modelos():
    """Público y privado parten el MISMO mes: sus aciertos son complementarios.

    Con ganancias totales casi iguales, si un modelo gana en el público casi seguro pierde
    en el privado (la diferencia total es fija). Con una diferencia clara, no se invierte.
    """
    es, rng = _mes(seed=1)
    base = es * 1.0
    parecido_a = base + rng.normal(scale=2.0, size=len(es))
    parecido_b = base + rng.normal(scale=2.0, size=len(es))
    r = simular_public_private(es, {"a": parecido_a, "b": parecido_b}, 600, 200, 3)
    assert r["prob_ganador_public_pierde_private"] > 0.8
    assert r["arrepentimiento_private_medio"] > 0
    assert r["candidatos"]["a"]["private_std"] > 0

    claramente_mejor = base + rng.normal(scale=0.5, size=len(es))
    mucho_peor = base + rng.normal(scale=4.0, size=len(es))
    r2 = simular_public_private(
        es, {"mejor": claramente_mejor, "peor": mucho_peor}, 600, 200, 3
    )
    assert r2["prob_ganador_public_pierde_private"] < 0.3
    assert (
        r2["candidatos"]["mejor"]["private_media"]
        > r2["candidatos"]["peor"]["private_media"]
    )


def test_simulacion_determinista():
    es, rng = _mes(seed=2)
    sc = {"a": rng.random(len(es)), "b": rng.random(len(es))}
    r1 = simular_public_private(es, sc, 500, 30, 11)
    assert r1 == simular_public_private(es, sc, 500, 30, 11)
    assert r1 != simular_public_private(es, sc, 500, 30, 12)


# ------------------------------------------------------ integración
@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    rng = np.random.default_rng(1)
    filas = []
    for mes in (202103, 202104, 202105, 202106):
        n = 4000
        clase = rng.choice(["CONTINUA", "BAJA+1", "BAJA+2"], n, p=[0.96, 0.02, 0.02])
        x = rng.normal(size=(n, 12)) + (clase == "BAJA+2")[:, None] * np.linspace(
            1.5, 0, 12
        )
        d = {f"x{i}": x[:, i] for i in range(12)}
        d.update(
            numero_de_cliente=np.arange(n, dtype=np.int64) + 48_000_000,
            foto_mes=np.full(n, mes, dtype=np.int32),
            clase_ternaria=clase,
        )
        filas.append(pl.DataFrame(d))
    p = tmp_path_factory.mktemp("est") / "f.parquet"
    pl.concat(filas).write_parquet(p)
    return p


@pytest.fixture
def entorno(parquet, tmp_path, monkeypatch):
    """Estudio de 3 trials en un SQLite temporal + carpetas de params/runs redirigidas."""
    cfg = _cfg()
    db = tmp_path / "e.db"
    feats = [f"x{i}" for i in range(12)]
    study, hechos = op.abrir_estudio(cfg, md.nombre_estudio(cfg, FE_HASH), db)
    preps = op.preparar_folds(cfg, parquet, feats, cfg.semilla_maestra)
    op.correr_estudio(cfg, study, preps, hechos)
    monkeypatch.setattr(est, "construir_features", lambda c: (parquet, FE_HASH))
    monkeypatch.setattr(est, "DB_PATH", db)
    monkeypatch.setattr(est, "PARAMS_DIR", tmp_path / "params")
    monkeypatch.setattr(md, "PARAMS_DIR", tmp_path / "params")
    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(tracking, "RUNS_CSV", tmp_path / "runs.csv")
    (tmp_path / "runs").mkdir()
    return cfg, tmp_path


def test_etapa_completa_y_export_reutilizable(entorno):
    cfg, tmp = entorno
    with tracking.Run("t_estab", {"x": 1}) as run:
        est.etapa_estabilidad(cfg, run)
    det = json.loads((run.dir / "estabilidad.json").read_text())
    assert len(det["trials"]) == 2 and det["elegido"] in map(int, det["trials"])
    for r in det["trials"].values():
        assert len(r["valores_por_semilla"]) == 3  # N semillas
        e = r["estadisticos"]
        assert e["min"] <= e["mediana"] <= e["max"]
        assert r["optimismo_optuna"] == pytest.approx(r["valor_optuna"] - e["media"])
        assert r["beneficio_semillerio"] == pytest.approx(
            r["valor_promedio_semillas"] - e["media"]
        )
    # una simulación public/private por fold, con trials y semillas
    assert [s["valid"] for s in det["public_private"]] == [202106, 202105]
    ps = det["public_private"][0]
    assert "promedio" in ps["entre_semillas"]["candidatos"]
    assert 0 <= ps["entre_trials"]["prob_ganador_public_pierde_private"] <= 1
    assert (run.dir / "estabilidad_trials.csv").exists()
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["estabilidad"]["elegido"] == det["elegido"]

    # el export se reutiliza con estable:auto y respeta el chequeo de fe_hash
    cfg_f = _cfg("final.params_desde=estable:auto")
    params = md.resolver_params(cfg_f, FE_HASH)
    assert params["num_leaves"] >= 4 and params["feature_fraction"] == 0.8
    guardado = json.loads(next((tmp / "params").glob("*__estable.json")).read_text())
    assert guardado["fe_hash"] == FE_HASH and guardado["trial"] == det["elegido"]
    with pytest.raises(FileNotFoundError):  # otras features -> otro estudio
        md.resolver_params(cfg_f, FE_HASH[::-1])


def test_top_k_se_acota_a_los_trials_disponibles(entorno):
    cfg, tmp = entorno
    study = est.cargar_estudio(md.nombre_estudio(cfg, FE_HASH), tmp / "e.db")
    assert len(est.top_trials(study, 10)) == 3
    vals = [t.value for t in est.top_trials(study, 3)]
    assert vals == sorted(vals, reverse=True)


def test_estudio_inexistente_da_error_claro(entorno):
    _, tmp = entorno
    with pytest.raises(FileNotFoundError, match="optuna"):
        est.cargar_estudio("no_existe", tmp / "e.db")


def test_validacion_de_config_de_estabilidad():
    with pytest.raises(ValueError, match="criterio"):
        _cfg("estabilidad.criterio=maximo")
    with pytest.raises(ValueError, match="estabilidad"):
        _cfg("estabilidad.n_semillas=21")
