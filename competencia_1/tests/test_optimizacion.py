import json

import numpy as np
import optuna
import polars as pl
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import modelo as md
from competencia_1.pipeline import optimizacion as op

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"
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
    ]
    return cfgmod.cargar_config(BASE_YAML, [*base, *ov])[0]


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
    p = tmp_path_factory.mktemp("opt") / "f.parquet"
    pl.concat(filas).write_parquet(p)
    return p


FEATURES = [f"x{i}" for i in range(12)]


def _estudio(cfg, parquet, db, n=None):
    nombre = md.nombre_estudio(cfg, "abcdef012345")
    study, hechos = op.abrir_estudio(cfg, nombre, db)
    preps = op.preparar_folds(cfg, parquet, FEATURES, cfg.semilla_maestra)
    op.correr_estudio(cfg, study, preps, hechos)
    return study


def test_sugerir_respeta_espacio_y_tipos():
    cfg = _cfg()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(30):
        t = study.ask()
        p = op.sugerir(t, cfg.optuna.espacio)
        assert isinstance(p["num_leaves"], int) and 4 <= p["num_leaves"] <= 16
        assert 5 <= p["num_iterations"] <= 20
        assert (
            isinstance(p["learning_rate"], float) and 0.05 <= p["learning_rate"] <= 0.3
        )
        study.tell(t, 0.0)


def test_valor_objetivo_normaliza_por_clientes():
    a = {"n": 100, "ganancia_meseta": 100.0}
    b = {"n": 200, "ganancia_meseta": 200.0}  # misma ganancia por cliente
    assert op.valor_objetivo([a, b]) == pytest.approx(150.0)


def test_nombre_estudio_depende_de_lo_que_invalida_trials():
    h = "abcdef012345"
    n0 = md.nombre_estudio(_cfg(), h)
    assert n0 == md.nombre_estudio(_cfg(), h)
    assert n0 == md.nombre_estudio(_cfg("lgbm.fijos.num_threads=7"), h)  # irrelevante
    assert n0 != md.nombre_estudio(_cfg("dataset.undersampling=0.2"), h)
    assert n0 != md.nombre_estudio(_cfg("target.positivos=[BAJA+2]"), h)
    assert n0 != md.nombre_estudio(_cfg(), "otrohash00000")
    assert md.nombre_estudio(_cfg("optuna.study_name=mio"), h) == "mio"


def test_estudio_se_reanuda_y_exporta(parquet, tmp_path):
    db = tmp_path / "e.db"
    s1 = _estudio(_cfg(), parquet, db)
    assert len(s1.trials) == 3
    # mismo objetivo: no corre nada nuevo
    s2 = _estudio(_cfg(), parquet, db)
    assert len(s2.trials) == 3
    # objetivo mayor: corre solo los que faltan
    s3 = _estudio(_cfg("optuna.n_trials=5"), parquet, db)
    assert len(s3.trials) == 5 and all(t.state.name == "COMPLETE" for t in s3.trials)
    # la reanudación no repite los sorteos de antes
    params = [json.dumps(t.params, sort_keys=True) for t in s3.trials]
    assert len(set(params)) == 5

    info = op.exportar_mejores(
        _cfg("optuna.n_trials=5"), s3, "abcdef012345", tmp_path / "params"
    )
    guardado = json.loads((tmp_path / "params" / f"{s3.study_name}.json").read_text())
    assert (
        guardado["fe_hash"] == "abcdef012345" and guardado["params"] == info["params"]
    )
    assert (
        guardado["params"]["feature_fraction"] == 0.8
    )  # lo que no se optimiza viene del manual
    assert guardado["valor"] == s3.best_value
    assert len(op.tabla_trials(s3)) == 5


def test_estudio_reproducible_con_la_misma_semilla(parquet, tmp_path):
    a = _estudio(_cfg(), parquet, tmp_path / "a.db")
    b = _estudio(_cfg(), parquet, tmp_path / "b.db")
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == [t.value for t in b.trials]


def test_dataset_reutilizado_da_lo_mismo_que_uno_nuevo(parquet):
    cfg = _cfg()
    preps = op.preparar_folds(cfg, parquet, FEATURES, cfg.semilla_maestra)
    params = {
        **cfg.lgbm.manual,
        "num_leaves": 8,
        "num_iterations": 10,
        "min_data_in_leaf": 20,
    }
    r1 = op.evaluar_trial(cfg, preps, params, 111623)
    r2 = op.evaluar_trial(cfg, preps, params, 111623)  # mismo Dataset, 2da vez
    assert [r["ganancia_meseta"] for r in r1] == [r["ganancia_meseta"] for r in r2]


def test_resolver_params_optuna_auto(parquet, tmp_path, monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(md, "PARAMS_DIR", tmp_path / "params")
    monkeypatch.setattr(op, "PARAMS_DIR", tmp_path / "params")
    s = _estudio(cfg, parquet, tmp_path / "e.db")
    op.exportar_mejores(cfg, s, "abcdef012345", tmp_path / "params")
    cfg_final = _cfg("final.params_desde=optuna:auto")
    assert md.resolver_params(cfg_final, "abcdef012345") == {
        **cfg.lgbm.manual,
        **s.best_trial.params,
    }
    # otras features + optuna:auto -> es OTRO estudio, que no existe
    with pytest.raises(FileNotFoundError):
        md.resolver_params(cfg_final, "abcdef012345"[::-1])
    # estudio pedido por nombre pero con otras features -> error explícito de fe_hash
    por_nombre = _cfg(f"final.params_desde=optuna:{s.study_name}")
    with pytest.raises(ValueError, match="fe_hash"):
        md.resolver_params(por_nombre, "otrohash00000")
