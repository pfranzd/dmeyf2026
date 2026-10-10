"""Folds explícitos, objetivo AUC, pruning y ensamble de conjuntos de hiperparámetros."""

import itertools
import json

import numpy as np
import optuna
import polars as pl
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import final as fin
from competencia_1.pipeline import modelo as md
from competencia_1.pipeline import optimizacion as op
from competencia_1.pipeline.dataset import contar_filas
from competencia_1.pipeline.metricas import auc

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"
FE_HASH = "abcdef012345"
FOLDS = (
    "periodos.folds=[{train: [202103], valid: 202105}, "
    "{train: [202103, 202104], valid: 202106}]"
)
ESPACIO = (
    "optuna.espacio={num_leaves: {tipo: int, low: 4, high: 16, log: false}, "
    "num_iterations: {tipo: int, low: 5, high: 20, log: false}, "
    "learning_rate: {tipo: float, low: 0.05, high: 0.3, log: true}}"
)


def _cfg(*ov):
    base = ["lgbm.fijos.num_threads=2", "dataset.undersampling=0.5"]
    return cfgmod.cargar_config(BASE_YAML, [*base, *ov])[0]


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    rng = np.random.default_rng(3)
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
    p = tmp_path_factory.mktemp("exp") / "f.parquet"
    pl.concat(filas).write_parquet(p)
    return p


FEATURES = [f"x{i}" for i in range(12)]


# ---------------------------------------------------------------- folds explícitos


def test_folds_explicitos_en_el_orden_declarado():
    cfg = _cfg(FOLDS)
    assert [(f.train, f.valid) for f in cfgmod.folds(cfg)] == [
        ((202103,), 202105),
        ((202103, 202104), 202106),
    ]


def test_sin_folds_explicitos_se_derivan_como_siempre():
    assert [(f.train, f.valid) for f in cfgmod.folds(_cfg())] == [
        ((202104,), 202106),
        ((202103,), 202105),
    ]


def test_folds_explicitos_rechazan_gap_1_y_formas_invalidas():
    with pytest.raises(ValueError, match="gap insuficiente"):
        _cfg("periodos.folds=[{train: [202103, 202104], valid: 202105}]")
    with pytest.raises(ValueError, match="lista vacía"):
        _cfg("periodos.folds=[]")
    with pytest.raises(ValueError, match="cada fold"):
        _cfg("periodos.folds=[{train: [202103]}]")


def test_el_estudio_cambia_con_los_folds_y_con_el_objetivo():
    sin = md.nombre_estudio(_cfg(), FE_HASH)
    con = md.nombre_estudio(_cfg(FOLDS), FE_HASH)
    auc_ = md.nombre_estudio(_cfg(FOLDS, "optuna.objetivo=auc"), FE_HASH)
    assert len({sin, con, auc_}) == 3
    # el pruning no cambia el significado de un trial: mismo estudio
    assert md.nombre_estudio(_cfg(FOLDS, "optuna.pruning=true"), FE_HASH) == con


def test_objetivo_ganancia_no_cambia_el_nombre_de_estudios_previos():
    """Nombre calculado con el código anterior al cambio (huella sin `objetivo`)."""
    cfg = _cfg()
    fijos = {k: v for k, v in cfg.lgbm.fijos.items() if k != "num_threads"}
    huella = {
        "folds": [(list(f.train), f.valid) for f in cfgmod.folds(cfg)],
        "positivos": cfg.target.positivos,
        "undersampling": cfg.dataset.undersampling,
        "espacio": cfg.optuna.espacio,
        "fijos": fijos,
        "ventana": cfg.optuna.ventana_meseta,
        "semilla": cfg.semilla_maestra,
    }
    esperado = f"{cfg.experimento}_{FE_HASH[:8]}_{cfgmod.hash_dict(huella, 8)}"
    assert md.nombre_estudio(cfg, FE_HASH) == esperado


def test_config_congelada_sin_claves_posteriores_sigue_cargando():
    d = cfgmod.cargar_dict(BASE_YAML)
    del d["periodos"]["folds"]
    del d["optuna"]["objetivo"], d["optuna"]["pruning"]
    cfg = cfgmod.desde_dict(d)
    assert cfg.periodos.folds is None
    assert (cfg.optuna.objetivo, cfg.optuna.pruning) == ("ganancia", False)
    d["optuna"]["typo_inventado"] = 1  # lo desconocido sigue siendo error
    with pytest.raises(ValueError, match="desconocidas"):
        cfgmod.desde_dict(d)


def test_validaciones_de_objetivo_y_params_desde():
    with pytest.raises(ValueError, match="objetivo"):
        _cfg("optuna.objetivo=f1")
    with pytest.raises(ValueError, match="top:<k>"):
        _cfg("final.params_desde=top:0")
    with pytest.raises(ValueError, match="reescalar_min_data"):
        _cfg("final.reescalar_min_data=casi")
    assert _cfg("final.reescalar_min_data=filas").final.reescalar_min_data == "filas"


# ---------------------------------------------------------------- AUC


def _auc_por_pares(y, s):
    pos, neg = s[y], s[~y]
    v = [(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)]
    return float(np.mean(v))


def test_auc_casos_limite():
    y = np.array([True, True, False, False, False])
    assert auc(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == 1.0
    assert auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 0.0
    assert auc(y, np.full(5, 0.5)) == 0.5  # todo empatado


def test_auc_coincide_con_el_calculo_por_pares_con_empates():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.2
    s = np.round(rng.random(200), 1)  # muchos empates
    assert auc(y, s) == pytest.approx(_auc_por_pares(y, s))


def test_auc_exige_las_dos_clases():
    with pytest.raises(ValueError, match="AUC indefinido"):
        auc(np.zeros(4, dtype=bool), np.arange(4.0))


def test_valor_objetivo_auc_es_media_simple_y_ganancia_no_cambia():
    a = {"n": 100, "ganancia_meseta": 100.0, "auc": 0.80}
    b = {"n": 200, "ganancia_meseta": 400.0, "auc": 0.90}
    assert op.valor_objetivo([a, b], "auc") == pytest.approx(0.85)
    assert op.valor_objetivo([a, b]) == pytest.approx(225.0)
    with pytest.raises(ValueError, match="objetivo"):
        op.valor_objetivo([a], "otro")


# ---------------------------------------------------------------- optuna con pruning + top-k


@pytest.fixture(scope="module")
def estudio(parquet, tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "e.db"
    cfg = _cfg(
        FOLDS,
        ESPACIO,
        "optuna.objetivo=auc",
        "optuna.pruning=true",
        "optuna.n_trials=16",
    )
    study, hechos = op.abrir_estudio(cfg, md.nombre_estudio(cfg, FE_HASH), db)
    preps = op.preparar_folds(cfg, parquet, FEATURES, cfg.semilla_maestra)
    assert [p.fold.valid for p in preps] == [202105, 202106]
    op.correr_estudio(cfg, study, preps, hechos)
    return cfg, db, study


def test_optuna_auc_con_pruning(estudio):
    study = estudio[2]
    estados = {t.state for t in study.trials}
    assert estados <= {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    completos = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(study.trials) == 16 and len(completos) >= 10
    for t in completos:
        assert len(t.user_attrs["auc_folds"]) == 2
        assert t.value == pytest.approx(np.mean(t.user_attrs["auc_folds"]))
    assert 0.5 < study.best_value <= 1.0
    # los podados quedan con el primer fold nada más
    for t in study.trials:
        if t.state.name == "PRUNED":
            assert len(t.user_attrs["auc_folds"]) == 1


def test_reanudar_cuenta_los_podados(estudio):
    cfg, db, _ = estudio
    _, hechos = op.abrir_estudio(cfg, md.nombre_estudio(cfg, FE_HASH), db)
    assert hechos == 16  # completos + podados: nada más para correr


def test_resolver_conjunto_top_k(estudio, monkeypatch):
    _, db, study = estudio
    monkeypatch.setattr(fin, "DB_PATH", db)
    cfg_f = _cfg(FOLDS, ESPACIO, "optuna.objetivo=auc", "final.params_desde=top:3")
    conjuntos = fin.resolver_conjunto(cfg_f, FE_HASH)
    top = op.top_trials(study, 3)
    assert [e for e, _ in conjuntos] == [f"t{t.number}" for t in top]
    p0 = conjuntos[0][1]
    assert p0["num_leaves"] == top[0].params["num_leaves"]
    assert (
        p0["feature_fraction"] == cfg_f.lgbm.manual["feature_fraction"]
    )  # lo fijo viene de manual
    with pytest.raises(ValueError, match="resolver_conjunto"):
        md.resolver_params(cfg_f, FE_HASH)


def test_resolver_conjunto_desde_archivo_de_ensamble(tmp_path):
    ruta = tmp_path / "params.json"
    miembros = [
        {"etiqueta": "t3", "params": {"num_leaves": 9, "num_iterations": 10}},
        {"etiqueta": "t7", "params": {"num_leaves": 5, "num_iterations": 20}},
    ]
    ruta.write_text(
        json.dumps({"fe_hash": FE_HASH, "ensamble": miembros}), encoding="utf-8"
    )
    cfg = _cfg(f"final.params_desde=archivo:{ruta.as_posix()}")
    assert fin.resolver_conjunto(cfg, FE_HASH) == [
        ("t3", miembros[0]["params"]),
        ("t7", miembros[1]["params"]),
    ]
    with pytest.raises(ValueError, match="ensamble"):
        md.resolver_params(cfg, FE_HASH)
    with pytest.raises(ValueError, match="fe_hash"):
        fin.resolver_conjunto(cfg, "otras_features")


def test_un_solo_conjunto_conserva_el_flujo_de_siempre():
    cfg = _cfg()  # params_desde=manual
    assert fin.resolver_conjunto(cfg, FE_HASH) == [("", dict(cfg.lgbm.manual))]


# ---------------------------------------------------------------- reescalado por filas


def test_contar_filas_con_y_sin_undersampling(parquet):
    df = pl.read_parquet(parquet, columns=["foto_mes", "clase_ternaria"])
    df = df.filter(pl.col("foto_mes").is_in([202103, 202104]))
    n_baja = df["clase_ternaria"].is_in(["BAJA+1", "BAJA+2"]).sum()
    assert contar_filas(parquet, [202103, 202104]) == df.height
    assert contar_filas(parquet, [202103, 202104], 0.25) == pytest.approx(
        n_baja + 0.25 * (df.height - n_baja)
    )


def test_factor_filas_es_final_sobre_train_del_ultimo_fold(parquet):
    cfg = _cfg(FOLDS, "dataset.undersampling=0.25")
    meses = [202103, 202104, 202105, 202106]
    esperado = contar_filas(parquet, meses) / contar_filas(
        parquet, [202103, 202104], 0.25
    )
    assert fin._factor_filas(cfg, parquet, meses) == pytest.approx(esperado)
    assert esperado > 4  # 4 meses completos contra 2 meses al 25 %
    p = md.escalar_min_data_factor({"min_data_in_leaf": 100, "x": 1}, 7.4)
    assert p == {"min_data_in_leaf": 740, "x": 1}
    assert md.escalar_min_data_factor({"min_data_in_leaf": 100}, 1) == {
        "min_data_in_leaf": 100
    }
