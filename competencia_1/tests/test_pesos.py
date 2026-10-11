"""target.peso_baja1: pesos de BAJA+1 al entrenar (fijos o tuneados por Optuna)."""

import numpy as np
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import modelo as md
from competencia_1.pipeline import optimizacion as op
from competencia_1.pipeline.dataset import cargar_particion, pesos_filas
from competencia_1.tests.test_expansiva import (  # noqa: F401
    FE_HASH,
    FEATURES,
    FOLDS,
    _cfg,
    parquet,
)

ESPACIO_PESO = (
    "optuna.espacio={num_leaves: {tipo: int, low: 4, high: 16, log: false}, "
    "num_iterations: {tipo: int, low: 5, high: 20, log: false}, "
    "peso_baja1: {tipo: float, low: 0.0, high: 1.0, log: false}}"
)


def test_default_sin_pesos_y_nombre_de_estudio_intacto():
    cfg = _cfg(FOLDS)
    assert cfg.target.peso_baja1 == 1.0
    con = _cfg(FOLDS, "target.peso_baja1=0.5")
    assert md.nombre_estudio(cfg, FE_HASH) != md.nombre_estudio(con, FE_HASH)
    # una config congelada sin la clave se completa con 1.0
    d = cfgmod.cargar_dict(cfgmod.CONFIGS_DIR / "base.yaml")
    del d["target"]["peso_baja1"]
    assert cfgmod.desde_dict(d).target.peso_baja1 == 1.0


def test_validaciones():
    with pytest.raises(ValueError, match="peso_baja1"):
        _cfg("target.peso_baja1=-1")
    with pytest.raises(ValueError, match="peso_baja1"):
        _cfg(
            "optuna.espacio={num_iterations: {tipo: int, low: 5, high: 20, log: false}, "
            "peso_baja1: {tipo: int, low: 0, high: 2, log: false}}"
        )


def test_pesos_filas(parquet):  # noqa: F811
    part = cargar_particion(parquet, FEATURES, [202103], ["BAJA+1", "BAJA+2"])
    assert pesos_filas(part, 1.0) is None
    w = pesos_filas(part, 0.25)
    assert np.all(w[part.es_baja1] == 0.25) and np.all(w[~part.es_baja1] == 1.0)
    assert md.peso_baja1(_cfg(), {"peso_baja1": 0.3}) == 0.3
    assert md.peso_baja1(_cfg("target.peso_baja1=0.7"), {}) == 0.7
    p, _ = md.params_lgbm(_cfg(), {"num_iterations": 5, "peso_baja1": 0.3}, 1)
    assert "peso_baja1" not in p


def test_set_weight_sobre_dataset_reusado_equivale_a_construirlo(parquet):  # noqa: F811
    """Optuna reusa el Dataset binarizado y le cambia los pesos por trial: tiene que dar
    exactamente lo mismo que construirlo con esos pesos, y volver a 1 lo mismo que sin pesos."""
    cfg = _cfg(FOLDS)
    params = {"num_iterations": 15, "num_leaves": 8, "learning_rate": 0.1}
    preps = op.preparar_folds(cfg, parquet, FEATURES, cfg.semilla_maestra)
    base = op.puntuar_folds(cfg, preps, params, 7)
    con_peso = op.puntuar_folds(cfg, preps, {**params, "peso_baja1": 0.2}, 7)
    assert not np.allclose(base[1], con_peso[1])
    de_nuevo = op.puntuar_folds(cfg, preps, params, 7)
    for a, b in zip(base, de_nuevo, strict=True):
        assert np.array_equal(a, b)
    # referencia: Dataset construido desde cero con los pesos
    cfg_w = _cfg(FOLDS, "target.peso_baja1=0.2")
    preps_w = op.preparar_folds(cfg_w, parquet, FEATURES, cfg.semilla_maestra)
    ref = op.puntuar_folds(cfg_w, preps_w, params, 7)
    for a, b in zip(con_peso, ref, strict=True):
        assert np.array_equal(a, b)


def test_optuna_tunea_el_peso(parquet, tmp_path):  # noqa: F811
    cfg = _cfg(FOLDS, ESPACIO_PESO, "optuna.objetivo=auc", "optuna.n_trials=4")
    study, hechos = op.abrir_estudio(
        cfg, md.nombre_estudio(cfg, FE_HASH), tmp_path / "e.db"
    )
    preps = op.preparar_folds(cfg, parquet, FEATURES, cfg.semilla_maestra)
    op.correr_estudio(cfg, study, preps, hechos)
    pesos = {t.params["peso_baja1"] for t in study.trials}
    assert len(study.trials) == 4 and len(pesos) == 4
