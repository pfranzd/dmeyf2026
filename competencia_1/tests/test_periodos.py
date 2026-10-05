import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.periodos import (
    derivar_folds,
    derivar_meses_final,
    mes_offset,
    rango_meses,
    validar_gap,
    validar_target_completo,
)
from competencia_1.pipeline.semillas import generar_semillas
from config.semillas import SEMILLAS


def test_mes_offset_cruza_anio():
    assert mes_offset(202108, -2) == 202106
    assert mes_offset(202101, -1) == 202012
    assert mes_offset(202012, 1) == 202101
    assert mes_offset(202108, -20) == 201912


def test_rango_meses():
    assert rango_meses(202011, 202102) == [202011, 202012, 202101, 202102]


def test_folds_default():
    folds = derivar_folds(202108, 2, 1, 2)
    assert [(f.train, f.valid) for f in folds] == [
        ((202104,), 202106),
        ((202103,), 202105),
    ]
    assert derivar_meses_final(202108, 2, 1) == [202106]
    assert derivar_meses_final(202108, 2, 3) == [202104, 202105, 202106]


def test_validar_gap():
    validar_gap([202104], 202106, 2)
    with pytest.raises(ValueError):
        validar_gap([202105], 202106, 2)


def test_target_incompleto():
    with pytest.raises(ValueError, match="202107"):
        validar_target_completo([202106, 202107], {202106}, "final")


def test_semillas():
    assert generar_semillas(1, 3) == SEMILLAS[:3]
    s20 = generar_semillas(111623, 20)
    assert len(set(s20)) == 20 and s20[:5] == SEMILLAS
    assert s20 == generar_semillas(111623, 20)
    assert all(100003 <= s <= 999983 for s in s20)


def test_base_yaml_valida_y_overrides():
    cfg, _ = cfgmod.cargar_config(
        cfgmod.CONFIGS_DIR / "base.yaml", ["final.n_semillas=7"]
    )
    assert cfg.final.n_semillas == 7 and len(cfgmod.semillas_finales(cfg)) == 7
    with pytest.raises(ValueError):
        cfgmod.cargar_config(cfgmod.CONFIGS_DIR / "base.yaml", ["periodos.gap=1"])
    with pytest.raises(ValueError):
        cfgmod.cargar_config(cfgmod.CONFIGS_DIR / "base.yaml", ["final.n_semilas=3"])
    with pytest.raises(ValueError):
        cfgmod.cargar_config(cfgmod.CONFIGS_DIR / "base.yaml", ["final.n_semillas=21"])
