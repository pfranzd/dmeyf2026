import json

import numpy as np
import polars as pl
import pytest

from competencia_1.pipeline import ablacion as ab
from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import tracking
from competencia_1.pipeline.features import columnas_seleccionadas
from competencia_1.pipeline.features.seleccion import (
    seleccionar_columnas,
)

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"
MANUAL = (
    "lgbm.manual={learning_rate: 0.1, num_iterations: 25, feature_fraction: 0.8, "
    "num_leaves: 8, min_data_in_leaf: 20}"
)
ORIG = [f"o{i}" for i in range(5)]
LAGS = [f"lag_1_o{i}" for i in range(3)]  # señal fuerte
RUIDO = [f"pr_o{i}" for i in range(3)]  # ruido puro


def _cfg(*ov):
    base = [
        MANUAL,
        "lgbm.fijos.num_threads=2",
        "dataset.undersampling=0.5",
        "ablacion.n_semillas=3",
        "optuna.ventana_meseta=101",
    ]
    return cfgmod.cargar_config(BASE_YAML, [*base, *ov])[0]


# ------------------------------------------------------- selección pura
COLS = [*ORIG, *LAGS, *RUIDO]
BLOQUES = {
    **dict.fromkeys(ORIG, "originales"),
    **dict.fromkeys(LAGS, "3_lags"),
    **dict.fromkeys(RUIDO, "2_rank_intrames"),
}


def test_seleccion_por_bloques_y_pseudobloques():
    assert seleccionar_columnas(COLS, BLOQUES) == COLS
    assert seleccionar_columnas(COLS, BLOQUES, excluir_bloques=["3_lags"]) == [
        *ORIG,
        *RUIDO,
    ]
    assert seleccionar_columnas(COLS, BLOQUES, excluir_bloques=["derivadas"]) == ORIG
    assert seleccionar_columnas(
        COLS, BLOQUES, solo_bloques=["originales", "3_lags"]
    ) == [*ORIG, *LAGS]
    assert seleccionar_columnas(COLS, BLOQUES, solo_bloques=["derivadas"]) == [
        *LAGS,
        *RUIDO,
    ]
    sel = seleccionar_columnas(COLS, BLOQUES, excluir_features=["o0", "pr_o1"])
    assert "o0" not in sel and "pr_o1" not in sel and len(sel) == len(COLS) - 2


def test_seleccion_rechaza_typos_y_vacios():
    with pytest.raises(ValueError, match="bloque inexistente"):
        seleccionar_columnas(COLS, BLOQUES, excluir_bloques=["3_lag"])
    with pytest.raises(ValueError, match="inexistentes"):
        seleccionar_columnas(COLS, BLOQUES, excluir_features=["zzz"])
    with pytest.raises(ValueError, match="0 features"):
        seleccionar_columnas(
            COLS, BLOQUES, solo_bloques=["originales"], excluir_bloques=["originales"]
        )
    with pytest.raises(ValueError, match="claves"):
        seleccionar_columnas([*COLS, "foto_mes"], {**BLOQUES, "foto_mes": "originales"})


# ------------------------------------------------------------- datos
@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    d = tmp_path_factory.mktemp("abl")
    rng = np.random.default_rng(5)
    filas = []
    for mes in (202103, 202104, 202105, 202106):
        n = 4000
        clase = rng.choice(["CONTINUA", "BAJA+1", "BAJA+2"], n, p=[0.96, 0.02, 0.02])
        pos = (clase == "BAJA+2")[:, None]
        cols = {c: rng.normal(size=n) + pos[:, 0] * 0.3 for c in ORIG}
        cols.update({c: rng.normal(size=n) + pos[:, 0] * 2.0 for c in LAGS})
        cols.update({c: rng.normal(size=n) for c in RUIDO})
        cols.update(
            numero_de_cliente=np.arange(n, dtype=np.int64) + 48_000_000,
            foto_mes=np.full(n, mes, dtype=np.int32),
            clase_ternaria=clase,
        )
        filas.append(pl.DataFrame(cols))
    p = d / "features.parquet"
    pl.concat(filas).write_parquet(p)
    pl.DataFrame(
        {
            "bloque": ["3_lags"] * len(LAGS) + ["2_rank_intrames"] * len(RUIDO),
            "feature": [*LAGS, *RUIDO],
            "expresion": "x",
            "origen": "x",
        }
    ).write_csv(d / "catalogo.csv")
    return p


def test_dataset_excluir_aplica_a_las_columnas_del_modelo(parquet):
    assert len(columnas_seleccionadas(_cfg(), parquet)) == len(COLS)
    sel = columnas_seleccionadas(
        _cfg("dataset.excluir_bloques=[2_rank_intrames]"), parquet
    )
    assert sel == [*ORIG, *LAGS]
    sel = columnas_seleccionadas(_cfg("dataset.excluir_features=[o0]"), parquet)
    assert "o0" not in sel
    with pytest.raises(ValueError, match="inexistente"):
        columnas_seleccionadas(_cfg("dataset.excluir_bloques=[8_canaritos]"), parquet)


def test_config_de_variantes():
    ov = "ablacion.variantes={a: {solo_bloques: null, excluir_bloques: [], excluir_features: []}}"
    cfg = _cfg(ov, "ablacion.referencia=a")
    assert list(cfgmod.variantes_ablacion(cfg)) == ["a"]
    # `null` quita una variante heredada de base.yaml
    cfg2 = _cfg(
        "ablacion.variantes={completo: null, b: {solo_bloques: null, excluir_bloques: [], excluir_features: []}}"
    )
    assert list(cfgmod.variantes_ablacion(cfg2)) == ["b"]
    with pytest.raises(ValueError, match="desconocidas"):
        _cfg("ablacion.variantes={a: {solo: [x]}}")
    with pytest.raises(ValueError, match="referencia"):
        _cfg(
            ov, "etapas.ablacion=true"
        )  # la referencia por defecto ('completo') no existe


def test_nombre_estudio_cambia_solo_si_hay_seleccion():
    from competencia_1.pipeline import modelo as md

    h = "abcdef012345"
    assert md.nombre_estudio(_cfg(), h) == md.nombre_estudio(
        _cfg("dataset.excluir_bloques=[]"), h
    )
    assert md.nombre_estudio(_cfg(), h) != md.nombre_estudio(
        _cfg("dataset.excluir_bloques=[3_lags]"), h
    )


# ------------------------------------------------------------ ablación
VARIANTES = {
    "completo": ([*ORIG, *LAGS, *RUIDO]),
    "sin_ruido": ([*ORIG, *LAGS]),
    "sin_lags": ([*ORIG, *RUIDO]),
}
PARAMS = {
    "learning_rate": 0.1,
    "num_iterations": 25,
    "feature_fraction": 0.8,
    "num_leaves": 8,
    "min_data_in_leaf": 20,
}


def test_evaluar_y_comparar_pareado(parquet):
    cfg = _cfg()
    semillas = [111623, 116239, 116423]
    res = ab.evaluar_variantes(cfg, parquet, COLS, VARIANTES, PARAMS, semillas)
    assert set(res) == set(VARIANTES) and res["sin_lags"]["n_features"] == len(
        ORIG
    ) + len(RUIDO)
    assert set(res["completo"]["valores"]) == set(semillas)
    assert all(len(v) == 2 for v in res["completo"]["por_fold"].values())  # 2 folds

    filas = {f["variante"]: f for f in ab.comparar(res, "completo", semillas)}
    ref = filas["completo"]
    assert ref["delta_media"] == 0 and ref["veredicto"] == "referencia"
    # quitar la familia con señal fuerte empeora en todas las semillas y en los dos folds
    sl = filas["sin_lags"]
    assert sl["delta_media"] < 0 and sl["semillas_mejores"] == 0
    assert all(x < 0 for x in sl["delta_por_fold"]) and sl["veredicto"] == "peor"
    # quitar solo ruido no puede ser una caída grande
    assert abs(filas["sin_ruido"]["delta_media"]) < abs(sl["delta_media"]) / 2

    # determinismo: misma corrida, mismos números
    res2 = ab.evaluar_variantes(cfg, parquet, COLS, VARIANTES, PARAMS, semillas)
    assert res2["sin_lags"]["valores"] == res["sin_lags"]["valores"]


def test_etapa_ablacion_completa(parquet, tmp_path, monkeypatch):
    cfg = _cfg(
        "ablacion.variantes={completo: {solo_bloques: null, excluir_bloques: [], excluir_features: []},"
        " sin_lags: {solo_bloques: null, excluir_bloques: [3_lags], excluir_features: []},"
        " solo_orig_y_lags: {solo_bloques: [originales, 3_lags], excluir_bloques: [], excluir_features: []}}",
        "etapas.ablacion=true",
    )
    monkeypatch.setattr(ab, "construir_features", lambda c: (parquet, "hash"))
    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(tracking, "RUNS_CSV", tmp_path / "runs.csv")
    (tmp_path / "runs").mkdir()
    with tracking.Run("t_abl", {"x": 1}) as run:
        ab.etapa_ablacion(cfg, run)
    det = json.loads((run.dir / "ablacion.json").read_text())
    assert det["referencia"] == "completo" and len(det["semillas"]) == 3
    assert det["columnas"]["solo_orig_y_lags"] == [*ORIG, *LAGS]
    tabla = pl.read_csv(run.dir / "ablacion.csv")
    assert tabla.height == 3 and set(tabla["veredicto"]) >= {"referencia"}
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["ablacion"]["variantes"]["sin_lags"]["veredicto"] == "peor"
    assert meta["ganancia_valid"] > 0


# ------------------------------------------------- veredicto, listas, flags
def _res(deltas_por_fold, std=0.5):
    """Resultados mínimos para `comparar`: 5 semillas, 2 folds, delta fijo contra la referencia."""
    sem = [1, 2, 3, 4, 5]
    ruido = [-std, -std / 2, 0.0, std / 2, std]
    base = {
        "n_features": 10,
        "valores": dict.fromkeys(sem, 100.0),
        "por_fold": {s: [100.0, 100.0] for s in sem},
        "valor_promedio_semillas": 100.0,
        "por_fold_promedio": [100.0, 100.0],
    }
    var = {
        "n_features": 5,
        "valores": {
            s: 100.0 + sum(deltas_por_fold) / 2 + r
            for s, r in zip(sem, ruido, strict=True)
        },
        "por_fold": {
            s: [100.0 + d + r for d in deltas_por_fold]
            for s, r in zip(sem, ruido, strict=True)
        },
        "valor_promedio_semillas": 100.0,
        "por_fold_promedio": [100.0, 100.0],
    }
    return {"ref": base, "v": var}, sem


@pytest.mark.parametrize(
    ("deltas", "esperado"),
    [
        ([10.0, 12.0], "mejor"),
        ([-10.0, -12.0], "peor"),
        ([-30.0, 8.0], "mixto"),  # efecto grande que cambia de signo según el mes
        ([0.1, -0.1], "indistinguible"),
    ],
)
def test_veredicto(deltas, esperado):
    res, sem = _res(deltas)
    filas = {f["variante"]: f for f in ab.comparar(res, "ref", sem)}
    assert filas["v"]["veredicto"] == esperado
    assert filas["ref"]["veredicto"] == "referencia"


def test_listas_de_features_desde_archivo(tmp_path):
    from competencia_1.pipeline.features.seleccion import expandir_lista

    (tmp_path / "l.txt").write_text("# comentario\no0\n\n  o1  \n")
    assert expandir_lista(["x", "archivo:l.txt"], tmp_path) == ["x", "o0", "o1"]
    assert expandir_lista(["a", "b"], tmp_path) == ["a", "b"]
    with pytest.raises(FileNotFoundError):
        expandir_lista(["archivo:no_existe.txt"], tmp_path)


def test_ignorar_fe_hash_solo_con_el_flag(tmp_path, caplog):
    from competencia_1.pipeline import modelo as md

    f = tmp_path / "p.json"
    f.write_text(json.dumps({"fe_hash": "otro", "params": {"num_leaves": 9}}))
    ref = f"final.params_desde=archivo:{f.as_posix()}"
    with pytest.raises(ValueError, match="fe_hash"):
        md.resolver_params(_cfg(ref), "actual")
    with caplog.at_level("WARNING"):
        assert md.resolver_params(
            _cfg(ref, "final.params_ignorar_fe_hash=true"), "actual"
        ) == {"num_leaves": 9}
    assert "params_ignorar_fe_hash" in caplog.text


# ------------------------------------------------------------ canaritos
def test_canaritos_detecta_ruido_y_señal(tmp_path, monkeypatch):
    from competencia_1.pipeline import canaritos as cn

    rng = np.random.default_rng(9)
    filas = []
    for mes in (202103, 202104, 202105, 202106):
        n = 4000
        clase = rng.choice(["CONTINUA", "BAJA+1", "BAJA+2"], n, p=[0.96, 0.02, 0.02])
        pos = clase == "BAJA+2"
        cols = {"senal": rng.normal(size=n) + pos * 2.0, "ruido": rng.normal(size=n)}
        cols.update({f"canarito_{i}": rng.normal(size=n) for i in range(1, 6)})
        cols.update(
            numero_de_cliente=np.arange(n, dtype=np.int64) + 48_000_000,
            foto_mes=np.full(n, mes, dtype=np.int32),
            clase_ternaria=clase,
        )
        filas.append(pl.DataFrame(cols))
    parquet = tmp_path / "features.parquet"
    pl.concat(filas).write_parquet(parquet)

    cfg = _cfg("ablacion.n_semillas=2")
    feats = ["senal", "ruido", *[f"canarito_{i}" for i in range(1, 6)]]
    imp = cn.importancias_promedio(cfg, parquet, feats, PARAMS, [111623, 116239])
    assert imp["gain_medio"].sum() == pytest.approx(1.0, abs=1e-6)
    tabla, umbral = cn.clasificar(imp)
    fila = {r["feature"]: r for r in tabla.iter_rows(named=True)}
    assert not fila["senal"]["bajo_umbral"]  # supera al mejor canarito
    assert fila["senal"]["ranking"] == 1
    assert fila["canarito_1"]["es_canarito"] and umbral == max(
        fila[f"canarito_{i}"]["gain_medio"] for i in range(1, 6)
    )

    imp_sin = imp.filter(~pl.col("feature").str.starts_with("canarito_"))
    with pytest.raises(ValueError, match="canaritos"):
        cn.clasificar(imp_sin)

    # etapa completa: escribe la lista de ruidosas y registra el resumen
    monkeypatch.setattr(cn, "construir_features", lambda c: (parquet, "hash"))
    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(tracking, "RUNS_CSV", tmp_path / "runs.csv")
    (tmp_path / "runs").mkdir()
    with tracking.Run("t_can", {"x": 1}) as run:
        cn.etapa_canaritos(cfg, run)
    lista = (run.dir / "features_ruidosas.txt").read_text().splitlines()
    assert lista[0].startswith("#") and "senal" not in lista[1:]
    assert not any(x.startswith("canarito_") for x in lista[1:])
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["canaritos"]["n_canaritos"] == 5
    assert (run.dir / "canaritos_bloques.csv").exists()
