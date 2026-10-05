import json

import numpy as np
import polars as pl
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import modelo as md
from competencia_1.pipeline.dataset import (
    azar_uniforme,
    cargar_particion,
    validar_features,
)
from dmeyf.metrics import (
    COSTO_ESTIMULO,
    GANANCIA_ACIERTO,
    curva_ganancia,
    ganancia_meseta,
)

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"


def _cfg(*ov):
    return cfgmod.cargar_config(BASE_YAML, list(ov))[0]


# ------------------------------------------------------------- métricas
def test_curva_ganancia_ordena_por_score():
    curva = curva_ganancia([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    # orden por score: pos, pos, neg, neg
    assert curva.tolist() == [
        GANANCIA_ACIERTO,
        2 * GANANCIA_ACIERTO,
        2 * GANANCIA_ACIERTO - COSTO_ESTIMULO,
        2 * GANANCIA_ACIERTO - 2 * COSTO_ESTIMULO,
    ]


def test_curva_empates_estable_y_nan():
    a = curva_ganancia([1, 0, 0], [0.5, 0.5, 0.5])
    assert a[0] == GANANCIA_ACIERTO  # el empate respeta el orden del input
    with pytest.raises(ValueError):
        curva_ganancia([1, 0], [0.5, np.nan])


def test_meseta_ignora_pico_aislado():
    n = 2001
    curva = np.concatenate([np.linspace(0, 1000, 1000), np.linspace(1000, 0, n - 1000)])
    curva[200] = 5000  # pico espurio
    valor, envios = ganancia_meseta(curva, ventana=101)
    assert 900 < valor < 1001 and abs(envios - 1000) < 60
    assert ganancia_meseta([5.0], 401) == (5.0, 1)


# ------------------------------------------------------------- dataset
def test_azar_uniforme_determinista_y_uniforme():
    ids = np.arange(48_000_000, 48_200_000, dtype=np.int64)
    meses = np.full(ids.size, 202106, dtype=np.int32)
    a, b = azar_uniforme(ids, meses, 7), azar_uniforme(ids, meses, 7)
    assert (a == b).all() and (a >= 0).all() and (a < 1).all()
    assert abs(a.mean() - 0.5) < 0.005 and abs((a <= 0.1).mean() - 0.1) < 0.005
    assert (azar_uniforme(ids, meses, 8) != a).any()


@pytest.fixture
def parquet_chico(tmp_path):
    rng = np.random.default_rng(0)
    n = 20_000
    clase = rng.choice(["CONTINUA", "BAJA+1", "BAJA+2"], n, p=[0.98, 0.01, 0.01])
    df = pl.DataFrame(
        {
            "numero_de_cliente": np.arange(n, dtype=np.int64) + 48_000_000,
            "foto_mes": np.full(n, 202106, dtype=np.int32),
            "clase_ternaria": clase,
            "x1": rng.normal(size=n) + (clase == "BAJA+2") * 2,
            "x2": rng.integers(0, 5, n),
        }
    ).with_columns(
        pl.when(pl.col("x2") == 4).then(None).otherwise(pl.col("x2")).alias("x2")
    )
    p = tmp_path / "f.parquet"
    df.write_parquet(p)
    return p


def test_undersampling_conserva_todas_las_bajas(parquet_chico):
    full = cargar_particion(parquet_chico, ["x1", "x2"], [202106], ["BAJA+2"])
    sub = cargar_particion(
        parquet_chico, ["x1", "x2"], [202106], ["BAJA+2"], undersampling=0.1, seed=3
    )
    assert sub.es_baja2.sum() == full.es_baja2.sum()  # todas las BAJA+2
    assert len(sub) < 0.2 * len(full)
    sub2 = cargar_particion(
        parquet_chico, ["x1", "x2"], [202106], ["BAJA+2"], undersampling=0.1, seed=3
    )
    assert (sub.ids == sub2.ids).all()  # determinístico
    assert np.isnan(full.X[:, 1]).any() and full.X.dtype == np.float32


def test_target_binario_segun_positivos(parquet_chico):
    b2 = cargar_particion(parquet_chico, ["x1"], [202106], ["BAJA+2"])
    b12 = cargar_particion(parquet_chico, ["x1"], [202106], ["BAJA+1", "BAJA+2"])
    assert b12.y.sum() > b2.y.sum() and b12.es_baja2.sum() == b2.es_baja2.sum()
    sin = cargar_particion(parquet_chico, ["x1"], [202106], [], con_target=False)
    assert sin.y is None and sin.es_baja2 is None


def test_claves_no_pueden_ser_features():
    for k in ("numero_de_cliente", "foto_mes", "clase_ternaria"):
        with pytest.raises(ValueError, match="no pueden ser features"):
            validar_features(["x1", k])


# --------------------------------------------------------------- modelo
def test_entrenamiento_reproducible_y_semilla_importa():
    # con pocas features feature_fraction no tiene entre qué sortear: se usan 30
    cfg = _cfg()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 30)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] + rng.normal(size=20_000)) > 2.5).astype(np.int8)
    features = [f"x{i}" for i in range(30)]
    params = {
        "learning_rate": 0.1,
        "num_iterations": 30,
        "feature_fraction": 0.5,
        "num_leaves": 8,
        "min_data_in_leaf": 20,
    }

    def pred(seed):
        p, n = md.params_lgbm(cfg, params, seed)
        return md.predecir(md.entrenar_lgbm(X, y, features, p, n), X)

    assert (pred(111623) == pred(111623)).all()  # misma semilla: idéntico
    assert (
        pred(111623) != pred(116239)
    ).any()  # otra semilla: otro modelo (semillerío)
    assert np.corrcoef(pred(111623), y)[0, 1] > 0.3


def test_escalar_min_data():
    assert (
        md.escalar_min_data({"min_data_in_leaf": 500}, 0.1)["min_data_in_leaf"] == 5000
    )
    assert (
        md.escalar_min_data({"min_data_in_leaf": 500}, 1.0)["min_data_in_leaf"] == 500
    )


def test_resolver_params_chequea_fe_hash(tmp_path):
    f = tmp_path / "best.json"
    f.write_text(json.dumps({"fe_hash": "abc", "params": {"num_leaves": 9}}))
    cfg = _cfg(f"final.params_desde=archivo:{f.as_posix()}")
    assert md.resolver_params(cfg, "abc") == {"num_leaves": 9}
    with pytest.raises(ValueError, match="fe_hash"):
        md.resolver_params(cfg, "otro")
    assert md.resolver_params(_cfg(), "x")["num_leaves"] == 8
    with pytest.raises(FileNotFoundError):
        md.resolver_params(_cfg("final.params_desde=archivo:/no/existe.json"), "x")
