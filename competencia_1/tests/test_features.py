"""Tests de FE sobre una muestra de clientes reales (se saltean si no hay caché base)."""

import duckdb
import polars as pl
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.datos import CACHE_BASE
from competencia_1.pipeline.features import calcular_fe_hash, construir_features
from competencia_1.pipeline.features.query import verificar_sin_futuro

pytestmark = pytest.mark.skipif(
    not CACHE_BASE.exists(), reason="falta work/.../cache/base.parquet"
)

BASE_YAML = cfgmod.CONFIGS_DIR / "base.yaml"
# FE chico para que los tests corran rápido
CHICO = [
    "fe.campos_serie=[ctrx_quarter, mcaja_ahorro, cproductos]",
    "fe.rankings.campos=[ctrx_quarter, mcaja_ahorro]",
    "fe.rankings.ntile10=[ctrx_quarter]",
]


def _cfg(*overrides):
    return cfgmod.cargar_config(BASE_YAML, [*CHICO, *overrides])[0]


@pytest.fixture(scope="module")
def muestra(tmp_path_factory):
    d = tmp_path_factory.mktemp("fe")
    out = d / "muestra.parquet"
    b = CACHE_BASE.as_posix()
    duckdb.sql(
        f"""
        copy (
            select * from read_parquet('{b}')
            where numero_de_cliente in (
                select numero_de_cliente from (select distinct numero_de_cliente from read_parquet('{b}'))
                order by hash(numero_de_cliente) limit 3000)
            or numero_de_cliente in (
                select numero_de_cliente from read_parquet('{b}') where foto_mes = 202103
                intersect select numero_de_cliente from read_parquet('{b}') where foto_mes = 202105
                except select numero_de_cliente from read_parquet('{b}') where foto_mes = 202104)
        ) to '{out.as_posix()}' (format parquet)
        """
    )
    return out


def _build(cfg, muestra, tmp_path):
    pq, h = construir_features(cfg, base=muestra, raiz=tmp_path / "feat")
    return pl.read_parquet(pq), pq, h


def test_filas_target_y_catalogo(muestra, tmp_path):
    df, pq, _ = _build(_cfg(), muestra, tmp_path)
    assert df.height == pl.read_parquet(muestra).height
    assert (pq.parent / "catalogo.csv").exists() and (pq.parent / "query.sql").exists()
    assert not {"pk_cliente", "pk_mes", "mes_0"} & set(df.columns)
    assert df.schema["numero_de_cliente"] == pl.Int64


def test_query_sin_informacion_futura(muestra, tmp_path):
    _, pq, _ = _build(_cfg(), muestra, tmp_path)
    sql = (pq.parent / "query.sql").read_text(encoding="utf-8")
    verificar_sin_futuro(sql)
    with pytest.raises(ValueError, match="futura"):
        verificar_sin_futuro("select lead(x, 1) over () from t")
    with pytest.raises(ValueError, match="futura"):
        verificar_sin_futuro("rows between current row and 2 following")


def test_lag_correcto_con_meses_faltantes(muestra, tmp_path):
    df, _, _ = _build(_cfg(), muestra, tmp_path)
    orig = pl.read_parquet(muestra)
    # clientes presentes en 03 y 05 pero no en 04
    s03 = set(orig.filter(pl.col("foto_mes") == 202103)["numero_de_cliente"])
    s04 = set(orig.filter(pl.col("foto_mes") == 202104)["numero_de_cliente"])
    s05 = set(orig.filter(pl.col("foto_mes") == 202105)["numero_de_cliente"])
    con_hueco = sorted((s03 & s05) - s04)
    assert con_hueco, "la muestra debería incluir clientes con un mes faltante"
    for cli in con_hueco[:10]:
        fila = df.filter(
            (pl.col("numero_de_cliente") == cli) & (pl.col("foto_mes") == 202105)
        )
        v03 = orig.filter(
            (pl.col("numero_de_cliente") == cli) & (pl.col("foto_mes") == 202103)
        )["ctrx_quarter"][0]
        assert (
            fila["lag_1_ctrx_quarter"][0] is None
        )  # t-1 no existe: NO es el de 202103
        assert fila["lag_2_ctrx_quarter"][0] == v03  # t-2 sí es 202103
        assert fila["delta_1_ctrx_quarter"][0] is None


def test_deltas_son_x_menos_lag(muestra, tmp_path):
    df, _, _ = _build(_cfg(), muestra, tmp_path)
    ok = df.filter(pl.col("lag_1_ctrx_quarter").is_not_null())
    assert ok.height > 0
    assert (
        ok["delta_1_ctrx_quarter"] == ok["ctrx_quarter"] - ok["lag_1_ctrx_quarter"]
    ).all()


def test_toggles_de_familias(muestra, tmp_path):
    # lags apagados pero deltas activos: el lag auxiliar no se publica
    df, _, _ = _build(
        _cfg("fe.lags.activo=false", "fe.deltas.n=[1]", "fe.ventanas.activo=false"),
        muestra,
        tmp_path,
    )
    assert "delta_1_ctrx_quarter" in df.columns
    assert not [c for c in df.columns if c.startswith(("lag_", "avg_", "ratioavg_"))]

    # solo derivadas: ninguna columna original más allá de claves/target
    df2, _, _ = _build(_cfg("fe.originales=false"), muestra, tmp_path)
    assert "ctrx_quarter" not in df2.columns and "r_tc_uso_limite" in df2.columns
    assert {"numero_de_cliente", "foto_mes", "clase_ternaria"} <= set(df2.columns)

    # todo apagado salvo originales
    df3, _, _ = _build(
        _cfg(
            *[
                f"fe.intrafila.{k}=false"
                for k in ("tc_consolidado", "tc_fechas", "ratios", "agregados", "flags")
            ],
            "fe.lags.activo=false",
            "fe.deltas.activo=false",
            "fe.ventanas.activo=false",
            "fe.tendencia.activo=false",
            "fe.historia=false",
            "fe.rankings.activo=false",
        ),
        muestra,
        tmp_path,
    )
    assert df3.width == pl.read_parquet(muestra).width - len(_cfg().fe.drop_drift)


def test_drop_drift_tambien_de_las_derivadas(muestra, tmp_path):
    df, _, _ = _build(_cfg(), muestra, tmp_path)
    for c in (
        "Master_Finiciomora",
        "Visa_Finiciomora",
        "tc_finiciomora_menor",
        "tc_fultimo_cierre_menor",
    ):
        assert c not in df.columns
    assert "f_mora" in df.columns  # el flag de mora se conserva
    df2, _, _ = _build(_cfg("fe.drop_drift=[]"), muestra, tmp_path)
    assert "Master_Finiciomora" in df2.columns and "tc_finiciomora_menor" in df2.columns


def test_rankings_null_safe(muestra, tmp_path):
    df, _, _ = _build(_cfg(), muestra, tmp_path)
    nulos = df.filter(pl.col("mcaja_ahorro").is_null())
    assert (nulos["pr_mcaja_ahorro"].is_null()).all()
    con_valor = df.filter(pl.col("mcaja_ahorro").is_not_null())
    assert con_valor["pr_mcaja_ahorro"].is_between(0, 1).all()
    assert con_valor["d10_ctrx_quarter"].is_between(1, 10).all()
    # valores iguales dentro de un mes -> mismo decil (ntile los repartía al azar)
    por_valor = df.group_by(["foto_mes", "ctrx_quarter"]).agg(
        pl.col("d10_ctrx_quarter").n_unique()
    )
    assert (por_valor["d10_ctrx_quarter"] <= 1).all()


def test_canaritos_deterministicos(muestra, tmp_path):
    cfg = _cfg("fe.canaritos.activo=true", "fe.canaritos.n=2")
    a, _, _ = _build(cfg, muestra, tmp_path / "a")
    b, _, _ = _build(cfg, muestra, tmp_path / "b")
    assert a["canarito_1"].to_list() == b["canarito_1"].to_list()
    assert a["canarito_1"].to_list() != a["canarito_2"].to_list()
    assert a["canarito_1"].is_between(0, 1).all()
    _, h1 = calcular_fe_hash(cfg, muestra)
    _, h2 = calcular_fe_hash(
        _cfg("fe.canaritos.activo=true", "fe.canaritos.n=2", "semilla_maestra=116239"),
        muestra,
    )
    assert h1 != h2  # los canaritos dependen de la semilla


def test_fe_hash_estable_y_sensible(muestra):
    h1 = calcular_fe_hash(_cfg(), muestra)[0]
    assert h1 == calcular_fe_hash(_cfg(), muestra)[0]
    assert h1 != calcular_fe_hash(_cfg("fe.lags.n=[1]"), muestra)[0]


def test_cache_por_hash(muestra, tmp_path):
    _, pq, _ = _build(_cfg(), muestra, tmp_path)
    m = pq.stat().st_mtime_ns
    _, pq2, _ = _build(_cfg(), muestra, tmp_path)
    assert pq2 == pq and pq.stat().st_mtime_ns == m


def test_validaciones_de_config():
    with pytest.raises(ValueError, match="ratio_avg"):
        _cfg("fe.ventanas.stats=[max]")
    with pytest.raises(ValueError, match="ratios requiere"):
        _cfg("fe.intrafila.agregados=false")
    with pytest.raises(ValueError, match="curado"):
        _cfg("fe.campos_serie=casi")
    with pytest.raises(ValueError, match="stats"):
        _cfg("fe.ventanas.stats=[mediana]")


def test_campos_inexistentes_son_error(muestra, tmp_path):
    with pytest.raises(ValueError, match="inexistentes"):
        _build(_cfg("fe.campos_serie=[no_existe]"), muestra, tmp_path)
    with pytest.raises(ValueError, match="inexistentes"):
        _build(_cfg("fe.drop_extra=[no_existe]"), muestra, tmp_path)
    assert not list((tmp_path / "feat").glob("*.tmp"))  # no quedan cachés a medias


def test_montos_rank_cero_fijo(muestra, tmp_path):
    crudo, _, h0 = _build(_cfg(), muestra, tmp_path)
    df, _, h1 = _build(_cfg("fe.montos=rank_cero_fijo"), muestra, tmp_path)
    assert h0 != h1
    assert df.height == crudo.height
    j = crudo.select("numero_de_cliente", "foto_mes", c="mcaja_ahorro").join(
        df.select("numero_de_cliente", "foto_mes", r="mcaja_ahorro"),
        on=["numero_de_cliente", "foto_mes"],
    )
    # signo y NULL preservados, cero fijo en 0, rango en [-1, 1]
    assert (j["c"].is_null() == j["r"].is_null()).all()
    nn = j.drop_nulls()
    assert ((nn["c"] == 0) == (nn["r"] == 0)).all()
    assert ((nn["c"] > 0) == (nn["r"] > 0)).all()
    assert nn["r"].min() >= -1 and nn["r"].max() <= 1
    # monótono dentro de cada mes y signo
    for (_, s), g in nn.filter(pl.col("c") != 0).group_by(
        "foto_mes", pl.col("c").sign().alias("s")
    ):
        g = g.sort("c")
        assert (g["r"].diff().drop_nulls() >= 0).all()
    # los no montos no cambian; los deltas se calculan en la escala nueva
    assert crudo["ctrx_quarter"].equals(df["ctrx_quarter"])
    d = df.drop_nulls(["delta_1_mcaja_ahorro", "lag_1_mcaja_ahorro"])
    assert (
        d["delta_1_mcaja_ahorro"] - (d["mcaja_ahorro"] - d["lag_1_mcaja_ahorro"])
    ).abs().max() < 1e-9


def test_montos_deflactado(muestra, tmp_path):
    crudo, _, _ = _build(_cfg(), muestra, tmp_path)
    df, _, _ = _build(_cfg("fe.montos=deflactado"), muestra, tmp_path)
    med = (
        df.filter(pl.col("mpayroll") > 0)
        .group_by("foto_mes")
        .agg(pl.col("mpayroll").median())
    )
    assert (
        med["mpayroll"] - 1
    ).abs().max() < 0.05  # mediana de positivos ~1 en todos los meses
    assert crudo["cproductos"].equals(df["cproductos"])


def test_montos_crudo_no_cambia_fe_hash(muestra):
    import dataclasses

    from competencia_1.pipeline.features import _fe_para_hash

    fe = _cfg().fe
    assert "montos" not in _fe_para_hash(fe)
    assert _fe_para_hash(fe) == {
        k: v
        for k, v in dataclasses.asdict(fe).items()
        if k not in ("montos", "comportamiento")
    }
    with pytest.raises(ValueError, match="fe.montos"):
        _cfg("fe.montos=otro")


def test_comportamiento(muestra, tmp_path):
    crudo, _, h0 = _build(_cfg(), muestra, tmp_path)
    df, pq, h1 = _build(_cfg("fe.comportamiento=true"), muestra, tmp_path)
    assert h0 != h1 and df.height == crudo.height
    nuevas = set(df.columns) - set(crudo.columns)
    assert nuevas == {
        "f_perdio_payroll",
        "n_productos_perdidos",
        "f_visa_status_sube",
        "f_master_status_sube",
        "meses_desde_payroll",
        "meses_desde_trx",
        "meses_desde_consumo_tc",
        "r_saldo_vs_max",
        "r_activos_vs_max",
        "racha_caida_ctrx",
        "r_transf_emitidas_recibidas",
        "rfm_score",
    }
    cat = pl.read_csv(pq.parent / "catalogo.csv")
    assert set(cat.filter(pl.col("bloque") == "9_comportamiento")["feature"]) == nuevas
    # recency en [0, 3]; NULL solo si la ventana no está completa (202103, 202104)
    for c in ("meses_desde_payroll", "meses_desde_trx", "meses_desde_consumo_tc"):
        v = df[c].drop_nulls()
        assert v.min() >= 0 and v.max() <= 3
        assert df.filter(pl.col("foto_mes") >= 202105)[c].null_count() == 0
    assert df["rfm_score"].min() >= 3 and df["rfm_score"].max() <= 15
    assert set(df["racha_caida_ctrx"].unique().to_list()) <= {0, 1, 2}

    # f_perdio_payroll a mano para clientes presentes en 3 meses seguidos
    b = pl.read_parquet(muestra).select(
        "numero_de_cliente",
        "foto_mes",
        p=(pl.col("cpayroll_trx").fill_null(0) + pl.col("cpayroll2_trx").fill_null(0))
        > 0,
    )
    t = b.filter(pl.col("foto_mes") == 202106)
    for mes, s in ((202105, "p1"), (202104, "p2")):
        t = t.join(
            b.filter(pl.col("foto_mes") == mes).select(
                "numero_de_cliente", pl.col("p").alias(s)
            ),
            on="numero_de_cliente",
        )
    esperado = t.select(
        "numero_de_cliente",
        e=((pl.col("p1") | pl.col("p2")) & ~pl.col("p")).cast(pl.Int32),
    )
    j = esperado.join(
        df.filter(pl.col("foto_mes") == 202106).select(
            "numero_de_cliente", "f_perdio_payroll"
        ),
        on="numero_de_cliente",
    )
    assert j.height > 1000 and (j["e"] == j["f_perdio_payroll"]).all()
    assert j["e"].sum() > 0


def test_comportamiento_apagado_no_cambia_fe_hash():
    from competencia_1.pipeline.features import _fe_para_hash

    assert "comportamiento" not in _fe_para_hash(_cfg().fe)
    with pytest.raises(ValueError, match="comportamiento requiere"):
        _cfg(
            "fe.comportamiento=true",
            "fe.intrafila.agregados=false",
            "fe.intrafila.ratios=false",
        )
