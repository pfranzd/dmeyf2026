import duckdb
import pytest

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import datos

BASE = cfgmod.CONFIGS_DIR / "base.yaml"


def _csv_crudo(path):
    # cliente 1: todos los meses; cliente 2: se va tras 202103 (BAJA+1 en 03);
    # cliente 3: presente 03 y 04, ausente 05 (BAJA+2 en 03 -> 04 está, 05 no)
    filas = []
    for mes in (202103, 202104, 202105, 202106, 202107):
        filas.append((48_000_000, mes, 1.5))
        if mes <= 202103:
            filas.append((2, mes, 2.0))
        if mes <= 202104:
            filas.append((3, mes, 3.0))
    path.write_text(
        "numero_de_cliente,foto_mes,x\n"
        + "\n".join(",".join(map(str, f)) for f in filas)
        + "\n"
    )


def _cfg(tmp_path, **ov):
    over = [f"datos.{k}={v}" for k, v in ov.items()]
    return cfgmod.cargar_config(BASE, over)[0]


def test_reconstruye_target_y_tipos(tmp_path):
    crudo = tmp_path / "crudo.csv"
    _csv_crudo(crudo)
    cfg = _cfg(tmp_path, csv_crudo=crudo.as_posix(), reconstruir_target="true")
    pq = datos.construir_cache(cfg, tmp_path / "base.parquet")
    resumen = {r["foto_mes"]: r for r in datos.validar_base(pq)}
    assert resumen[202103]["baja1"] == 1 and resumen[202103]["baja2"] == 1
    assert resumen[202103]["continua"] == 1
    # los dos últimos meses no tienen "futuro" suficiente
    assert resumen[202106]["sin_target"] == 1 and resumen[202107]["sin_target"] == 1
    assert (
        duckdb.sql(
            f"select typeof(numero_de_cliente) from '{pq.as_posix()}' limit 1"
        ).fetchone()[0]
        == "BIGINT"
    )
    assert datos.meses_con_target_completo(list(resumen.values())) == {
        202103,
        202104,
        202105,
    }


def test_cache_vigente_no_se_regenera(tmp_path):
    crudo = tmp_path / "crudo.csv"
    _csv_crudo(crudo)
    cfg = _cfg(tmp_path, csv_crudo=crudo.as_posix(), reconstruir_target="true")
    destino = tmp_path / "base.parquet"
    datos.construir_cache(cfg, destino)
    m = destino.stat().st_mtime_ns
    datos.construir_cache(cfg, destino)
    assert destino.stat().st_mtime_ns == m


def test_duplicados_se_detectan(tmp_path):
    pq = tmp_path / "dup.parquet"
    duckdb.sql(
        f"""copy (select 1::BIGINT as numero_de_cliente, 202103::INTEGER as foto_mes,
                  'CONTINUA' as clase_ternaria
                  union all select 1, 202103, 'CONTINUA') to '{pq.as_posix()}' (format parquet)"""
    )
    with pytest.raises(ValueError, match="duplicados"):
        datos.validar_base(pq)


def test_validar_periodos_detecta_mes_incompleto(tmp_path):
    cfg = _cfg(tmp_path)
    mk = lambda mes, nulos: {"foto_mes": mes, "sin_target": nulos}
    ok = [mk(m, 0) for m in (202103, 202104, 202105, 202106)] + [
        mk(202107, 5),
        mk(202108, 9),
    ]
    datos.validar_periodos(cfg, ok)
    malo = [mk(m, 0) for m in (202103, 202104, 202105)] + [mk(202106, 3), mk(202108, 9)]
    with pytest.raises(ValueError, match="202106"):
        datos.validar_periodos(cfg, malo)
    with pytest.raises(ValueError, match="no existe"):
        datos.validar_periodos(cfg, ok[:-1])
