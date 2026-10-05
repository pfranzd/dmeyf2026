"""Etapa datos: CSV procesado -> parquet tipado en caché + validaciones.

Decisiones:
- `numero_de_cliente` se fuerza a BIGINT y `foto_mes` a INTEGER desde la lectura,
  para que los IDs nunca pasen por float (riesgo de notación científica).
- La caché se regenera solo si el origen es más nuevo (o si se reconstruye el target).
- Un mes tiene "target completo" si ninguna de sus filas tiene clase_ternaria NULL
  (hoy: 202103-202106; 202107 solo conoce BAJA+1 y 202108 es el mes a predecir).
"""

import logging
from pathlib import Path

import duckdb

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline import periodos
from competencia_1.pipeline.tracking import WORK, Run

log = logging.getLogger("competencia_1.datos")

CACHE_BASE = WORK / "cache" / "base.parquet"
TIPOS_FORZADOS = {"numero_de_cliente": "BIGINT", "foto_mes": "INTEGER"}
CLASES = {"CONTINUA", "BAJA+1", "BAJA+2"}

# Portado de exp/z101_target_sql/target_sql.py (upstream monday/z101_target_sql,
# commit 21402a7 de dmecoyfin/dmeyf2026). Panel denso cliente x mes + lead.
SQL_TARGET = """
    with periodos as (
        select distinct foto_mes from crudo
    ), clientes as (
        select distinct numero_de_cliente from crudo
    ), todo as (
        select numero_de_cliente, foto_mes from clientes cross join periodos
    ), con_target as (
        select
            c.*
            , if(c.numero_de_cliente is null, 0, 1) as mes_0
            , lead(mes_0, 1) over (partition by t.numero_de_cliente order by foto_mes) as mes_1
            , lead(mes_0, 2) over (partition by t.numero_de_cliente order by foto_mes) as mes_2
            , case
                when mes_2 = 1 then 'CONTINUA'
                when mes_1 = 1 and mes_2 = 0 then 'BAJA+2'
                when mes_1 = 0 then 'BAJA+1'
                else null
              end as clase_ternaria
        from todo t
        left join crudo c using (numero_de_cliente, foto_mes)
    )
    select * exclude (mes_0, mes_1, mes_2)
    from con_target
    where mes_0 = 1
"""


def _ruta(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else cfgmod.RAIZ / p


def _leer_csv_sql(path: Path) -> str:
    tipos = ", ".join(f"'{k}': '{v}'" for k, v in TIPOS_FORZADOS.items())
    return f"read_csv('{path.as_posix()}', sample_size = -1, types = {{{tipos}}})"


def construir_cache(cfg: cfgmod.Config, destino: Path = CACHE_BASE) -> Path:
    """Escribe el parquet base tipado (si hace falta) y devuelve su ruta."""
    d = cfg.datos
    origen = _ruta(d.csv_crudo if d.reconstruir_target else d.csv_procesado)
    if not origen.exists():
        raise FileNotFoundError(f"no existe el origen de datos: {origen}")
    if destino.exists() and destino.stat().st_mtime >= origen.stat().st_mtime:
        log.info("caché vigente, no se regenera: %s", destino)
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp.parquet")
    con = duckdb.connect()
    try:
        if d.reconstruir_target:
            log.info("reconstruyendo clase_ternaria desde %s", origen)
            con.execute(f"create table crudo as select * from {_leer_csv_sql(origen)}")
            query = SQL_TARGET
        else:
            query = f"select * from {_leer_csv_sql(origen)}"
        con.execute(
            f"copy ({query}) to '{tmp.as_posix()}' (format parquet, compression zstd)"
        )
    finally:
        con.close()
    tmp.replace(destino)
    log.info("caché escrita: %s (%.0f MB)", destino, destino.stat().st_size / 1e6)
    return destino


def resumen_meses(parquet: Path) -> list[dict]:
    """Por mes: filas, clientes y conteo por clase (incluye NULL)."""
    con = duckdb.connect()
    try:
        filas = con.execute(
            f"""
            select foto_mes
                 , count(*) as filas
                 , count(distinct numero_de_cliente) as clientes
                 , count(*) filter (clase_ternaria = 'CONTINUA') as continua
                 , count(*) filter (clase_ternaria = 'BAJA+1') as baja1
                 , count(*) filter (clase_ternaria = 'BAJA+2') as baja2
                 , count(*) filter (clase_ternaria is null) as sin_target
            from read_parquet('{parquet.as_posix()}')
            group by 1 order by 1
            """
        ).fetchall()
        cols = [
            "foto_mes",
            "filas",
            "clientes",
            "continua",
            "baja1",
            "baja2",
            "sin_target",
        ]
        return [dict(zip(cols, f, strict=True)) for f in filas]
    finally:
        con.close()


def validar_base(parquet: Path) -> list[dict]:
    """Chequea esquema, unicidad (cliente, mes) y clases válidas. Devuelve el resumen."""
    con = duckdb.connect()
    try:
        esquema = {
            r[0]: r[1]
            for r in con.execute(
                f"describe select * from read_parquet('{parquet.as_posix()}')"
            ).fetchall()
        }
        for col, tipo in TIPOS_FORZADOS.items():
            if col not in esquema:
                raise ValueError(f"falta la columna obligatoria: {col}")
            if esquema[col] != tipo:
                raise ValueError(f"{col} debe ser {tipo}, es {esquema[col]}")
        if "clase_ternaria" not in esquema:
            raise ValueError("falta la columna clase_ternaria")
        n, n_unicos = con.execute(
            f"""select count(*), count(distinct (numero_de_cliente, foto_mes))
                from read_parquet('{parquet.as_posix()}')"""
        ).fetchone()
        if n != n_unicos:
            raise ValueError(f"hay {n - n_unicos:,} duplicados de (cliente, mes)")
        clases = {
            r[0]
            for r in con.execute(
                f"select distinct clase_ternaria from read_parquet('{parquet.as_posix()}')"
            ).fetchall()
        }
        raros = clases - CLASES - {None}
        if raros:
            raise ValueError(f"clase_ternaria con valores inesperados: {sorted(raros)}")
        nulos_id = con.execute(
            f"""select count(*) from read_parquet('{parquet.as_posix()}')
                where numero_de_cliente is null or foto_mes is null"""
        ).fetchone()[0]
        if nulos_id:
            raise ValueError(f"{nulos_id:,} filas con cliente o mes nulo")
    finally:
        con.close()
    return resumen_meses(parquet)


def meses_con_target_completo(resumen: list[dict]) -> set[int]:
    return {r["foto_mes"] for r in resumen if r["sin_target"] == 0}


def validar_periodos(cfg: cfgmod.Config, resumen: list[dict]) -> None:
    """Cruza los períodos del config con lo que realmente hay en los datos."""
    presentes = {r["foto_mes"] for r in resumen}
    completos = meses_con_target_completo(resumen)
    objetivo = cfg.periodos.target
    if objetivo not in presentes:
        raise ValueError(f"el target_period {objetivo} no existe en los datos")
    for i, f in enumerate(cfgmod.folds(cfg)):
        periodos.validar_target_completo(list(f.train), completos, f"fold {i} train")
        periodos.validar_target_completo([f.valid], completos, f"fold {i} valid")
    periodos.validar_target_completo(cfgmod.meses_final(cfg), completos, "final")
    if objetivo in completos:
        log.warning(
            "el target_period %s ya tiene target conocido: se está prediciendo un mes etiquetado",
            objetivo,
        )


def etapa_datos(cfg: cfgmod.Config, run: Run) -> None:
    parquet = construir_cache(cfg)
    resumen = validar_base(parquet)
    for r in resumen:
        log.info(
            "mes %s: %s filas | CONTINUA=%s BAJA+1=%s BAJA+2=%s sin_target=%s",
            r["foto_mes"],
            r["filas"],
            r["continua"],
            r["baja1"],
            r["baja2"],
            r["sin_target"],
        )
    validar_periodos(cfg, resumen)
    run.registrar(
        datos={
            "parquet": parquet.relative_to(cfgmod.RAIZ).as_posix(),
            "meses": resumen,
            "meses_target_completo": sorted(meses_con_target_completo(resumen)),
        }
    )
