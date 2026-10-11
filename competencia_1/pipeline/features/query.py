"""Ensambla la query final de FE: panel -> base -> series -> reales -> final.

panel   : cliente x mes denso (filas fantasma incluidas)
base    : panel + familia 1 (intra-fila)
escala  : base con los montos en escala relativa por mes (solo si fe.montos != crudo)
series  : base + lags / ventanas / tendencia / historia (ventanas hacia atrás)
reales  : series sin filas fantasma (mes_0 = 1)
final   : reales + rankings intra-mes + deltas + canaritos + RFM, sin columnas auxiliares,
          ordenado por (foto_mes, numero_de_cliente) para que el parquet sea reproducible
"""

import logging
import re

import duckdb

from competencia_1.pipeline.config import FE
from competencia_1.pipeline.features import campos as cm
from competencia_1.pipeline.features.catalogo import Catalogo
from competencia_1.pipeline.features.comportamiento import (
    sql_comportamiento_final,
    sql_comportamiento_mes,
    sql_comportamiento_series,
)
from competencia_1.pipeline.features.extras import sql_canaritos
from competencia_1.pipeline.features.intrafila import sql_intrafila
from competencia_1.pipeline.features.rankings import materializar_montos, sql_rankings
from competencia_1.pipeline.features.temporales import (
    SQL_PANEL,
    sql_deltas,
    sql_series,
    ventanas_sql,
)

log = logging.getLogger("competencia_1.fe")

CLAVES = ("numero_de_cliente", "foto_mes", "clase_ternaria")
_FUTURO = re.compile(r"\blead\s*\(|\bfollowing\b", re.IGNORECASE)


def verificar_sin_futuro(sql: str) -> None:
    """Ninguna feature puede mirar hacia adelante en el tiempo."""
    m = _FUTURO.search(sql)
    if m:
        raise ValueError(f"la query de FE usa información futura ({m.group(0)!r})")


def armar_query(
    fe: FE, semilla: int, con: duckdb.DuckDBPyConnection
) -> tuple[str, Catalogo, list[str]]:
    """Devuelve (sql, catálogo, columnas excluidas del resultado).

    Requiere en `con` la vista `fuente` y las macros ya creadas.
    """
    cat = Catalogo()
    drop = set(fe.drop_drift) | set(fe.drop_extra)

    b1 = sql_intrafila(fe.intrafila, cat, drop)
    if fe.comportamiento:
        b1 += sql_comportamiento_mes(cat)
    con.execute(f"create or replace view base as {SQL_PANEL} select *{b1} from panel")
    esquema = {r[0]: r[1] for r in con.execute("describe base").fetchall()}

    desconocidas = sorted(drop - set(esquema))
    if desconocidas:
        raise ValueError(
            f"drop_drift/drop_extra con columnas inexistentes: {desconocidas}"
        )

    campos_serie = cm.resolver_campos(
        fe.campos_serie, cm.CAMPOS_SERIE_CURADO, esquema, drop, "campos_serie"
    )
    campos_rank = (
        cm.resolver_campos(
            fe.rankings.campos, cm.CAMPOS_RANK_CURADO, esquema, drop, "rankings.campos"
        )
        if fe.rankings.activo
        else []
    )
    for c in fe.rankings.ntile10 if fe.rankings.activo else []:
        if c not in esquema:
            raise ValueError(f"rankings.ntile10: campo inexistente: {c}")

    montos = cm.campos_monto(esquema, drop) if fe.montos != "crudo" else []
    b_escala = materializar_montos(con, fe.montos, montos)  # tabla temporal `rel`
    if b_escala:  # montos en escala relativa: se unen al panel por (cliente, mes)
        escala = f"""   , escala as (
        select base.*{b_escala}
        from base
        left join rel
          on rel.pk_cliente = base.pk_cliente and rel.pk_mes = base.pk_mes
     )"""
    else:
        escala = """   , escala as (
        select *
        from base
     )"""
    b_series, lags_aux = sql_series(fe, campos_serie, cat)
    if fe.comportamiento:
        b_series += sql_comportamiento_series(cat)
    b_rank = sql_rankings(fe, campos_rank, cat)
    b_delta = sql_deltas(fe, campos_serie, cat)
    b_canar = sql_canaritos(fe, semilla, cat)
    b_comp = sql_comportamiento_final(cat) if fe.comportamiento else ""

    originales = [
        c for c in con.execute("describe fuente").fetchall() if c[0] not in CLAVES
    ]
    excluidas = ["pk_cliente", "pk_mes", "mes_0", *lags_aux, *sorted(drop)]
    if not fe.originales:
        excluidas += [c[0] for c in originales if c[0] not in drop]

    sql = f"""
-- `rel`: tabla temporal con los montos en escala relativa (rankings.materializar_montos)
{SQL_PANEL}
   , base as (
        select *{b1}
        from panel
     )
{escala}
   , series as (
        select *{b_series}
        from escala{ventanas_sql(fe)}
     )
   , reales as (
        select *
        from series
        where mes_0 = 1
     )
select * exclude ({", ".join(excluidas)}){b_rank}{b_delta}{b_canar}{b_comp}
from reales
order by foto_mes, numero_de_cliente
"""
    verificar_sin_futuro(sql)
    return sql, cat, excluidas
