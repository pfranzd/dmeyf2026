"""Familias temporales: lags, deltas, ventanas móviles, tendencia e historia.

Todo se calcula sobre el PANEL DENSO (cliente x mes, incluidos los meses
faltantes como filas fantasma) para que `lag(x, n)` signifique "n meses
atrás" y no "n filas atrás". Solo se mira hacia atrás (`preceding`): ninguna
ventana usa `lead` ni `following`.
"""

from competencia_1.pipeline.config import FE
from competencia_1.pipeline.features.catalogo import Catalogo

# `fuente` es la vista sobre el parquet base. pk_* sobreviven a las filas
# fantasma (donde numero_de_cliente y foto_mes son NULL).
SQL_PANEL = """
    with periodos as (
        select distinct foto_mes from fuente
    ), clientes as (
        select distinct numero_de_cliente from fuente
    ), todo as (
        select numero_de_cliente, foto_mes from clientes cross join periodos
    ), panel as (
        select
            t.numero_de_cliente as pk_cliente
            , t.foto_mes as pk_mes
            , if(c.numero_de_cliente is null, 0, 1) as mes_0
            , c.*
        from todo t
        left join fuente c
          on c.numero_de_cliente = t.numero_de_cliente
         and c.foto_mes = t.foto_mes
    )
"""

FUNC_SQL = {"avg": "avg", "max": "max", "min": "min", "std": "stddev"}


def lags_requeridos(fe: FE) -> tuple[list[int], list[int]]:
    """(lags a publicar, lags auxiliares solo para calcular deltas)."""
    publicos = sorted(set(fe.lags.n)) if fe.lags.activo else []
    para_deltas = set(fe.deltas.n) if fe.deltas.activo else set()
    return publicos, sorted(para_deltas - set(publicos))


def tamanios_ventana(fe: FE) -> list[int]:
    t = set()
    if fe.ventanas.activo:
        t.add(fe.ventanas.tamanio)
    if fe.tendencia.activo:
        t.add(fe.tendencia.tamanio)
    return sorted(t)


def ventanas_sql(fe: FE) -> str:
    """Cláusula WINDOW con solo las ventanas nombradas que se usan."""
    publicos, aux = lags_requeridos(fe)
    defs = []
    if publicos or aux or fe.historia:
        defs.append("ventana_orden as (partition by pk_cliente order by pk_mes)")
    for k in tamanios_ventana(fe):
        defs.append(
            f"ventana_{k} as (partition by pk_cliente order by pk_mes "
            f"rows between {k} preceding and current row)"
        )
    if fe.historia:
        defs.append(
            "historia as (partition by pk_cliente order by pk_mes "
            "rows between unbounded preceding and current row)"
        )
    return ("\n        window " + "\n             , ".join(defs)) if defs else ""


def sql_series(fe: FE, campos: list[str], cat: Catalogo) -> tuple[str, list[str]]:
    """Lags, ventanas y tendencia + historia. Devuelve (fragmento, lags auxiliares)."""
    publicos, aux = lags_requeridos(fe)
    frag = ""
    for c in campos:
        for n in publicos:
            frag += cat.reg(
                "3_lags", f"lag_{n}_{c}", f"lag({c}, {n}) over ventana_orden", c
            )
        if fe.ventanas.activo:
            k = fe.ventanas.tamanio
            for stat in fe.ventanas.stats:
                frag += cat.reg(
                    "5_ventanas",
                    f"{stat}_{k}_{c}",
                    f"{FUNC_SQL[stat]}({c}) over ventana_{k}",
                    c,
                )
        if fe.tendencia.activo:
            k = fe.tendencia.tamanio
            frag += cat.reg(
                "6_tendencia",
                f"slope_{k}_{c}",
                f"regr_slope({c}, cliente_antiguedad) over ventana_{k}",
                f"{c} vs cliente_antiguedad",
            )
    aux_nombres = []
    # lags solo necesarios para los deltas: se calculan pero no se publican
    for c in campos:
        for n in aux:
            nombre = f"lag_{n}_{c}"
            aux_nombres.append(nombre)
            frag += f"\n    , lag({c}, {n}) over ventana_orden as {nombre}"
    if fe.historia:
        frag += cat.reg(
            "7_historia",
            "meses_en_panel",
            "sum(mes_0) over historia",
            "presencias acumuladas",
        )
        frag += cat.reg(
            "7_historia",
            "antiguedad_panel",
            "row_number() over ventana_orden",
            "posición del mes dentro del panel",
        )
    return frag, aux_nombres


def sql_deltas(fe: FE, campos: list[str], cat: Catalogo) -> str:
    """Deltas y ratio contra media móvil. Van después de filtrar las filas reales."""
    frag = ""
    for c in campos:
        if fe.deltas.activo:
            for n in sorted(set(fe.deltas.n)):
                frag += cat.reg(
                    "4_deltas",
                    f"delta_{n}_{c}",
                    f"{c} - lag_{n}_{c}",
                    f"{c}, lag_{n}_{c}",
                )
                if fe.deltas.pct:
                    frag += cat.reg(
                        "4_deltas",
                        f"deltapct_{n}_{c}",
                        f"delta_pct({c}, lag_{n}_{c})",
                        f"variación % contra t-{n}",
                    )
        if fe.ventanas.activo and fe.ventanas.ratio_avg:
            k = fe.ventanas.tamanio
            frag += cat.reg(
                "5_ventanas",
                f"ratioavg_{k}_{c}",
                f"ratio_seguro({c}, avg_{k}_{c})",
                f"{c} contra su media móvil",
            )
    return frag
