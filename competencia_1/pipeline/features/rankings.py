"""Familia 2: rankings intra-mes (defensa contra data drifting).

La inflación mueve todos los montos nominales pero no la posición relativa del
cliente frente a sus contemporáneos. Se calculan sobre filas REALES (después de
descartar las fantasma) y son NULL-safe: un NULL queda NULL en vez de recibir el
ranking más alto (comportamiento por defecto de `percent_rank` en z402).

El decil NO usa `ntile(10)`: ntile reparte los empates según el orden físico de las
filas, y con variables muy repetidas (ctrx_quarter) daba deciles distintos entre
corridas. Con `percent_rank` los valores iguales caen siempre en el mismo decil.
"""

from competencia_1.pipeline.config import FE
from competencia_1.pipeline.features.catalogo import Catalogo


def sql_rankings(fe: FE, campos: list[str], cat: Catalogo) -> str:
    if not fe.rankings.activo:
        return ""
    frag = ""
    for c in campos:
        frag += cat.reg(
            "2_rank_intrames",
            f"pr_{c}",
            f"case when {c} is null then null else "
            f"percent_rank() over (partition by foto_mes, ({c} is null) order by {c}) end",
            f"{c} — posición relativa dentro del mes",
        )
    for c in fe.rankings.ntile10:
        frag += cat.reg(
            "2_rank_intrames",
            f"d10_{c}",
            f"case when {c} is null then null else "
            f"least(1 + floor(10 * percent_rank() over "
            f"(partition by foto_mes, ({c} is null) order by {c})), 10)::INTEGER end",
            f"{c} — decil dentro del mes (por percent_rank: empates -> mismo decil)",
        )
    return frag


# Columnas de monto que se escalan por consulta. Cada ventana ordenada arrastra todas las
# columnas ya calculadas, así que el costo crece más que linealmente con el tamaño del lote:
# con 91 montos en una sola consulta tardaba >1 h; en lotes de 5 son ~10 s (mismos valores).
LOTE_MONTOS = 5


def _expresion_monto(modo: str) -> str:
    if modo == "rank_cero_fijo":
        return (
            "case when {c} is null then null when {c} = 0 then 0 "
            "when {c} > 0 then cume_dist() over (partition by pk_mes, sign({c}) order by {c}) "
            "else -cume_dist() over (partition by pk_mes, sign({c}) order by {c} desc) end"
        )
    if modo == "deflactado":
        return "{c} / nullif(median(case when {c} > 0 then {c} end) over (partition by pk_mes), 0)"
    raise ValueError(f"fe.montos inválido: {modo}")


def materializar_montos(con, modo: str, campos: list[str]) -> str:
    """Crea la tabla temporal `rel` (pk_cliente, pk_mes + los montos en escala relativa, solo
    filas reales) y devuelve la cláusula `replace` que la aplica sobre `base`.

    Lleva los montos a una escala comparable entre meses; se aplica sobre el panel, antes de
    lags, deltas y ventanas, que quedan expresados en la escala nueva. Las filas fantasma
    quedan en NULL (no están en `rel`).
    - rank_cero_fijo: negativos -> [-1, 0), cero -> 0, positivos -> (0, 1], por mes y por
      signo con `cume_dist` (los empates reciben el mismo valor: determinístico).
    - deflactado: monto / mediana de los montos positivos del mes (quita inflación y aguinaldo
      sin perder la forma de la distribución).
    Requiere la vista `base` en `con`. Devuelve "" si no hay nada que escalar.
    """
    if modo == "crudo" or not campos:
        return ""
    expr = _expresion_monto(modo)
    con.execute(
        "create or replace temp table montos_crudos as select pk_cliente, pk_mes, "
        + ", ".join(campos)
        + " from base where mes_0 = 1"
    )
    lotes = [campos[i : i + LOTE_MONTOS] for i in range(0, len(campos), LOTE_MONTOS)]
    for j, lote in enumerate(lotes):
        partes = ", ".join(f"{expr.format(c=c)} as {c}" for c in lote)
        con.execute(
            f"create or replace temp table montos_lote_{j} as "
            f"select pk_cliente, pk_mes, {partes} from montos_crudos"
        )
    seleccion = ", ".join(f"l{j}.{c}" for j, lote in enumerate(lotes) for c in lote)
    uniones = " ".join(
        f"join montos_lote_{j} l{j} using (pk_cliente, pk_mes)"
        for j in range(1, len(lotes))
    )
    con.execute(
        "create or replace temp table rel as select l0.pk_cliente, l0.pk_mes, "
        f"{seleccion} from montos_lote_0 l0 {uniones}"
    )
    con.execute("drop table montos_crudos")
    for j in range(len(lotes)):
        con.execute(f"drop table montos_lote_{j}")
    return " replace (" + ", ".join(f"rel.{c} as {c}" for c in campos) + ")"
