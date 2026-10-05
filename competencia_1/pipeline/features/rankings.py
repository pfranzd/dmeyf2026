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
