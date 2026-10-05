"""Canaritos (arboles/z0607_ParadigmShift): columnas aleatorias para detectar sobreajuste."""

from competencia_1.pipeline.config import FE
from competencia_1.pipeline.features.catalogo import Catalogo


def sql_canaritos(fe: FE, semilla: int, cat: Catalogo) -> str:
    """`n` columnas U(0,1) determinísticas: hash(cliente, mes, semilla + i).

    No se usa `random()` de DuckDB porque con ejecución multihilo no es
    reproducible; un hash de la clave sí lo es.
    """
    if not fe.canaritos.activo:
        return ""
    frag = ""
    for i in range(1, fe.canaritos.n + 1):
        frag += cat.reg(
            "8_canaritos",
            f"canarito_{i}",
            f"(hash(numero_de_cliente, foto_mes, {semilla + i}) % 1000003)::DOUBLE / 1000003",
            f"hash(cliente, mes, semilla+{i})",
        )
    return frag
