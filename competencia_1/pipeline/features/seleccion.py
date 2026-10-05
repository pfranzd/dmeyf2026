"""Selección de columnas por bloque del catálogo (base de las ablaciones de FE).

Cada familia de FE se calcula de forma independiente de las demás, así que construir
el parquet con una familia apagada produce EXACTAMENTE las mismas columnas que elegir
las del parquet completo sin esa familia. Por eso las ablaciones no reconstruyen el FE
(~10 min cada una): eligen columnas de la caché completa.

Bloques: los del catálogo (`3_lags`, `4_deltas`, ...) más dos seudo-bloques:
    originales : las columnas crudas (todo lo que no figura en el catálogo)
    derivadas  : todas las creadas por FE (la unión de los bloques del catálogo)
"""

from pathlib import Path

import polars as pl

from competencia_1.pipeline.features.query import CLAVES

ORIGINALES = "originales"
DERIVADAS = "derivadas"


def expandir_lista(items: list[str], raiz: Path) -> list[str]:
    """Reemplaza cada `archivo:<ruta>` por las líneas de ese archivo (sin vacías ni #comentarios).

    Permite excluir cientos de features (p. ej. la salida de los canaritos) sin llenar el YAML.
    """
    out: list[str] = []
    for it in items:
        if not it.startswith("archivo:"):
            out.append(it)
            continue
        ruta = Path(it.removeprefix("archivo:"))
        ruta = ruta if ruta.is_absolute() else raiz / ruta
        if not ruta.exists():
            raise FileNotFoundError(f"no existe la lista de features: {ruta}")
        out += [
            ln.strip()
            for ln in ruta.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    return out


def cargar_bloques(parquet: Path, columnas: list[str]) -> dict[str, str]:
    """feature -> bloque, para todas las columnas candidatas (las crudas son 'originales')."""
    catalogo = Path(parquet).parent / "catalogo.csv"
    derivadas = {}
    if catalogo.exists():
        cat = pl.read_csv(catalogo)
        derivadas = dict(
            zip(cat["feature"].to_list(), cat["bloque"].to_list(), strict=True)
        )
    return {c: derivadas.get(c, ORIGINALES) for c in columnas}


def bloques_validos(bloques: dict[str, str]) -> set[str]:
    return set(bloques.values()) | {ORIGINALES, DERIVADAS}


def _pertenece(bloque: str, nombre: str) -> bool:
    if nombre == DERIVADAS:
        return bloque != ORIGINALES
    return bloque == nombre


def seleccionar_columnas(
    columnas: list[str],
    bloques: dict[str, str],
    solo_bloques: list[str] | None = None,
    excluir_bloques: list[str] | None = None,
    excluir_features: list[str] | None = None,
) -> list[str]:
    """Filtra `columnas` conservando su orden. Los nombres inexistentes son error (typos)."""
    solo_bloques = solo_bloques or []
    excluir_bloques = excluir_bloques or []
    excluir_features = excluir_features or []

    validos = bloques_validos(bloques)
    for b in [*solo_bloques, *excluir_bloques]:
        if b not in validos:
            raise ValueError(
                f"bloque inexistente: '{b}' (disponibles: {sorted(validos)})"
            )
    faltan = sorted(set(excluir_features) - set(columnas))
    if faltan:
        raise ValueError(f"excluir_features con columnas inexistentes: {faltan}")
    claves = sorted(set(columnas) & set(CLAVES))
    if claves:
        raise ValueError(f"las claves/target no son candidatas a feature: {claves}")

    out = []
    for c in columnas:
        b = bloques[c]
        if solo_bloques and not any(_pertenece(b, s) for s in solo_bloques):
            continue
        if any(_pertenece(b, e) for e in excluir_bloques) or c in excluir_features:
            continue
        out.append(c)
    if not out:
        raise ValueError("la selección dejó 0 features")
    return out
