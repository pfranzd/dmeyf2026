"""Ensamble de runs: combina las probabilidades finales de varios runs en un solo ranking.

Cada run aporta su `probas/promedio.parquet` (o el parquet que se indique). Dos métodos:
- rank (default): cada modelo se convierte en su rank normalizado en (0, 1] y se promedia.
  Es el adecuado cuando los modelos tienen escalas de probabilidad distintas (p. ej. 15 vs
  113 hojas, o distinto min_data_in_leaf): solo importa el orden, que es lo que decide el corte.
- prob: promedio ponderado de probabilidades (sesga hacia el modelo más "confiado").
"""

import itertools
import statistics

import polars as pl

from competencia_1.pipeline.salida import seleccionar_top, validar_probas

METODOS = ("rank", "prob")


def normalizar_rank(probas: pl.DataFrame) -> pl.DataFrame:
    """prob -> rank promedio / n, en (0, 1]: mayor prob, mayor valor. Empates comparten rank."""
    n = probas.height
    return probas.with_columns(
        (pl.col("prob").rank(method="average") / n).alias("prob")
    )


def combinar(
    probas: list[pl.DataFrame], metodo: str = "rank", pesos: list[float] | None = None
) -> pl.DataFrame:
    """Promedio ponderado de los modelos sobre exactamente el mismo conjunto de clientes."""
    if metodo not in METODOS:
        raise ValueError(f"metodo={metodo!r}; opciones: {METODOS}")
    if len(probas) < 2:
        raise ValueError("un ensamble necesita al menos 2 modelos")
    pesos = pesos or [1.0] * len(probas)
    if len(pesos) != len(probas) or any(p <= 0 for p in pesos):
        raise ValueError(f"pesos inválidos: {pesos} para {len(probas)} modelos")
    for p in probas:
        validar_probas(p)
    ids = set(probas[0]["numero_de_cliente"].to_list())
    for i, p in enumerate(probas[1:], start=1):
        if set(p["numero_de_cliente"].to_list()) != ids:
            raise ValueError(
                f"el modelo {i} no tiene los mismos clientes que el modelo 0"
            )

    total = sum(pesos)
    partes = []
    for i, (p, w) in enumerate(zip(probas, pesos, strict=True)):
        q = normalizar_rank(p) if metodo == "rank" else p
        partes.append(
            q.select("numero_de_cliente", (pl.col("prob") * w / total).alias(f"m{i}"))
        )
    out = partes[0]
    for q in partes[1:]:
        out = out.join(q, on="numero_de_cliente", how="inner")
    out = out.select(
        "numero_de_cliente",
        pl.sum_horizontal(pl.exclude("numero_de_cliente")).alias("prob"),
    ).with_columns(pl.col("prob").clip(0.0, 1.0))  # redondeo de punto flotante
    validar_probas(out)
    return out.sort("numero_de_cliente")


def coincidencias(
    modelos: dict[str, pl.DataFrame], ensamble: pl.DataFrame, envios: list[int]
) -> dict:
    """Jaccard del top-N entre pares de modelos y de cada modelo contra el ensamble.

    Si los modelos ya coinciden casi del todo (Jaccard ~1), el ensamble no puede aportar.
    """
    out = {}
    for n in envios:
        tops = {k: set(seleccionar_top(v, n).to_list()) for k, v in modelos.items()}
        top_ens = set(seleccionar_top(ensamble, n).to_list())
        pares = {
            f"{a}|{b}": len(tops[a] & tops[b]) / len(tops[a] | tops[b])
            for a, b in itertools.combinations(tops, 2)
        }
        out[str(n)] = {
            "jaccard_pares": pares,
            "jaccard_medio_pares": statistics.fmean(pares.values()),
            "jaccard_vs_ensamble": {
                k: len(t & top_ens) / len(t | top_ens) for k, t in tops.items()
            },
        }
    return out
