"""Etapa canaritos: ¿qué features no son mejores que el azar? (arboles/z0607_ParadigmShift).

La caché de features debe incluir columnas aleatorias (`fe.canaritos.activo`, bloque
`8_canaritos`). Se entrena con TODAS las features más los canaritos y se promedia la
importancia (ganancia normalizada de cada modelo) entre semillas y folds. Una feature
cuya importancia no supera a la del mejor canarito no distingue señal de ruido: es
candidata a salir. Se escribe `features_ruidosas.txt`, que `excluir_features` acepta como
`archivo:<ruta>` para evaluarlo con la etapa ablacion o usarlo en el modelo final.

Usa `ablacion.n_semillas` y los hiperparámetros de `final.params_desde`.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.features import columnas_seleccionadas, construir_features
from competencia_1.pipeline.features.seleccion import cargar_bloques
from competencia_1.pipeline.modelo import (
    crear_dataset,
    entrenar_dataset,
    params_lgbm,
    resolver_params,
)
from competencia_1.pipeline.semillas import generar_semillas
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.canaritos")

PREFIJO = "canarito_"


def importancias_promedio(
    cfg: cfgmod.Config,
    parquet: Path,
    features: list[str],
    params: dict,
    semillas: list[int],
) -> pl.DataFrame:
    """Ganancia normalizada (suma 1 por modelo) promediada entre semillas x folds."""
    por_modelo = []
    for seed in semillas:
        p, n_iter = params_lgbm(cfg, params, seed)
        for fold in cfgmod.folds(cfg):
            tr = cargar_particion(
                parquet,
                features,
                list(fold.train),
                cfg.target.positivos,
                undersampling=cfg.dataset.undersampling,
                seed=seed,
            )
            modelo = entrenar_dataset(
                crear_dataset(tr.X, tr.y, features, cfg.lgbm.fijos), p, n_iter
            )
            gain = modelo.feature_importance(importance_type="gain").astype(float)
            por_modelo.append(gain / gain.sum() if gain.sum() > 0 else gain)
        log.info("semilla %d: modelos entrenados", seed)
    m = np.vstack(por_modelo)
    return pl.DataFrame(
        {"feature": features, "gain_medio": m.mean(axis=0), "gain_std": m.std(axis=0)}
    )


def clasificar(imp: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Marca las features con importancia <= la del mejor canarito. Devuelve (tabla, umbral)."""
    es_can = pl.col("feature").str.starts_with(PREFIJO)
    canarios = imp.filter(es_can)
    if canarios.height < 3:
        raise ValueError(
            f"hacen falta al menos 3 canaritos en las features (hay {canarios.height}): "
            "activá fe.canaritos y no los excluyas con dataset.excluir_*"
        )
    umbral = float(canarios["gain_medio"].max())
    tabla = (
        imp.sort("gain_medio", descending=True)
        .with_row_index("ranking", offset=1)
        .with_columns(
            es_canarito=es_can,
            bajo_umbral=pl.col("gain_medio") <= umbral,
        )
    )
    return tabla, umbral


def etapa_canaritos(cfg: cfgmod.Config, run: Run) -> None:
    parquet, fe_hash = construir_features(cfg)
    features = columnas_seleccionadas(cfg, parquet)
    params = resolver_params(cfg, fe_hash)
    semillas = generar_semillas(cfg.semilla_maestra, cfg.ablacion.n_semillas)
    n_can = sum(f.startswith(PREFIJO) for f in features)
    log.info(
        "canaritos: %d features (%d canaritos) x %d semillas x %d folds",
        len(features),
        n_can,
        len(semillas),
        cfg.periodos.n_folds,
    )
    imp = importancias_promedio(cfg, parquet, features, params, semillas)
    tabla, umbral = clasificar(imp)

    bloques = cargar_bloques(parquet, features)
    tabla = tabla.with_columns(
        bloque=pl.col("feature").map_elements(
            lambda f: "canarito" if f.startswith(PREFIJO) else bloques[f],
            return_dtype=pl.Utf8,
        )
    )
    reales = tabla.filter(~pl.col("es_canarito"))
    ruidosas = reales.filter(pl.col("bajo_umbral"))
    resumen = (
        reales.group_by("bloque")
        .agg(
            n=pl.len(),
            ruidosas=pl.col("bajo_umbral").sum(),
            gain_total=pl.col("gain_medio").sum(),
        )
        .with_columns(pct_ruidosas=(pl.col("ruidosas") / pl.col("n") * 100).round(1))
        .sort("bloque")
    )
    log.info(
        "umbral (mejor canarito): %.3e | %d de %d features reales no lo superan (%.0f%%)",
        umbral,
        ruidosas.height,
        reales.height,
        100 * ruidosas.height / reales.height,
    )
    for r in resumen.iter_rows(named=True):
        log.info(
            "  %-20s %3d features, %3d bajo el umbral (%5.1f%%), gain total %.3f",
            r["bloque"],
            r["n"],
            r["ruidosas"],
            r["pct_ruidosas"],
            r["gain_total"],
        )

    tabla.write_csv(run.dir / "canaritos.csv")
    resumen.write_csv(run.dir / "canaritos_bloques.csv")
    (run.dir / "features_ruidosas.txt").write_text(
        "# features cuya importancia no supera a la del mejor canarito "
        f"(fe_hash={fe_hash}, umbral={umbral:.3e})\n"
        + "\n".join(ruidosas["feature"])
        + "\n",
        encoding="utf-8",
    )
    run.registrar(
        fe_hash=fe_hash,
        canaritos={
            "umbral": umbral,
            "n_canaritos": n_can,
            "n_features_reales": reales.height,
            "n_ruidosas": ruidosas.height,
            "semillas": semillas,
            "lista": "features_ruidosas.txt",
        },
    )
