"""Etapa ablacion: ¿qué familias de features aportan? (comparación pareada por semilla).

Cada variante es un subconjunto de columnas de la caché de features (ver
features/seleccion.py: equivale exactamente a reconstruir el FE sin esas familias, sin
pagar 10 minutos por variante). Para que las diferencias no se confundan con el ruido:
- todas las variantes usan las MISMAS N semillas y, por semilla, la MISMA muestra de
  undersampling: la diferencia entre dos variantes está pareada;
- se reportan la diferencia media ± error estándar, en cuántas semillas gana la variante,
  la diferencia para el promedio de semillas y la diferencia en CADA fold (mes).

Limitaciones: los hiperparámetros se fijan (los de `final.params_desde`, normalmente
optimizados con el conjunto completo: eso favorece a la referencia) y hay solo 2 meses de
validación: el error estándar mide el ruido de las semillas, no la variación entre meses.
"""

import gc
import json
import logging
import math
import statistics
from pathlib import Path

import numpy as np
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.features import columnas_seleccionadas, construir_features
from competencia_1.pipeline.features.seleccion import (
    cargar_bloques,
    expandir_lista,
    seleccionar_columnas,
)
from competencia_1.pipeline.metricas import resumir_scores
from competencia_1.pipeline.modelo import (
    crear_dataset,
    entrenar_dataset,
    params_lgbm,
    predecir,
    resolver_params,
)
from competencia_1.pipeline.optimizacion import valor_objetivo
from competencia_1.pipeline.semillas import generar_semillas
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.ablacion")


def _m(x: float) -> str:
    return f"{x / 1e6:.1f}M"


def valor_folds(
    cfg: cfgmod.Config, es_baja2: list[np.ndarray], scores: list[np.ndarray]
):
    """(valor objetivo, ganancia meseta normalizada por fold) de un modelo."""
    res = []
    for es, sc in zip(es_baja2, scores, strict=True):
        r = resumir_scores(es, sc, cfg.optuna.ventana_meseta)
        res.append({"n": r["n"], "ganancia_meseta": r["ganancia_meseta"]})
    n_ref = float(np.mean([r["n"] for r in res]))
    por_fold = [r["ganancia_meseta"] * n_ref / r["n"] for r in res]
    return valor_objetivo(res), por_fold


def evaluar_variantes(
    cfg: cfgmod.Config,
    parquet: Path,
    features: list[str],
    columnas: dict[str, list[str]],
    params: dict,
    semillas: list[int],
) -> dict[str, dict]:
    """Por variante: valor por semilla, valor de cada fold y score del promedio de semillas."""
    pos = {c: i for i, c in enumerate(features)}
    indices = {n: [pos[c] for c in cols] for n, cols in columnas.items()}
    folds = cfgmod.folds(cfg)
    valid = [
        cargar_particion(parquet, features, [f.valid], cfg.target.positivos)
        for f in folds
    ]
    es_baja2 = [v.es_baja2 for v in valid]

    valores = {n: {} for n in columnas}  # variante -> semilla -> valor
    por_fold = {
        n: {} for n in columnas
    }  # variante -> semilla -> [valor fold 0, 1, ...]
    suma_scores = {n: [np.zeros(len(e)) for e in es_baja2] for n in columnas}

    for i, seed in enumerate(semillas):
        trains = [
            cargar_particion(
                parquet,
                features,
                list(f.train),
                cfg.target.positivos,
                undersampling=cfg.dataset.undersampling,
                seed=seed,  # misma muestra de undersampling para todas las variantes
            )
            for f in folds
        ]
        p, n_iter = params_lgbm(cfg, params, seed)
        for nombre, ix in indices.items():
            nombres = [features[j] for j in ix]
            scores = []
            for tr, v in zip(trains, valid, strict=True):
                ds = crear_dataset(tr.X[:, ix], tr.y, nombres, cfg.lgbm.fijos)
                scores.append(predecir(entrenar_dataset(ds, p, n_iter), v.X[:, ix]))
            for f, sc in enumerate(scores):
                suma_scores[nombre][f] += sc
            valores[nombre][seed], por_fold[nombre][seed] = valor_folds(
                cfg, es_baja2, scores
            )
        log.info(
            "semilla %d (%d/%d): %d variantes entrenadas",
            seed,
            i + 1,
            len(semillas),
            len(indices),
        )
        del trains
        gc.collect()

    out = {}
    for nombre, cols in columnas.items():
        promedio = [s / len(semillas) for s in suma_scores[nombre]]
        valor_ens, fold_ens = valor_folds(cfg, es_baja2, promedio)
        out[nombre] = {
            "n_features": len(cols),
            "valores": valores[nombre],
            "por_fold": por_fold[nombre],
            "valor_promedio_semillas": valor_ens,
            "por_fold_promedio": fold_ens,
        }
    return out


def comparar(
    resultados: dict[str, dict], referencia: str, semillas: list[int]
) -> list[dict]:
    """Estadísticos por variante y diferencias pareadas contra `referencia`."""
    ref = resultados[referencia]
    filas = []
    for nombre, r in resultados.items():
        v = [r["valores"][s] for s in semillas]
        d = [r["valores"][s] - ref["valores"][s] for s in semillas]
        n = len(semillas)
        d_std = statistics.stdev(d) if n > 1 else 0.0
        ee = d_std / math.sqrt(n) if n > 1 else 0.0
        n_folds = len(r["por_fold_promedio"])
        d_fold = [
            statistics.fmean(
                r["por_fold"][s][f] - ref["por_fold"][s][f] for s in semillas
            )
            for f in range(n_folds)
        ]
        media_d = statistics.fmean(d)
        if nombre == referencia:
            veredicto = "referencia"
        elif abs(media_d) <= 2 * ee:
            veredicto = "indistinguible"
        elif all(x * media_d > 0 for x in d_fold):
            veredicto = "mejor" if media_d > 0 else "peor"
        else:
            veredicto = "mixto"  # efecto grande pero de signo distinto según el mes
        filas.append(
            {
                "variante": nombre,
                "n_features": r["n_features"],
                "media": statistics.fmean(v),
                "mediana": statistics.median(v),
                "std": statistics.pstdev(v),
                "delta_media": media_d,
                "delta_ee": ee,
                "semillas_mejores": sum(x > 0 for x in d),
                "delta_promedio_semillas": r["valor_promedio_semillas"]
                - ref["valor_promedio_semillas"],
                "delta_por_fold": d_fold,
                "veredicto": veredicto,
            }
        )
    return filas


def etapa_ablacion(cfg: cfgmod.Config, run: Run) -> None:
    parquet, fe_hash = construir_features(cfg)
    features = columnas_seleccionadas(cfg, parquet)
    bloques = cargar_bloques(parquet, features)
    variantes = cfgmod.variantes_ablacion(cfg)
    columnas = {
        nombre: seleccionar_columnas(
            features,
            bloques,
            v.solo_bloques,
            v.excluir_bloques,
            expandir_lista(v.excluir_features, cfgmod.RAIZ),
        )
        for nombre, v in variantes.items()
    }
    params = resolver_params(cfg, fe_hash)
    semillas = generar_semillas(cfg.semilla_maestra, cfg.ablacion.n_semillas)
    log.info(
        "ablación: %d variantes x %d semillas, params=%s",
        len(columnas),
        len(semillas),
        params,
    )
    for nombre, cols in columnas.items():
        log.info("  %-28s %4d features", nombre, len(cols))

    resultados = evaluar_variantes(cfg, parquet, features, columnas, params, semillas)
    filas = comparar(resultados, cfg.ablacion.referencia, semillas)

    log.info(
        "referencia: %s (%s de media)",
        cfg.ablacion.referencia,
        _m(next(f["media"] for f in filas if f["variante"] == cfg.ablacion.referencia)),
    )
    for f in sorted(filas, key=lambda f: -f["delta_media"]):
        log.info(
            "%-28s %4d feat | media=%s std=%s | delta=%+.1fM ± %.1fM (mejor en %d/%d semillas) | "
            "folds=%s | promedio de semillas: %+.1fM | %s",
            f["variante"],
            f["n_features"],
            _m(f["media"]),
            _m(f["std"]),
            f["delta_media"] / 1e6,
            f["delta_ee"] / 1e6,
            f["semillas_mejores"],
            len(semillas),
            "/".join(f"{x / 1e6:+.1f}" for x in f["delta_por_fold"]),
            f["delta_promedio_semillas"] / 1e6,
            f["veredicto"],
        )

    pl.DataFrame(
        [
            {**f, "delta_por_fold": json.dumps([round(x) for x in f["delta_por_fold"]])}
            for f in filas
        ]
    ).write_csv(run.dir / "ablacion.csv")
    (run.dir / "ablacion.json").write_text(
        json.dumps(
            {
                "referencia": cfg.ablacion.referencia,
                "semillas": semillas,
                "params": params,
                "columnas": columnas,
                "resultados": resultados,
                "comparacion": filas,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ref = next(f for f in filas if f["variante"] == cfg.ablacion.referencia)
    run.registrar(
        fe_hash=fe_hash,
        ganancia_valid=round(ref["media"]),
        ablacion={
            "referencia": cfg.ablacion.referencia,
            "semillas": semillas,
            "variantes": {
                f["variante"]: {
                    "n_features": f["n_features"],
                    "media": f["media"],
                    "delta_media": f["delta_media"],
                    "delta_ee": f["delta_ee"],
                    "veredicto": f["veredicto"],
                }
                for f in filas
            },
        },
    )
