"""Etapa final (semillerío): entrena N modelos con los mismos params/features y distinta semilla.

Cada modelo predice el mes objetivo y deja `probas/s<seed>.parquet`; si hay 2 o más
semillas también `probas/promedio.parquet` (promedio de probabilidades).
Entrena SIN undersampling y reescala min_data_in_leaf si se ajustó con undersampling.
"""

import logging

import numpy as np
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.features import columnas_modelo, construir_features
from competencia_1.pipeline.modelo import (
    entrenar_lgbm,
    escalar_min_data,
    importancias,
    params_lgbm,
    predecir,
    resolver_params,
)
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.final")


def etapa_final(cfg: cfgmod.Config, run: Run) -> None:
    parquet, fe_hash = construir_features(cfg)
    features = columnas_modelo(parquet)
    params = resolver_params(cfg, fe_hash)
    if cfg.final.reescalar_min_data:
        params = escalar_min_data(params, cfg.dataset.undersampling)
    semillas = cfgmod.semillas_finales(cfg)
    meses = cfgmod.meses_final(cfg)
    log.info("final: meses=%s semillas=%s params=%s", meses, semillas, params)

    train = cargar_particion(parquet, features, meses, cfg.target.positivos)
    objetivo = cargar_particion(
        parquet, features, [cfg.periodos.target], cfg.target.positivos, con_target=False
    )
    log.info(
        "train: %d filas, %d positivos (%.2f%%) | a predecir: %d clientes",
        len(train),
        train.y.sum(),
        100 * train.y.mean(),
        len(objetivo),
    )

    dir_probas, dir_modelos = run.dir / "probas", run.dir / "modelos"
    dir_probas.mkdir(exist_ok=True)
    dir_modelos.mkdir(exist_ok=True)

    todas = []
    for i, seed in enumerate(semillas):
        p, n_iter = params_lgbm(cfg, params, seed)
        modelo = entrenar_lgbm(train.X, train.y, features, p, n_iter)
        prob = predecir(modelo, objetivo.X)
        todas.append(prob)
        df = pl.DataFrame({"numero_de_cliente": objetivo.ids, "prob": prob})
        destino = dir_probas / f"s{seed}.parquet"
        df.write_parquet(destino)
        modelo.save_model(str(dir_modelos / f"s{seed}.txt"))
        if i == 0:
            importancias(modelo).write_csv(run.dir / "importancias.csv")
        for f in (destino, dir_modelos / f"s{seed}.txt"):
            run.registrar_archivo(f)
        log.info(
            "semilla %d (%d/%d): prob media=%.5f",
            seed,
            i + 1,
            len(semillas),
            prob.mean(),
        )

    if len(todas) >= 2:
        promedio = pl.DataFrame(
            {"numero_de_cliente": objetivo.ids, "prob": np.mean(todas, axis=0)}
        )
        destino = dir_probas / "promedio.parquet"
        promedio.write_parquet(destino)
        run.registrar_archivo(destino)

    run.registrar(
        modelo={
            "params_final": params,
            "n_features": len(features),
            "meses_final": meses,
        }
    )
