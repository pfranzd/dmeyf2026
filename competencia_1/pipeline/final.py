"""Etapa final (semillerío): entrena N modelos con los mismos params/features y distinta semilla.

Cada modelo predice el mes objetivo y deja `probas/s<seed>.parquet`; si hay 2 o más
semillas también `probas/promedio.parquet` (promedio de probabilidades).
Entrena SIN undersampling y reescala min_data_in_leaf si se ajustó con undersampling.

Ensamble de hiperparámetros: con `final.params_desde: top:<k>` (los k mejores trials del
estudio) o un json con la clave `ensamble`, se entrena cada conjunto × cada semilla. Los
modelos sueltos quedan en `probas/ensamble/<conjunto>_s<seed>.parquet` y solo el
`promedio.parquet` queda en `probas/` (de ahí sale el CSV).
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion, contar_filas
from competencia_1.pipeline.features import columnas_seleccionadas, construir_features
from competencia_1.pipeline.modelo import (
    cargar_json_params,
    entrenar_lgbm,
    escalar_min_data,
    escalar_min_data_factor,
    importancias,
    nombre_estudio,
    params_lgbm,
    predecir,
    resolver_params,
    ruta_params,
)
from competencia_1.pipeline.optimizacion import DB_PATH, cargar_estudio, top_trials
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.final")


def resolver_conjunto(cfg: cfgmod.Config, fe_hash: str) -> list[tuple[str, dict]]:
    """[(etiqueta, params)] a entrenar. Un solo conjunto => etiqueta vacía (flujo de siempre)."""
    origen = cfg.final.params_desde
    if origen.startswith("top:"):
        estudio = cargar_estudio(nombre_estudio(cfg, fe_hash), DB_PATH)
        trials = top_trials(estudio, int(origen[4:]))
        return [(f"t{t.number}", {**cfg.lgbm.manual, **t.params}) for t in trials]
    if origen.startswith("archivo:"):
        data = cargar_json_params(cfg, ruta_params(cfg, fe_hash), fe_hash)
        if "ensamble" in data:
            return [(m["etiqueta"], dict(m["params"])) for m in data["ensamble"]]
    return [("", resolver_params(cfg, fe_hash))]


def _factor_filas(cfg: cfgmod.Config, parquet: Path, meses_final: list[int]) -> float:
    """Filas del final / filas de train del último fold (con su undersampling)."""
    ultimo = cfgmod.folds(cfg)[-1]
    filas_fold = contar_filas(parquet, list(ultimo.train), cfg.dataset.undersampling)
    return contar_filas(parquet, meses_final) / filas_fold


def etapa_final(cfg: cfgmod.Config, run: Run) -> None:
    parquet, fe_hash = construir_features(cfg)
    features = columnas_seleccionadas(cfg, parquet)
    conjuntos = resolver_conjunto(cfg, fe_hash)
    semillas = cfgmod.semillas_finales(cfg)
    meses = cfgmod.meses_final(cfg)
    if cfg.final.reescalar_min_data == "filas":
        factor = _factor_filas(cfg, parquet, meses)
        log.info("min_data_in_leaf x %.3f (filas del final / filas de train)", factor)
        conjuntos = [(e, escalar_min_data_factor(p, factor)) for e, p in conjuntos]
    elif cfg.final.reescalar_min_data:
        conjuntos = [
            (e, escalar_min_data(p, cfg.dataset.undersampling)) for e, p in conjuntos
        ]
    ensamble = len(conjuntos) > 1
    log.info(
        "final: meses=%s semillas=%s conjuntos=%d params=%s",
        meses,
        semillas,
        len(conjuntos),
        conjuntos[0][1] if not ensamble else "(ver meta.json)",
    )

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
    dir_sueltas = dir_probas / "ensamble" if ensamble else dir_probas
    dir_sueltas.mkdir(exist_ok=True)

    todas = []
    total = len(conjuntos) * len(semillas)
    for etiqueta, params in conjuntos:
        for seed in semillas:
            nombre = f"{etiqueta}_s{seed}" if etiqueta else f"s{seed}"
            p, n_iter = params_lgbm(cfg, params, seed)
            modelo = entrenar_lgbm(train.X, train.y, features, p, n_iter)
            prob = predecir(modelo, objetivo.X)
            todas.append(prob)
            df = pl.DataFrame({"numero_de_cliente": objetivo.ids, "prob": prob})
            destino = dir_sueltas / f"{nombre}.parquet"
            df.write_parquet(destino)
            modelo.save_model(str(dir_modelos / f"{nombre}.txt"))
            if len(todas) == 1:
                importancias(modelo).write_csv(run.dir / "importancias.csv")
            for f in (destino, dir_modelos / f"{nombre}.txt"):
                run.registrar_archivo(f)
            log.info(
                "modelo %s (%d/%d): prob media=%.5f",
                nombre,
                len(todas),
                total,
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
            "params_final": (
                [{"etiqueta": e, "params": p} for e, p in conjuntos]
                if ensamble
                else conjuntos[0][1]
            ),
            "n_features": len(features),
            "meses_final": meses,
        }
    )
