"""Etapa validacion: evalúa unos hiperparámetros en los folds temporales.

Cada fold entrena con meses anteriores (con el gap de 2 meses) y puntúa el mes de
validación completo. Se mide ganancia meseta sobre BAJA+2. Esta misma función
`evaluar_fold` es la que usará Optuna como objetivo.
"""

import logging
from pathlib import Path

import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.features import columnas_seleccionadas, construir_features
from competencia_1.pipeline.metricas import curva_submuestreada, resumir_scores
from competencia_1.pipeline.modelo import (
    entrenar_lgbm,
    params_lgbm,
    predecir,
    resolver_params,
)
from competencia_1.pipeline.periodos import Fold
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.validacion")


def evaluar_fold(
    cfg: cfgmod.Config,
    parquet: Path,
    features: list[str],
    fold: Fold,
    params: dict,
    seed: int,
) -> dict:
    """Entrena en `fold.train` (con undersampling) y evalúa en `fold.valid`."""
    train = cargar_particion(
        parquet,
        features,
        list(fold.train),
        cfg.target.positivos,
        undersampling=cfg.dataset.undersampling,
        seed=seed,
    )
    valid = cargar_particion(parquet, features, [fold.valid], cfg.target.positivos)
    p, n_iter = params_lgbm(cfg, params, seed)
    modelo = entrenar_lgbm(train.X, train.y, features, p, n_iter)
    res = resumir_scores(
        valid.es_baja2, predecir(modelo, valid.X), cfg.optuna.ventana_meseta
    )
    res.update(
        valid=fold.valid,
        train=list(fold.train),
        n_train=len(train),
        n_train_pos=int(train.y.sum()),
    )
    return res


def etapa_validacion(cfg: cfgmod.Config, run: Run) -> None:
    parquet, fe_hash = construir_features(cfg)
    features = columnas_seleccionadas(cfg, parquet)
    params = resolver_params(cfg, fe_hash)
    log.info("validando %d features con params %s", len(features), params)

    folds, curvas = [], []
    for i, fold in enumerate(cfgmod.folds(cfg)):
        r = evaluar_fold(cfg, parquet, features, fold, params, cfg.semilla_maestra)
        for k, g in curva_submuestreada(r.pop("curva")):
            curvas.append({"fold": i, "valid": fold.valid, "envios": k, "ganancia": g})
        log.info(
            "fold %d (train=%s valid=%s): meseta=%.0f en %d envíos | max crudo=%d en %d | "
            "train n=%d pos=%d",
            i,
            fold.train,
            fold.valid,
            r["ganancia_meseta"],
            r["envios_optimos"],
            r["ganancia_max_cruda"],
            r["envios_max_crudo"],
            r["n_train"],
            r["n_train_pos"],
        )
        folds.append(r)

    pl.DataFrame(curvas).write_csv(run.dir / "curva_ganancia_valid.csv")
    media = sum(f["ganancia_meseta"] for f in folds) / len(folds)
    run.registrar(
        ganancia_valid=round(media),
        validacion={
            "folds": folds,
            "envios_optimos": [f["envios_optimos"] for f in folds],
            "n_clientes_valid": round(sum(f["n"] for f in folds) / len(folds)),
            "ganancia_meseta_media": media,
            "params": params,
        },
    )
    log.info("ganancia meseta media en validación: %.0f", media)
