"""Etapa optuna: búsqueda de hiperparámetros con validación temporal.

Alineado con la metodología de la materia, no un Optuna genérico:
- Cada trial se evalúa en los folds temporales del config (train en meses anteriores,
  validación en el mes que está `gap` meses después): nada de CV dentro del mismo mes.
- El objetivo es la ganancia meseta sobre BAJA+2 (no AUC), promediada entre folds y
  normalizada por la cantidad de clientes de cada mes.
- `num_iterations` es un hiperparámetro más: no hay early stopping sobre el mes de
  validación, que lo filtraría hacia el modelo.
- Todos los trials usan la misma semilla de LightGBM, así comparan hiperparámetros y
  no azar. El estudio vive en SQLite y se reanuda: `n_trials` es el TOTAL deseado.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.features import columnas_modelo, construir_features
from competencia_1.pipeline.metricas import resumir_scores
from competencia_1.pipeline.modelo import (
    PARAMS_DIR,
    crear_dataset,
    entrenar_dataset,
    nombre_estudio,
    params_lgbm,
    predecir,
)
from competencia_1.pipeline.periodos import Fold
from competencia_1.pipeline.tracking import Run, ruta_relativa

log = logging.getLogger("competencia_1.optuna")

DB_PATH = cfgmod.RAIZ / "db" / "competencia_1.db"


@dataclass
class FoldPreparado:
    """Datos de un fold ya cargados: se arman una vez y se reusan en todos los trials."""

    fold: Fold
    dtrain: lgb.Dataset  # ya binarizado
    n_train: int
    n_train_pos: int
    X_valid: np.ndarray
    es_baja2_valid: np.ndarray


def preparar_folds(
    cfg: cfgmod.Config, parquet: Path, features: list[str], seed: int
) -> list[FoldPreparado]:
    out = []
    for fold in cfgmod.folds(cfg):
        train = cargar_particion(
            parquet,
            features,
            list(fold.train),
            cfg.target.positivos,
            undersampling=cfg.dataset.undersampling,
            seed=seed,
        )
        valid = cargar_particion(parquet, features, [fold.valid], cfg.target.positivos)
        dtrain = crear_dataset(train.X, train.y, features, cfg.lgbm.fijos)
        dtrain.construct()
        out.append(
            FoldPreparado(
                fold, dtrain, len(train), int(train.y.sum()), valid.X, valid.es_baja2
            )
        )
        log.info(
            "fold preparado: train=%s (n=%d, pos=%d) valid=%s (n=%d)",
            fold.train,
            len(train),
            int(train.y.sum()),
            fold.valid,
            len(valid),
        )
    return out


def sugerir(trial: optuna.Trial, espacio: dict) -> dict:
    """Hiperparámetros de un trial según `optuna.espacio` del config."""
    params = {}
    for nombre, e in espacio.items():
        log_scale = bool(e.get("log", False))
        if e["tipo"] == "int":
            params[nombre] = trial.suggest_int(
                nombre, int(e["low"]), int(e["high"]), log=log_scale
            )
        else:
            params[nombre] = trial.suggest_float(
                nombre, float(e["low"]), float(e["high"]), log=log_scale
            )
    return params


def valor_objetivo(resultados: list[dict]) -> float:
    """Media de la ganancia meseta entre folds, llevada a un mes de tamaño promedio.

    La ganancia crece con la cantidad de clientes del mes: se normaliza por n para que
    ningún fold pese más solo por tener más clientes.
    """
    n_ref = np.mean([r["n"] for r in resultados])
    return float(np.mean([r["ganancia_meseta"] * n_ref / r["n"] for r in resultados]))


def evaluar_trial(
    cfg: cfgmod.Config, preps: list[FoldPreparado], params: dict, seed: int
) -> list[dict]:
    p, n_iter = params_lgbm(cfg, params, seed)
    resultados = []
    for prep in preps:
        modelo = entrenar_dataset(prep.dtrain, p, n_iter)
        r = resumir_scores(
            prep.es_baja2_valid,
            predecir(modelo, prep.X_valid),
            cfg.optuna.ventana_meseta,
        )
        r.pop("curva")
        r["valid"] = prep.fold.valid
        resultados.append(r)
    return resultados


def abrir_estudio(
    cfg: cfgmod.Config, nombre: str, storage: Path
) -> tuple[optuna.Study, int]:
    """Crea o reanuda el estudio. Devuelve (estudio, trials completos ya hechos).

    El sampler se resiembra con semilla + trials hechos: si se reanudara con la misma
    semilla, el generador volvería a empezar y repetiría los primeros sorteos al azar.
    """
    storage.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=nombre,
        storage=f"sqlite:///{storage.as_posix()}",
        direction="maximize",
        load_if_exists=True,
    )
    hechos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.sampler = optuna.samplers.TPESampler(seed=cfg.semilla_maestra + hechos)
    return study, hechos


def correr_estudio(
    cfg: cfgmod.Config,
    study: optuna.Study,
    preps: list[FoldPreparado],
    hechos: int,
) -> None:
    restantes = cfg.optuna.n_trials - hechos
    if restantes <= 0:
        log.info(
            "el estudio ya tiene %d trials (objetivo %d): nada que correr",
            hechos,
            cfg.optuna.n_trials,
        )
        return

    def objetivo(trial: optuna.Trial) -> float:
        params = sugerir(trial, cfg.optuna.espacio)
        completos = {
            **cfg.lgbm.manual,
            **params,
        }  # num_iterations fuera del espacio -> manual
        resultados = evaluar_trial(cfg, preps, completos, cfg.semilla_maestra)
        trial.set_user_attr(
            "ganancia_meseta_folds", [r["ganancia_meseta"] for r in resultados]
        )
        trial.set_user_attr(
            "envios_optimos_folds", [r["envios_optimos"] for r in resultados]
        )
        trial.set_user_attr("n_valid_folds", [r["n"] for r in resultados])
        return valor_objetivo(resultados)

    def reportar(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        log.info(
            "trial %d: valor=%.0f envios=%s | mejor=%.0f (trial %d) | %s",
            trial.number,
            trial.value,
            trial.user_attrs["envios_optimos_folds"],
            study.best_value,
            study.best_trial.number,
            trial.params,
        )

    log.info(
        "optimizando: %d trials más (hechos: %d, objetivo: %d)",
        restantes,
        hechos,
        cfg.optuna.n_trials,
    )
    study.optimize(objetivo, n_trials=restantes, callbacks=[reportar])


def tabla_trials(study: optuna.Study) -> pl.DataFrame:
    filas = []
    for t in study.trials:
        fila = {"trial": t.number, "estado": t.state.name, "valor": t.value}
        fila.update({f"param_{k}": v for k, v in t.params.items()})
        fila["ganancia_meseta_folds"] = json.dumps(
            t.user_attrs.get("ganancia_meseta_folds")
        )
        fila["envios_optimos_folds"] = json.dumps(
            t.user_attrs.get("envios_optimos_folds")
        )
        filas.append(fila)
    return pl.DataFrame(filas)


def exportar_mejores(
    cfg: cfgmod.Config,
    study: optuna.Study,
    fe_hash: str,
    dir_params: Path,
) -> dict:
    """Escribe params/<estudio>.json: lo que la etapa final reutiliza y verifica."""
    mejor = study.best_trial
    params = {**cfg.lgbm.manual, **mejor.params}
    info = {
        "study_name": study.study_name,
        "fe_hash": fe_hash,
        "valor": mejor.value,
        "trial": mejor.number,
        "n_trials": len(study.trials),
        "params": params,
        "ganancia_meseta_folds": mejor.user_attrs.get("ganancia_meseta_folds"),
        "envios_optimos_folds": mejor.user_attrs.get("envios_optimos_folds"),
        "undersampling": cfg.dataset.undersampling,
        "positivos": cfg.target.positivos,
        "folds": [
            {"train": list(f.train), "valid": f.valid} for f in cfgmod.folds(cfg)
        ],
        "semilla": cfg.semilla_maestra,
    }
    dir_params.mkdir(parents=True, exist_ok=True)
    (dir_params / f"{study.study_name}.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info


def etapa_optuna(cfg: cfgmod.Config, run: Run) -> None:
    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )  # los trials los loguea `reportar`
    parquet, fe_hash = construir_features(cfg)
    features = columnas_modelo(parquet)
    nombre = nombre_estudio(cfg, fe_hash)
    study, hechos = abrir_estudio(cfg, nombre, DB_PATH)
    log.info("estudio %s (%d features)", nombre, len(features))

    if hechos < cfg.optuna.n_trials:  # no cargar datos si no hay nada que correr
        preps = preparar_folds(cfg, parquet, features, cfg.semilla_maestra)
        correr_estudio(cfg, study, preps, hechos)
    else:
        correr_estudio(cfg, study, [], hechos)

    info = exportar_mejores(cfg, study, fe_hash, PARAMS_DIR)
    tabla_trials(study).write_csv(run.dir / "optuna_trials.csv")
    (run.dir / "best_params.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run.registrar(
        fe_hash=fe_hash,
        optuna={
            "study_name": nombre,
            "storage": ruta_relativa(DB_PATH),
            "n_trials": info["n_trials"],
            "mejor_valor": info["valor"],
            "mejor_trial": info["trial"],
            "mejores_params": info["params"],
        },
    )
    log.info(
        "mejor trial %d: %.0f | para usarlo: final.params_desde=optuna:%s (o optuna:auto)",
        info["trial"],
        info["valor"],
        nombre,
    )
