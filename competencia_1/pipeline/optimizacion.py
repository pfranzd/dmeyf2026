"""Etapa optuna: búsqueda de hiperparámetros con validación temporal.

Alineado con la metodología de la materia, no un Optuna genérico:
- Cada trial se evalúa en los folds temporales del config (train en meses anteriores,
  validación en el mes que está `gap` meses después): nada de CV dentro del mismo mes.
- El objetivo (`optuna.objetivo`) es por defecto la ganancia meseta sobre BAJA+2,
  promediada entre folds y normalizada por la cantidad de clientes de cada mes. También
  puede ser el AUC sobre BAJA+2 (media simple entre folds).
- Con `optuna.pruning` el primer fold decide si el trial sigue (MedianPruner): los folds
  deben ir de menor a mayor costo.
- `num_iterations` es un hiperparámetro más: no hay early stopping sobre el mes de
  validación, que lo filtraría hacia el modelo.
- Todos los trials usan la misma semilla de LightGBM, así comparan hiperparámetros y
  no azar. El estudio vive en SQLite y se reanuda: `n_trials` es el TOTAL deseado.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.dataset import cargar_particion
from competencia_1.pipeline.features import columnas_seleccionadas, construir_features
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


def _resolver_db() -> Path:
    """Base de Optuna. DMEYF_DB la cambia (p. ej. work/entrega/db para aislar una reproducción)."""
    p = Path(os.environ.get("DMEYF_DB") or cfgmod.RAIZ / "db" / "competencia_1.db")
    return p if p.is_absolute() else cfgmod.RAIZ / p


DB_PATH = _resolver_db()


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
    cfg: cfgmod.Config,
    parquet: Path,
    features: list[str],
    seed: int,
    reusar_valid: list[FoldPreparado] | None = None,
) -> list[FoldPreparado]:
    """Carga cada fold (train con undersampling de semilla `seed`, valid completo).

    `reusar_valid`: folds ya preparados con otra semilla; el mes de validación no
    depende de la semilla, así que se copia en vez de volver a leerlo del parquet.
    """
    out = []
    for i, fold in enumerate(cfgmod.folds(cfg)):
        train = cargar_particion(
            parquet,
            features,
            list(fold.train),
            cfg.target.positivos,
            undersampling=cfg.dataset.undersampling,
            seed=seed,
        )
        if reusar_valid is not None:
            X_valid, es_baja2 = reusar_valid[i].X_valid, reusar_valid[i].es_baja2_valid
        else:
            valid = cargar_particion(
                parquet, features, [fold.valid], cfg.target.positivos
            )
            X_valid, es_baja2 = valid.X, valid.es_baja2
        dtrain = crear_dataset(train.X, train.y, features, cfg.lgbm.fijos)
        dtrain.construct()
        out.append(
            FoldPreparado(
                fold, dtrain, len(train), int(train.y.sum()), X_valid, es_baja2
            )
        )
        log.info(
            "fold preparado (seed %d): train=%s (n=%d, pos=%d) valid=%s (n=%d)",
            seed,
            fold.train,
            len(train),
            int(train.y.sum()),
            fold.valid,
            len(es_baja2),
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


def valor_objetivo(resultados: list[dict], objetivo: str = "ganancia") -> float:
    """Valor que maximiza Optuna a partir del resumen de cada fold.

    `ganancia`: media de la ganancia meseta entre folds, llevada a un mes de tamaño
    promedio. La ganancia crece con la cantidad de clientes del mes: se normaliza por n
    para que ningún fold pese más solo por tener más clientes. `auc`: media simple.
    """
    if objetivo == "auc":
        return float(np.mean([r["auc"] for r in resultados]))
    if objetivo != "ganancia":
        raise ValueError(f"objetivo inválido: {objetivo}")
    n_ref = np.mean([r["n"] for r in resultados])
    return float(np.mean([r["ganancia_meseta"] * n_ref / r["n"] for r in resultados]))


def puntuar_folds(
    cfg: cfgmod.Config, preps: list[FoldPreparado], params: dict, seed: int
) -> list[np.ndarray]:
    """Entrena con `params` y `seed` y devuelve el score del mes de validación de cada fold."""
    p, n_iter = params_lgbm(cfg, params, seed)
    return [
        predecir(entrenar_dataset(prep.dtrain, p, n_iter), prep.X_valid)
        for prep in preps
    ]


def resumir_folds(
    cfg: cfgmod.Config, preps: list[FoldPreparado], scores: list[np.ndarray]
) -> list[dict]:
    """Ganancia meseta y envíos óptimos por fold a partir de los scores."""
    resultados = []
    for prep, score in zip(preps, scores, strict=True):
        r = resumir_scores(prep.es_baja2_valid, score, cfg.optuna.ventana_meseta)
        r.pop("curva")
        r["valid"] = prep.fold.valid
        resultados.append(r)
    return resultados


def evaluar_trial(
    cfg: cfgmod.Config, preps: list[FoldPreparado], params: dict, seed: int
) -> list[dict]:
    return resumir_folds(cfg, preps, puntuar_folds(cfg, preps, params, seed))


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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        if cfg.optuna.pruning
        else optuna.pruners.NopPruner(),
    )
    # los podados cuentan: son trials ya gastados del presupuesto
    terminados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    hechos = sum(t.state in terminados for t in study.trials)
    study.sampler = optuna.samplers.TPESampler(seed=cfg.semilla_maestra + hechos)
    return study, hechos


def _guardar_attrs(trial: optuna.Trial, resultados: list[dict]) -> None:
    trial.set_user_attr(
        "ganancia_meseta_folds", [r["ganancia_meseta"] for r in resultados]
    )
    trial.set_user_attr(
        "envios_optimos_folds", [r["envios_optimos"] for r in resultados]
    )
    trial.set_user_attr("auc_folds", [r["auc"] for r in resultados])
    trial.set_user_attr("n_valid_folds", [r["n"] for r in resultados])


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

    objetivo_cfg = cfg.optuna.objetivo
    fmt = "%.4f" if objetivo_cfg == "auc" else "%.0f"

    def objetivo(trial: optuna.Trial) -> float:
        params = sugerir(trial, cfg.optuna.espacio)
        completos = {
            **cfg.lgbm.manual,
            **params,
        }  # num_iterations fuera del espacio -> manual
        p, n_iter = params_lgbm(cfg, completos, cfg.semilla_maestra)
        resultados: list[dict] = []
        for i, prep in enumerate(preps):
            score = predecir(entrenar_dataset(prep.dtrain, p, n_iter), prep.X_valid)
            resultados += resumir_folds(cfg, [prep], [score])
            if cfg.optuna.pruning and i < len(preps) - 1:
                trial.report(valor_objetivo(resultados, objetivo_cfg), i)
                if trial.should_prune():
                    _guardar_attrs(trial, resultados)
                    raise optuna.TrialPruned
        _guardar_attrs(trial, resultados)
        return valor_objetivo(resultados, objetivo_cfg)

    def reportar(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            log.info(
                "trial %d: podado tras el primer fold (%s) | %s",
                trial.number,
                trial.user_attrs.get("ganancia_meseta_folds"),
                trial.params,
            )
            return
        log.info(
            "trial %d: valor=" + fmt + " envios=%s | mejor=" + fmt + " (trial %d) | %s",
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


def cargar_estudio(nombre: str, storage: Path) -> optuna.Study:
    try:
        return optuna.load_study(
            study_name=nombre, storage=f"sqlite:///{storage.as_posix()}"
        )
    except KeyError as e:
        raise FileNotFoundError(
            f"no existe el estudio '{nombre}' en {storage}: corré antes la etapa 'optuna'"
        ) from e


def top_trials(study: optuna.Study, k: int) -> list[optuna.trial.FrozenTrial]:
    completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completos:
        raise ValueError(f"el estudio '{study.study_name}' no tiene trials completos")
    return sorted(completos, key=lambda t: t.value, reverse=True)[:k]


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
        fila["auc_folds"] = json.dumps(t.user_attrs.get("auc_folds"))
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
        "auc_folds": mejor.user_attrs.get("auc_folds"),
        "objetivo": cfg.optuna.objetivo,
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
    features = columnas_seleccionadas(cfg, parquet)
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
        "mejor trial %d: %s | para usarlo: final.params_desde=optuna:%s (o optuna:auto)",
        info["trial"],
        f"{info['valor']:.4f}"
        if cfg.optuna.objetivo == "auc"
        else f"{info['valor']:.0f}",
        nombre,
    )
