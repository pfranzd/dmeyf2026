"""Etapa estabilidad: elegir entre los mejores trials por robustez, no por un número afortunado.

Optuna reporta, para cada trial, UNA corrida con UNA semilla: el mejor trial de un
estudio es en parte el más afortunado (maldición del ganador). Acá los top-k trials se
re-entrenan con N semillas (cada una con su propio muestreo de undersampling) sobre los
mismos folds temporales y se miden:
- mediana / media / desvío / mín / máx del objetivo entre semillas,
- el optimismo de Optuna (valor de Optuna - media entre semillas),
- el beneficio de promediar las N semillas (el semillerío) contra un modelo suelto,
- la simulación public/private 30/70 (qué tanto puede engañar el leaderboard público).
Se elige por `estabilidad.criterio` (mediana o media-menos-desvío) y se exporta
params/<estudio>__estable.json, utilizable con `final.params_desde: estable:auto`.
"""

import gc
import json
import logging
import statistics
from pathlib import Path

import numpy as np
import optuna
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.features import columnas_seleccionadas, construir_features
from competencia_1.pipeline.modelo import PARAMS_DIR, nombre_estudio
from competencia_1.pipeline.optimizacion import (
    DB_PATH,
    preparar_folds,
    puntuar_folds,
    resumir_folds,
    valor_objetivo,
)
from competencia_1.pipeline.semillas import generar_semillas
from competencia_1.pipeline.simulacion import simular_public_private
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.estabilidad")


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


def estadisticos(valores: list[float]) -> dict:
    return {
        "media": statistics.fmean(valores),
        "mediana": statistics.median(valores),
        "std": statistics.pstdev(valores),
        "min": min(valores),
        "max": max(valores),
    }


def puntaje(est: dict, criterio: str) -> float:
    """Criterio de selección: nunca el máximo, que es el valor más afortunado."""
    if criterio == "mediana":
        return est["mediana"]
    if criterio == "media_menos_desvio":
        return est["media"] - est["std"]
    raise ValueError(f"criterio inválido: {criterio}")


def elegir(resumenes: dict[int, dict], criterio: str) -> int:
    """Número del trial con mejor puntaje (desempata por el trial más antiguo)."""
    return min(
        resumenes, key=lambda n: (-puntaje(resumenes[n]["estadisticos"], criterio), n)
    )


def _m(x: float) -> str:
    return f"{x / 1e6:.1f}M"


def evaluar_candidatos(
    cfg: cfgmod.Config,
    parquet: Path,
    features: list[str],
    trials: list[optuna.trial.FrozenTrial],
    semillas: list[int],
) -> tuple[
    dict[int, dict], dict[int, list[np.ndarray]], dict[int, dict[int, list]], list
]:
    """Entrena cada trial con cada semilla. Devuelve (resumen por trial, score del promedio
    por fold, scores por trial/semilla, folds preparados con la 1ª semilla)."""
    preps0 = preparar_folds(cfg, parquet, features, semillas[0])
    scores: dict[int, dict[int, list[np.ndarray]]] = {t.number: {} for t in trials}
    for i, seed in enumerate(semillas):
        preps = (
            preps0
            if i == 0
            else preparar_folds(cfg, parquet, features, seed, reusar_valid=preps0)
        )
        for t in trials:
            params = {**cfg.lgbm.manual, **t.params}
            scores[t.number][seed] = puntuar_folds(cfg, preps, params, seed)
        log.info(
            "semilla %d (%d/%d): %d trials entrenados",
            seed,
            i + 1,
            len(semillas),
            len(trials),
        )
        if i > 0:
            del preps
            gc.collect()

    resumen, ensembles = {}, {}
    for t in trials:
        por_semilla = scores[t.number]
        valores, envios = [], []
        for seed in semillas:
            res = resumir_folds(cfg, preps0, por_semilla[seed])
            valores.append(valor_objetivo(res))
            envios.append([r["envios_optimos"] for r in res])
        n_folds = len(preps0)
        promedio = [
            np.mean([por_semilla[s][f] for s in semillas], axis=0)
            for f in range(n_folds)
        ]
        ensembles[t.number] = promedio
        res_ens = resumir_folds(cfg, preps0, promedio)
        valor_ens = valor_objetivo(res_ens)
        est = estadisticos(valores)
        resumen[t.number] = {
            "trial": t.number,
            "params": t.params,
            "valor_optuna": t.value,
            "valores_por_semilla": dict(zip(semillas, valores, strict=True)),
            "estadisticos": est,
            "optimismo_optuna": t.value - est["media"],
            "valor_promedio_semillas": valor_ens,
            "beneficio_semillerio": valor_ens - est["media"],
            "envios_optimos_promedio": [r["envios_optimos"] for r in res_ens],
            "envios_optimos_semillas": envios,
        }
    return resumen, ensembles, scores, preps0


def simular(
    cfg: cfgmod.Config,
    preps: list,
    elegido: int,
    ensembles: dict[int, list[np.ndarray]],
    scores: dict[int, dict[int, list[np.ndarray]]],
    resumen: dict[int, dict],
) -> list[dict]:
    """Public/private por fold: (a) trials entre sí, (b) semillas sueltas del trial elegido."""
    salida = []
    for f, prep in enumerate(preps):
        n = len(prep.es_baja2_valid)
        envios_trials = int(
            np.clip(
                round(
                    statistics.median(
                        r["envios_optimos_promedio"][f] for r in resumen.values()
                    )
                ),
                1,
                n,
            )
        )
        envios_sem = int(np.clip(resumen[elegido]["envios_optimos_promedio"][f], 1, n))
        entre_trials = simular_public_private(
            prep.es_baja2_valid,
            {f"trial_{k}": ensembles[k][f] for k in ensembles},
            envios_trials,
            cfg.estabilidad.n_simulaciones,
            cfg.semilla_maestra,
        )
        candidatos = {f"semilla_{s}": sc[f] for s, sc in scores[elegido].items()}
        candidatos["promedio"] = ensembles[elegido][f]
        entre_semillas = simular_public_private(
            prep.es_baja2_valid,
            candidatos,
            envios_sem,
            cfg.estabilidad.n_simulaciones,
            cfg.semilla_maestra,
        )
        salida.append(
            {
                "valid": prep.fold.valid,
                "entre_trials": entre_trials,
                "entre_semillas": entre_semillas,
            }
        )
    return salida


def exportar_estable(
    cfg: cfgmod.Config,
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    resumen_trial: dict,
    semillas: list[int],
    fe_hash: str,
    dir_params: Path,
) -> Path:
    """params/<estudio>__estable.json: mismo formato que el export de Optuna."""
    info = {
        "study_name": study.study_name,
        "fe_hash": fe_hash,
        "valor": resumen_trial["estadisticos"]["mediana"],
        "trial": trial.number,
        "criterio": cfg.estabilidad.criterio,
        "params": {**cfg.lgbm.manual, **trial.params},
        "estadisticos": resumen_trial["estadisticos"],
        "valor_optuna": trial.value,
        "semillas": semillas,
        "undersampling": cfg.dataset.undersampling,
        "positivos": cfg.target.positivos,
    }
    dir_params.mkdir(parents=True, exist_ok=True)
    path = dir_params / f"{study.study_name}__estable.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def etapa_estabilidad(cfg: cfgmod.Config, run: Run) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parquet, fe_hash = construir_features(cfg)
    features = columnas_seleccionadas(cfg, parquet)
    study = cargar_estudio(nombre_estudio(cfg, fe_hash), DB_PATH)
    trials = top_trials(study, cfg.estabilidad.top_k)
    semillas = generar_semillas(cfg.semilla_maestra, cfg.estabilidad.n_semillas)
    log.info(
        "estabilidad: top %d trials x %d semillas (criterio: %s)",
        len(trials),
        len(semillas),
        cfg.estabilidad.criterio,
    )

    resumen, ensembles, scores, preps = evaluar_candidatos(
        cfg, parquet, features, trials, semillas
    )
    for n, r in resumen.items():
        e = r["estadisticos"]
        log.info(
            "trial %d: optuna=%s | semillas: mediana=%s media=%s std=%s min=%s max=%s | "
            "optimismo=%s | promedio de semillas=%s (%+.1fM)",
            n,
            _m(r["valor_optuna"]),
            _m(e["mediana"]),
            _m(e["media"]),
            _m(e["std"]),
            _m(e["min"]),
            _m(e["max"]),
            _m(r["optimismo_optuna"]),
            _m(r["valor_promedio_semillas"]),
            r["beneficio_semillerio"] / 1e6,
        )

    elegido = elegir(resumen, cfg.estabilidad.criterio)
    mejor_optuna = max(resumen, key=lambda n: resumen[n]["valor_optuna"])
    log.info(
        "elegido: trial %d (criterio %s)%s",
        elegido,
        cfg.estabilidad.criterio,
        ""
        if elegido == mejor_optuna
        else f" | distinto del mejor de Optuna (trial {mejor_optuna})",
    )

    sim = simular(cfg, preps, elegido, ensembles, scores, resumen)
    for s in sim:
        log.info(
            "public/private (valid %s): P(ganador public != ganador private) entre trials=%.2f "
            "(arrepentimiento %s) | entre semillas del trial %d=%.2f | private std promedio=%s",
            s["valid"],
            s["entre_trials"]["prob_ganador_public_pierde_private"],
            _m(s["entre_trials"]["arrepentimiento_private_medio"]),
            elegido,
            s["entre_semillas"]["prob_ganador_public_pierde_private"],
            _m(
                float(
                    np.mean(
                        [
                            c["private_std"]
                            for c in s["entre_semillas"]["candidatos"].values()
                        ]
                    )
                )
            ),
        )

    trial_elegido = next(t for t in trials if t.number == elegido)
    ruta = exportar_estable(
        cfg, study, trial_elegido, resumen[elegido], semillas, fe_hash, PARAMS_DIR
    )
    detalle = {
        "criterio": cfg.estabilidad.criterio,
        "elegido": elegido,
        "mejor_optuna": mejor_optuna,
        "semillas": semillas,
        "trials": resumen,
        "public_private": sim,
    }
    (run.dir / "estabilidad.json").write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    pl.DataFrame(
        [
            {
                "trial": n,
                "valor_optuna": r["valor_optuna"],
                **{f"semillas_{k}": v for k, v in r["estadisticos"].items()},
                "optimismo_optuna": r["optimismo_optuna"],
                "valor_promedio_semillas": r["valor_promedio_semillas"],
                "beneficio_semillerio": r["beneficio_semillerio"],
                "elegido": n == elegido,
            }
            for n, r in resumen.items()
        ]
    ).write_csv(run.dir / "estabilidad_trials.csv")
    run.registrar(
        fe_hash=fe_hash,
        estabilidad={
            "elegido": elegido,
            "criterio": cfg.estabilidad.criterio,
            "mejor_optuna": mejor_optuna,
            "estadisticos_elegido": resumen[elegido]["estadisticos"],
            "beneficio_semillerio_elegido": resumen[elegido]["beneficio_semillerio"],
            "params_exportados": ruta.name,
        },
    )
    log.info("exportado %s | usar con final.params_desde=estable:auto", ruta.name)
