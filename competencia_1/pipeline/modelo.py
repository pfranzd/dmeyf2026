"""LightGBM: entrenamiento reproducible, predicción, importancias y resolución de params."""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.tracking import WORK

log = logging.getLogger("competencia_1.modelo")

PARAMS_DIR = WORK / "params"
_ENTEROS = ("num_leaves", "min_data_in_leaf", "max_bin", "bagging_freq", "max_depth")


def params_lgbm(cfg: cfgmod.Config, params: dict, seed: int) -> tuple[dict, int]:
    """Mezcla fijos + hiperparámetros + semilla. Devuelve (params, num_iterations).

    `seed` fija bagging, feature_fraction y el resto de las semillas internas de
    LightGBM; junto con `deterministic` y `num_threads` fijos hace el entrenamiento
    reproducible bit a bit.
    """
    p = {**cfg.lgbm.fijos, **params, "seed": int(seed)}
    if "num_iterations" not in p:
        raise ValueError("falta num_iterations en los hiperparámetros")
    n_iter = int(p.pop("num_iterations"))
    for k in _ENTEROS:
        if k in p:
            p[k] = int(p[k])
    return p, n_iter


def crear_dataset(
    X: np.ndarray, y: np.ndarray, features: list[str], params: dict
) -> lgb.Dataset:
    """Dataset de LightGBM. max_bin y feature_pre_filter se fijan acá (al construirlo)."""
    ds_params = {k: params[k] for k in ("max_bin", "feature_pre_filter") if k in params}
    return lgb.Dataset(
        X, label=y, feature_name=features, params=ds_params, free_raw_data=True
    )


def entrenar_dataset(dtrain: lgb.Dataset, params: dict, n_iter: int) -> lgb.Booster:
    """Entrena sobre un Dataset (reutilizable entre trials de Optuna: se binariza una vez)."""
    return lgb.train(params, dtrain, num_boost_round=n_iter)


def entrenar_lgbm(
    X: np.ndarray, y: np.ndarray, features: list[str], params: dict, n_iter: int
) -> lgb.Booster:
    return entrenar_dataset(crear_dataset(X, y, features, params), params, n_iter)


def predecir(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return booster.predict(X)


def importancias(booster: lgb.Booster) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort("gain", descending=True)


def escalar_min_data(params: dict, undersampling: float) -> dict:
    """min_data_in_leaf se ajustó sobre datos submuestreados: al entrenar con todos
    los datos se divide por la fracción de undersampling (z494, 'punto SUTIL')."""
    out = dict(params)
    if "min_data_in_leaf" in out and undersampling < 1.0:
        out["min_data_in_leaf"] = max(1, round(out["min_data_in_leaf"] / undersampling))
    return out


def nombre_estudio(cfg: cfgmod.Config, fe_hash: str) -> str:
    """`<exp>_<fe_hash[:8]>_<hash de lo que invalida los trials>`.

    Si cambia algo que altera el significado de un trial (folds, target, undersampling,
    espacio, parámetros fijos, ventana, semilla) el estudio es otro y no se mezclan.
    num_threads se excluye: `deterministic` lo hace irrelevante para el resultado.
    """
    if cfg.optuna.study_name != "auto":
        return cfg.optuna.study_name
    fijos = {k: v for k, v in cfg.lgbm.fijos.items() if k != "num_threads"}
    huella = {
        "folds": [(list(f.train), f.valid) for f in cfgmod.folds(cfg)],
        "positivos": cfg.target.positivos,
        "undersampling": cfg.dataset.undersampling,
        "espacio": cfg.optuna.espacio,
        "fijos": fijos,
        "ventana": cfg.optuna.ventana_meseta,
        "semilla": cfg.semilla_maestra,
    }
    d = cfg.dataset
    if (
        d.excluir_bloques or d.excluir_features
    ):  # solo si hay selección: no cambia estudios previos
        huella["seleccion"] = {
            "bloques": sorted(d.excluir_bloques),
            "features": sorted(d.excluir_features),
        }
    return f"{cfg.experimento}_{fe_hash[:8]}_{cfgmod.hash_dict(huella, 8)}"


def resolver_params(cfg: cfgmod.Config, fe_hash: str) -> dict:
    """Hiperparámetros según `final.params_desde` (manual | optuna:<study|auto> | estable:<study|auto> | archivo:<path>).

    Los json de Optuna guardan el `fe_hash` con el que se optimizaron: usarlos con
    otras features es un error, no un warning.
    """
    origen = cfg.final.params_desde
    if origen == "manual":
        return dict(cfg.lgbm.manual)
    tipo, _, ref = origen.partition(":")
    if tipo in ("optuna", "estable"):
        nombre = nombre_estudio(cfg, fe_hash) if ref == "auto" else ref
        sufijo = "__estable" if tipo == "estable" else ""
        path = PARAMS_DIR / f"{nombre}{sufijo}.json"
    else:
        path = Path(ref) if Path(ref).is_absolute() else cfgmod.RAIZ / ref
    if not path.exists():
        raise FileNotFoundError(f"no existe el archivo de hiperparámetros: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fe_hash") != fe_hash:
        msg = (
            f"{path.name}: optimizado con fe_hash={data.get('fe_hash')} pero el actual es "
            f"{fe_hash}; esos hiperparámetros no corresponden a estas features"
        )
        if not cfg.final.params_ignorar_fe_hash:
            raise ValueError(msg)
        log.warning("%s (se usan igual por final.params_ignorar_fe_hash)", msg)
    return dict(data["params"])
