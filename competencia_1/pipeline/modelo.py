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


def entrenar_lgbm(
    X: np.ndarray, y: np.ndarray, features: list[str], params: dict, n_iter: int
) -> lgb.Booster:
    # feature_pre_filter y max_bin se fijan al construir el Dataset
    ds_params = {k: params[k] for k in ("max_bin", "feature_pre_filter") if k in params}
    dtrain = lgb.Dataset(
        X, label=y, feature_name=features, params=ds_params, free_raw_data=True
    )
    return lgb.train(params, dtrain, num_boost_round=n_iter)


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


def resolver_params(cfg: cfgmod.Config, fe_hash: str) -> dict:
    """Hiperparámetros según `final.params_desde` (manual | optuna:<study> | archivo:<path>).

    Los json de Optuna guardan el `fe_hash` con el que se optimizaron: usarlos con
    otras features es un error, no un warning.
    """
    origen = cfg.final.params_desde
    if origen == "manual":
        return dict(cfg.lgbm.manual)
    tipo, _, ref = origen.partition(":")
    if tipo == "optuna":
        path = PARAMS_DIR / f"{ref}.json"
    else:
        path = Path(ref) if Path(ref).is_absolute() else cfgmod.RAIZ / ref
    if not path.exists():
        raise FileNotFoundError(f"no existe el archivo de hiperparámetros: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fe_hash") != fe_hash:
        raise ValueError(
            f"{path.name}: optimizado con fe_hash={data.get('fe_hash')} pero el actual es "
            f"{fe_hash}; esos hiperparámetros no corresponden a estas features"
        )
    return dict(data["params"])
