"""Arma X / y / es_baja2 por lista de meses desde el parquet de features.

- Nunca entran como features `numero_de_cliente`, `foto_mes` ni `clase_ternaria`.
- `y` sale de `target.positivos` (lo que se entrena); `es_baja2` marca BAJA+2, la
  única clase que da ganancia (se usa para medir, nunca para entrenar).
- El undersampling de CONTINUA es determinístico: azar = hash(cliente, mes, semilla).
  Se conservan TODAS las filas BAJA+1 y BAJA+2 (como z494).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

CLAVES = ("numero_de_cliente", "foto_mes", "clase_ternaria")
CLASES_BAJA = ("BAJA+1", "BAJA+2")


@dataclass
class Particion:
    X: np.ndarray  # float32, NaN donde hay nulos
    ids: np.ndarray  # int64
    meses: np.ndarray  # int32
    features: list[str]
    y: np.ndarray | None  # int8; None si no hay target (mes a predecir)
    es_baja2: np.ndarray | None  # bool; None si no hay target

    def __len__(self) -> int:
        return len(self.ids)


def azar_uniforme(ids: np.ndarray, meses: np.ndarray, seed: int) -> np.ndarray:
    """U[0,1) determinístico por (cliente, mes, semilla) con mezcla splitmix64.

    Se implementa a mano (y no con el hash de polars/duckdb) porque estos no
    garantizan estabilidad entre versiones y acá importa reproducir exacto.
    """
    with np.errstate(over="ignore"):
        x = ids.astype(np.uint64) * np.uint64(1_000_003) + meses.astype(np.uint64)
        x = x ^ np.uint64(seed)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def validar_features(features: list[str]) -> None:
    prohibidas = set(features) & set(CLAVES)
    if prohibidas:
        raise ValueError(f"claves/target no pueden ser features: {sorted(prohibidas)}")
    if len(set(features)) != len(features):
        raise ValueError("lista de features con duplicados")


def cargar_particion(
    parquet: Path,
    features: list[str],
    meses: list[int],
    positivos: list[str],
    undersampling: float = 1.0,
    seed: int = 0,
    con_target: bool = True,
) -> Particion:
    validar_features(features)
    cols = [
        "numero_de_cliente",
        "foto_mes",
        *(["clase_ternaria"] if con_target else []),
        *features,
    ]
    df = (
        pl.scan_parquet(parquet)
        .filter(pl.col("foto_mes").is_in(meses))
        .select(cols)
        .collect()
    )
    if df.height == 0:
        raise ValueError(f"no hay filas para los meses {meses}")
    faltan = set(meses) - set(df["foto_mes"].unique().to_list())
    if faltan:
        raise ValueError(f"meses sin filas en el parquet: {sorted(faltan)}")

    if con_target:
        if df["clase_ternaria"].null_count():
            raise ValueError(
                f"meses {meses}: hay filas sin clase_ternaria (target incompleto)"
            )
        if undersampling < 1.0:
            azar = azar_uniforme(
                df["numero_de_cliente"].to_numpy(), df["foto_mes"].to_numpy(), seed
            )
            es_baja = df["clase_ternaria"].is_in(CLASES_BAJA).to_numpy()
            df = df.filter(pl.Series(es_baja | (azar <= undersampling)))
        y = df["clase_ternaria"].is_in(positivos).to_numpy().astype(np.int8)
        es_baja2 = (df["clase_ternaria"] == "BAJA+2").to_numpy()
    else:
        y = es_baja2 = None

    X = np.ascontiguousarray(
        df.select([pl.col(c).cast(pl.Float32) for c in features]).to_numpy(),
        dtype=np.float32,
    )
    return Particion(
        X=X,
        ids=df["numero_de_cliente"].to_numpy().astype(np.int64),
        meses=df["foto_mes"].to_numpy().astype(np.int32),
        features=list(features),
        y=y,
        es_baja2=es_baja2,
    )
