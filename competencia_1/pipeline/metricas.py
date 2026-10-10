"""Métricas de negocio sobre scores: curva de ganancia, meseta y envíos óptimos.

La lógica numérica vive en dmeyf/metrics.py (reutilizable por cualquier experimento);
acá solo se arma el resumen que consumen validación, Optuna y los logs.
"""

import numpy as np
import polars as pl

from dmeyf.metrics import curva_ganancia, ganancia_meseta


def auc(es_baja2: np.ndarray, score: np.ndarray) -> float:
    """AUC (Mann-Whitney con rangos promedio: los empates cuentan 0,5) de BAJA+2 contra el resto."""
    y = np.asarray(es_baja2, dtype=bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC indefinido: falta una de las dos clases en el mes")
    rangos = pl.Series(np.asarray(score, dtype=float)).rank("average").to_numpy()
    return float((rangos[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def resumir_scores(es_baja2: np.ndarray, score: np.ndarray, ventana: int) -> dict:
    """Resumen de un mes de validación (la ganancia se mide siempre sobre BAJA+2)."""
    curva = curva_ganancia(es_baja2, score)
    meseta, envios_meseta = ganancia_meseta(curva, ventana)
    i_max = int(np.argmax(curva))
    return {
        "n": len(curva),
        "n_baja2": int(es_baja2.sum()),
        "auc": auc(es_baja2, score),
        "ganancia_meseta": meseta,
        "envios_optimos": envios_meseta,
        "ganancia_max_cruda": int(curva[i_max]),
        "envios_max_crudo": i_max + 1,
        "curva": curva,
    }


def curva_submuestreada(
    curva: np.ndarray, hasta: int = 30_000, paso: int = 100
) -> list[tuple]:
    """Puntos (envíos, ganancia) para guardar la curva sin escribir 160k filas."""
    n = min(hasta, len(curva))
    return [(k, int(curva[k - 1])) for k in range(paso, n + 1, paso)]
