"""Métricas de negocio sobre scores: curva de ganancia, meseta y envíos óptimos.

La lógica numérica vive en dmeyf/metrics.py (reutilizable por cualquier experimento);
acá solo se arma el resumen que consumen validación, Optuna y los logs.
"""

import numpy as np

from dmeyf.metrics import curva_ganancia, ganancia_meseta


def resumir_scores(es_baja2: np.ndarray, score: np.ndarray, ventana: int) -> dict:
    """Resumen de un mes de validación (la ganancia se mide siempre sobre BAJA+2)."""
    curva = curva_ganancia(es_baja2, score)
    meseta, envios_meseta = ganancia_meseta(curva, ventana)
    i_max = int(np.argmax(curva))
    return {
        "n": len(curva),
        "n_baja2": int(es_baja2.sum()),
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
