"""
dmeyf/metrics.py — Métrica de negocio: ganancia esperada de un modelo de churn.

Valores oficiales de la cátedra 2026 (NO reusar los de ediciones anteriores,
p. ej. 273.000/-7.000 — esos son de otro año). Encontrados de forma
consistente, sin ambigüedad, en 9+ notebooks oficiales del repo, entre
ellos:
  - monday/z401_Fuego_contra_fuego.ipynb:118-119,183-188
  - arboles/z0201_ComparandoModelos.ipynb:3915
  - arboles/z0333_OptimizacionHiperparametros.ipynb:772,950,1542

Uso:
    from dmeyf.metrics import ganancia_prob, GANANCIA_ACIERTO, COSTO_ESTIMULO
"""

import numpy as np

GANANCIA_ACIERTO = 1_072_500  # ganancia por acertar un BAJA+2 estimulado
COSTO_ESTIMULO = 27_500  # costo de estimular a quien no se iba a dar de baja
THRESHOLD_DEFAULT = 0.025  # = 1/40, punto de corte oficial 2026


def ganancia_prob(y_true, y_score, threshold: float = THRESHOLD_DEFAULT) -> float:
    """Ganancia de negocio de estimular a los clientes con y_score >= threshold.

    y_true: array-like de labels reales. Acepta `clase_ternaria` como string
        (compara contra "BAJA+2") o un array 0/1 ya binarizado.
    y_score: array-like de probabilidad estimada de BAJA+2 para cada fila.

    Cada acierto (estimulado y realmente BAJA+2) suma GANANCIA_ACIERTO; cada
    falso positivo (estimulado y no era BAJA+2) resta COSTO_ESTIMULO. Los no
    estimulados no suman ni restan.

    Calcular SIEMPRE mes a mes (nunca agregado sobre varios foto_mes): la
    ganancia no es comparable entre meses con distinta cantidad de clientes
    o distinta tasa de baja base. Recall y lift sí pueden agregarse.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.dtype.kind in ("U", "S", "O"):
        es_positivo = y_true == "BAJA+2"
    else:
        es_positivo = y_true.astype(bool)

    estimulado = y_score >= threshold
    aciertos = np.sum(estimulado & es_positivo)
    falsos_positivos = np.sum(estimulado & ~es_positivo)
    return float(aciertos * GANANCIA_ACIERTO - falsos_positivos * COSTO_ESTIMULO)
