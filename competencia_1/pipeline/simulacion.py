"""Simulación public/private de Kaggle sobre un mes de validación (monday/z301).

En la competencia el leaderboard público se calcula sobre ~30 % de los clientes y el
privado sobre el ~70 % restante. Acá se repite ese reparto estratificado muchas veces
sobre un mes con target conocido para medir cuánto puede engañar el público:
- cuánto varía la ganancia de un mismo modelo entre repartos, y
- con qué frecuencia el ganador del público NO es el ganador del privado.

Con cada reparto, la ganancia pública se escala por 1/proporción pública y la privada
por 1/proporción privada, así ambas quedan comparables con la de un mes completo.
"""

import numpy as np

from dmeyf.metrics import COSTO_ESTIMULO, GANANCIA_ACIERTO


def mascara_top(score: np.ndarray, envios: int) -> np.ndarray:
    """True en los `envios` clientes de mayor score (desempate estable, como la curva)."""
    if not 0 < envios <= len(score):
        raise ValueError(f"envios={envios} fuera de (0, {len(score)}]")
    orden = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    mascara = np.zeros(len(score), dtype=bool)
    mascara[orden[:envios]] = True
    return mascara


def _reparto(
    es_baja2: np.ndarray, prop_public: float, rng: np.random.Generator
) -> np.ndarray:
    """Máscara de 'público' con la misma proporción de BAJA+2 y del resto (estratificado)."""
    publico = np.zeros(len(es_baja2), dtype=bool)
    for idx in (np.flatnonzero(es_baja2), np.flatnonzero(~es_baja2)):
        elegidos = rng.permutation(idx)[: round(prop_public * len(idx))]
        publico[elegidos] = True
    return publico


def simular_public_private(
    es_baja2: np.ndarray,
    scores: dict[str, np.ndarray],
    envios: int,
    n_sim: int,
    seed: int,
    prop_public: float = 0.3,
) -> dict:
    """Compara candidatos (nombre -> score del mes) enviando a los top `envios` de cada uno.

    Devuelve, por candidato, media y desvío de la ganancia pública y privada, y a nivel
    global la fracción de simulaciones en que el ganador del público no gana el privado
    y el 'arrepentimiento' privado medio de haberlo elegido.
    """
    es = np.asarray(es_baja2, dtype=bool)
    nombres = list(scores)
    sel = {k: mascara_top(scores[k], envios) for k in nombres}
    rng = np.random.default_rng(seed)
    pub = np.empty((n_sim, len(nombres)))
    priv = np.empty((n_sim, len(nombres)))
    for i in range(n_sim):
        publico = _reparto(es, prop_public, rng)
        p_real = publico.mean()
        for j, k in enumerate(nombres):
            s = sel[k]
            aciertos_pub = np.count_nonzero(s & publico & es)
            falsos_pub = np.count_nonzero(s & publico & ~es)
            aciertos_priv = np.count_nonzero(s & ~publico & es)
            falsos_priv = np.count_nonzero(s & ~publico & ~es)
            pub[i, j] = (
                aciertos_pub * GANANCIA_ACIERTO - falsos_pub * COSTO_ESTIMULO
            ) / p_real
            priv[i, j] = (
                aciertos_priv * GANANCIA_ACIERTO - falsos_priv * COSTO_ESTIMULO
            ) / (1 - p_real)

    ganador_pub = pub.argmax(axis=1)
    ganador_priv = priv.argmax(axis=1)
    elegido_priv = priv[np.arange(n_sim), ganador_pub]
    return {
        "envios": envios,
        "n_sim": n_sim,
        "candidatos": {
            k: {
                "public_media": float(pub[:, j].mean()),
                "public_std": float(pub[:, j].std()),
                "private_media": float(priv[:, j].mean()),
                "private_std": float(priv[:, j].std()),
            }
            for j, k in enumerate(nombres)
        },
        "prob_ganador_public_pierde_private": float(
            np.mean(ganador_pub != ganador_priv)
        ),
        "arrepentimiento_private_medio": float(
            np.mean(priv.max(axis=1) - elegido_priv)
        ),
    }
