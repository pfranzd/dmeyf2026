"""
config/semillas.py — Fuente única de verdad de las semillas del proyecto.

Requisito de la cátedra (DMEyF 2026): 5 números primos entre 100003 y 999983.
Los cinco valores de abajo fueron verificados como primos y dentro de rango.

Uso:
    from config.semillas import SEMILLAS, SEMILLA_PRIMARIA, fijar_semilla
"""

# Las 5 semillas del proyecto. Usadas para promediar resultados (seed averaging)
# y reducir la varianza del modelo.
SEMILLAS = [111623, 116239, 116423, 116923, 123169]

# Semilla por defecto para corridas simples / desarrollo.
SEMILLA_PRIMARIA = SEMILLAS[0]


def fijar_semilla(seed: int) -> None:
    """Fija el azar global de Python y numpy para reproducibilidad.

    Nota: LightGBM/XGBoost NO leen el estado global; a esos modelos hay que
    pasarles la semilla por parámetro (p. ej. seed=... o random_state=...).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)