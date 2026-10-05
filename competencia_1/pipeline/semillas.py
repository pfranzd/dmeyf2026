"""Semillas reproducibles: primos en [100003, 999983] derivados de la maestra."""

import random

from config.semillas import SEMILLAS

_MIN, _MAX = 100003, 999983


def _es_primo(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def validar_semilla_curso(semilla: int, etiqueta: str = "semilla") -> None:
    """La cátedra exige primos entre 100003 y 999983."""
    if not isinstance(semilla, int) or isinstance(semilla, bool):
        raise TypeError(f"{etiqueta}: debe ser un entero, es {semilla!r}")
    if not _MIN <= semilla <= _MAX or not _es_primo(semilla):
        raise ValueError(f"{etiqueta}={semilla}: debe ser primo entre {_MIN} y {_MAX}")


def generar_semillas(maestra: int, n: int) -> list[int]:
    """Devuelve `n` primos distintos; los primeros son las SEMILLAS del curso.

    Las 5 semillas oficiales van primero (en orden) y el resto se sortea de
    forma determinística a partir de `maestra`, así un semillerío de 20 modelos
    extiende al de 5 sin cambiar sus primeros modelos.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    fijas = list(SEMILLAS)
    if n <= len(fijas):
        return fijas[:n]
    primos = [p for p in range(_MIN, _MAX + 1) if _es_primo(p) and p not in fijas]
    extra = random.Random(maestra).sample(primos, n - len(fijas))
    return fijas + extra
