"""Aritmética de períodos YYYYMM y derivación de folds temporales.

Con target = T y gap = g:
  - meses finales : [T-g-(n-1) .. T-g]            (n = n_meses_train)
  - fold k        : valid = T-g-k, train = [valid-g-(n-1) .. valid-g]
El gap garantiza que el target del último mes de train (que mira 2 meses
hacia adelante) no se solape con el mes de validación.
"""

from dataclasses import dataclass


def mes_offset(foto_mes: int, delta: int) -> int:
    """Suma `delta` meses (puede ser negativo) a un período YYYYMM."""
    anio, mes = divmod(foto_mes, 100)
    if not 1 <= mes <= 12:
        raise ValueError(f"foto_mes inválido: {foto_mes}")
    total = anio * 12 + (mes - 1) + delta
    return (total // 12) * 100 + (total % 12) + 1


def rango_meses(desde: int, hasta: int) -> list[int]:
    """Lista de períodos consecutivos entre `desde` y `hasta` inclusive."""
    n = (hasta // 100 - desde // 100) * 12 + (hasta % 100 - desde % 100)
    if n < 0:
        raise ValueError(f"rango invertido: {desde}..{hasta}")
    return [mes_offset(desde, i) for i in range(n + 1)]


@dataclass(frozen=True)
class Fold:
    train: tuple[int, ...]
    valid: int


def derivar_folds(
    target: int, gap: int, n_meses_train: int, n_folds: int
) -> list[Fold]:
    folds = []
    for k in range(n_folds):
        valid = mes_offset(target, -gap - k)
        fin = mes_offset(valid, -gap)
        train = rango_meses(mes_offset(fin, -(n_meses_train - 1)), fin)
        folds.append(Fold(train=tuple(train), valid=valid))
    return folds


def derivar_meses_final(target: int, gap: int, n_meses_train: int) -> list[int]:
    fin = mes_offset(target, -gap)
    return rango_meses(mes_offset(fin, -(n_meses_train - 1)), fin)


def validar_gap(train: list[int] | tuple[int, ...], valid: int, gap: int) -> None:
    """Falla si el último mes de train queda a menos de `gap` meses de valid."""
    distancia = (valid // 100 - max(train) // 100) * 12 + (
        valid % 100 - max(train) % 100
    )
    if distancia < gap:
        raise ValueError(
            f"gap insuficiente: train termina en {max(train)}, valid={valid}, "
            f"distancia={distancia} < gap={gap}"
        )


def validar_target_completo(
    meses: list[int], meses_completos: set[int], etiqueta: str
) -> None:
    """Falla si algún mes usado para entrenar no tiene el target completo."""
    malos = sorted(set(meses) - set(meses_completos))
    if malos:
        raise ValueError(
            f"{etiqueta}: meses con target incompleto o inexistente: {malos}"
        )
