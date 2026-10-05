"""Etapa salida: probabilidades -> CSV de entrega (solo IDs enteros, sin header).

Garantías del formato:
- `numero_de_cliente` viaja como Int64 de punta a punta (nunca float/pandas).
- Una línea por cliente, solo dígitos: se relee el archivo escrito como TEXTO y
  cada línea debe cumplir ^\\d+$ (descarta 4.8e+07, decimales, comillas, signos).
- Si alguna validación falla, el archivo se borra y se lanza la excepción.
"""

import logging
import re
import statistics
from pathlib import Path

import duckdb
import polars as pl

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.datos import CACHE_BASE
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.salida")

_SOLO_DIGITOS = re.compile(r"^\d+$")
REDONDEO_ENVIOS = 500


def validar_probas(probas: pl.DataFrame) -> None:
    """Contrato de entrada: numero_de_cliente Int64 único y sin nulos; prob en [0, 1]."""
    if set(probas.columns) != {"numero_de_cliente", "prob"}:
        raise ValueError(
            f"columnas esperadas (numero_de_cliente, prob), llegaron {probas.columns}"
        )
    if probas.schema["numero_de_cliente"] != pl.Int64:
        raise TypeError(
            f"numero_de_cliente debe ser Int64, es {probas.schema['numero_de_cliente']}"
        )
    if probas["numero_de_cliente"].null_count() or probas["prob"].null_count():
        raise ValueError("hay nulos en numero_de_cliente o prob")
    if probas["numero_de_cliente"].n_unique() != probas.height:
        raise ValueError("numero_de_cliente duplicado en las probabilidades")
    if probas["prob"].is_nan().any() or not probas["prob"].is_between(0, 1).all():
        raise ValueError("prob fuera de [0, 1] o NaN")


def seleccionar_top(probas: pl.DataFrame, envios: int) -> pl.Series:
    """IDs de los `envios` clientes con mayor prob. Desempata por ID: determinístico."""
    if not 0 < envios <= probas.height:
        raise ValueError(f"envios={envios} fuera de (0, {probas.height}]")
    return (
        probas.sort(["prob", "numero_de_cliente"], descending=[True, False])
        .head(envios)
        .get_column("numero_de_cliente")
    )


def validar_csv(path: Path, envios: int, ids_validos: set[int] | None = None) -> None:
    """Relee `path` como texto y verifica el formato exigido por la competencia."""
    texto = path.read_bytes().decode("utf-8")
    if "\r" in texto:
        raise ValueError(f"{path.name}: contiene \\r (se esperan fines de línea \\n)")
    lineas = texto.split("\n")
    if lineas and lineas[-1] == "":
        lineas.pop()
    malas = [(i + 1, x) for i, x in enumerate(lineas) if not _SOLO_DIGITOS.match(x)]
    if malas:
        raise ValueError(f"{path.name}: líneas que no son enteros planos: {malas[:5]}")
    if len(lineas) != envios:
        raise ValueError(f"{path.name}: {len(lineas)} líneas, se esperaban {envios}")
    if len(set(lineas)) != len(lineas):
        raise ValueError(f"{path.name}: IDs duplicados")
    if ids_validos is not None:
        fuera = [x for x in lineas if int(x) not in ids_validos]
        if fuera:
            raise ValueError(
                f"{path.name}: {len(fuera)} IDs no pertenecen al período objetivo: {fuera[:5]}"
            )


def escribir_csv_ids(
    probas: pl.DataFrame,
    envios: int,
    path: Path,
    ids_validos: set[int] | None = None,
) -> Path:
    """Escribe y valida un CSV de entrega. Borra el archivo si no pasa la validación."""
    validar_probas(probas)
    ids = seleccionar_top(probas, envios)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids.to_frame().write_csv(path, include_header=False, line_terminator="\n")
    try:
        validar_csv(path, envios, ids_validos)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def resolver_envios(
    spec: str | int | list[int],
    envios_optimos_valid: list[int] | None = None,
    n_target: int | None = None,
    n_valid: int | None = None,
) -> list[int]:
    """Convierte `salida.envios` en una lista concreta de cortes.

    auto: mediana de los envíos óptimos de validación, escalada por la razón de
    clientes (target/valid) y redondeada a múltiplos de 500.
    """
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return sorted(set(spec))
    if spec != "auto":
        raise ValueError(f"salida.envios inválido: {spec!r}")
    if not envios_optimos_valid or not n_target or not n_valid:
        raise ValueError(
            "salida.envios=auto requiere envíos óptimos de validación (corré optuna/"
            "evaluación antes) o indicá un número / lista de envíos"
        )
    escalado = statistics.median(envios_optimos_valid) * n_target / n_valid
    return [max(REDONDEO_ENVIOS, round(escalado / REDONDEO_ENVIOS) * REDONDEO_ENVIOS)]


def ids_del_periodo(parquet: Path, foto_mes: int) -> set[int]:
    con = duckdb.connect()
    try:
        filas = con.execute(
            f"select numero_de_cliente from read_parquet('{parquet.as_posix()}') where foto_mes = ?",
            [foto_mes],
        ).fetchall()
    finally:
        con.close()
    return {f[0] for f in filas}


def generar_submits(
    probas_por_modelo: dict[str, pl.DataFrame],
    envios: list[int],
    destino: Path,
    prefijo: str,
    ids_validos: set[int] | None = None,
) -> list[Path]:
    """Un CSV por (modelo, corte): `<prefijo>_<modelo>_e<envios>.csv`."""
    generados = []
    for modelo, probas in probas_por_modelo.items():
        for n in envios:
            path = destino / f"{prefijo}_{modelo}_e{n}.csv"
            generados.append(escribir_csv_ids(probas, n, path, ids_validos))
            log.info("submit escrito: %s", path.name)
    return generados


def etapa_salida(cfg: cfgmod.Config, run: Run) -> None:
    """Lee run/probas/*.parquet (contrato de la etapa final) y escribe los CSV."""
    dir_probas = run.dir / "probas"
    archivos = sorted(dir_probas.glob("*.parquet"))
    if not archivos:
        raise FileNotFoundError(
            f"no hay probabilidades en {dir_probas}: corré la etapa 'final'"
        )
    probas = {a.stem: pl.read_parquet(a) for a in archivos}

    objetivo = cfg.periodos.target
    validos = ids_del_periodo(CACHE_BASE, objetivo)
    ref = run.meta.get("validacion", {})
    envios = resolver_envios(
        cfg.salida.envios,
        ref.get("envios_optimos"),
        len(validos),
        ref.get("n_clientes_valid"),
    )
    log.info("cortes de envíos: %s", envios)

    for h in probas.values():
        validar_probas(h)
        if set(h["numero_de_cliente"].to_list()) != validos:
            raise ValueError(
                "las probabilidades no cubren exactamente los clientes del target"
            )

    nombres = {k: probas[k] for k in probas if cfg.final.promedio or k != "promedio"}
    archivos_csv = generar_submits(
        nombres, envios, run.dir / "submits", run.run_id, validos
    )
    for a in archivos_csv:
        run.registrar_archivo(a)
    run.registrar(envios=envios)
