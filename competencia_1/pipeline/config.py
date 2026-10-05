"""Configuración centralizada: YAML con herencia + overrides CLI -> dataclasses frozen.

`configs/base.yaml` es la única fuente de defaults: los dataclasses no tienen
valores por defecto, así que una clave faltante o desconocida (typo) es error.
"""

import copy
import dataclasses
import hashlib
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

from competencia_1.pipeline import periodos

RAIZ = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class Periodos:
    target: int
    gap: int
    n_meses_train: int
    n_folds: int
    meses_final: list[int] | None  # override explícito


@dataclass(frozen=True)
class Target:
    positivos: list[str]


@dataclass(frozen=True)
class Datos:
    csv_procesado: str
    csv_crudo: str
    reconstruir_target: bool


@dataclass(frozen=True)
class IntraFila:
    tc_consolidado: bool
    tc_fechas: bool
    ratios: bool
    agregados: bool
    flags: bool


@dataclass(frozen=True)
class Lags:
    activo: bool
    n: list[int]


@dataclass(frozen=True)
class Deltas:
    activo: bool
    n: list[int]
    pct: bool


@dataclass(frozen=True)
class Ventanas:
    activo: bool
    tamanio: int
    stats: list[str]
    ratio_avg: bool


@dataclass(frozen=True)
class Tendencia:
    activo: bool
    tamanio: int


@dataclass(frozen=True)
class Rankings:
    activo: bool
    campos: str | list[str]
    ntile10: list[str]


@dataclass(frozen=True)
class Canaritos:
    activo: bool
    n: int


@dataclass(frozen=True)
class FE:
    originales: bool
    drop_drift: list[str]
    drop_extra: list[str]
    intrafila: IntraFila
    campos_serie: str | list[str]
    lags: Lags
    deltas: Deltas
    ventanas: Ventanas
    tendencia: Tendencia
    historia: bool
    rankings: Rankings
    canaritos: Canaritos


@dataclass(frozen=True)
class DatasetCfg:
    undersampling: float


@dataclass(frozen=True)
class LGBM:
    fijos: dict
    manual: dict


@dataclass(frozen=True)
class OptunaCfg:
    n_trials: int
    study_name: str  # "auto" -> derivado de experimento + fe_hash + folds
    ventana_meseta: int
    espacio: dict


@dataclass(frozen=True)
class Estabilidad:
    top_k: int
    n_semillas: int


@dataclass(frozen=True)
class Final:
    params_desde: str  # manual | optuna:<study> | archivo:<path>
    n_semillas: int
    semillas: list[int] | None
    promedio: bool
    reescalar_min_data: bool


@dataclass(frozen=True)
class Salida:
    envios: str | int | list[int]  # "auto" | N | [N, ...]
    formato: str


@dataclass(frozen=True)
class Etapas:
    datos: bool
    features: bool
    validacion: bool
    optuna: bool
    estabilidad: bool
    final: bool
    salida: bool


@dataclass(frozen=True)
class Config:
    experimento: str
    semilla_maestra: int
    periodos: Periodos
    target: Target
    datos: Datos
    fe: FE
    dataset: DatasetCfg
    lgbm: LGBM
    optuna: OptunaCfg
    estabilidad: Estabilidad
    final: Final
    salida: Salida
    etapas: Etapas


# ---------------------------------------------------------------- carga


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cargar_dict(path: Path, _vistos: tuple = ()) -> dict:
    """Lee un YAML resolviendo `hereda: otro.yaml` (relativo al archivo)."""
    path = Path(path).resolve()
    if path in _vistos:
        raise ValueError(f"herencia circular en {path}")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    padre = d.pop("hereda", None)
    if padre is None:
        return d
    return _deep_merge(cargar_dict(path.parent / padre, _vistos + (path,)), d)


def aplicar_overrides(d: dict, overrides: list[str]) -> dict:
    """Aplica `clave.anidada=valor` (el valor se interpreta como YAML)."""
    d = copy.deepcopy(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override inválido (falta '='): {ov}")
        clave, valor = ov.split("=", 1)
        partes = clave.strip().split(".")
        nodo = d
        for p in partes[:-1]:
            if not isinstance(nodo.get(p), dict):
                raise TypeError(f"override {clave}: '{p}' no es una sección")
            nodo = nodo[p]
        if partes[-1] not in nodo:
            raise ValueError(f"override {clave}: clave inexistente")
        nodo[partes[-1]] = yaml.safe_load(valor)
    return d


def _construir(cls, d, ruta: str = ""):
    """dict -> dataclass recursivo; rechaza claves faltantes y desconocidas."""
    if not isinstance(d, dict):
        raise TypeError(f"{ruta or 'config'}: se esperaba una sección, llegó {d!r}")
    hints = typing.get_type_hints(cls)
    campos = {f.name for f in dataclasses.fields(cls)}
    desconocidas = set(d) - campos
    faltantes = campos - set(d)
    if desconocidas:
        raise ValueError(
            f"{ruta or 'config'}: claves desconocidas {sorted(desconocidas)}"
        )
    if faltantes:
        raise ValueError(f"{ruta or 'config'}: faltan claves {sorted(faltantes)}")
    kwargs = {}
    for nombre in campos:
        tipo = hints[nombre]
        valor = d[nombre]
        if dataclasses.is_dataclass(tipo):
            valor = _construir(tipo, valor, f"{ruta}.{nombre}".lstrip("."))
        elif isinstance(tipo, (types.UnionType, typing._UnionGenericAlias)):
            pass  # unión de tipos simples; validada en `validar`
        kwargs[nombre] = valor
    return cls(**kwargs)


def cargar_config(
    path: Path, overrides: list[str] | None = None
) -> tuple[Config, dict]:
    """Devuelve (Config validada, dict resuelto) listo para congelar a YAML."""
    d = aplicar_overrides(cargar_dict(path), overrides or [])
    cfg = _construir(Config, d)
    validar(cfg)
    return cfg, d


def desde_dict(d: dict) -> Config:
    cfg = _construir(Config, d)
    validar(cfg)
    return cfg


# ------------------------------------------------------------ validación


def folds(cfg: Config) -> list[periodos.Fold]:
    p = cfg.periodos
    return periodos.derivar_folds(p.target, p.gap, p.n_meses_train, p.n_folds)


def meses_final(cfg: Config) -> list[int]:
    p = cfg.periodos
    if p.meses_final is not None:
        return list(p.meses_final)
    return periodos.derivar_meses_final(p.target, p.gap, p.n_meses_train)


def semillas_finales(cfg: Config) -> list[int]:
    from competencia_1.pipeline.semillas import generar_semillas

    if cfg.final.semillas is not None:
        return list(cfg.final.semillas)
    return generar_semillas(cfg.semilla_maestra, cfg.final.n_semillas)


def validar(cfg: Config) -> None:
    p = cfg.periodos
    if p.gap < 2:
        raise ValueError(f"periodos.gap={p.gap}: el gap mínimo es 2 meses")
    if p.n_meses_train < 1 or p.n_folds < 1:
        raise ValueError("periodos.n_meses_train y n_folds deben ser >= 1")
    if not 190001 <= p.target <= 299912 or not 1 <= p.target % 100 <= 12:
        raise ValueError(f"periodos.target inválido: {p.target}")
    for f in folds(cfg):
        periodos.validar_gap(f.train, f.valid, p.gap)
    for m in meses_final(cfg):
        periodos.validar_gap([m], p.target, p.gap)
    if not cfg.target.positivos or not set(cfg.target.positivos) <= {
        "CONTINUA",
        "BAJA+1",
        "BAJA+2",
    }:
        raise ValueError(f"target.positivos inválido: {cfg.target.positivos}")
    if not 0 < cfg.dataset.undersampling <= 1:
        raise ValueError("dataset.undersampling debe estar en (0, 1]")
    if not 1 <= cfg.final.n_semillas <= 20:
        raise ValueError("final.n_semillas debe estar entre 1 y 20")
    from competencia_1.pipeline.semillas import validar_semilla_curso

    validar_semilla_curso(cfg.semilla_maestra, "semilla_maestra")
    if cfg.final.semillas is not None:
        if not cfg.final.semillas:
            raise ValueError("final.semillas no puede ser una lista vacía")
        if len(set(cfg.final.semillas)) != len(cfg.final.semillas):
            raise ValueError("final.semillas tiene semillas repetidas")
        if len(cfg.final.semillas) > 20:
            raise ValueError("final.semillas admite como máximo 20 semillas")
        for x in cfg.final.semillas:
            validar_semilla_curso(x, "final.semillas")
    fp = cfg.final.params_desde
    if fp != "manual" and not fp.startswith(("optuna:", "archivo:")):
        raise ValueError(f"final.params_desde inválido: {fp}")
    if cfg.optuna.n_trials < 1:
        raise ValueError("optuna.n_trials debe ser >= 1")
    for nombre, esp in cfg.optuna.espacio.items():
        if esp.get("tipo") not in ("int", "float"):
            raise ValueError(f"optuna.espacio.{nombre}: tipo debe ser int o float")
        if esp["low"] >= esp["high"]:
            raise ValueError(f"optuna.espacio.{nombre}: low >= high")
        if esp.get("log", False) and esp["low"] <= 0:
            raise ValueError(f"optuna.espacio.{nombre}: log=true requiere low > 0")
    if (
        "num_iterations" not in cfg.optuna.espacio
        and "num_iterations" not in cfg.lgbm.manual
    ):
        raise ValueError("num_iterations debe estar en optuna.espacio o en lgbm.manual")
    if cfg.salida.formato != "solo_ids":
        raise ValueError(f"salida.formato no soportado: {cfg.salida.formato}")
    e = cfg.salida.envios
    if not (e == "auto" or isinstance(e, int) or (isinstance(e, list) and e)):
        raise ValueError(f"salida.envios inválido: {e!r}")
    validar_fe(cfg.fe)


def validar_fe(fe: FE) -> None:
    if fe.lags.activo and (not fe.lags.n or min(fe.lags.n) < 1):
        raise ValueError("fe.lags.n debe tener enteros >= 1 con lags activos")
    if fe.deltas.activo and (not fe.deltas.n or min(fe.deltas.n) < 1):
        raise ValueError("fe.deltas.n debe tener enteros >= 1 con deltas activos")
    v = fe.ventanas
    if v.activo:
        if v.tamanio < 1:
            raise ValueError("fe.ventanas.tamanio debe ser >= 1")
        if not v.stats or not set(v.stats) <= {"avg", "max", "min", "std"}:
            raise ValueError(f"fe.ventanas.stats inválido: {v.stats} (avg|max|min|std)")
        if v.ratio_avg and "avg" not in v.stats:
            raise ValueError(
                "fe.ventanas.ratio_avg requiere 'avg' en fe.ventanas.stats"
            )
    if fe.tendencia.activo and fe.tendencia.tamanio < 2:
        raise ValueError("fe.tendencia.tamanio debe ser >= 2 (se ajusta una recta)")
    if fe.intrafila.ratios and not (
        fe.intrafila.tc_consolidado and fe.intrafila.agregados
    ):
        raise ValueError(
            "fe.intrafila.ratios requiere tc_consolidado y agregados activos"
        )
    if fe.canaritos.activo and fe.canaritos.n < 1:
        raise ValueError("fe.canaritos.n debe ser >= 1 con canaritos activos")
    for nombre, spec in (
        ("campos_serie", fe.campos_serie),
        ("rankings.campos", fe.rankings.campos),
    ):
        if isinstance(spec, str) and spec not in ("curado", "todos"):
            raise ValueError(f"fe.{nombre}: '{spec}' inválido (curado | todos | lista)")


def hash_dict(d: dict, largo: int = 12) -> str:
    """Hash estable de un dict (YAML canónico con claves ordenadas)."""
    canon = yaml.safe_dump(d, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:largo]
