"""Entrega definitiva: congela un run ya ejecutado para que otra persona lo reproduzca.

`promover` toma un run (su config_resuelta.yaml, los hiperparámetros finales de meta.json y el
sha256 de su CSV) y escribe en competencia_1/definitiva/ una configuración autocontenida:
no depende de configs/exp, de work/ ni de db/ (el estudio de Optuna no hace falta).
`reproducir` corre esa configuración en un directorio de trabajo aparte y compara el sha256.
"""

import copy
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.tracking import (
    RAIZ,
    info_git,
    ruta_relativa,
    sha256_archivo,
)

log = logging.getLogger("competencia_1.entrega")

DIR_DEFINITIVA = RAIZ / "competencia_1" / "definitiva"
ETAPAS_ENTREGA = ("datos", "features", "final", "salida")
CABECERA = """\
# ENTREGA DEFINITIVA (generada por `python -m competencia_1.entrega promover`; no editar a mano).
# Reproducir: python -m competencia_1.entrega reproducir
# Origen y sha256 esperado: ver entrega.json. Los hiperparámetros ya vienen escalados en
# params.json (reescalar_min_data: false) y no hace falta Optuna.
"""


def _cargar_run(run_dir: Path) -> tuple[dict, dict]:
    meta_p, cfg_p = run_dir / "meta.json", run_dir / "config_resuelta.yaml"
    for p in (meta_p, cfg_p):
        if not p.exists():
            raise FileNotFoundError(f"no existe {p}: ¿es la carpeta de un run?")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    return meta, yaml.safe_load(cfg_p.read_text(encoding="utf-8"))


def _csv_del_run(
    meta: dict, envios: int, modelo: str, run_dir: Path | None = None
) -> tuple[str, str]:
    """(ruta, sha256) del CSV `*_<modelo>_e<envios>.csv` registrado en el run.

    Si el corte no se registró (p. ej. un CSV extra generado después con las probabilidades
    del run), se toma de `<run>/submits/`: el contenido es el mismo que escribiría la etapa
    salida con ese corte, y el sha256 se calcula del archivo.
    """
    sufijo = f"_{modelo}_e{envios}.csv"
    candidatos = {
        k: v for k, v in meta.get("archivos", {}).items() if k.endswith(sufijo)
    }
    if not candidatos and run_dir is not None:
        extra = sorted((run_dir / "submits").glob(f"*{sufijo}"))
        if extra:
            return ruta_relativa(extra[0]), sha256_archivo(extra[0])
    if not candidatos:
        disponibles = sorted(
            Path(k).name for k in meta.get("archivos", {}) if k.endswith(".csv")
        )
        raise ValueError(
            f"el run no tiene un CSV '{modelo}' con {envios} envíos. "
            f"Cortes disponibles: {disponibles[:6]}"
        )
    ruta = min(candidatos)
    return ruta, candidatos[ruta]["sha256"]


def _referencias_archivo(valor) -> list[str]:
    """Rutas de los valores `archivo:<ruta>` que aparezcan en cualquier parte de un config."""
    if isinstance(valor, str):
        return [valor.removeprefix("archivo:")] if valor.startswith("archivo:") else []
    if isinstance(valor, dict):
        valor = list(valor.values())
    if isinstance(valor, list):
        return [r for v in valor for r in _referencias_archivo(v)]
    return []


def cadena_config(path: Path) -> list[Path]:
    """Archivos de los que depende un config de experimento: él, sus `hereda` y los `archivo:`."""
    vistos: list[Path] = []
    pendientes = [Path(path).resolve()]
    while pendientes:
        p = pendientes.pop()
        if p in vistos:
            continue
        if not p.exists():
            raise FileNotFoundError(
                f"el config referencia un archivo que no existe: {p}"
            )
        vistos.append(p)
        if p.suffix not in (".yaml", ".yml"):
            continue
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if d.get("hereda"):
            pendientes.append((p.parent / d["hereda"]).resolve())
        for ref in _referencias_archivo(d):
            r = Path(ref)
            pendientes.append((r if r.is_absolute() else RAIZ / r).resolve())
    return sorted(vistos)


def _experimento_reproducible(meta: dict, cfg: dict) -> tuple[Path | None, bool]:
    """(yaml del experimento, ¿se puede rehacer de punta a punta con ese solo config?).

    Es automático si el experimento optimiza en su propia corrida y toma los hiperparámetros
    de su propio estudio (no de otra corrida): así `reproducir --completo` rehace todo.
    """
    nombre = f"{meta.get('experimento')}.yaml"
    yaml_exp = RAIZ / "competencia_1" / "configs" / "exp" / nombre
    if not yaml_exp.exists():
        return None, False
    desde = cfg.get("final", {}).get("params_desde", "")
    propio = bool(cfg.get("etapas", {}).get("optuna")) and bool(
        re.fullmatch(r"(optuna|estable):auto|top:\d+", desde)
    )
    return yaml_exp, propio


def promover(
    run_dir: Path,
    envios: int,
    modelo: str = "promedio",
    destino: Path = DIR_DEFINITIVA,
    desde_crudo: bool = True,
    con_modelos: bool = False,
    resultado_publico: float | None = None,
) -> Path:
    """Escribe config.yaml, params.json y entrega.json en `destino` a partir de un run.

    Con `con_modelos` copia los modelos entrenados del run a `destino/modelos/` y el config
    queda en modo "solo predecir" (`final.modelos_desde`): reproducir tarda minutos en vez de
    reentrenar. Pesa del orden de 100 MB: queda a criterio de quien promueve.
    """
    run_dir = Path(run_dir)
    run_dir = run_dir if run_dir.is_absolute() else RAIZ / run_dir
    meta, cfg = _cargar_run(run_dir)

    if meta.get("estado") != "ok":
        raise ValueError(f"el run terminó con estado {meta.get('estado')!r}, no 'ok'")
    if meta.get("etapas", {}).get("final") != "ok":
        raise ValueError(
            "el run no ejecutó la etapa 'final': no hay modelo que promover"
        )
    params = meta.get("modelo", {}).get("params_final")
    if not params:
        raise ValueError("meta.json no tiene modelo.params_final")
    csv_ruta, csv_sha = _csv_del_run(meta, envios, modelo, run_dir)
    yaml_exp, completo_auto = _experimento_reproducible(meta, cfg)

    # Reconstruye desde el crudo solo si se pide: la base debe dar el mismo contenido
    # (se verificó para esta entrega); si no, se usa el CSV procesado local.
    entrega = copy.deepcopy(cfg)
    entrega["experimento"] = "entrega"
    entrega["datos"]["reconstruir_target"] = bool(desde_crudo)
    entrega["final"].update(
        {
            "params_desde": f"archivo:{ruta_relativa(destino / 'params.json')}",
            # Los params ya salen escalados del run: no volver a dividir por el undersampling.
            "reescalar_min_data": False,
            # fe_hash depende de los bytes de base.parquet (cambian con la versión de duckdb);
            # las features quedan fijadas por este mismo archivo.
            "params_ignorar_fe_hash": True,
        }
    )
    entrega["final"]["modelos_desde"] = None
    if con_modelos:
        origen = run_dir / "modelos"
        archivos = sorted(origen.glob("*.txt"))
        if not archivos:
            raise ValueError(f"el run no tiene modelos guardados en {origen}")
        entrega["final"]["modelos_desde"] = ruta_relativa(destino / "modelos")
    entrega["salida"]["envios"] = [envios]
    entrega["etapas"] = {k: k in ETAPAS_ENTREGA for k in entrega["etapas"]}
    cfgmod.desde_dict(entrega)  # valida: falla antes de escribir nada

    destino.mkdir(parents=True, exist_ok=True)
    viejos = destino / "modelos"
    if (
        viejos.exists()
    ):  # los modelos de una entrega anterior no se mezclan con los nuevos
        shutil.rmtree(viejos)
    if con_modelos:
        viejos.mkdir()
        for a in archivos:
            shutil.copy2(a, viejos / a.name)
    (destino / "config.yaml").write_text(
        CABECERA + yaml.safe_dump(entrega, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (destino / "params.json").write_text(
        json.dumps(
            {
                "nota": f"hiperparámetros finales (ya escalados) del run {run_dir.name}",
                # lista [{etiqueta, params}] => ensamble de varios conjuntos (resolver_conjunto)
                ("ensamble" if isinstance(params, list) else "params"): params,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # Si se vuelve a promover el mismo run con el mismo corte, se conserva el resultado público.
    previo = destino / "entrega.json"
    if resultado_publico is None and previo.exists():
        anterior = json.loads(previo.read_text(encoding="utf-8"))
        # el puntaje es de un corte concreto: solo se conserva si es el mismo run y el mismo corte
        if (
            anterior.get("run") == ruta_relativa(run_dir)
            and anterior.get("envios") == envios
        ):
            resultado_publico = anterior.get("resultado_publico")
    info = {
        "run": ruta_relativa(run_dir),
        "experimento": meta.get("experimento"),
        "commit_del_run": meta.get("git", {}).get("commit"),
        "commit_actual": info_git()["commit"],
        "promovido": datetime.now().astimezone().isoformat(timespec="seconds"),
        "envios": envios,
        "modelo": modelo,
        "csv_original": Path(csv_ruta).name,
        "sha256_esperado": csv_sha,
        "versiones": meta.get("versiones"),
        "resultado_publico": resultado_publico,
        "modelos_incluidos": len(archivos) if con_modelos else 0,
        "config_experimento": ruta_relativa(yaml_exp) if yaml_exp else None,
        "completo_automatico": completo_auto,
    }
    (destino / "entrega.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log.info(
        "entrega escrita en %s (run %s, %d envíos)",
        ruta_relativa(destino),
        run_dir.name,
        envios,
    )
    return destino


def _correr_y_comparar(args: list[str], info: dict, limpiar: bool) -> bool:
    """Corre el pipeline con `args` y compara el sha256 del CSV con `info`."""
    # Import diferido: main arrastra todo el pipeline y WORK ya quedó fijado por el entorno.
    from competencia_1 import main as main_mod
    from competencia_1.pipeline.tracking import RUNS_DIR, WORK

    if limpiar and WORK.exists():
        shutil.rmtree(WORK)
    antes = {p.name for p in RUNS_DIR.glob("*")} if RUNS_DIR.exists() else set()
    main_mod.main(args)
    nuevos = sorted({p.name for p in RUNS_DIR.glob("*")} - antes)
    if not nuevos:
        raise RuntimeError("la corrida no generó ninguna carpeta de run")
    run = RUNS_DIR / nuevos[-1]
    csvs = list((run / "submits").glob(f"*_{info['modelo']}_e{info['envios']}.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"se esperaba 1 CSV en {run / 'submits'}, hay {len(csvs)}")
    obtenido = sha256_archivo(csvs[0])
    ok = obtenido == info["sha256_esperado"]
    log.info("CSV generado: %s", ruta_relativa(csvs[0]))
    log.info("sha256 esperado: %s", info["sha256_esperado"])
    log.info("sha256 obtenido: %s", obtenido)
    log.info("RESULTADO: %s", "OK, coincide" if ok else "DIFIERE")
    return ok


def _info(destino: Path) -> dict:
    info_p = destino / "entrega.json"
    if not info_p.exists():
        raise FileNotFoundError(f"no existe {info_p}: corré primero `promover`")
    return json.loads(info_p.read_text(encoding="utf-8"))


def reproducir(
    destino: Path = DIR_DEFINITIVA, limpiar: bool = False, entrenar: bool = False
) -> bool:
    """Corre config.yaml y compara el sha256 del CSV con entrega.json. True si coincide.

    Si la entrega incluye modelos entrenados, solo predice con ellos; `entrenar=True` los
    ignora y reentrena todo desde cero.
    """
    info = _info(destino)
    args = ["--config", str(destino / "config.yaml")]
    if entrenar:
        args += ["--set", "final.modelos_desde=null"]
    return _correr_y_comparar(args, info, limpiar)


def reproducir_completo(destino: Path = DIR_DEFINITIVA, limpiar: bool = False) -> bool:
    """Rehace TODO: Optuna, selección de los mejores trials, modelos finales y CSV.

    Corre el config original del experimento (el que dio origen a los hiperparámetros) con un
    estudio de Optuna nuevo (DMEYF_DB aislado) y el target reconstruido desde el crudo, y
    compara el sha256 con la entrega. Solo es automático si el experimento es autocontenido.
    """
    info = _info(destino)
    if not info.get("completo_automatico"):
        raise ValueError(
            "esta entrega no se puede rehacer de punta a punta con un solo config "
            "(sus hiperparámetros vienen de otra corrida): ver los pasos en el README"
        )
    config = Path(info["config_experimento"])
    config = config if config.is_absolute() else RAIZ / config
    args = [
        "--config",
        str(config),
        "--set",
        f"salida.envios=[{info['envios']}]",
        "--set",
        "datos.reconstruir_target=true",
    ]
    return _correr_y_comparar(args, info, limpiar)
