"""Punto de entrada: orquesta las etapas del pipeline de la Primera Competencia.

    python -m competencia_1.main --config competencia_1/configs/exp/e001_baseline_catedra.yaml
    python -m competencia_1.main --config <exp>.yaml --set final.n_semillas=20 --set etapas.optuna=true
    python -m competencia_1.main --reproducir work/competencia_1/runs/<run_id>

Este archivo solo orquesta: la lógica vive en competencia_1/pipeline/.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.datos import etapa_datos
from competencia_1.pipeline.evaluacion import etapa_validacion
from competencia_1.pipeline.features import etapa_features
from competencia_1.pipeline.final import etapa_final
from competencia_1.pipeline.salida import etapa_salida
from competencia_1.pipeline.tracking import Run

log = logging.getLogger("competencia_1.main")

# La consola de Windows usa cp1252 por defecto y rompe las tildes de los logs.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ORDEN_ETAPAS = [
    "datos",
    "features",
    "validacion",
    "optuna",
    "estabilidad",
    "final",
    "salida",
]


def _etapa_pendiente(nombre: str, cfg: cfgmod.Config, run: Run) -> None:
    # Las etapas se irán enchufando una a una (ver plan, secciones 2-5).
    raise NotImplementedError(f"etapa '{nombre}' todavía no implementada")


# nombre de etapa -> callable(cfg, run). Cada módulo nuevo se registra acá.
ETAPAS = {
    nombre: (lambda cfg, run, n=nombre: _etapa_pendiente(n, cfg, run))
    for nombre in ORDEN_ETAPAS
}
ETAPAS["datos"] = etapa_datos
ETAPAS["features"] = etapa_features
ETAPAS["validacion"] = etapa_validacion
ETAPAS["final"] = etapa_final
ETAPAS["salida"] = etapa_salida


def parsear_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--config", type=Path, help="YAML del experimento (hereda de base.yaml)"
    )
    g.add_argument("--reproducir", type=Path, help="carpeta de un run previo")
    ap.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="CLAVE=VALOR"
    )
    ap.add_argument(
        "--etapas", help="lista separada por comas; reemplaza lo de etapas.* del config"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parsear_args(argv)

    if args.reproducir:
        resuelta = args.reproducir / "config_resuelta.yaml"
        with open(resuelta, encoding="utf-8") as f:
            d = cfgmod.aplicar_overrides(yaml.safe_load(f), args.overrides)
        cfg = cfgmod.desde_dict(d)
    else:
        cfg, d = cfgmod.cargar_config(args.config, args.overrides)

    if args.etapas is not None:
        pedidas = [e.strip() for e in args.etapas.split(",") if e.strip()]
        desconocidas = set(pedidas) - set(ORDEN_ETAPAS)
        if desconocidas:
            raise SystemExit(f"etapas desconocidas: {sorted(desconocidas)}")
        d["etapas"] = {e: e in pedidas for e in ORDEN_ETAPAS}
        cfg = cfgmod.desde_dict(d)

    with Run(cfg.experimento, d) as run:
        run.registrar(
            target=cfg.periodos.target,
            semillas=cfgmod.semillas_finales(cfg) if cfg.etapas.final else None,
            periodos={
                "folds": [
                    {"train": list(f.train), "valid": f.valid}
                    for f in cfgmod.folds(cfg)
                ],
                "meses_final": cfgmod.meses_final(cfg),
            },
        )
        for nombre in ORDEN_ETAPAS:
            if not getattr(cfg.etapas, nombre):
                log.info("etapa %s: desactivada", nombre)
                continue
            log.info("etapa %s: inicio", nombre)
            ETAPAS[nombre](cfg, run)
            run.meta["etapas"][nombre] = "ok"
            run.guardar_meta()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
