"""Entrega definitiva: congelar un run (promover) y comprobar que se reproduce (reproducir).

    # Elegir qué run es la entrega y con cuántos envíos (genera competencia_1/definitiva/):
    python -m competencia_1.entrega promover --run work/competencia_1/runs/<run_id> --envios 11000

    # Reproducirla en un directorio aparte (work/entrega) y comparar el sha256. Si la entrega
    # incluye modelos entrenados (promover --con-modelos) solo predice (~12 min):
    python -m competencia_1.entrega reproducir

    # Reentrenar todo desde cero aunque haya modelos guardados (~80 min):
    python -m competencia_1.entrega reproducir --entrenar

    # Rehacer TODO, incluida la optimización de hiperparámetros con Optuna (~2,5 h):
    python -m competencia_1.entrega reproducir --completo

`reproducir` usa DMEYF_WORK=work/entrega por defecto para no tocar la caché de los experimentos;
exportar DMEYF_WORK antes (p. ej. un disco montado en GCP) para cambiarlo.
La lógica vive en competencia_1/pipeline/entrega.py.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Debe fijarse antes de importar el pipeline: tracking.WORK se resuelve al importarlo.
if len(sys.argv) > 1 and sys.argv[1] == "reproducir":
    os.environ.setdefault("DMEYF_WORK", "work/entrega")
    os.environ.setdefault("DMEYF_DB", "work/entrega/db/competencia_1.db")

from competencia_1.pipeline.entrega import (
    DIR_DEFINITIVA,
    promover,
    reproducir,
    reproducir_completo,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("promover", help="congela un run como entrega definitiva")
    p.add_argument("--run", type=Path, required=True, help="carpeta del run")
    p.add_argument(
        "--envios", type=int, required=True, help="corte de envíos a entregar"
    )
    p.add_argument(
        "--modelo", default="promedio", help="promedio (default) o s<semilla>"
    )
    p.add_argument(
        "--con-modelos",
        action="store_true",
        help="copia los modelos entrenados a definitiva/modelos (~140 MB) para solo predecir",
    )
    p.add_argument(
        "--csv-procesado",
        action="store_true",
        help="usar datasets/processed/competencia_01.csv en vez de reconstruir desde el crudo",
    )

    r = sub.add_parser("reproducir", help="corre la entrega y compara el sha256")
    r.add_argument(
        "--entrenar",
        action="store_true",
        help="reentrena todo aunque la entrega traiga modelos ya entrenados",
    )
    r.add_argument(
        "--completo",
        action="store_true",
        help="rehace también la optimización de hiperparámetros (config original del experimento)",
    )
    r.add_argument(
        "--limpiar", action="store_true", help="borra el directorio de trabajo antes"
    )

    args = ap.parse_args(argv)
    if args.cmd == "promover":
        promover(
            args.run,
            args.envios,
            args.modelo,
            desde_crudo=not args.csv_procesado,
            con_modelos=args.con_modelos,
        )
        return 0
    if args.completo:
        ok = reproducir_completo(DIR_DEFINITIVA, args.limpiar)
    else:
        ok = reproducir(DIR_DEFINITIVA, args.limpiar, args.entrenar)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
