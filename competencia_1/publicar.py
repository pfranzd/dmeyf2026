"""Publica la entrega oficial en el repositorio de entregas (validada, con commit y push).

    # Un run pasa a ser la entrega oficial con N envíos:
    python -m competencia_1.publicar --run work/competencia_1/runs/<run_id> --envios 10500

Antes de commitear corre la entrega exportada en un entorno virtual nuevo (versiones fijas de
requirements-lock.txt) y exige que el CSV sea idéntico, byte a byte, al del run. Si no coincide,
no se commitea ni se sube nada. Opciones: --validar predecir|entrenar|completo (default predecir),
--repo-entregas <ruta> (default ../dmeyf2026-entregas), --inicializar (primer commit del repo),
--sin-push, --crudo-local <csv> (evita descargar el dataset al validar).
La lógica vive en competencia_1/pipeline/publicacion.py.
"""

import argparse
import logging
import sys
from pathlib import Path

from competencia_1.pipeline.publicacion import (
    CARPETA,
    MODOS,
    REPO_ENTREGAS,
    PublicacionError,
    publicar,
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
    ap.add_argument(
        "--run", type=Path, required=True, help="carpeta del run a publicar"
    )
    ap.add_argument(
        "--envios", type=int, required=True, help="corte de envíos entregado"
    )
    ap.add_argument("--repo-entregas", type=Path, default=REPO_ENTREGAS)
    ap.add_argument(
        "--carpeta", default=CARPETA, help="carpeta dentro del repo de entregas"
    )
    ap.add_argument("--validar", choices=MODOS, default="predecir")
    ap.add_argument("--inicializar", action="store_true")
    ap.add_argument("--sin-push", action="store_true")
    ap.add_argument("--crudo-local", type=Path, default=None)
    ap.add_argument(
        "--publico",
        type=float,
        default=None,
        help="puntaje público de este corte, en millones (se anota en entrega.json)",
    )
    args = ap.parse_args(argv)
    try:
        r = publicar(
            args.run,
            args.envios,
            repo=args.repo_entregas,
            carpeta=args.carpeta,
            modo=args.validar,
            push=not args.sin_push,
            inicializar=args.inicializar,
            crudo_local=args.crudo_local,
            resultado_publico=args.publico,
        )
    except PublicacionError as e:
        print(f"\nNO SE PUBLICÓ: {e}", file=sys.stderr)
        return 1
    print(f"\nPUBLICADO (push={r['push']}): commit {r['commit_entregas']}")
    print(f"CSV validado: {r['csv_generado']}")
    print(f"sha256: {r['sha256']}")
    print(f"Link para la cátedra: {r['link']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
