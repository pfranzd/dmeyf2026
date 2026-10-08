"""Ensambla las probabilidades finales de varios runs y escribe los CSV de entrega.

    python -m competencia_1.ensamble --nombre ens01 --envios 10000 11000 \\
        --runs work/competencia_1/runs/<run_a> work/competencia_1/runs/<run_b>

Opciones: --metodo rank|prob (default rank), --pesos 2 1 (uno por run), --probas promedio
(parquet a leer dentro de <run>/probas/). El resultado queda en
work/competencia_1/ensambles/<timestamp>_<nombre>/ con submits/, probas.parquet y meta.json
(runs de entrada con el sha256 de cada parquet, para poder reproducirlo).
La lógica vive en competencia_1/pipeline/ensamble.py.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.datos import CACHE_BASE
from competencia_1.pipeline.ensamble import METODOS, coincidencias, combinar
from competencia_1.pipeline.salida import escribir_csv_ids, ids_del_periodo
from competencia_1.pipeline.tracking import (
    WORK,
    info_git,
    ruta_relativa,
    sha256_archivo,
)

log = logging.getLogger("competencia_1.ensamble")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _parsear(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--runs", nargs="+", required=True, help="carpetas de run (2 o más)"
    )
    ap.add_argument("--nombre", required=True, help="nombre corto del ensamble")
    ap.add_argument("--envios", nargs="+", type=int, required=True)
    ap.add_argument("--metodo", choices=METODOS, default="rank")
    ap.add_argument("--pesos", nargs="+", type=float, default=None)
    ap.add_argument(
        "--probas", default="promedio", help="parquet dentro de <run>/probas/"
    )
    return ap.parse_args(argv)


def _target_del_run(run_dir: Path) -> int:
    d = yaml.safe_load((run_dir / "config_resuelta.yaml").read_text(encoding="utf-8"))
    cfg = cfgmod.desde_dict(d)
    return cfg.periodos.target


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    args = _parsear(argv)
    runs = [Path(r) if Path(r).is_absolute() else cfgmod.RAIZ / r for r in args.runs]

    entradas, probas, targets = [], {}, set()
    for i, run in enumerate(runs):
        path = run / "probas" / f"{args.probas}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no existe {path}: ¿el run corrió la etapa final?")
        targets.add(_target_del_run(run))
        clave = f"{i}_{run.name}"
        probas[clave] = pl.read_parquet(path)
        entradas.append(
            {
                "run": ruta_relativa(run),
                "parquet": ruta_relativa(path),
                "sha256": sha256_archivo(path),
                "peso": args.pesos[i] if args.pesos else 1.0,
            }
        )
        log.info(
            "modelo %s: %s (%d clientes)",
            clave,
            ruta_relativa(path),
            probas[clave].height,
        )
    if len(targets) != 1:
        raise ValueError(f"los runs predicen períodos distintos: {sorted(targets)}")
    target = targets.pop()

    ens = combinar(list(probas.values()), args.metodo, args.pesos)

    ens_id = f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_{args.nombre}"
    destino = WORK / "ensambles" / ens_id
    (destino / "submits").mkdir(parents=True, exist_ok=False)
    ens.write_parquet(destino / "probas.parquet")

    validos = ids_del_periodo(CACHE_BASE, target)
    archivos = {}
    for n in sorted(set(args.envios)):
        p = escribir_csv_ids(
            ens, n, destino / "submits" / f"{ens_id}_e{n}.csv", validos
        )
        archivos[ruta_relativa(p)] = sha256_archivo(p)
        log.info("submit escrito: %s", ruta_relativa(p))

    resumen = coincidencias(probas, ens, sorted(set(args.envios)))
    for n, r in resumen.items():
        log.info(
            "corte %s: jaccard medio entre modelos=%.3f | vs ensamble=%s",
            n,
            r["jaccard_medio_pares"],
            {k: round(v, 3) for k, v in r["jaccard_vs_ensamble"].items()},
        )

    meta = {
        "ensamble_id": ens_id,
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target": target,
        "metodo": args.metodo,
        "entradas": entradas,
        "envios": sorted(set(args.envios)),
        "coincidencias": resumen,
        "archivos": archivos,
        "git": info_git(),
        "comando": "python -m competencia_1.ensamble "
        + " ".join(sys.argv[1:] if argv is None else argv),
    }
    (destino / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("ensamble listo: %s", ruta_relativa(destino))


if __name__ == "__main__":
    main()
