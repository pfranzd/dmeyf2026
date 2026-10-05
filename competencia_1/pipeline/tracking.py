"""Tracking por corrida: carpeta de run, logging, meta.json y registro global.

Cada run vive en work/competencia_1/runs/<run_id>/ con su config congelada,
run.log y meta.json (actualizado de forma incremental, así un run caído deja
rastro). `runs.csv` resume todas las corridas en una fila cada una.
"""

import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import yaml

from competencia_1.pipeline.config import RAIZ

WORK = RAIZ / "work" / "competencia_1"
RUNS_DIR = WORK / "runs"
RUNS_CSV = WORK / "runs.csv"
CAMPOS_RUNS_CSV = [
    "run_id",
    "inicio",
    "experimento",
    "estado",
    "target",
    "fe_hash",
    "semillas",
    "ganancia_valid",
    "envios",
    "ruta",
]

log = logging.getLogger("competencia_1")


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def ruta_relativa(path: Path) -> str:
    """Ruta relativa al repo (con /) si está adentro; si no, la ruta absoluta."""
    path = Path(path)
    return (
        path.relative_to(RAIZ).as_posix()
        if path.is_relative_to(RAIZ)
        else path.as_posix()
    )


def _git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def info_git() -> dict:
    sucio = _git("status", "--porcelain", "--", "competencia_1")
    return {
        "commit": _git("rev-parse", "HEAD") or None,
        "dirty": bool(sucio),
        "dirty_archivos": sucio.splitlines()[:20],
    }


def versiones() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("lightgbm", "optuna", "polars", "duckdb", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - no instalado o sin __version__
            out[mod] = None
    return out


class Run:
    """Contexto de una corrida. Úsese con `with Run(...) as run:`."""

    def __init__(
        self, experimento: str, config_resuelta: dict, run_id: str | None = None
    ):
        self.inicio = datetime.now(UTC).astimezone()
        self.run_id = run_id or f"{self.inicio:%Y%m%d-%H%M%S}_{experimento}"
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta: dict = {
            "run_id": self.run_id,
            "experimento": experimento,
            "inicio": self.inicio.isoformat(timespec="seconds"),
            "fin": None,
            "estado": "running",
            "git": info_git(),
            "versiones": versiones(),
            "archivos": {},
            "etapas": {},
        }
        with open(self.dir / "config_resuelta.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_resuelta, f, sort_keys=False, allow_unicode=True)
        self._config_logging()
        if self.meta["git"]["dirty"]:
            log.warning(
                "competencia_1 tiene cambios sin commitear: el run no es 100%% "
                "reproducible desde git (%s)",
                ", ".join(self.meta["git"]["dirty_archivos"][:5]),
            )
        self.guardar_meta()

    # -- logging ---------------------------------------------------------
    def _config_logging(self) -> None:
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        raiz = logging.getLogger("competencia_1")
        raiz.setLevel(logging.INFO)
        raiz.handlers.clear()
        for h in (
            logging.FileHandler(self.dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ):
            h.setFormatter(fmt)
            raiz.addHandler(h)
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").addHandler(raiz.handlers[0])

    # -- meta ------------------------------------------------------------
    def registrar(self, **kv) -> None:
        """Agrega claves de primer nivel a meta.json (se guarda al instante)."""
        self.meta.update(kv)
        self.guardar_meta()

    def registrar_archivo(self, path: Path) -> None:
        path = Path(path)
        self.meta["archivos"][ruta_relativa(path)] = {
            "sha256": sha256_archivo(path),
            "bytes": path.stat().st_size,
        }
        self.guardar_meta()

    def guardar_meta(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(self.dir / "meta.json")

    # -- ciclo de vida ---------------------------------------------------
    def __enter__(self) -> Self:
        log.info("run %s iniciado -> %s", self.run_id, self.dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.meta["fin"] = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
        if exc_type is None:
            self.meta["estado"] = "ok"
        else:
            self.meta["estado"] = "failed"
            self.meta["error"] = f"{exc_type.__name__}: {exc}"
            log.error("run %s falló", self.run_id, exc_info=(exc_type, exc, tb))
        self.guardar_meta()
        self._agregar_runs_csv()
        log.info("run %s terminó: %s", self.run_id, self.meta["estado"])
        return False  # no tragar la excepción

    def _agregar_runs_csv(self) -> None:
        m = self.meta
        fila = {
            "run_id": self.run_id,
            "inicio": m["inicio"],
            "experimento": m["experimento"],
            "estado": m["estado"],
            "target": m.get("target"),
            "fe_hash": m.get("fe_hash"),
            "semillas": json.dumps(m.get("semillas")),
            "ganancia_valid": m.get("ganancia_valid"),
            "envios": json.dumps(m.get("envios")),
            "ruta": ruta_relativa(self.dir),
        }
        nuevo = not RUNS_CSV.exists()
        with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS_RUNS_CSV)
            if nuevo:
                w.writeheader()
            w.writerow(fila)
