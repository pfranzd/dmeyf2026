"""Etapa features: parquet base -> parquet de features (caché por hash de configuración).

El resultado vive en work/competencia_1/features/<fe_hash>/ con:
    features.parquet   (claves + target + originales/derivadas según fe.*)
    catalogo.csv       (una fila por feature creada: bloque, expresión, origen)
    fe_config.yaml     (sección fe + versión + tamaño del parquet base)
    query.sql          (la query exacta ejecutada)
Un mismo `fe_hash` se reutiliza entre corridas sin recalcular.
"""

import dataclasses
import logging
import shutil
from pathlib import Path

import duckdb
import yaml

from competencia_1.pipeline import config as cfgmod
from competencia_1.pipeline.datos import CACHE_BASE
from competencia_1.pipeline.features.macros import crear_macros
from competencia_1.pipeline.features.query import CLAVES, armar_query
from competencia_1.pipeline.tracking import WORK, Run, ruta_relativa

log = logging.getLogger("competencia_1.fe")

# Subir al cambiar cualquier SQL de FE: invalida las cachés (y los estudios de Optuna).
FE_VERSION = (
    "3"  # 2: salida ordenada; 3: decil por percent_rank (ntile no era reproducible)
)
FEATURES_DIR = WORK / "features"
PREFIJOS_RATIO = ("r_", "ratioavg_", "deltapct_")


def calcular_fe_hash(cfg: cfgmod.Config, base: Path) -> tuple[str, dict]:
    payload = {
        "version": FE_VERSION,
        "fe": dataclasses.asdict(cfg.fe),
        "base_bytes": base.stat().st_size,
    }
    if cfg.fe.canaritos.activo:
        payload["semilla"] = cfg.semilla_maestra  # los canaritos dependen de la semilla
    return cfgmod.hash_dict(payload), payload


def columnas_modelo(parquet: Path) -> list[str]:
    """Columnas candidatas a feature: todo salvo claves y target."""
    con = duckdb.connect()
    try:
        cols = [
            r[0]
            for r in con.execute(
                f"describe select * from read_parquet('{parquet.as_posix()}')"
            ).fetchall()
        ]
    finally:
        con.close()
    return [c for c in cols if c not in CLAVES]


def verificar_salida(base: Path, salida: Path, features_creadas: list[str]) -> None:
    """Chequeos post-build: filas, target intacto, unicidad, sin infinitos en ratios."""
    con = duckdb.connect()
    try:
        b, s = base.as_posix(), salida.as_posix()
        n_in = con.execute(f"select count(*) from read_parquet('{b}')").fetchone()[0]
        n_out, n_unicos = con.execute(
            f"""select count(*), count(distinct (numero_de_cliente, foto_mes))
                from read_parquet('{s}')"""
        ).fetchone()
        if n_in != n_out:
            raise ValueError(f"FE alteró la cantidad de filas: {n_in:,} -> {n_out:,}")
        if n_out != n_unicos:
            raise ValueError("FE produjo duplicados de (cliente, mes)")
        dif = con.execute(
            f"""select count(*) from (
                    (select foto_mes, clase_ternaria, count(*) n from read_parquet('{b}') group by all
                     except
                     select foto_mes, clase_ternaria, count(*) n from read_parquet('{s}') group by all)
                    union all
                    (select foto_mes, clase_ternaria, count(*) n from read_parquet('{s}') group by all
                     except
                     select foto_mes, clase_ternaria, count(*) n from read_parquet('{b}') group by all)
                )"""
        ).fetchone()[0]
        if dif:
            raise ValueError("FE alteró la distribución del target")
        ratios = [f for f in features_creadas if f.startswith(PREFIJOS_RATIO)]
        if ratios:
            chk = " + ".join(f"count(*) filter (isinf({c}))" for c in ratios)
            n_inf = con.execute(f"select {chk} from read_parquet('{s}')").fetchone()[0]
            if n_inf:
                raise ValueError(f"hay {n_inf} valores infinitos en columnas de ratio")
    finally:
        con.close()


def construir_features(
    cfg: cfgmod.Config, base: Path | None = None, raiz: Path | None = None
) -> tuple[Path, str]:
    """Devuelve (features.parquet, fe_hash), construyéndolo si no está en caché."""
    base = Path(base or CACHE_BASE)
    raiz = Path(raiz or FEATURES_DIR)
    if not base.exists():
        raise FileNotFoundError(f"falta la base: {base} (corré la etapa 'datos')")
    h, payload = calcular_fe_hash(cfg, base)
    destino = raiz / h
    if (destino / "features.parquet").exists():
        log.info("features en caché (fe_hash=%s)", h)
        return destino / "features.parquet", h

    tmp = raiz / f"{h}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    spill = WORK / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"set temp_directory = '{spill.as_posix()}'")
        con.execute(
            f"create or replace view fuente as select * from read_parquet('{base.as_posix()}')"
        )
        crear_macros(con)
        sql, cat, _ = armar_query(cfg.fe, cfg.semilla_maestra, con)
        log.info(
            "FE: %d features nuevas, query de %d líneas",
            len(cat.filas),
            sql.count("\n"),
        )
        salida = tmp / "features.parquet"
        con.execute(
            f"copy ({sql}) to '{salida.as_posix()}' (format parquet, compression zstd)"
        )
        (tmp / "query.sql").write_text(sql, encoding="utf-8")
        cat.guardar(tmp / "catalogo.csv")
        (tmp / "fe_config.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        verificar_salida(base, salida, cat.nombres())
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)  # no dejar una caché a medias
        raise
    finally:
        con.close()
    shutil.rmtree(destino, ignore_errors=True)
    tmp.replace(destino)
    log.info("features escritas: %s", destino / "features.parquet")
    return destino / "features.parquet", h


def etapa_features(cfg: cfgmod.Config, run: Run) -> None:
    parquet, h = construir_features(cfg)
    cols = columnas_modelo(parquet)
    (run.dir / "features.txt").write_text("\n".join(cols) + "\n", encoding="utf-8")
    run.registrar(
        fe_hash=h,
        features={
            "parquet": ruta_relativa(parquet),
            "n_features": len(cols),
        },
    )
    log.info("fe_hash=%s | %d columnas candidatas a feature", h, len(cols))
