"""Publicación de la entrega oficial en el repositorio de entregas.

`publicar` toma un run ya ejecutado y deja en `<repo_entregas>/<carpeta>/` todo lo necesario para
replicar exactamente su CSV: el código, el entorno con versiones fijas, los hiperparámetros, los
modelos y la optimización que los originó. Antes de hacer commit y push **valida** que la carpeta
exportada, corrida en un entorno virtual nuevo, genere el mismo CSV (mismo sha256 y mismos bytes)
que el del run. Si algo falla no se commitea ni se sube nada.

Pasos: precondiciones → promover (congela el run en competencia_1/definitiva) → exportar →
chequear que no entren datos ni archivos pesados → validar → commit → push → verificar el remoto.
"""

import filecmp
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

from competencia_1.pipeline import entrega as ent
from competencia_1.pipeline.tracking import RAIZ, ruta_relativa, sha256_archivo

log = logging.getLogger("competencia_1.publicacion")

URL_REPO = "https://github.com/pfranzd/dmeyf2026-entregas"
REPO_ENTREGAS = RAIZ.parent / "dmeyf2026-entregas"
CARPETA = "competencia_01"
PLANTILLAS = RAIZ / "competencia_1" / "plantillas_entrega"
CO_AUTOR = "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"

# Código que viaja a la carpeta de entrega (el resto de competencia_1 no hace falta).
CODIGO = (
    "competencia_1/__init__.py",
    "competencia_1/main.py",
    "competencia_1/entrega.py",
    "competencia_1/pipeline",
    "dmeyf",
    "config",
)
# Debe estar commiteado: la carpeta publicada se identifica con ese commit.
RUTAS_A_COMMITEAR = (
    *CODIGO,
    "competencia_1/configs",
    "competencia_1/plantillas_entrega",
)
IGNORAR_AL_COPIAR = shutil.ignore_patterns("__pycache__", "*.pyc", "publicacion.py")
# Lo que genera la reproducción y no se toca al reemplazar la carpeta.
CONSERVAR = {"datasets", "work", ".venv", ".git"}
PROHIBIDOS = re.compile(r"(^|/)(datasets|work|db)/|\.(csv|parquet)$")
MAX_BYTES = 50 * 1024 * 1024  # GitHub rechaza archivos de más de 100 MB; avisamos antes
MODOS = ("predecir", "entrenar", "completo")


class PublicacionError(RuntimeError):
    """Una precondición o validación falló: no se publicó nada."""


def _git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if check and r.returncode != 0:
        raise PublicacionError(f"git {' '.join(args)} falló: {r.stderr.strip()}")
    return r.stdout.strip()


# ---------------------------------------------------------------- precondiciones


def verificar_codigo_commiteado() -> str:
    """El código que se publica debe estar commiteado; devuelve el commit actual."""
    sucio = _git(RAIZ, "status", "--porcelain", "--", *RUTAS_A_COMMITEAR)
    if sucio:
        raise PublicacionError(
            "hay cambios sin commitear en el código que se publica; commitearlos antes:\n"
            + sucio
        )
    return _git(RAIZ, "rev-parse", "HEAD")


def verificar_repo_entregas(repo: Path, inicializar: bool) -> bool:
    """Repo de entregas existente, limpio, con origin y al día. Devuelve True si no tiene commits."""
    if not (repo / ".git").exists():
        raise PublicacionError(f"{repo} no es un repositorio git (clonar {URL_REPO})")
    if not _git(repo, "remote", "get-url", "origin", check=False):
        raise PublicacionError(f"{repo} no tiene remote 'origin'")
    sin_commits = _git(repo, "rev-parse", "--verify", "HEAD", check=False) == ""
    if sin_commits and not inicializar:
        raise PublicacionError(
            "el repo de entregas no tiene commits todavía: usar --inicializar"
        )
    if sin_commits:  # rama main desde el primer commit
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    if _git(repo, "status", "--porcelain"):
        raise PublicacionError(f"{repo} tiene cambios sin commitear")
    if not sin_commits:
        rama = _git(repo, "symbolic-ref", "--short", "HEAD")
        if rama != "main":
            raise PublicacionError(f"{repo} está en la rama {rama!r}, no en 'main'")
        _git(repo, "fetch", "origin")
        atras = _git(repo, "rev-list", "--count", "HEAD..origin/main", check=False)
        if atras not in ("", "0"):
            raise PublicacionError(
                f"el repo de entregas está {atras} commits atrás de origin/main: hacer pull"
            )
    return sin_commits


# ---------------------------------------------------------------- entorno (lock)


def _nombre_canonico(nombre: str) -> str:
    return re.sub(r"[-_.]+", "-", nombre).lower()


def generar_lock(requirements: Path) -> str:
    """`pip freeze` de las dependencias directas y todas sus transitivas, del entorno actual."""
    pendientes = [
        Requirement(linea.split("#")[0].strip()).name
        for linea in requirements.read_text(encoding="utf-8").splitlines()
        if linea.split("#")[0].strip()
    ]
    vistos: dict[str, tuple[str, str]] = {}
    while pendientes:
        nombre = _nombre_canonico(pendientes.pop())
        if nombre in vistos:
            continue
        dist = metadata.distribution(nombre)
        vistos[nombre] = (dist.metadata["Name"], dist.version)
        for r in dist.requires or []:
            req = Requirement(r)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            pendientes.append(req.name)
    lineas = sorted(f"{n}=={v}" for n, v in vistos.values())
    return (
        "# Entorno congelado con el que se generó la entrega (Python 3.11).\n"
        "# Instalar:  python -m pip install -r requirements-lock.txt\n"
        + "\n".join(lineas)
        + "\n"
    )


# ---------------------------------------------------------------- plantillas


def _render(plantilla: str, valores: dict[str, str]) -> str:
    texto = (PLANTILLAS / plantilla).read_text(encoding="utf-8")
    for k, v in valores.items():
        texto = texto.replace(f"@@{k}@@", v)
    faltan = set(re.findall(r"@@([A-Z0-9_]+)@@", texto))
    if faltan:
        raise PublicacionError(f"{plantilla}: placeholders sin valor {sorted(faltan)}")
    return texto


def _escribir(path: Path, texto: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8", newline="\n")


def _resumen_modelo(cfg: dict, meta: dict) -> str:
    p, d = cfg["periodos"], cfg["dataset"]
    folds = p.get("folds") or [{"train": "(derivado)", "valid": "(derivado)"}]
    filas = "\n".join(
        f"  - train {f['train']} → validación {f['valid']}" for f in folds
    )
    n_modelos = len(meta.get("modelo", {}).get("params_final") or [])
    ensamble = (
        f"\n- Ensamble final: {n_modelos} conjuntos de hiperparámetros × "
        f"{cfg['final']['n_semillas']} semillas; se promedian las probabilidades."
        if n_modelos > 1
        else ""
    )
    return (
        f"- LightGBM, objetivo `binary` sobre `{' + '.join(cfg['target']['positivos'])}`; la ganancia "
        "se mide sobre BAJA+2 (+780.000 / −20.000).\n"
        f"- Features: datos originales + familias temporales (lags, deltas, ventanas, tendencia) "
        f"según `competencia_1/configs/exp/{meta['experimento']}.yaml`.\n"
        f"- Optimización: validación temporal con gap de {p['gap']} meses; objetivo "
        f"`{cfg['optuna'].get('objetivo', 'ganancia')}` promedio de los folds:\n{filas}\n"
        f"- Undersampling de CONTINUA en la optimización: {d['undersampling']} "
        "(el modelo final entrena sin undersampling).\n"
        f"- Modelo final entrenado con los meses {p['meses_final']}; predice {p['target']}."
        f"{ensamble}"
    )


def _texto_optimizacion(info: dict, meta: dict, resumen: dict | None) -> str:
    if resumen is not None:
        top = resumen["top"]
        return (
            "Los hiperparámetros **no están fijados a mano**: salen de una optimización bayesiana "
            f"con Optuna, incluida en esta carpeta.\n\n"
            f"- Estudio: `optimizacion/estudio.db` (SQLite de Optuna, {resumen['n_trials']} trials, "
            f"{resumen['n_completos']} completos y {resumen['n_podados']} podados) y "
            "`optimizacion/resumen.json`.\n"
            f"- La entrega usa los {len(top)} mejores trials: "
            f"{', '.join('#' + str(t['trial']) for t in top)}.\n"
            f"- Para rehacerla de punta a punta: `bash reproducir.sh completo` (corre "
            f"`competencia_1/configs/exp/{meta['experimento']}.yaml`: Optuna → mejores trials → "
            "modelos → CSV) y comprueba el sha256.\n"
        )
    cadena = "\n".join(f"  - `{c}`" for c in info.get("cadena_config") or [])
    return (
        "Estos hiperparámetros provienen de **otra corrida de optimización** y no se rehacen "
        "automáticamente con un solo comando.\n\nArchivos de configuración involucrados:\n"
        f"{cadena or '  - (ver competencia_1/configs/exp)'}\n"
    )


def _resumen_optuna(meta: dict, cfg: dict, destino: Path) -> dict | None:
    """Exporta el estudio de Optuna del run; None si el run no optimizó."""
    import optuna

    o = meta.get("optuna")
    if not o:
        return None
    db = Path(o["storage"])
    db = db if db.is_absolute() else RAIZ / db
    if not db.exists():
        raise PublicacionError(f"no existe el estudio de Optuna del run: {db}")
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / "estudio.db"
    salida.unlink(missing_ok=True)
    optuna.copy_study(
        from_study_name=o["study_name"],
        from_storage=f"sqlite:///{db.as_posix()}",
        to_storage=f"sqlite:///{salida.as_posix()}",
    )
    estudio = optuna.load_study(
        study_name=o["study_name"], storage=f"sqlite:///{salida.as_posix()}"
    )
    completos = [t for t in estudio.trials if t.state.name == "COMPLETE"]
    k = cfg["final"]["params_desde"]
    n_top = int(k.split(":")[1]) if k.startswith("top:") else 1
    top = sorted(completos, key=lambda t: t.value, reverse=True)[:n_top]
    resumen = {
        "estudio": o["study_name"],
        "objetivo": cfg["optuna"].get("objetivo", "ganancia"),
        "n_trials": len(estudio.trials),
        "n_completos": len(completos),
        "n_podados": sum(t.state.name == "PRUNED" for t in estudio.trials),
        "top": [{"trial": t.number, "valor": t.value, "params": t.params} for t in top],
    }
    _escribir(
        destino / "resumen.json",
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n",
    )
    return resumen


# ---------------------------------------------------------------- exportación


def _vaciar_carpeta(destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for hijo in destino.iterdir():
        if hijo.name in CONSERVAR:
            continue
        shutil.rmtree(hijo) if hijo.is_dir() else hijo.unlink()


def exportar(
    repo: Path,
    carpeta: str,
    run_dir: Path,
    commit: str,
    definitiva: Path = ent.DIR_DEFINITIVA,
) -> dict:
    """Escribe `<repo>/<carpeta>` a partir de competencia_1/definitiva y el run. Devuelve el resumen."""
    info = json.loads((definitiva / "entrega.json").read_text(encoding="utf-8"))
    meta, cfg = ent._cargar_run(run_dir)
    destino = repo / carpeta
    _vaciar_carpeta(destino)

    for rel in CODIGO:
        origen = RAIZ / rel
        if origen.is_dir():
            shutil.copytree(origen, destino / rel, ignore=IGNORAR_AL_COPIAR)
        else:
            (destino / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origen, destino / rel)

    # Configuración del experimento: él, lo que hereda y los archivos que referencia.
    base = RAIZ / "competencia_1" / "configs" / "base.yaml"
    fuentes = {base}
    if info.get("config_experimento"):
        fuentes |= set(ent.cadena_config(RAIZ / info["config_experimento"]))
        info["cadena_config"] = sorted(ruta_relativa(p) for p in fuentes)
    for f in fuentes:
        (destino / ruta_relativa(f)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, destino / ruta_relativa(f))

    shutil.copytree(definitiva, destino / "competencia_1" / "definitiva")
    if info.get(
        "cadena_config"
    ):  # entrega.json de la carpeta incluye la cadena de configs
        _escribir(
            destino / "competencia_1" / "definitiva" / "entrega.json",
            json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        )

    requirements = RAIZ / "competencia_1" / "requirements.txt"
    shutil.copy2(requirements, destino / "requirements.txt")
    _escribir(destino / "requirements-lock.txt", generar_lock(requirements))
    _escribir(destino / ".python-version", "3.11\n")

    resumen = _resumen_optuna(meta, cfg, destino / "optimizacion")
    n_modelos = info.get("modelos_incluidos") or 1
    n_trials = (resumen or {}).get("n_trials") or cfg["optuna"]["n_trials"]
    valores = {
        "EXPERIMENTO": str(meta["experimento"]),
        "ENVIOS": str(info["envios"]),
        "CSV_ORIGINAL": info["csv_original"],
        "SHA256": info["sha256_esperado"],
        "PUBLICO": (
            f"{info['resultado_publico']:.4f} M"
            if info.get("resultado_publico") is not None
            else "(no registrado)"
        ),
        "N_MODELOS": str(n_modelos),
        "N_TRIALS": str(n_trials),
        "URL_REPO": URL_REPO + ".git",
        "CARPETA": carpeta,
        "PYTHON": "3.11",
        "RESUMEN": _resumen_modelo(cfg, meta),
        "OPTIMIZACION": _texto_optimizacion(info, meta, resumen),
        "FECHA": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "COMMIT": commit[:10],
        "RUN_ID": run_dir.name,
    }
    script = _render("entrega.py.tpl", valores)
    _escribir(destino / "entrega.py", script)
    _escribir(destino / "entrega.ipynb", _a_notebook(destino / "entrega.py"))
    _escribir(destino / "reproducir.sh", _render("reproducir.sh.tpl", valores))
    _escribir(destino / "README.md", _render("README_carpeta.md.tpl", valores))
    _escribir(repo / ".gitignore", _render("gitignore.tpl", {}))
    _escribir(repo / ".gitattributes", _render("gitattributes.tpl", {}))
    _escribir(repo / "README.md", _readme_raiz(repo))
    return {"info": info, "optimizacion": resumen, "destino": destino}


def _a_notebook(script: Path) -> str:
    """entrega.py (formato percent) -> entrega.ipynb con el mismo código."""
    import jupytext

    nb = jupytext.read(script, fmt="py:percent")
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return jupytext.writes(nb, fmt="ipynb")


def _readme_raiz(repo: Path) -> str:
    filas = []
    for entrega_json in sorted(repo.glob("*/competencia_1/definitiva/entrega.json")):
        carpeta = entrega_json.parents[2].name
        i = json.loads(entrega_json.read_text(encoding="utf-8"))
        publico = i.get("resultado_publico")
        filas.append(
            f"| {carpeta.replace('competencia_', 'Competencia ')} | [`{carpeta}/`]({carpeta}/) | "
            f"{i['experimento']} | {i['envios']} | `{i['sha256_esperado'][:12]}…` | "
            f"{f'{publico:.2f} M' if publico is not None else '—'} |"
        )
    return _render("README_raiz.md.tpl", {"FILAS": "\n".join(filas)})


# ---------------------------------------------------------------- chequeos de contenido


def chequear_indice(repo: Path) -> None:
    """Nada de datos (csv, parquet, datasets, work) ni archivos que GitHub rechazaría."""
    _git(repo, "add", "-A")
    archivos = _git(repo, "ls-files", "--cached", "-z").split("\0")
    malos = [a for a in archivos if a and PROHIBIDOS.search(a)]
    if malos:
        raise PublicacionError(f"archivos que no deben subirse: {malos[:10]}")
    pesados = [
        f"{a} ({(repo / a).stat().st_size / 1e6:.0f} MB)"
        for a in archivos
        if a and (repo / a).exists() and (repo / a).stat().st_size > MAX_BYTES
    ]
    if pesados:
        raise PublicacionError(f"archivos de más de 50 MB: {pesados}")


# ---------------------------------------------------------------- validación


def _python_del_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def preparar_venv(carpeta: Path) -> Path:
    """venv nuevo desde requirements-lock.txt (se reutiliza si el lock no cambió)."""
    venv = carpeta / ".venv"
    lock = carpeta / "requirements-lock.txt"
    sello = venv / ".lock_sha256"
    huella = hashlib.sha256(lock.read_bytes()).hexdigest()
    if not _python_del_venv(venv).exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        sello.unlink(missing_ok=True)
    if not sello.exists() or sello.read_text() != huella:
        py = str(_python_del_venv(venv))
        subprocess.run(
            [py, "-m", "pip", "install", "--quiet", "-r", str(lock)], check=True
        )
        sello.write_text(huella)
    return _python_del_venv(venv)


def validar_en_limpio(
    carpeta: Path, info: dict, csv_run: Path, modo: str, crudo_local: Path | None
) -> Path:
    """Corre entrega.py en un venv construido desde el lock y exige el mismo CSV que el run."""
    py = preparar_venv(carpeta)
    env = {**os.environ, "ENTREGA_MODO": modo, "PYTHONIOENCODING": "utf-8"}
    for k in ("DMEYF_WORK", "DMEYF_DB"):  # que el script use los suyos
        env.pop(k, None)
    if crudo_local is not None:
        env["ENTREGA_CRUDO_LOCAL"] = str(crudo_local)
    log.info("validando (modo %s) con %s", modo, py)
    r = subprocess.run([str(py), "entrega.py"], cwd=carpeta, env=env, check=False)
    if r.returncode != 0:
        raise PublicacionError(
            f"entrega.py terminó con código {r.returncode}: el CSV no coincide o falló la corrida"
        )
    csvs = sorted(
        (carpeta / "work" / "entrega" / "runs").glob(
            f"*/submits/*_{info['modelo']}_e{info['envios']}.csv"
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not csvs:
        raise PublicacionError("la validación no generó el CSV esperado")
    generado = csvs[-1]
    sha_gen = sha256_archivo(generado)
    sha_run = sha256_archivo(csv_run)
    if not (sha_gen == info["sha256_esperado"] == sha_run):
        raise PublicacionError(
            f"sha256 distintos: generado={sha_gen} entrega.json={info['sha256_esperado']} "
            f"run={sha_run}"
        )
    if not filecmp.cmp(generado, csv_run, shallow=False):
        raise PublicacionError("el CSV generado y el del run difieren byte a byte")
    log.info("CSV validado: %s (sha256 %s)", generado.name, sha_gen)
    return generado


# ---------------------------------------------------------------- git


def commit_y_push(repo: Path, carpeta: str, mensaje: str, push: bool) -> str:
    """Commit (si hay cambios) y push de main; verifica que el remoto quede en el mismo commit."""
    _git(repo, "add", "-A")
    sh = f"{carpeta}/reproducir.sh"
    _git(repo, "update-index", "--add", "--chmod=+x", sh)
    if _git(repo, "diff", "--cached", "--name-only"):
        _git(repo, "commit", "-q", "-m", f"{mensaje}\n\n{CO_AUTOR}")
    else:
        log.info("sin cambios respecto del último commit del repo de entregas")
    local = _git(repo, "rev-parse", "HEAD")
    if not push:
        return local
    _git(repo, "push", "-u", "origin", "main")
    remoto = _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
    if remoto != local:
        raise PublicacionError(f"el remoto quedó en {remoto} y el local en {local}")
    return local


# ---------------------------------------------------------------- orquestación


def _restaurar(repo: Path) -> None:
    """Deja el repo de entregas como estaba (sin tocar lo ignorado: .venv, work, datasets)."""
    if _git(repo, "rev-parse", "--verify", "HEAD", check=False):
        _git(repo, "reset", "-q", "--hard", "HEAD", check=False)
    else:
        _git(repo, "rm", "-rq", "--cached", "--ignore-unmatch", ".", check=False)
    _git(repo, "clean", "-fdq", check=False)


def publicar(
    run_dir: Path,
    envios: int,
    repo: Path = REPO_ENTREGAS,
    carpeta: str = CARPETA,
    modo: str = "predecir",
    push: bool = True,
    inicializar: bool = False,
    crudo_local: Path | None = None,
    validador=validar_en_limpio,
    definitiva: Path = ent.DIR_DEFINITIVA,
) -> dict:
    if modo not in MODOS:
        raise PublicacionError(f"modo de validación inválido: {modo}")
    run_dir = Path(run_dir)
    run_dir = run_dir if run_dir.is_absolute() else RAIZ / run_dir
    repo = Path(repo)

    commit = verificar_codigo_commiteado()
    verificar_repo_entregas(repo, inicializar)
    meta, _ = ent._cargar_run(run_dir)  # falla pronto si no es un run
    ent._csv_del_run(meta, envios, "promedio", run_dir)

    ent.promover(run_dir, envios, "promedio", destino=definitiva, con_modelos=True)
    try:
        salida = exportar(repo, carpeta, run_dir, commit, definitiva)
        info, destino = salida["info"], salida["destino"]
        chequear_indice(repo)
        csv_run = next((run_dir / "submits").glob(info["csv_original"]), None) or next(
            run_dir.rglob(info["csv_original"])
        )
        generado = validador(destino, info, csv_run, modo, crudo_local)
    except BaseException:
        _restaurar(repo)  # el repo de entregas queda como estaba: nada a medias
        raise

    mensaje = (
        f"entrega {carpeta}: {info['experimento']} con {info['envios']} envíos "
        f"(sha256 {info['sha256_esperado'][:12]})"
    )
    head = commit_y_push(repo, carpeta, mensaje, push)
    return {
        "commit_entregas": head,
        "link": f"{URL_REPO}/tree/main/{carpeta}",
        "sha256": info["sha256_esperado"],
        "csv_generado": str(generado),
        "push": push,
    }
