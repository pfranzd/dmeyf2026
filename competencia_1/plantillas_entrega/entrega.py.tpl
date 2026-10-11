# %% [markdown]
# # Entrega Competencia 01 — @@EXPERIMENTO@@ (@@ENVIOS@@ envíos)
#
# Este archivo (y su copia `entrega.ipynb`, que es el mismo código) reproduce **exactamente** el
# CSV entregado: `@@CSV_ORIGINAL@@` (sha256 `@@SHA256@@`).
#
# Parte del dataset de la competencia (`competencia_01_crudo.csv`, se descarga solo si falta) y
# comprueba al final que el CSV generado tenga el mismo sha256. Elegí el modo en la celda
# siguiente o con la variable de entorno `ENTREGA_MODO`:
#
# | MODO | Qué hace | Tiempo aprox. |
# |---|---|---|
# | `predecir` (default) | datos → features → predice con los @@N_MODELOS@@ modelos ya entrenados (`competencia_1/definitiva/modelos`) | ~16 min |
# | `entrenar` | datos → features → reentrena los @@N_MODELOS@@ modelos con los hiperparámetros de `params.json` | ~80 min |
# | `completo` | rehace TODO: la optimización de hiperparámetros con Optuna (@@N_TRIALS@@ trials), selección de los mejores y modelos finales | ~1,5 h |

# %%
import os

MODO = os.environ.get("ENTREGA_MODO", "predecir")  # "predecir" | "entrenar" | "completo"
assert MODO in ("predecir", "entrenar", "completo"), MODO

# %%
import importlib.metadata as metadata
import json
import shutil
import sys
import urllib.request
from pathlib import Path

# Carpeta de esta entrega (en un notebook __file__ no existe: se usa el directorio actual).
AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
os.chdir(AQUI)
sys.path.insert(0, str(AQUI))
# Todo lo que se genera va a work/entrega (y la base de Optuna también): nada se mezcla.
os.environ["DMEYF_WORK"] = str(AQUI / "work" / "entrega")
os.environ["DMEYF_DB"] = str(AQUI / "work" / "entrega" / "db" / "competencia_1.db")

URL_CRUDO = "https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv"
CRUDO = AQUI / "datasets" / "raw" / "competencia_01_crudo.csv"

# %% [markdown]
# ## 1. Versiones de las librerías
# La igualdad byte a byte requiere las mismas versiones con las que se generó la entrega
# (`requirements-lock.txt` las instala). Si alguna difiere se avisa, pero se sigue.

# %%
info = json.loads((AQUI / "competencia_1" / "definitiva" / "entrega.json").read_text(encoding="utf-8"))
esperadas = info.get("versiones") or {}
print(f"python {sys.version.split()[0]} (esperado {esperadas.get('python')})")
for paquete, esperada in esperadas.items():
    if paquete == "python":
        continue
    instalada = metadata.version(paquete)
    print(f"{paquete:10s} {instalada:10s} esperado {esperada}", "" if instalada == esperada else "  <-- DISTINTA")

# %% [markdown]
# ## 2. Dataset crudo
# Se usa tal cual lo publica la cátedra (no se sube al repositorio). Si no está, se descarga.

# %%
if not CRUDO.exists():
    CRUDO.parent.mkdir(parents=True, exist_ok=True)
    local = os.environ.get("ENTREGA_CRUDO_LOCAL")  # atajo para no descargar de nuevo
    if local and Path(local).exists():
        shutil.copyfile(local, CRUDO)
    else:
        print("descargando", URL_CRUDO)
        urllib.request.urlretrieve(URL_CRUDO, CRUDO)
print(CRUDO, f"{CRUDO.stat().st_size / 1e6:.0f} MB")

# %% [markdown]
# ## 3. Reproducción y comparación del sha256
# La lógica vive en `competencia_1/` (el mismo pipeline con el que se hicieron los experimentos).

# %%
from competencia_1.pipeline.entrega import DIR_DEFINITIVA, reproducir, reproducir_completo

if MODO == "completo":
    ok = reproducir_completo(DIR_DEFINITIVA, limpiar=True)
else:
    ok = reproducir(DIR_DEFINITIVA, limpiar=True, entrenar=(MODO == "entrenar"))

# %%
print("RESULTADO:", "OK, el CSV coincide con la entrega" if ok else "DIFIERE")
if not ok:
    raise SystemExit(1)
