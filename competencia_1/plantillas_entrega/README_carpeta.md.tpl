# Competencia 01 — entrega final: @@EXPERIMENTO@@ con @@ENVIOS@@ envíos

Esta carpeta contiene todo lo necesario para replicar **exactamente** el archivo que se subió
como entrega final de la competencia:

| | |
|---|---|
| Archivo entregado | `@@CSV_ORIGINAL@@` (solo `numero_de_cliente`, sin encabezado, @@ENVIOS@@ filas) |
| sha256 | `@@SHA256@@` |
| Puntaje público | @@PUBLICO@@ |
| Período objetivo | 202108 |

El dataset de la competencia **no está en el repositorio**: el script lo descarga solo desde la
URL pública de la cátedra (`competencia_01_crudo.csv`).

## Cómo replicarla

Hace falta Python 3.11 y unos 10 GB de RAM. Hay tres niveles; todos terminan comparando el
sha256 del CSV generado con el de arriba (`RESULTADO: OK, coincide`):

| Modo | Qué rehace | Tiempo aprox. |
|---|---|---|
| `predecir` (default) | Datos y features desde el crudo, y **predice** con los @@N_MODELOS@@ modelos ya entrenados | ~15 min |
| `entrenar` | Lo anterior, pero **reentrena** los @@N_MODELOS@@ modelos con los hiperparámetros de `competencia_1/definitiva/params.json` | ~80 min |
| `completo` | **Todo, incluida la optimización de hiperparámetros** (Optuna, @@N_TRIALS@@ trials) que dio origen a esos hiperparámetros | ~2,5 h |

**Linux o VM de GCP:**

```bash
git clone @@URL_REPO@@ && cd dmeyf2026-entregas/@@CARPETA@@
bash reproducir.sh              # o: bash reproducir.sh entrenar | bash reproducir.sh completo
```

**Windows (PowerShell):**

```powershell
git clone @@URL_REPO@@; cd dmeyf2026-entregas\@@CARPETA@@
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
$env:ENTREGA_MODO = "predecir"      # o "entrenar" / "completo"
.venv\Scripts\python entrega.py
```

También se puede abrir `entrega.ipynb` (es el mismo código que `entrega.py`) y ejecutarlo
completo; el modo se elige en la segunda celda.

El CSV generado queda en `work/entrega/runs/<run>/submits/` y el script compara su sha256 con
`competencia_1/definitiva/entrega.json`.

## Entorno

- `requirements-lock.txt`: **versiones exactas** de todas las librerías (incluidas las
  transitivas) con las que se generó la entrega. `requirements.txt` lista solo las directas.
- Python @@PYTHON@@ (`.python-version`). El script avisa si alguna versión instalada difiere.
- La igualdad byte a byte depende de las versiones de LightGBM y DuckDB y de
  `lgbm.fijos.num_threads` (8) en `competencia_1/configs/base.yaml`; no los cambies.

## Qué hace el modelo

@@RESUMEN@@

## De dónde salen los hiperparámetros (optimización)

@@OPTIMIZACION@@

## Contenido de la carpeta

```
entrega.py / entrega.ipynb     punto de entrada (mismo código)
reproducir.sh                  atajo para Linux / GCP
requirements*.txt, .python-version
competencia_1/                 pipeline (datos → features → Optuna → modelos → CSV)
  configs/base.yaml            defaults
  configs/exp/@@EXPERIMENTO@@.yaml   configuración del experimento (optimización + final)
  definitiva/                  config.yaml, params.json, entrega.json y modelos/ (entrega congelada)
dmeyf/, config/                métricas y semillas
optimizacion/                  evidencia de la optimización (estudio de Optuna + resumen)
```

## Trazabilidad

Generada el @@FECHA@@ desde el repositorio de trabajo, commit `@@COMMIT@@`, run
`@@RUN_ID@@`. El CSV de ese run coincide byte a byte con el que reproduce esta carpeta.
