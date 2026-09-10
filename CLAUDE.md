# Proyecto DMEyF 2026 — UBA (implementación en Python)

## Reglas críticas
- Este repo es un FORK PÚBLICO de github.com/dmecoyfin/dmeyf2026.
- Materiales oficiales de la cátedra en `monday/`, `arboles/`, `ensembles/`,
  `modelos/`, `git-zero-to-hero/` y futuras carpetas de clase. NO editar nada
  de esas carpetas: se sincronizan con upstream vía `-Xtheirs`, que pisa
  cambios locales. Para trabajar sobre un notebook oficial, copiarlo a
  `clases/` (para estudiar/anotar) o a `exp/` (para implementar) y editar
  la copia.
- Al copiar un archivo oficial, dejar comentario con la ruta origen y el hash
  de commit del upstream del que salió.
- NUNCA commitear credenciales, tokens ni datasets. El repo es público.

## Contexto
Fork público de github.com/dmecoyfin/dmeyf2026. Competencia semestral de
churn/retención sobre datos financieros reales. Materia: Data Mining en
Economía y Finanzas, Maestría en Explotación de Datos, UBA.

## Estructura del repo
| Carpeta | Origen | Propósito |
|---|---|---|
| `monday/` (y futuras clases) | CÁTEDRA · read-only | Notebooks oficiales, sincronizan con upstream |
| `arboles/` | CÁTEDRA · read-only | Material oficial de la cátedra |
| `ensembles/` | CÁTEDRA · read-only | Material oficial de la cátedra |
| `modelos/` | CÁTEDRA · read-only | Material oficial de la cátedra |
| `git-zero-to-hero/` | CÁTEDRA · read-only | Documentación de git de la cátedra |
| `clases/` | MÍO | Copias de notebooks oficiales para estudiar y anotar |
| `exp/` | MÍO | Experimentos generales, uno por subcarpeta, scripts `.py` |
| `CazaTalentos/` | CÁTEDRA · read-only | Trabajo sobre desafíos de la competencia paralela CazaTalentos |
| `zero2hero/` | CÁTEDRA · read-only | Material oficial de la cátedra |
| `dmeyf/` | MÍO | Paquete Python con utilidades reutilizables |
| `config/` | MÍO | Semillas (`semillas.py`) y configuración global |
| `notebooks/` | MÍO | EDA inicial, pareado con `.py` vía jupytext |
| `notas/` | MÍO | Apuntes en markdown |
| `docs/` | MÍO | Libro de cátedra y material extenso (en `.gitignore`, solo local) |
| `datasets/` | Datos | Datos crudos, en `.gitignore` |
| `db/` | Datos | Estudios de Optuna en SQLite, en `.gitignore` |
| `work/` | Salidas | Modelos serializados, submits, imágenes, artefactos, en `.gitignore` |

## Stack y convenciones de código
- Escribí Python en scripts `.py` con formato percent (`# %%`) para uso
  interactivo en VS Code. Notebooks `.ipynb` solo para EDA inicial, pareados
  con jupytext.
- Usá `duckdb` y `polars` para datos. No uses `pandas` salvo compatibilidad
  con librerías que lo exijan, y en ese caso documentá el porqué en un comentario.
- Modelos: `lightgbm`, `xgboost`, `scikit-learn`.
- Tuning: `optuna` con persistencia SQLite en `db/<experimento>.db`.
- Tracking: `mlflow` con backend local.
- Formato: `ruff format` y `ruff check` antes de commitear.
- Environment: `.venv` local (Windows), versiones fijadas en `pyproject.toml`.

## Reglas del dominio DMEyF
- Target: `clase_ternaria` con valores {CONTINUA, BAJA+1, BAJA+2}. No
  reencodear sin razón documentada en el README del experimento.
- Métrica de negocio: `ganancia_prob(y_true, y_score, threshold)` en
  `dmeyf/metrics.py`. Umbral default: 0.025 (revisable por experimento).
- Split temporal: siempre con gap de 2 meses entre train y test. Usá
  `StratifiedShuffleSplit` respetando el orden temporal.
- Toda función que use aleatoriedad recibe `seed: int` como parámetro. NUNCA
  hardcodear. Las 5 semillas válidas del curso están en `config/semillas.py`
  (primos entre 100003 y 999983).

## Workflow para un experimento nuevo
1. Crear `exp/<nombre>/` con `config.yaml`, `train.py`, `README.md`.
2. En `train.py`: cargar config, fijar semilla, cargar datos con polars/DuckDB,
   split temporal con gap, entrenar, logear a MLflow, guardar modelo en `work/`.
3. Estudio de Optuna con storage en `db/<nombre>.db`.
4. Al cerrar el experimento: completar `README.md` con hipótesis, decisiones,
   resultados y próximos pasos.

## Git
- Repo público. Cero credenciales, cero datos.
- Sincronizar upstream: `git fetch upstream && git merge -Xtheirs upstream/main`.
- Commits en español, presente imperativo. Ejemplo: "agrega split temporal con gap".
- Al copiar código de cátedra a `exp/` o `clases/`, dejar comentario con la ruta
  origen y el hash de commit del upstream.

   ## Documentación extendida (cargar bajo demanda)
   <!-- Descomentar cuando existan los archivos:
   - @notas/metrica_ganancia.md
   - @notas/leakage_notebook.md
   - @notas/tracking_setup.md
   -->