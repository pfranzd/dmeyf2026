# Proyecto DMEyF 2026 — UBA (implementación en Python)

## Reglas críticas
- Este repo es un FORK PÚBLICO de github.com/dmecoyfin/dmeyf2026.
- Los materiales oficiales de la cátedra están en `monday/` (notebooks .ipynb en
  Python) y otras carpetas de clase que se irán sumando. `git-zero-to-hero/` es
  documentación de git. NO editar nada de esas carpetas: se sincronizan con
  upstream vía `-Xtheirs`, que pisa cambios locales. Para trabajar sobre un
  notebook oficial, copiarlo a `exp/` y editar la copia.
- Al copiar un script oficial, anotar en un comentario de qué archivo de src/ proviene
  y en qué commit del upstream estaba.
- NUNCA commitear credenciales, tokens ni datasets. El repo es público.
- Datos: polars o DuckDB (millones de registros; no pandas por defecto).
  Modelos: lightgbm / xgboost / scikit-learn. Opt. bayesiana: Optuna. Tracking: MLflow.
- Reproducibilidad total: semillas en config/semillas.py; versiones fijas en pyproject.toml.

## Estructura
monday/            → CÁTEDRA · notebooks oficiales · read-only · sincroniza con upstream
(otras de clase)   → CÁTEDRA · idem, se irán sumando
git-zero-to-hero/  → CÁTEDRA · documentación de git · read-only
clases/            → MÍO · copias de los notebooks de la cátedra para estudiar/anotar
exp/               → MÍO · scripts .py: un experimento por carpeta (implementación real)
dmeyf/             → MÍO · paquete Python con funciones reutilizables
config/            → MÍO · semillas.py y configuración
notebooks/         → MÍO · EDA (emparejado con .py vía jupytext)
datasets/          → datos locales (en .gitignore, NO se suben)
work/              → salidas, modelos, submits (en .gitignore, NO se suben)
notas/             → apuntes markdown
docs/              → documento con el libro de cátedra