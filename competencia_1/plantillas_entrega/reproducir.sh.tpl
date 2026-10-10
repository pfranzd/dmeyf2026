#!/usr/bin/env bash
# Reproduce la entrega (Linux / VM de GCP). Uso:
#   bash reproducir.sh              # predice con los modelos ya entrenados (~15 min)
#   bash reproducir.sh entrenar     # reentrena los modelos (~80 min)
#   bash reproducir.sh completo     # rehace también la optimización de Optuna (~2,5 h)
# Crea un entorno virtual .venv con las versiones exactas de requirements-lock.txt (Python 3.11),
# descarga el dataset si falta y termina con código 0 solo si el sha256 coincide.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Hace falta Python >= 3.11 (instalá python3.11 o exportá PYTHON=<ruta>)"; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements-lock.txt
ENTREGA_MODO="${1:-predecir}" exec .venv/bin/python entrega.py
