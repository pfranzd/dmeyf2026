#!/usr/bin/env bash
# Prepara el entorno para correr competencia_1 en Linux / una VM de GCP.
# Idempotente: crea .venv, instala las dependencias fijadas y descarga el dataset crudo.
#   bash competencia_1/scripts/preparar_entorno.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || { echo "Se necesita Python >= 3.11 (actual: $("$PYTHON" --version))"; exit 1; }

[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r competencia_1/requirements.txt

CRUDO=datasets/raw/competencia_01_crudo.csv
URL=https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv
if [ ! -f "$CRUDO" ]; then
    mkdir -p datasets/raw
    echo "Descargando $URL (~500 MB)..."
    curl -fL --retry 3 -o "$CRUDO.tmp" "$URL"
    mv "$CRUDO.tmp" "$CRUDO"
fi
echo "Entorno listo. Python: $(.venv/bin/python --version); dataset: $CRUDO"
