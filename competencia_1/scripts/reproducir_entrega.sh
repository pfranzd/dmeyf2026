#!/usr/bin/env bash
# Reproduce la entrega definitiva y compara el sha256 del CSV (Linux / GCP).
#   bash competencia_1/scripts/reproducir_entrega.sh              # predice con los modelos guardados
#   bash competencia_1/scripts/reproducir_entrega.sh --entrenar   # reentrena todo desde cero
# Usa work/entrega como directorio de trabajo (se cambia exportando DMEYF_WORK).
# Tarda ~15 min (con --entrenar ~80 min) y necesita ~10 GB de RAM. Sale con código 0 solo si el sha256 coincide.
set -euo pipefail
cd "$(dirname "$0")/../.."

bash competencia_1/scripts/preparar_entorno.sh
exec .venv/bin/python -m competencia_1.entrega reproducir "$@"
