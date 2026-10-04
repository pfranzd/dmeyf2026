"""Barrido de 9 combinaciones para k=3.

Ejecuta z501_analisis_video.py con distintas combinaciones de
min_samples_leaf, filtro de features y PCA. Cada corrida genera su
propia carpeta con nombre único (el script z501_analisis_video.py ya
maneja los sufijos automáticamente).

Uso desde la línea de comandos:
    python barrer_k3.py --csv ../datasets/processed/competencia_01_fe.csv

Uso desde un notebook:
    !python barrer_k3.py --csv ../datasets/processed/competencia_01_fe.csv

Opciones:
    --skip-existing   salta combinaciones cuya carpeta ya existe (útil para
                      no re-correr los baselines)
    --solo N          corre solo la combinación N (1-9)

Tiempo total estimado: 9 x 5 min ≈ 45 min. Usa --skip-existing si ya
tenés algunas combinaciones corridas.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


# (label, msl, modo, pca)
# modo: "limpio" | "filtrado" | "sin_filtro"
COMBINACIONES = [
    # --- Baseline (ya lo tenés) ---
    ("01_baseline_limpio_sinpca",   40, "limpio",   0),
    ("02_baseline_limpio_pca20",    40, "limpio",  20),
    ("03_baseline_filtrado_sinpca", 50, "filtrado", 0),
    # --- Sensibilidad a msl (hojas del RF) ---
    ("04_msl30_limpio_pca20",       30, "limpio",  20),
    ("05_msl60_limpio_pca20",       60, "limpio",  20),
    # --- Sensibilidad a PCA (dimensiones del espacio) ---
    ("06_pca10_limpio",             40, "limpio",  10),
    ("07_pca30_limpio",             40, "limpio",  30),
    # --- Filtrado con PCA (conservar pr_/d10_) ---
    ("08_filtrado_pca20",           40, "filtrado", 20),
    # --- Combinación agresiva ---
    ("09_agresivo_msl30_filtrado_pca10", 30, "filtrado", 10),
]


def sufijo_esperado(modo: str, pca: int) -> str:
    """Reconstruye el sufijo que z501_analisis_video.py le pone al out_dir."""
    if modo == "limpio":
        sufijo = "_limpio"
    elif modo == "filtrado":
        sufijo = "_filtrado"
    else:
        sufijo = ""
    if pca > 0:
        sufijo += f"_pca{pca}"
    return sufijo


def construir_comando(csv: str, msl: int, modo: str, pca: int) -> list[str]:
    """Arma la línea de comando para invocar z501_analisis_video.py."""
    cmd = [
        "python", "z501_analisis_video_pca_sin_rent.py",
        "--csv", csv,
        "--k", "3",
        "--min-samples-leaf", str(msl),
    ]
    if modo == "limpio":
        cmd.append("--modo-limpio")
    elif modo == "sin_filtro":
        cmd.append("--sin-filtrar-redundantes")
    # modo "filtrado" es el default: no se pasa flag
    if pca > 0:
        cmd.extend(["--pca", str(pca)])
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="ruta al CSV del FE (competencia_01_fe.csv)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="saltar combinaciones cuya carpeta ya existe")
    ap.add_argument("--solo", type=int, default=None,
                    help="correr solo la combinación N (1-9)")
    args = ap.parse_args()

    t0 = time.time()
    resultados = []

    for i, (label, msl, modo, pca) in enumerate(COMBINACIONES, 1):
        if args.solo is not None and i != args.solo:
            continue

        print(f"\n{'='*72}")
        print(f"[{i}/9] {label}")
        print(f"       msl={msl}, modo={modo}, pca={pca}")
        print('='*72)

        # Verificar si la carpeta ya existe
        sufijo = sufijo_esperado(modo, pca)
        out_dir = Path(f"analisis_k3_msl{msl}{sufijo}")

        if args.skip_existing and out_dir.exists():
            print(f"  ya existe: {out_dir} (skip)")
            resultados.append((label, "skip", out_dir))
            continue

        # Ejecutar
        t_inicio = time.time()
        cmd = construir_comando(args.csv, msl, modo, pca)
        result = subprocess.run(cmd)
        duracion = time.time() - t_inicio

        if result.returncode != 0:
            print(f"  ERROR (código {result.returncode})")
            resultados.append((label, "error", out_dir))
        else:
            print(f"  listo en {duracion:.0f}s → {out_dir}")
            resultados.append((label, "ok", out_dir))

    # Resumen final
    print(f"\n{'='*72}")
    print(f"BARRIDO COMPLETO en {(time.time() - t0)/60:.1f} min")
    print('='*72)
    for label, status, out_dir in resultados:
        print(f"  [{status:5s}] {label:40s} → {out_dir}")

    print("\nPróximo paso: correr comparar_k3.py para armar la tabla comparativa.")


if __name__ == "__main__":
    main()
