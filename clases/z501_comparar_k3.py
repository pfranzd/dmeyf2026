"""Comparación de los 9 experimentos del barrido k=3.

Lee automáticamente los CSVs de cada carpeta (rentabilidad_por_cluster.csv
y variables_diferenciadoras.csv) y arma:

  1. Una tabla comparativa por cluster con:
     - Tamaño de cada cluster
     - Ratio vs fiel
     - Top-1 variable diferenciadora + su Cohen's d y lift
     - Suma total de |Cohen's d| del top-5 (proxy de "qué tan distintivo" es)

  2. Ranking automático por criterio narrativo:
     - Los 3 clusters tienen tamaños razonables (>150 cada uno)
     - Los 3 clusters tienen top-1 con |Cohen's d| >= 1.0
     - Los top-1 de los 3 clusters son variables DISTINTAS entre sí
     - El total en juego es alto

Uso:
    python comparar_k3.py

Salida: comparacion_k3.csv + tabla en pantalla.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Mismo mapeo que en barrer_k3.py — mantener sincronizado
COMBINACIONES = [
    ("01_baseline_limpio_sinpca",   40, "limpio",   0),
    ("02_baseline_limpio_pca20",    40, "limpio",  20),
    ("03_baseline_filtrado_sinpca", 50, "filtrado", 0),
    ("04_msl30_limpio_pca20",       30, "limpio",  20),
    ("05_msl60_limpio_pca20",       60, "limpio",  20),
    ("06_pca10_limpio",             40, "limpio",  10),
    ("07_pca30_limpio",             40, "limpio",  30),
    ("08_filtrado_pca20",           40, "filtrado", 20),
    ("09_agresivo_msl30_filtrado_pca10", 30, "filtrado", 10),
]


def sufijo_esperado(modo: str, pca: int) -> str:
    if modo == "limpio":
        sufijo = "_limpio"
    elif modo == "filtrado":
        sufijo = "_filtrado"
    else:
        sufijo = ""
    if pca > 0:
        sufijo += f"_pca{pca}"
    return sufijo


def analizar_experimento(carpeta: Path, label: str) -> dict | None:
    """Extrae métricas comparativas de una carpeta de experimento."""
    rent_path = carpeta / "rentabilidad_por_cluster.csv"
    dif_path = carpeta / "variables_diferenciadoras.csv"

    if not rent_path.exists() or not dif_path.exists():
        return None

    rent = pd.read_csv(rent_path).sort_values("cluster")
    dif = pd.read_csv(dif_path)

    # Métricas globales
    metricas = {
        "experimento": label,
        "n_total_baja2": int(rent["n_clientes"].sum()),
        "total_juego_M": round(rent["rent_anual_cluster_total"].sum() / 1e6, 1),
        "n_clusters_grandes": int((rent["n_clientes"] >= 150).sum()),
    }

    # Por cluster: tamaño, ratio, top variable, suma de |d| top-5
    top_vars = {}
    suma_abs_d = 0.0
    all_d_geq_1 = True
    min_cohen_d_top1 = float("inf")

    for c in sorted(rent["cluster"].unique()):
        rent_row = rent[rent["cluster"] == c].iloc[0]
        metricas[f"c{int(c)}_n"] = int(rent_row["n_clientes"])
        metricas[f"c{int(c)}_ratio"] = round(rent_row["ratio_vs_fiel"], 2)
        metricas[f"c{int(c)}_M"] = round(rent_row["rent_anual_cluster_total"] / 1e6, 1)

        dif_c = dif[dif["cluster"] == c].sort_values(
            "cohen_d", key=lambda s: s.abs(), ascending=False
        )
        if len(dif_c) == 0:
            metricas[f"c{int(c)}_top1_var"] = "(sin señal)"
            metricas[f"c{int(c)}_top1_d"] = 0.0
            metricas[f"c{int(c)}_top1_lift"] = 0.0
            all_d_geq_1 = False
        else:
            top1 = dif_c.iloc[0]
            metricas[f"c{int(c)}_top1_var"] = top1["variable"]
            metricas[f"c{int(c)}_top1_d"] = round(top1["cohen_d"], 2)
            metricas[f"c{int(c)}_top1_lift"] = round(top1["lift"], 2)
            top_vars[int(c)] = top1["variable"]
            suma_abs_d += dif_c.head(5)["cohen_d"].abs().sum()
            min_cohen_d_top1 = min(min_cohen_d_top1, abs(top1["cohen_d"]))

    metricas["suma_abs_d_top5"] = round(suma_abs_d, 1)
    metricas["min_top1_d"] = round(min_cohen_d_top1, 2) if min_cohen_d_top1 != float("inf") else 0.0

    # ¿Los top-1 de los 3 clusters son distintos?
    metricas["top1_distintos"] = len(set(top_vars.values())) == len(top_vars)

    return metricas


def puntuar(m: dict) -> float:
    """Puntaje compuesto para rankear experimentos.

    Criterios (a mayor puntaje mejor):
      + Los 3 clusters tienen >= 150 clientes (evita clusters basura)
      + Los top-1 son variables DISTINTAS entre sí (perfiles diferentes)
      + El |Cohen's d| mínimo del top-1 es alto (todos los clusters nítidos)
      + Suma de |d| del top-5 alta (perfiles ricos, no una sola señal)
    """
    puntaje = 0.0

    # 1. Clusters chicos penalizan fuerte
    if m["n_clusters_grandes"] == 3:
        puntaje += 5
    elif m["n_clusters_grandes"] == 2:
        puntaje += 1

    # 2. Top-1 distintos entre clusters
    if m["top1_distintos"]:
        puntaje += 5

    # 3. Nitidez mínima
    puntaje += m["min_top1_d"] * 3

    # 4. Riqueza global (suma de d top-5 de los 3 clusters)
    puntaje += m["suma_abs_d_top5"] / 5.0

    return round(puntaje, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="comparacion_k3.csv")
    args = ap.parse_args()

    filas = []
    for label, msl, modo, pca in COMBINACIONES:
        sufijo = sufijo_esperado(modo, pca)
        carpeta = Path(f"analisis_k3_msl{msl}{sufijo}")
        m = analizar_experimento(carpeta, label)
        if m is None:
            print(f"[skip] {label}: carpeta {carpeta} no encontrada o incompleta")
            continue
        m["puntaje"] = puntuar(m)
        filas.append(m)

    if not filas:
        print("No se encontró ninguna carpeta con resultados. Corré barrer_k3.py primero.")
        return

    df = pd.DataFrame(filas).sort_values("puntaje", ascending=False)
    df.to_csv(args.out, index=False)

    # Vista en pantalla
    print("\n" + "=" * 100)
    print("RANKING DE EXPERIMENTOS (a mayor puntaje mejor)")
    print("=" * 100)

    for _, r in df.iterrows():
        print(f"\n{r['experimento']}  [puntaje: {r['puntaje']}]")
        print(f"  Total en juego: ${r['total_juego_M']}M | n total BAJA+2: {r['n_total_baja2']}")
        print(f"  Clusters con >=150 clientes: {r['n_clusters_grandes']}/3 | "
              f"Top-1 distintos: {r['top1_distintos']} | "
              f"Min |d| top-1: {r['min_top1_d']}")
        for c in [1, 2, 3]:
            n = r.get(f"c{c}_n")
            if n is None:
                continue
            print(f"  Cluster {c}: n={n}, ratio={r[f'c{c}_ratio']}, "
                  f"${r[f'c{c}_M']}M → {r[f'c{c}_top1_var']} "
                  f"(d={r[f'c{c}_top1_d']}, lift={r[f'c{c}_top1_lift']})")

    print(f"\n\nTabla completa guardada en: {args.out}")

    # Sugerencia del ganador
    ganador = df.iloc[0]
    print(f"\n{'='*100}")
    print(f"GANADOR SUGERIDO: {ganador['experimento']}")
    print(f"  Puntaje {ganador['puntaje']}, ${ganador['total_juego_M']}M en juego,")
    print(f"  3 clusters con variables top-1 distintas y nítidas.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
