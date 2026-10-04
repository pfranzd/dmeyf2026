"""Marca cada fila cliente-mes de BAJA+2 con su cluster asignado.

Reejecuta el pipeline ganador del análisis (experimento 9):
    k=3, msl=30, filtro liviano de variantes redundantes de ctrx_quarter,
    PCA=10 componentes antes de KMeans.

Guarda un CSV con la HISTORIA COMPLETA de cada cliente BAJA+2 (todas sus
filas cliente-mes) más la columna 'cluster' con el label asignado (1, 2 ó 3).
Cada cliente tiene el mismo cluster en todos sus meses, porque el clustering
es a nivel cliente (agrega toda su historia en un solo perfil).

Uso:
    python z502_marcar_clusters.py --csv ../datasets/processed/competencia_01_fe.csv

    # Con ruta de salida personalizada:
    python z502_marcar_clusters.py --csv ../datasets/processed/competencia_01_fe.csv \\
        --out ./datasets/clusters_marcados.csv

Salida por defecto: clusters_marcados.csv en el directorio actual.

Estructura del CSV de salida:
    numero_de_cliente | foto_mes | clase_ternaria | cluster | [~630 features del FE]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reutiliza el pipeline del script principal
from z501_cluster_rf import (
    ID_COL, MES_COL, TARGET_COL, GRUPO_COL,
    cargar_muestra, columnas_features, entrenar_rf,
    contar_hojas_por_id, normalizar, log,
)
from z501_analisis_video_pca_sin_rent import (
    EXCLUIR_REDUNDANTES,
    clusterizar_con_pca,
)


# --- Parámetros ganadores del experimento 9 ---
# No se exponen como CLI: son el clustering FINAL, no se tocan.
K = 3
MSL = 30
PCA_N = 10
N_TREES = 300
SEED = 214363


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, required=True,
                    help="ruta al CSV del FE (competencia_01_fe.csv)")
    ap.add_argument("--out", type=Path, default=Path("clusters_marcados.csv"),
                    help="ruta del CSV de salida "
                         "(default: clusters_marcados.csv en directorio actual)")
    args = ap.parse_args()

    # --- Pipeline (RF + PCA + KMeans) ---
    log(f"leyendo {args.csv.name}")
    df = cargar_muestra(args.csv, SEED)
    feats_todas = columnas_features(df)

    # Filtro liviano: excluye las 11 variantes redundantes de ctrx_quarter
    # (avg_3, min_3, max_3, std_3, d10, ratioavg_3, slope_3, lag_1, lag_2,
    # delta_1, delta_2). Es el modo "filtrado" del script principal.
    feats = [f for f in feats_todas if f not in EXCLUIR_REDUNDANTES]
    log(f"filtro: {len(feats_todas) - len(feats)} variantes redundantes "
        f"de ctrx_quarter excluidas ({len(feats)} features finales)")

    X = df[feats].to_numpy(dtype=np.float32)
    y = df[GRUPO_COL].to_numpy()
    ids = df[ID_COL].to_numpy()

    log(f"entrenando RF (msl={MSL}, n_trees={N_TREES})")
    rf = entrenar_rf(X, y, N_TREES, MSL, SEED)
    log(f"OOB={rf.oob_score_:.4f}")

    log("contando hojas por cliente")
    P = normalizar(contar_hojas_por_id(rf, X, ids, feats))
    ids_churn = pd.Index(df.loc[df[GRUPO_COL] == 1, ID_COL].unique())
    P_churn = P.loc[ids_churn]

    log(f"clusterizando con PCA({PCA_N}) + KMeans(k={K})")
    labels, pca = clusterizar_con_pca(P_churn, K, PCA_N, SEED)
    var_exp = pca.explained_variance_ratio_.sum()
    log(f"PCA({pca.n_components_}) varianza explicada = {var_exp:.3f} "
        f"({var_exp*100:.1f}%)")
    log("tamaños: " + ", ".join(
        f"cluster_{c}={n:,}"
        for c, n in labels.value_counts().sort_index().items()
    ))

    # --- Armar dataset final: historia completa de BAJA+2 + cluster ---
    log("armando dataset (historia completa cliente-mes)")
    df_baja2 = df[df[GRUPO_COL] == 1].copy()
    df_baja2 = df_baja2.merge(
        labels.rename("cluster"),
        left_on=ID_COL, right_index=True, how="inner",
    )

    # Reordenar columnas: identificadores + cluster al frente para lectura
    # cómoda. La columna 'grupo' (siempre 1 acá) se descarta.
    cols_frente = [ID_COL, MES_COL, TARGET_COL, "cluster"]
    otras_cols = [
        c for c in df_baja2.columns
        if c not in cols_frente + [GRUPO_COL]
    ]
    df_baja2 = df_baja2[cols_frente + otras_cols]
    df_baja2 = df_baja2.sort_values([ID_COL, MES_COL])

    # --- Guardar ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log(f"guardando → {args.out}")
    df_baja2.to_csv(args.out, index=False)

    # --- Reporte final ---
    log("=" * 60)
    log(f"dataset final: {len(df_baja2):,} filas × {df_baja2.shape[1]} columnas")
    log(f"  {df_baja2[ID_COL].nunique():,} clientes BAJA+2 únicos")
    log(f"  {df_baja2[MES_COL].nunique()} meses distintos "
        f"({df_baja2[MES_COL].min()} a {df_baja2[MES_COL].max()})")
    log("  filas por cluster:")
    for c, n in df_baja2["cluster"].value_counts().sort_index().items():
        n_clientes = df_baja2[df_baja2["cluster"] == c][ID_COL].nunique()
        log(f"    cluster_{c}: {n:,} filas ({n_clientes:,} clientes únicos)")
    log("=" * 60)
    log("listo. Para análisis rápidos:")
    log("  import pandas as pd")
    log(f"  df = pd.read_csv('{args.out}')")
    log("  df.groupby('cluster').agg(...)")


if __name__ == "__main__":
    main()
