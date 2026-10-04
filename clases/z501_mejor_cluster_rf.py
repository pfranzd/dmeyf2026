"""Barrido de hiperparámetros para clusters_rf_alumnos.py.

Ejecuta el pipeline con distintas combinaciones de (k, seed_kmeans,
min_samples_leaf, n_trees) y devuelve un CSV con métricas de calidad y
un ranking por cada criterio.

Uso mínimo (barre solo k, RF fijo):
    python buscar_mejor_cluster.py --csv competencia_01_ct_julia.csv

Barrido completo (RF también):
    python buscar_mejor_cluster.py --csv competencia_01_ct_julia.csv \\
        --ks 3 4 5 6 7 8 \\
        --seeds-kmeans 42 214363 2024 \\
        --min-samples-leaf 30 50 100 \\
        --n-trees 200 300

El RF se cachea entre configuraciones: si solo cambia k o seed_kmeans, no se
re-entrena. Los resultados se guardan incrementalmente, así que se puede
interrumpir el proceso sin perder lo hecho.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ID_COL = "numero_de_cliente"
MES_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
GRUPO_COL = "grupo"
NON_FEATURE_COLS = (ID_COL, MES_COL, TARGET_COL, GRUPO_COL)
RAIZ_COL = "_sin_split"

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Funciones reutilizadas del script original (copiadas para hacer standalone)
# --------------------------------------------------------------------------- #

_QUERY_MUESTRA = """
WITH raw AS (
    SELECT * REPLACE (CAST({id} AS BIGINT) AS {id})
    FROM read_csv('{csv}', sample_size = -1)
),
churn AS (SELECT DISTINCT {id} FROM raw WHERE {target} = 'BAJA+2'),
n_meses AS (SELECT COUNT(DISTINCT {mes}) AS n FROM raw),
fieles AS (
    SELECT {id} FROM raw GROUP BY {id}
    HAVING COUNT(*) = (SELECT n FROM n_meses)
       AND SUM(CASE WHEN {target} IN ('BAJA+1', 'BAJA+2') THEN 1 ELSE 0 END) = 0
    ORDER BY hash({id} + {seed}) LIMIT (SELECT COUNT(*) FROM churn)
)
SELECT raw.*, 1 AS {grupo} FROM raw JOIN churn USING ({id})
UNION ALL
SELECT raw.*, 0 AS {grupo} FROM raw JOIN fieles USING ({id})
ORDER BY {id}, {mes}
"""


def cargar_muestra(csv_path: Path, seed: int) -> pd.DataFrame:
    query = _QUERY_MUESTRA.format(
        csv=csv_path.as_posix(), id=ID_COL, mes=MES_COL,
        target=TARGET_COL, grupo=GRUPO_COL, seed=seed,
    )
    df = duckdb.connect().execute(query).df()
    df[TARGET_COL] = df[TARGET_COL].astype("string")
    return df


def columnas_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def entrenar_rf(X, y, n_trees, min_samples_leaf, seed):
    rf = RandomForestClassifier(
        n_estimators=n_trees, min_samples_leaf=min_samples_leaf,
        max_features="sqrt", class_weight="balanced", oob_score=True,
        n_jobs=-1, random_state=seed,
    )
    rf.fit(X, y)
    return rf


def atributo_de_hoja(estimator) -> np.ndarray:
    t = estimator.tree_
    mapa = np.full(t.node_count, -1, dtype=np.int64)
    internos = np.flatnonzero(t.children_left != -1)
    mapa[t.children_left[internos]] = t.feature[internos]
    mapa[t.children_right[internos]] = t.feature[internos]
    return mapa


def contar_hojas_por_id(rf, X, ids, feature_names):
    hojas = rf.apply(X)
    n_feat = len(feature_names)
    codigos = np.empty_like(hojas)
    for j, est in enumerate(rf.estimators_):
        codigos[:, j] = atributo_de_hoja(est)[hojas[:, j]]
    codigos[codigos < 0] = n_feat
    idx, ids_unicos = pd.factorize(ids)
    C = np.zeros((len(ids_unicos), n_feat + 1), dtype=np.int64)
    np.add.at(C, (np.repeat(idx, hojas.shape[1]), codigos.ravel()), 1)
    return pd.DataFrame(
        C, index=pd.Index(ids_unicos, name=ID_COL),
        columns=feature_names + [RAIZ_COL],
    )


def normalizar(C):
    P = C.drop(columns=[RAIZ_COL], errors="ignore")
    P = P.loc[:, P.sum(axis=0) > 0]
    return P.div(P.sum(axis=1), axis=0)


def clusterizar(P, k, seed):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(P.values)
    orden = pd.Series(km.labels_).value_counts().index.tolist()
    remap = {viejo: nuevo + 1 for nuevo, viejo in enumerate(orden)}
    return pd.Series(
        [remap[l] for l in km.labels_], index=P.index, name="cluster",
    )


# --------------------------------------------------------------------------- #
# Evaluación de calidad del clustering
# --------------------------------------------------------------------------- #

def calcular_lifts(P_churn, labels, top_n=3):
    """Para cada cluster, devuelve el top-n de lifts (share_c / share_global)
    entre atributos con presencia mínima. Devuelve dict cluster -> [lifts]."""
    share_global = P_churn.mean(axis=0)
    min_share = 0.5 / P_churn.shape[1]
    out = {}
    for c in sorted(labels.unique()):
        Pc = P_churn[labels == c]
        share_c = Pc.mean(axis=0)
        lifts = (share_c / share_global.replace(0, np.nan)).dropna()
        # solo atributos con presencia real en el cluster
        lifts = lifts[share_c[lifts.index] >= min_share]
        out[c] = lifts.sort_values(ascending=False).head(top_n).tolist()
    return out


def evaluar_clustering(P_churn, labels, top_n=3):
    """Diccionario con métricas de calidad de una partición dada."""
    X = P_churn.values

    sizes = labels.value_counts()
    lifts = calcular_lifts(P_churn, labels, top_n)
    lifts_top1 = [ls[0] for ls in lifts.values() if ls]
    lifts_top3 = [np.mean(ls) for ls in lifts.values() if ls]

    return {
        # geometría en el espacio de hojas
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),  # menor es mejor
        # calidad interpretativa
        "lift_max_top1": max(lifts_top1),
        "lift_mean_top1": np.mean(lifts_top1),
        "lift_min_top1": min(lifts_top1),  # peor cluster
        "lift_mean_top3": np.mean(lifts_top3),
        # tamaños
        "cluster_min_size": int(sizes.min()),
        "cluster_max_size": int(sizes.max()),
        "cluster_size_ratio": float(sizes.min() / sizes.max()),
    }


# --------------------------------------------------------------------------- #
# Barrido con caché del RF
# --------------------------------------------------------------------------- #

def barrer(csv, ks, seeds_kmeans, msls, n_trees_list, seed_rf, top_n, salida):
    log(f"leyendo {csv.name}")
    df = cargar_muestra(csv, seed_rf)
    feats = columnas_features(df)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df[GRUPO_COL].to_numpy()
    ids = df[ID_COL].to_numpy()
    ids_churn = pd.Index(df.loc[df[GRUPO_COL] == 1, ID_COL].unique())
    log(f"muestra: {len(df):,} filas × {len(feats)} features")

    # cargar resultados previos para reanudar
    if salida.exists():
        previos = pd.read_csv(salida)
        hechos = set(zip(
            previos["k"], previos["seed_kmeans"],
            previos["min_samples_leaf"], previos["n_trees"],
        ))
        log(f"reanudando: {len(hechos)} configuraciones ya evaluadas")
        resultados = previos.to_dict("records")
    else:
        hechos, resultados = set(), []

    rf_cache = {}  # (msl, n_trees) -> (oob, P_churn)

    for msl, nt in itertools.product(msls, n_trees_list):
        # skip si todas las combinaciones de este RF ya están hechas
        if all((k, sk, msl, nt) in hechos
               for k, sk in itertools.product(ks, seeds_kmeans)):
            log(f"RF (msl={msl}, n_trees={nt}) ya barrido, salteando")
            continue

        log(f"entrenando RF (min_samples_leaf={msl}, n_trees={nt})")
        rf = entrenar_rf(X, y, nt, msl, seed_rf)
        P = normalizar(contar_hojas_por_id(rf, X, ids, feats))
        P_churn = P.loc[ids_churn]
        oob = rf.oob_score_
        log(f"  OOB={oob:.4f}, matriz {P_churn.shape}")

        for k, sk in itertools.product(ks, seeds_kmeans):
            if (k, sk, msl, nt) in hechos:
                continue
            labels = clusterizar(P_churn, k, sk)
            metricas = evaluar_clustering(P_churn, labels, top_n)
            metricas.update({
                "k": k, "seed_kmeans": sk,
                "min_samples_leaf": msl, "n_trees": nt, "seed_rf": seed_rf,
                "oob_accuracy": oob,
            })
            resultados.append(metricas)
            hechos.add((k, sk, msl, nt))
            # guardado incremental
            pd.DataFrame(resultados).to_csv(salida, index=False)
            log(
                f"  k={k}, seed_km={sk}: "
                f"lift_max={metricas['lift_max_top1']:.2f}  "
                f"lift_mean1={metricas['lift_mean_top1']:.2f}  "
                f"lift_min1={metricas['lift_min_top1']:.2f}  "
                f"sil={metricas['silhouette']:.3f}  "
                f"n_min={metricas['cluster_min_size']}"
            )

    return pd.DataFrame(resultados)


def imprimir_rankings(df, top=10):
    orden = [
        ("lift_max_top1", False, "cluster más nítido (tu criterio actual)"),
        ("lift_mean_top1", False, "calidad promedio de definición"),
        ("lift_min_top1", False, "peor cluster (evita basura)"),
        ("lift_mean_top3", False, "definición robusta (promedio top-3)"),
        ("silhouette", False, "cohesión geométrica"),
        ("davies_bouldin", True, "separación (menor mejor)"),
    ]
    cols = [
        "k", "seed_kmeans", "min_samples_leaf", "n_trees",
        "lift_max_top1", "lift_mean_top1", "lift_min_top1",
        "silhouette", "cluster_min_size",
    ]
    for metrica, asc, glosa in orden:
        print(f"\n=== Top {top} por {metrica} — {glosa} ===")
        print(
            df.sort_values(metrica, ascending=asc)
              .head(top)[cols].to_string(index=False)
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--salida", type=Path, default=Path("resultados_barrido.csv"))
    p.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument(
        "--seeds-kmeans", type=int, nargs="+", default=[214363],
        help="varias detectan inestabilidad del kmeans",
    )
    p.add_argument("--min-samples-leaf", type=int, nargs="+", default=[50])
    p.add_argument("--n-trees", type=int, nargs="+", default=[300])
    p.add_argument("--seed-rf", type=int, default=214363)
    p.add_argument("--top-n", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    n_combos = (
        len(args.ks) * len(args.seeds_kmeans)
        * len(args.min_samples_leaf) * len(args.n_trees)
    )
    n_rf = len(args.min_samples_leaf) * len(args.n_trees)
    log(f"barrido: {n_combos} combinaciones ({n_rf} entrenamientos de RF)")

    df = barrer(
        csv=args.csv, ks=args.ks, seeds_kmeans=args.seeds_kmeans,
        msls=args.min_samples_leaf, n_trees_list=args.n_trees,
        seed_rf=args.seed_rf, top_n=args.top_n, salida=args.salida,
    )
    imprimir_rankings(df)
    log(f"listo → {args.salida}")


if __name__ == "__main__":
    main()
