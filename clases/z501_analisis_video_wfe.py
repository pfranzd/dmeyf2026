"""Análisis para el video de Miranda: rentabilidad + variables diferenciadoras
+ triggers candidatos + radares por cluster + radar global.

Reutiliza el pipeline de z501_cluster_rf.py (que debe estar en el mismo
directorio). Corre RF + KMeans y después ejecuta:

  1. Rentabilidad por cluster: promedio mensual, anualizado, total, y ratio
     contra el cliente fiel promedio. Excluye los últimos 2 meses de cada
     cliente BAJA+2 (comportamiento pre-baja distorsiona la métrica).

  2. Variables diferenciadoras por cluster: para cada cluster, ranking de
     las variables que más lo distinguen del cliente fiel promedio, con:
       - Cohen's d (diferencia estandarizada)
       - Lift de perfilado: P(característica | cluster) / P(característica).
         Para flags: lift = P(=1|cluster) / P(=1). Para continuas: lift =
         P(x del lado extremo | cluster) / 0.5 (o sea, cuánto más concentrado
         está el cluster en el lado alto/bajo que la población general).

  3. Triggers candidatos: para las top-3 variables diferenciadoras de cada
     cluster, tasa de BAJA+2 por decil. Si algún decil tiene tasa >= 75%, es
     un umbral candidato.

  4. Radares por cluster: un gráfico de telaraña por cluster con sus 5
     principales diferenciadoras + rentabilidad (6 ejes, siempre). Muestra
     el cluster protagonista + los otros 4 clusters + los fieles como
     referencia. Escala: percentil global (0 = fondo, 100 = borde).

  5. Radar global: un solo gráfico con las 6 variables más importantes del
     Random Forest (feature importance) y todas las series (5 clusters +
     fieles) superpuestas para comparación general.

Uso:
    python analisis_video.py --csv ../datasets/processed/competencia_01_fe.csv \\
        --k 5 --min-samples-leaf 50

Salida (en ./analisis_k{K}_msl{MSL}/):
    - rentabilidad_por_cluster.csv
    - variables_diferenciadoras.csv     (con columna 'lift')
    - triggers_candidatos.csv
    - resumen.txt
    - radares.pdf                       (todos los radares en un solo PDF)
    - radar_cluster_1.png ... radar_cluster_K.png
    - radar_global.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# Reutiliza el pipeline del script principal (debe estar en el mismo directorio)
from z501_cluster_rf import (
    ID_COL, MES_COL, TARGET_COL, GRUPO_COL, COLORES_CLUSTER, INK, INK_2, GRID,
    cargar_muestra, columnas_features, entrenar_rf,
    contar_hojas_por_id, normalizar, clusterizar,
    caracterizar_clusters, log,
)

# Variable de rentabilidad usada como eje fijo en los radares por cluster
RENTABILIDAD = "mrentabilidad"

# Variables excluidas del análisis de diferenciadoras (evita confundir causa
# con consecuencia; se analizan por separado en la tabla de rentabilidad)
EXCLUIR_DIFERENCIADORAS = {
    "mrentabilidad", "mrentabilidad_annual",
    "mcomisiones", "mactivos_margen", "mpasivos_margen",
}

# Variables redundantes de ctrx_quarter que dominan el bosque y hacen que los
# clusters se separen solo por "grado de inactividad transaccional". Excluir
# estas variantes fuerza al bosque a mirar otros aspectos (crédito, canales,
# productos) y da clusters con perfiles más distintos entre sí.
# Se conservan: ctrx_quarter (nivel actual), pr_ctrx_quarter (ranking intra-mes)
# y deltapct_1_ctrx_quarter (cambio relativo). Con esas tres alcanza para
# capturar la señal transaccional sin dominar el modelo.
EXCLUIR_REDUNDANTES = {
    "avg_3_ctrx_quarter", "min_3_ctrx_quarter", "max_3_ctrx_quarter",
    "std_3_ctrx_quarter", "d10_ctrx_quarter",
    "ratioavg_3_ctrx_quarter", "slope_3_ctrx_quarter",
    "lag_1_ctrx_quarter", "lag_2_ctrx_quarter",
    "delta_1_ctrx_quarter", "delta_2_ctrx_quarter",
}

# --- Modo limpio (--modo-limpio) ---
# Filtro más agresivo: saca TODAS las derivadas temporales del FE excepto
# deltapct_1_* (que captura cambio brusco en una sola señal interpretable).
# Se conservan: variables base originales + bloque 1 del FE (tc_*, m_*_totales,
# c_*_totales, r_*, f_*, meses_en_panel, antiguedad_panel) + deltapct_1_*.
# Objetivo: forzar al bosque a clusterizar por estructura (qué productos tiene,
# qué relación tiene con el banco) en lugar de por matices temporales.
PREFIJOS_DERIVADAS_TEMPORALES = (
    "lag_",         # lag_1_X, lag_2_X
    "delta_",       # delta_1_X, delta_2_X (deltapct_1_X se excluye por ser 'deltapct_' no 'delta_')
    "avg_3_",       # promedios móviles
    "max_3_",       # máximos móviles
    "min_3_",       # mínimos móviles
    "std_3_",       # desvíos móviles
    "ratioavg_3_",  # cociente vs promedio móvil
    "slope_3_",     # pendientes móviles
    "pr_",          # percentiles intra-mes
    "d10_",         # deciles intra-mes
)
# Excepciones: variables base cuyo nombre coincide accidentalmente con algún
# prefijo. Ninguna de las variables originales del dataset ni del bloque 1
# empieza con estos prefijos, así que no hace falta ninguna excepción por ahora.
EXCEPCIONES_DERIVADAS = set()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def es_binaria(serie: pd.Series) -> bool:
    """True si la serie solo contiene 0/1 (o NaN)."""
    vals = pd.unique(serie.dropna())
    return len(vals) <= 2 and set(vals.astype(float)).issubset({0.0, 1.0})


def a_percentil_global(valor: float, distribucion: pd.Series) -> float:
    """Percentil (0-100) del valor dentro de la distribución dada."""
    d = distribucion.dropna().values
    if len(d) == 0:
        return 50.0
    return 100.0 * np.searchsorted(np.sort(d), valor, side="right") / len(d)


# --------------------------------------------------------------------------- #
# 1. Rentabilidad por cluster
# --------------------------------------------------------------------------- #

def rentabilidad_por_cluster(df: pd.DataFrame, labels: pd.Series) -> tuple:
    """Rentabilidad promedio, anualizada y total del cluster.

    Excluye los últimos 2 meses de cada BAJA+2 (comportamiento distorsionado
    por la baja inminente). Devuelve tabla por cluster + baseline fiel.
    """
    baja2 = df[df[GRUPO_COL] == 1].copy()
    baja2 = baja2.merge(labels.rename("cluster"), left_on=ID_COL, right_index=True)

    baja2 = baja2.sort_values([ID_COL, MES_COL])
    baja2["rank_desde_baja"] = (
        baja2.groupby(ID_COL)[MES_COL].rank(method="dense", ascending=False)
    )
    baja2_sano = baja2[baja2["rank_desde_baja"] > 2]

    tabla = baja2_sano.groupby("cluster").agg(
        n_clientes=(ID_COL, "nunique"),
        rent_mensual_media=("mrentabilidad", "mean"),
        rent_mensual_mediana=("mrentabilidad", "median"),
        comisiones_media=("mcomisiones", "mean"),
        margen_activos_media=("mactivos_margen", "mean"),
        margen_pasivos_media=("mpasivos_margen", "mean"),
    ).reset_index()

    tabla["rent_anual_por_cliente"] = tabla["rent_mensual_media"] * 12
    tabla["rent_anual_cluster_total"] = (
        tabla["rent_anual_por_cliente"] * tabla["n_clientes"]
    )

    fiel = df[df[GRUPO_COL] == 0]
    rent_fiel_anual = fiel["mrentabilidad"].mean() * 12
    tabla["ratio_vs_fiel"] = tabla["rent_anual_por_cliente"] / rent_fiel_anual

    tabla["categoria_valor"] = pd.cut(
        tabla["ratio_vs_fiel"],
        bins=[-np.inf, 0.5, 1.0, 1.5, np.inf],
        labels=["Bajo valor", "Medio-bajo", "Medio-alto", "Alto valor"],
    )

    return (
        tabla.sort_values("rent_anual_cluster_total", ascending=False),
        rent_fiel_anual,
    )


# --------------------------------------------------------------------------- #
# 2. Variables diferenciadoras + LIFT
# --------------------------------------------------------------------------- #

def calcular_lift(cluster_vals: pd.Series, poblacion_vals: pd.Series,
                  binaria: bool, direccion: str) -> float:
    """Lift de perfilado.

    - Binaria:  P(=1 | cluster) / P(=1 | población)
    - Continua: si direccion='MAYOR', P(x > mediana | cluster) / 0.5
                si direccion='MENOR', P(x < mediana | cluster) / 0.5

    Interpretación: 1.0 = como la población, 2.0 = 100% del cluster del lado
    extremo.
    """
    c = cluster_vals.dropna()
    p = poblacion_vals.dropna()
    if len(c) == 0 or len(p) == 0:
        return np.nan

    if binaria:
        p_pob = p.mean()
        if p_pob == 0:
            return np.nan
        return c.mean() / p_pob

    mediana = p.median()
    if direccion == "MAYOR":
        p_c = (c > mediana).mean()
    else:
        p_c = (c < mediana).mean()
    p_pob = 0.5  # por construcción
    return p_c / p_pob


def variables_diferenciadoras(
    df: pd.DataFrame, labels: pd.Series, features: list[str],
    top_n: int = 10, min_effect: float = 0.3,
) -> pd.DataFrame:
    """Para cada cluster, ranking de variables por Cohen's d contra fieles,
    con columna 'lift' de perfilado.
    """
    baja2 = df[df[GRUPO_COL] == 1].copy()
    baja2 = baja2.merge(labels.rename("cluster"), left_on=ID_COL, right_index=True)
    fiel = df[df[GRUPO_COL] == 0]

    fiel_media = fiel[features].mean()
    fiel_var = fiel[features].var()

    # Precomputar qué variables son binarias (una sola vez)
    var_binaria = {v: es_binaria(df[v]) for v in features}

    resultados = []
    for c in sorted(labels.unique()):
        cli = baja2[baja2["cluster"] == c]
        cli_media = cli[features].mean()
        cli_var = cli[features].var()
        sd_pool = np.sqrt((cli_var + fiel_var) / 2).replace(0, np.nan)
        cohen_d = (cli_media - fiel_media) / sd_pool

        for var in features:
            d = cohen_d[var]
            if pd.isna(d) or abs(d) < min_effect:
                continue
            direccion = "MAYOR" if d > 0 else "MENOR"
            lift = calcular_lift(
                cli[var], fiel[var], var_binaria[var], direccion,
            )
            resultados.append({
                "cluster": c,
                "variable": var,
                "media_cluster": cli_media[var],
                "media_fiel": fiel_media[var],
                "diferencia": cli_media[var] - fiel_media[var],
                "cohen_d": d,
                "lift": lift,
                "abs_d": abs(d),
                "direccion": direccion,
                "es_binaria": var_binaria[var],
                "magnitud": (
                    "GRANDE" if abs(d) >= 0.8
                    else "MEDIO" if abs(d) >= 0.5
                    else "CHICO"
                ),
            })

    if not resultados:
        return pd.DataFrame()

    out = pd.DataFrame(resultados)
    out = out.sort_values(["cluster", "abs_d"], ascending=[True, False])
    out = out.groupby("cluster").head(top_n).reset_index(drop=True)
    return out.drop(columns="abs_d")


# --------------------------------------------------------------------------- #
# 3. Triggers candidatos
# --------------------------------------------------------------------------- #

def triggers_candidatos(
    df: pd.DataFrame, labels: pd.Series, diferenciadoras: pd.DataFrame,
    top_por_cluster: int = 3, n_deciles: int = 10,
) -> pd.DataFrame:
    """Para las top-N variables diferenciadoras de cada cluster, tasa de
    BAJA+2 por decil (según la distribución del fiel).
    """
    baja2 = df[df[GRUPO_COL] == 1].copy()
    baja2 = baja2.merge(labels.rename("cluster"), left_on=ID_COL, right_index=True)
    fiel = df[df[GRUPO_COL] == 0]

    resultados = []
    for c in sorted(labels.unique()):
        vars_c = (
            diferenciadoras[diferenciadoras["cluster"] == c]
            .head(top_por_cluster)["variable"].tolist()
        )
        for var in vars_c:
            fiel_vals = fiel[var].dropna()
            if fiel_vals.nunique() < n_deciles:
                continue
            _, bins = pd.qcut(fiel_vals, n_deciles, duplicates="drop", retbins=True)

            baja2_c = baja2[baja2["cluster"] == c]
            b_dec = pd.cut(baja2_c[var], bins=bins, include_lowest=True, labels=False)
            f_dec = pd.cut(fiel[var], bins=bins, include_lowest=True, labels=False)

            for d in range(len(bins) - 1):
                n_b = (b_dec == d).sum()
                n_f = (f_dec == d).sum()
                if n_b + n_f == 0:
                    continue
                tasa = n_b / (n_b + n_f)
                resultados.append({
                    "cluster": c,
                    "variable": var,
                    "decil": d + 1,
                    "rango_desde": bins[d],
                    "rango_hasta": bins[d + 1],
                    "n_baja2": n_b,
                    "n_fiel": n_f,
                    "tasa_baja2": tasa,
                    "es_trigger": tasa >= 0.75,
                })

    return pd.DataFrame(resultados)


# --------------------------------------------------------------------------- #
# 4-5. Radares
# --------------------------------------------------------------------------- #

def calcular_medias_por_grupo(
    df: pd.DataFrame, labels: pd.Series, variables: list[str],
) -> dict[str, pd.Series]:
    """Devuelve dict {nombre_grupo: media_por_variable}.

    Grupos: cluster_1, ..., cluster_K, fieles.
    """
    baja2 = df[df[GRUPO_COL] == 1].copy()
    baja2 = baja2.merge(labels.rename("cluster"), left_on=ID_COL, right_index=True)

    out = {}
    for c in sorted(labels.unique()):
        cli = baja2[baja2["cluster"] == c]
        out[f"cluster_{c}"] = cli[variables].mean()
    fiel = df[df[GRUPO_COL] == 0]
    out["fieles"] = fiel[variables].mean()
    return out


def dibujar_radar(
    ax, medias_por_grupo: dict, variables: list[str], distribuciones: dict,
    cluster_destacado: int | None = None, titulo: str = "",
) -> None:
    """Dibuja un radar en ax.

    - medias_por_grupo: {nombre_grupo: pd.Series con media por variable}
    - variables: lista ordenada de variables (ejes del radar)
    - distribuciones: {variable: pd.Series con distribución global para
      calcular percentiles}
    - cluster_destacado: si se especifica, ese cluster va con línea gruesa y
      relleno; los demás con línea fina y transparencia.
    - fieles siempre gris punteado.
    """
    n = len(variables)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angulos_cerrados = angulos + angulos[:1]

    # Convertir cada media a su percentil global
    def a_percentiles(medias):
        return [
            a_percentil_global(medias[v], distribuciones[v])
            for v in variables
        ]

    # Determinar orden de dibujado: primero los "de fondo", último el destacado
    grupos = list(medias_por_grupo.keys())
    if cluster_destacado is not None:
        nombre_destacado = f"cluster_{cluster_destacado}"
        grupos = [g for g in grupos if g != nombre_destacado] + [nombre_destacado]

    for grupo in grupos:
        percs = a_percentiles(medias_por_grupo[grupo]) + [None]
        # cerrar el polígono
        percs[-1] = percs[0]

        if grupo == "fieles":
            ax.plot(angulos_cerrados, percs, color=INK, linewidth=2.2,
                    linestyle="--", label="CONTINÚA")
        elif grupo == f"cluster_{cluster_destacado}":
            # Cluster destacado: línea gruesa + relleno
            idx = int(grupo.split("_")[1]) - 1
            color = COLORES_CLUSTER[idx]
            ax.plot(angulos_cerrados, percs, color=color, linewidth=3.0,
                    label=grupo, zorder=3)
            ax.fill(angulos_cerrados, percs, color=color, alpha=0.20, zorder=2)
        else:
            # Otros clusters: línea fina, semi-transparente
            idx = int(grupo.split("_")[1]) - 1
            color = COLORES_CLUSTER[idx]
            ax.plot(angulos_cerrados, percs, color=color, linewidth=1.2,
                    alpha=0.55, label=grupo)

    ax.set_xticks(angulos)
    ax.set_xticklabels(
        [v if len(v) <= 22 else v[:20] + ".." for v in variables],
        size=8, color=INK_2,
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=7, color=INK_2)
    ax.tick_params(pad=8)
    ax.set_title(titulo, size=13, color=INK, weight="bold", pad=25)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines["polar"].set_color(GRID)


def radar_de_cluster(
    df: pd.DataFrame, labels: pd.Series, dif: pd.DataFrame,
    cluster: int, out_png: Path, n_vars: int = 5,
) -> list[str]:
    """Genera un radar para un cluster: 5 diferenciadoras top + rentabilidad.

    Devuelve la lista de variables usadas (para reportar).
    """
    dif_c = dif[dif["cluster"] == cluster].copy()
    # Excluir rentabilidad si aparece (aunque ya está filtrada)
    dif_c = dif_c[~dif_c["variable"].isin(EXCLUIR_DIFERENCIADORAS)]
    top_vars = dif_c["variable"].head(n_vars).tolist()
    variables = top_vars + [RENTABILIDAD]

    medias = calcular_medias_por_grupo(df, labels, variables)
    distribuciones = {v: df[v] for v in variables}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")
    dibujar_radar(
        ax, medias, variables, distribuciones,
        cluster_destacado=cluster,
        titulo=f"Cluster {cluster} vs otros clusters vs CONTINÚA",
    )
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.10),
        fontsize=8, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return variables


def radar_del_bosque(
    df: pd.DataFrame, labels: pd.Series, rf, features: list[str],
    out_png: Path, n_vars: int = 6,
) -> list[str]:
    """Radar global con las top-N features por importancia del RF.

    Muestra todos los clusters + fieles con el mismo peso visual.
    """
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    top_vars = imp.head(n_vars).index.tolist()

    medias = calcular_medias_por_grupo(df, labels, top_vars)
    distribuciones = {v: df[v] for v in top_vars}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")
    # Sin cluster destacado: todas las líneas con peso similar
    dibujar_radar(
        ax, medias, top_vars, distribuciones,
        cluster_destacado=None,
        titulo="Variables más importantes del modelo — todos los grupos",
    )
    # Estilo específico para el global: líneas más marcadas para los clusters
    for line in ax.lines:
        lbl = line.get_label()
        if lbl.startswith("cluster_"):
            line.set_linewidth(2.0)
            line.set_alpha(0.85)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.10),
        fontsize=8, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return top_vars


def compilar_pdf_radares(
    out_pdf: Path, out_dir: Path, clusters: list[int],
) -> None:
    """Junta todos los PNGs generados en un solo PDF apaisado."""
    with PdfPages(out_pdf) as pdf:
        # Radar global primero
        for nombre, titulo in (
            ("radar_global.png", "Panorama general"),
            *[(f"radar_cluster_{c}.png", f"Cluster {c}") for c in clusters],
        ):
            img_path = out_dir / nombre
            if not img_path.exists():
                continue
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plt.imread(img_path))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


# --------------------------------------------------------------------------- #
# Reporte de texto
# --------------------------------------------------------------------------- #

def escribir_resumen(
    out_path: Path, tabla_rent: pd.DataFrame, rent_fiel: float,
    diferenciadoras: pd.DataFrame, triggers: pd.DataFrame,
    top_bosque: list[str],
) -> None:
    lineas = []
    lineas.append("=" * 78)
    lineas.append("RENTABILIDAD POR CLUSTER")
    lineas.append("=" * 78)
    lineas.append(
        f"\nCliente fiel promedio: ${rent_fiel:,.0f} anuales de rentabilidad.\n"
    )
    for _, r in tabla_rent.iterrows():
        lineas.append(f"Cluster {int(r['cluster'])}: {int(r['n_clientes']):,} clientes")
        lineas.append(f"  Rent. mensual promedio: ${r['rent_mensual_media']:,.0f}")
        lineas.append(f"  Rent. anual por cliente: ${r['rent_anual_por_cliente']:,.0f}")
        lineas.append(f"  Rent. anual del cluster: ${r['rent_anual_cluster_total']:,.0f}")
        lineas.append(f"  Ratio vs fiel: {r['ratio_vs_fiel']:.2f} ({r['categoria_valor']})")
        lineas.append("")

    lineas.append("=" * 78)
    lineas.append("VARIABLES DIFERENCIADORAS (vs cliente fiel promedio)")
    lineas.append("Cohen's d = diferencia estandarizada  |  Lift = concentración vs población")
    lineas.append("=" * 78)
    for c in sorted(diferenciadoras["cluster"].unique()):
        lineas.append(f"\nCluster {int(c)}:")
        for _, r in diferenciadoras[diferenciadoras["cluster"] == c].head(5).iterrows():
            lift_str = f"lift={r['lift']:.2f}" if pd.notna(r["lift"]) else "lift= n/a"
            lineas.append(
                f"  {r['variable']:40s}  {r['direccion']:5s}  "
                f"d={r['cohen_d']:+.2f} ({r['magnitud']})  {lift_str}"
            )
            lineas.append(
                f"    Cluster: {r['media_cluster']:>12,.2f}  |  "
                f"Fiel: {r['media_fiel']:>12,.2f}"
            )

    lineas.append("\n" + "=" * 78)
    lineas.append("RADAR GLOBAL: variables más importantes según el modelo")
    lineas.append("=" * 78)
    for v in top_bosque:
        lineas.append(f"  - {v}")

    if not triggers.empty:
        lineas.append("\n" + "=" * 78)
        lineas.append("TRIGGERS CANDIDATOS (deciles donde tasa BAJA+2 >= 75%)")
        lineas.append("=" * 78)
        candidatos = triggers[triggers["es_trigger"]].copy()
        if candidatos.empty:
            lineas.append("\n(Ninguna variable muestra un decil con trigger claro.")
            lineas.append(" Ver triggers_candidatos.csv para ver todos los deciles.)")
        else:
            for c in sorted(candidatos["cluster"].unique()):
                lineas.append(f"\nCluster {int(c)}:")
                for _, r in candidatos[candidatos["cluster"] == c].iterrows():
                    lineas.append(
                        f"  {r['variable']}  decil {int(r['decil'])}: "
                        f"[{r['rango_desde']:,.2f} .. {r['rango_hasta']:,.2f}]  "
                        f"→ {r['tasa_baja2']*100:.0f}% son BAJA+2"
                    )

    out_path.write_text("\n".join(lineas), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-samples-leaf", type=int, default=50)
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=214363)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--top-diferenciadoras", type=int, default=10)
    ap.add_argument("--n-vars-radar", type=int, default=5,
                    help="cantidad de variables por radar de cluster (se suma "
                         "rentabilidad, quedan 5+1=6 ejes por default)")
    ap.add_argument("--n-vars-radar-global", type=int, default=6)
    ap.add_argument("--sin-filtrar-redundantes", dest="filtrar_redundantes",
                    action="store_false",
                    help="por default excluye del RF las variantes redundantes "
                         "de ctrx_quarter (avg_3, min_3, max_3, d10, etc.) "
                         "para que el bosque no las use como votos separados "
                         "por la misma señal. Pasar este flag lo desactiva.")
    ap.set_defaults(filtrar_redundantes=True)
    ap.add_argument("--modo-limpio", dest="modo_limpio",
                    action="store_true",
                    help="filtro más agresivo: saca TODAS las derivadas "
                         "temporales del FE (lag, delta, avg_3, max_3, min_3, "
                         "std_3, ratioavg_3, slope_3, pr_, d10_) excepto "
                         "deltapct_1_*. Deja solo variables base + bloque 1 "
                         "del FE. Anula --sin-filtrar-redundantes porque el "
                         "modo limpio ya incluye ese filtro.")
    ap.set_defaults(modo_limpio=False)
    args = ap.parse_args()

    if args.out_dir is None:
        if args.modo_limpio:
            sufijo = "_limpio"
        elif args.filtrar_redundantes:
            sufijo = "_filtrado"
        else:
            sufijo = ""
        args.out_dir = Path(
            f"analisis_k{args.k}_msl{args.min_samples_leaf}{sufijo}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pipeline (RF + KMeans) ---
    log(f"leyendo {args.csv.name}")
    df = cargar_muestra(args.csv, args.seed)
    feats_todas = columnas_features(df)

    # Aplicar filtro de features al pipeline. El modo limpio prevalece sobre
    # el filtro de redundantes: si ambos están activos, se aplica el limpio.
    if args.modo_limpio:
        # Modo limpio: descarta TODAS las derivadas temporales del FE excepto
        # deltapct_1_*. Deja variables base + bloque 1 + deltapct_1_*.
        feats = [
            f for f in feats_todas
            if f in EXCEPCIONES_DERIVADAS
            or f.startswith("deltapct_1_")
            or not any(f.startswith(p) for p in PREFIJOS_DERIVADAS_TEMPORALES)
        ]
        n_excluidas = len(feats_todas) - len(feats)
        log(f"MODO LIMPIO: excluyendo {n_excluidas} derivadas temporales "
            f"(lag, avg_3, max_3, min_3, std_3, ratioavg_3, slope_3, delta_, "
            f"pr_, d10_), conservando deltapct_1_* "
            f"({len(feats)} features finales)")
    elif args.filtrar_redundantes:
        # Filtro liviano: solo variantes de ctrx_quarter
        feats = [f for f in feats_todas if f not in EXCLUIR_REDUNDANTES]
        n_excluidas = len(feats_todas) - len(feats)
        log(f"filtrando {n_excluidas} variantes redundantes de ctrx_quarter "
            f"({len(feats)} features finales)")
    else:
        feats = feats_todas
        log(f"sin filtro de features ({len(feats)} features)")

    feats_analisis = [f for f in feats if f not in EXCLUIR_DIFERENCIADORAS]

    X = df[feats].to_numpy(dtype=np.float32)
    y = df[GRUPO_COL].to_numpy()
    ids = df[ID_COL].to_numpy()

    log(f"entrenando RF (min_samples_leaf={args.min_samples_leaf})")
    rf = entrenar_rf(X, y, args.n_trees, args.min_samples_leaf, args.seed)
    log(f"OOB={rf.oob_score_:.4f}")

    log("contando hojas y clusterizando")
    P = normalizar(contar_hojas_por_id(rf, X, ids, feats))
    ids_churn = pd.Index(df.loc[df[GRUPO_COL] == 1, ID_COL].unique())
    P_churn = P.loc[ids_churn]
    labels = clusterizar(P_churn, args.k, args.seed)

    log("labels: " + ", ".join(
        f"cluster_{c}={n:,}"
        for c, n in labels.value_counts().sort_index().items()
    ))

    # --- Análisis 1: rentabilidad ---
    log("análisis 1: rentabilidad por cluster")
    tabla_rent, rent_fiel = rentabilidad_por_cluster(df, labels)
    tabla_rent.to_csv(args.out_dir / "rentabilidad_por_cluster.csv", index=False)

    # --- Análisis 2: variables diferenciadoras (con lift) ---
    log("análisis 2: variables diferenciadoras (Cohen's d + lift)")
    dif = variables_diferenciadoras(
        df, labels, feats_analisis, top_n=args.top_diferenciadoras,
    )
    dif.to_csv(args.out_dir / "variables_diferenciadoras.csv", index=False)

    # --- Análisis 3: triggers candidatos ---
    log("análisis 3: triggers candidatos por decil")
    trig = triggers_candidatos(df, labels, dif, top_por_cluster=3)
    trig.to_csv(args.out_dir / "triggers_candidatos.csv", index=False)

    # --- Análisis 4: radares por cluster ---
    log("análisis 4: radares por cluster")
    clusters = sorted(labels.unique())
    for c in clusters:
        out_png = args.out_dir / f"radar_cluster_{c}.png"
        vars_usadas = radar_de_cluster(
            df, labels, dif, c, out_png, n_vars=args.n_vars_radar,
        )
        log(f"  cluster_{c}: {out_png.name}")

    # --- Análisis 5: radar global ---
    log("análisis 5: radar global (top features del RF)")
    out_png_global = args.out_dir / "radar_global.png"
    top_bosque = radar_del_bosque(
        df, labels, rf, feats, out_png_global, n_vars=args.n_vars_radar_global,
    )
    log(f"  {out_png_global.name} (variables: {', '.join(top_bosque)})")

    # --- PDF que junta todos los radares ---
    log("compilando radares.pdf")
    compilar_pdf_radares(args.out_dir / "radares.pdf", args.out_dir, clusters)

    # --- Resumen legible ---
    escribir_resumen(
        args.out_dir / "resumen.txt",
        tabla_rent, rent_fiel, dif, trig, top_bosque,
    )

    log(f"listo → {args.out_dir}/")
    log("  rentabilidad_por_cluster.csv")
    log("  variables_diferenciadoras.csv   (con lift)")
    log("  triggers_candidatos.csv")
    log("  resumen.txt   ← empezá por acá")
    log("  radares.pdf   ← todos los radares juntos")
    log("  radar_cluster_*.png / radar_global.png   ← individuales para slides")


if __name__ == "__main__":
    main()
