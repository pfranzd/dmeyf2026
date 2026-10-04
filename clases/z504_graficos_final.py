"""Gráficos del análisis final: radares, tendencias, barras y distribuciones.

Mantiene el estilo visual ya establecido en z501_cluster_rf.py /
z501_analisis_video_pca_sin_rent.py (paleta COLORES_CLUSTER, tipografía,
tamaño de página A4 apaisada) para que estos gráficos convivan con los
radares ya generados en las 26 carpetas `analisis_k*`.

Uso: módulo de importación, sin `main`. Ver z505_analisis_final.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from z501_cluster_rf import _estilo
from z503_base_analisis import (
    CLUSTER_COL,
    COLORES_CLUSTER,
    GRID,
    ID_COL,
    INK,
    INK_2,
    NOMBRES_CLUSTER,
    calcular_medias_por_grupo,
    dibujar_radar,
    etiqueta,
)

FIGSIZE = (11.69, 8.27)  # A4 apaisado, igual que z501_*

# Colores para "por encima / por debajo del fiel" (polaridad de Cohen's d),
# distintos de COLORES_CLUSTER porque no comparten chart con los clusters.
COLOR_MAYOR = "#b23a2e"  # cálido: el cluster está POR ENCIMA del fiel
COLOR_MENOR = "#2a6b8f"  # frío: el cluster está POR DEBAJO del fiel
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "cohen_d", [COLOR_MENOR, "#f2f0ea", COLOR_MAYOR]
)


def _color_cluster(c: int) -> str:
    return COLORES_CLUSTER[c - 1]


def _nombre_cluster(c: int) -> str:
    return f"Cluster {c} — {NOMBRES_CLUSTER.get(c, '')}"


# --------------------------------------------------------------------------- #
# 1. Radares
# --------------------------------------------------------------------------- #


def radar_cluster(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    dif: pd.DataFrame,
    cluster: int,
    out_png: Path,
    n_vars: int = 8,
) -> list[str]:
    """Radar del cluster con sus top-N diferenciadoras vs. el fiel."""
    variables = dif[dif["cluster"] == cluster].head(n_vars)["variable"].tolist()
    medias = calcular_medias_por_grupo(df_combinado, labels, variables)
    distribuciones = {v: df_combinado[v] for v in variables}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")
    etiquetas_ejes = [etiqueta(v) for v in variables]
    dibujar_radar(
        ax,
        medias,
        variables,
        distribuciones,
        cluster_destacado=cluster,
        titulo=_nombre_cluster(cluster),
    )
    ax.set_xticklabels(
        [e if len(e) <= 24 else e[:22] + ".." for e in etiquetas_ejes],
        size=8,
        color=INK_2,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return variables


def radar_dominio(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    dominio_nombre: str,
    variables: list[str],
    out_png: Path,
) -> None:
    """Un radar por dominio de negocio, con los MISMOS ejes para los 3
    clusters (+ fiel de referencia): a diferencia de radar_cluster (que usa
    las top variables de CADA cluster, ejes distintos entre sí), este es
    comparable entre clusters porque los ejes son fijos."""
    medias = calcular_medias_por_grupo(df_combinado, labels, variables)
    distribuciones = {v: df_combinado[v] for v in variables}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")
    dibujar_radar(
        ax,
        medias,
        variables,
        distribuciones,
        cluster_destacado=None,
        titulo=f"{dominio_nombre} — los 3 clusters (percentil vs. fiel)",
    )
    ax.set_xticklabels(
        [
            etiqueta(v) if len(etiqueta(v)) <= 24 else etiqueta(v)[:22] + ".."
            for v in variables
        ],
        size=8,
        color=INK_2,
    )
    for line in ax.lines:
        if line.get_label().startswith("cluster_"):
            line.set_linewidth(2.0)
            line.set_alpha(0.85)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def radar_panorama(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    variables: list[str],
    out_png: Path,
) -> None:
    """Panorama general: variables con mayor |Cohen's d| en cualquier
    cluster, todos los grupos con peso visual similar. Reemplaza el "radar
    global de importancias del RF" del pipeline de exploración: acá no se
    reentrena el Random Forest (fuera de alcance), así que el ranking sale
    de las diferenciadoras ya calculadas."""
    medias = calcular_medias_por_grupo(df_combinado, labels, variables)
    distribuciones = {v: df_combinado[v] for v in variables}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")
    dibujar_radar(
        ax,
        medias,
        variables,
        distribuciones,
        cluster_destacado=None,
        titulo="Panorama general — variables más diferenciadoras (los 3 clusters)",
    )
    ax.set_xticklabels(
        [
            etiqueta(v) if len(etiqueta(v)) <= 24 else etiqueta(v)[:22] + ".."
            for v in variables
        ],
        size=8,
        color=INK_2,
    )
    for line in ax.lines:
        if line.get_label().startswith("cluster_"):
            line.set_linewidth(2.0)
            line.set_alpha(0.85)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Tendencias (alineadas a mes_relativo, ancladas al primer mes de
#    ausencia del cliente; ver z503_base_analisis.agregar_mes_relativo)
# --------------------------------------------------------------------------- #


def linea_tendencia(
    tend: pd.DataFrame,
    variable: str,
    rango: list[int],
    media_fiel: float,
    out_png: Path,
) -> None:
    """Una variable, los 3 clusters alineados al evento + referencia fiel
    (línea horizontal punteada, porque el fiel no tiene mes_relativo).
    Incluye la tabla de `n` al pie: la profundidad decrece hacia t negativo
    porque no todos los clientes tienen tanta historia previa a la baja."""
    clusters = sorted(tend.index.get_level_values(CLUSTER_COL).unique())
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.08, 0.30, 0.74, 0.52])
    x = np.arange(len(rango))
    for c in clusters:
        s = tend.xs(c, level=CLUSTER_COL)
        centro = s["centro"][variable].reindex(rango).values
        lo = s["lo"][variable].reindex(rango).values
        hi = s["hi"][variable].reindex(rango).values
        color = _color_cluster(c)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(
            x,
            centro,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6,
            label=f"cluster_{c}",
        )
    ax.axhline(
        media_fiel, color=INK, linewidth=2.0, linestyle="--", label="CONTINÚA (fiel)"
    )
    ax.set_xticks(x, [f"t{r:+d}" for r in rango])
    ax.set_xlim(-0.3, len(rango) - 0.7)
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylabel(etiqueta(variable), color=INK_2)

    fig.text(0.08, 0.945, etiqueta(variable), fontsize=20, color=INK, weight="bold")
    fig.text(
        0.08,
        0.925,
        "t = meses relativos al primer mes en que el cliente deja de aparecer en el "
        "panel (t=0, sin dato, no se grafica). t=-1 es su último mes real (BAJA+1), "
        "t=-2 la fila BAJA+2. Línea = media del cluster presente en ese mes relativo; "
        "banda = IC 95%; referencia = media de los clientes fieles.",
        fontsize=10,
        color=INK_2,
        va="top",
        linespacing=1.4,
    )

    n_tab = (
        tend["n"][variable]
        .unstack("mes_relativo")
        .reindex(index=clusters, columns=rango)
        .fillna(0)
        .astype(int)
    )
    ax_t = fig.add_axes([0.08, 0.06, 0.88, 0.16])
    ax_t.axis("off")
    tabla = ax_t.table(
        cellText=n_tab.values,
        rowLabels=[f"cluster_{c}" for c in clusters],
        colLabels=[f"t{r:+d}" for r in rango],
        loc="center",
        cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1, 1.15)
    for (r, _c), cell in tabla.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK_2 if r == 0 else INK)
    ax_t.set_title(
        "n de clientes del cluster presentes en cada mes relativo",
        loc="left",
        fontsize=9,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def panel_dominio(
    tend: pd.DataFrame,
    dominio_nombre: str,
    variables: list[str],
    rango: list[int],
    medias_fieles: pd.Series,
    out_png: Path,
) -> None:
    """Small multiples: todas las variables de un dominio en una sola
    figura, para escanear el dominio completo de un vistazo."""
    clusters = sorted(tend.index.get_level_values(CLUSTER_COL).unique())
    n = len(variables)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE)
    axes = np.atleast_2d(axes)
    x = np.arange(len(rango))
    for i, var in enumerate(variables):
        ax = axes[i // ncols, i % ncols]
        for c in clusters:
            s = tend.xs(c, level=CLUSTER_COL)
            centro = s["centro"][var].reindex(rango).values
            ax.plot(
                x,
                centro,
                color=_color_cluster(c),
                linewidth=1.8,
                marker="o",
                markersize=4,
                label=f"cluster_{c}",
            )
        ax.axhline(medias_fieles[var], color=INK, linewidth=1.4, linestyle="--")
        ax.set_xticks(x, [f"{r:+d}" for r in rango], fontsize=7)
        _estilo(ax)
        ax.set_title(etiqueta(var), fontsize=9, color=INK, loc="left")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", frameon=False, fontsize=9, ncol=1)
    fig.suptitle(
        f"{dominio_nombre} — tendencia por mes relativo a la baja",
        fontsize=15,
        color=INK,
        weight="bold",
        x=0.06,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Barras
# --------------------------------------------------------------------------- #


def barras_cohen_d(dif: pd.DataFrame, cluster: int, out_png: Path, n: int = 10) -> None:
    """Barras horizontales de Cohen's d (top-N |d|) para un cluster, vs. el
    fiel. Color = polaridad (por encima / por debajo del fiel), no el color
    del cluster: acá el eje relevante es la dirección, no la identidad."""
    d = dif[dif["cluster"] == cluster].head(n).iloc[::-1]
    colores = [COLOR_MAYOR if v > 0 else COLOR_MENOR for v in d["cohen_d"]]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.32, 0.10, 0.62, 0.78])
    y = np.arange(len(d))
    ax.barh(y, d["cohen_d"], color=colores, height=0.6)
    ax.set_yticks(y, [etiqueta(v) for v in d["variable"]], fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1.0)
    _estilo(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, d["cohen_d"]):
        ax.text(
            v + (0.03 if v > 0 else -0.03),
            yi,
            f"{v:+.2f}",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize=8,
            color=INK_2,
        )
    fig.text(
        0.06,
        0.945,
        f"{_nombre_cluster(cluster)} — variables diferenciadoras",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "Cohen's d vs. cliente fiel (rojo = el cluster está por encima; azul = por debajo).",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def heatmap_cohen_d(dif_todos: pd.DataFrame, out_png: Path, max_vars: int = 25) -> None:
    """variables x cluster, color = Cohen's d (diverging, centrado en 0)."""
    variables = (
        dif_todos.assign(abs_d=dif_todos["cohen_d"].abs())
        .sort_values("abs_d", ascending=False)["variable"]
        .drop_duplicates()
        .head(max_vars)
        .tolist()
    )
    clusters = sorted(dif_todos["cluster"].unique())
    pivot = (
        dif_todos[dif_todos["variable"].isin(variables)]
        .pivot_table(index="variable", columns="cluster", values="cohen_d")
        .reindex(index=variables, columns=clusters)
    )
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.32, 0.08, 0.40, 0.80])
    vmax = float(np.nanmax(np.abs(pivot.values)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(pivot.values, cmap=CMAP_DIVERGENTE, norm=norm, aspect="auto")
    ax.set_yticks(
        range(len(variables)), [etiqueta(v) for v in variables], fontsize=8.5, color=INK
    )
    ax.set_xticks(
        range(len(clusters)), [f"Cluster {c}" for c in clusters], fontsize=9, color=INK
    )
    for i in range(len(variables)):
        for j in range(len(clusters)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=INK if abs(val) < vmax * 0.6 else "white",
                )
    ax.spines[:].set_visible(False)
    cax = fig.add_axes([0.75, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cax, label="Cohen's d vs. fiel")
    fig.text(
        0.06,
        0.94,
        "Mapa de calor — variables diferenciadoras por cluster",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.905,
        "Rojo = el cluster está por encima del fiel; azul = por debajo.",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def barras_grupo_vs_fiel(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    variables: list[str],
    titulo: str,
    subtitulo: str,
    out_png: Path,
) -> None:
    """Barras agrupadas: promedio de cada variable por cluster + fiel.
    Sirve tanto para tenencia de producto (conteos) como para flags de
    riesgo (proporciones 0-1) — ambos son "promedio de una variable simple
    por grupo", solo cambia la escala del eje Y entre llamadas (nunca se
    mezclan dos escalas en el mismo eje)."""
    medias = calcular_medias_por_grupo(df_combinado, labels, variables)
    grupos = [f"cluster_{c}" for c in sorted(labels.unique())] + ["fieles"]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.08, 0.30, 0.88, 0.52])
    n_g = len(grupos)
    ancho = 0.8 / n_g
    x = np.arange(len(variables))
    for i, g in enumerate(grupos):
        color = INK if g == "fieles" else _color_cluster(int(g.split("_")[1]))
        offset = (i - (n_g - 1) / 2) * ancho
        ax.bar(
            x + offset,
            medias[g][variables].values,
            width=ancho * 0.92,
            color=color,
            alpha=1.0 if g != "fieles" else 0.75,
            label="CONTINÚA (fiel)" if g == "fieles" else g,
        )
    ax.set_xticks(
        x, [etiqueta(v) for v in variables], rotation=25, ha="right", fontsize=9
    )
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(0.06, 0.945, titulo, fontsize=18, color=INK, weight="bold")
    fig.text(0.06, 0.905, subtitulo, fontsize=10, color=INK_2)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Valor
# --------------------------------------------------------------------------- #


def barras_valor_anual(
    tabla_rent: pd.DataFrame, rent_fiel: float, out_png: Path
) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.10, 0.30, 0.85, 0.52])
    t = tabla_rent.sort_values("cluster")
    x = np.arange(len(t))
    colores = [_color_cluster(int(c)) for c in t["cluster"]]
    ax.bar(x, t["rent_anual_por_cliente"], color=colores, width=0.55)
    ax.axhline(
        rent_fiel, color=INK, linewidth=2.0, linestyle="--", label="CONTINÚA (fiel)"
    )
    for xi, v in zip(x, t["rent_anual_por_cliente"]):
        ax.text(xi, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(
        x,
        [f"Cluster {int(c)}\n{NOMBRES_CLUSTER.get(int(c), '')}" for c in t["cluster"]],
        fontsize=9,
    )
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylabel("$ anuales por cliente", color=INK_2)
    fig.text(
        0.06,
        0.945,
        "Rentabilidad anual por cliente, por cluster",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.905,
        f"Referencia: cliente fiel promedio = ${rent_fiel:,.0f}/año "
        f"(población fiel completa, no muestra).",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def descomposicion_margen(tabla_rent: pd.DataFrame, out_png: Path) -> None:
    """Activo (crédito) vs. pasivo (depósitos) vs. comisiones: de dónde
    viene la rentabilidad de cada cluster."""
    t = tabla_rent.sort_values("cluster")
    componentes = [
        ("margen_activos_media", "Margen activos (crédito)"),
        ("margen_pasivos_media", "Margen pasivos (depósitos)"),
        ("comisiones_media", "Comisiones"),
    ]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.10, 0.30, 0.85, 0.52])
    x = np.arange(len(t))
    ancho = 0.8 / len(componentes)
    tonos = [
        "#14181a",
        "#6a7a72",
        "#b7bdb4",
    ]  # INK -> gris, para 3 componentes de UN concepto
    for i, (col, etiqueta_comp) in enumerate(componentes):
        offset = (i - (len(componentes) - 1) / 2) * ancho
        ax.bar(
            x + offset, t[col], width=ancho * 0.92, color=tonos[i], label=etiqueta_comp
        )
    ax.axhline(0, color=INK_2, linewidth=1.0)
    ax.set_xticks(
        x,
        [f"Cluster {int(c)}\n{NOMBRES_CLUSTER.get(int(c), '')}" for c in t["cluster"]],
        fontsize=9,
    )
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_ylabel("$ mensuales promedio", color=INK_2)
    fig.text(
        0.06,
        0.945,
        "Composición del margen mensual, por cluster",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.905,
        "De dónde sale la rentabilidad de cada cluster: crédito, depósitos o comisiones.",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def barras_valor_en_riesgo(tabla_rent: pd.DataFrame, out_png: Path) -> None:
    t = tabla_rent.sort_values("rent_anual_cluster_total", ascending=False)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.28, 0.12, 0.66, 0.76])
    y = np.arange(len(t))
    colores = [_color_cluster(int(c)) for c in t["cluster"]]
    ax.barh(y, t["rent_anual_cluster_total"] / 1e6, color=colores, height=0.55)
    ax.set_yticks(
        y,
        [
            f"Cluster {int(c)} — {NOMBRES_CLUSTER.get(int(c), '')}\n"
            f"({int(n):,} clientes)"
            for c, n in zip(t["cluster"], t["n_clientes"])
        ],
        fontsize=9,
    )
    for yi, v in zip(y, t["rent_anual_cluster_total"] / 1e6):
        ax.text(v, yi, f"  ${v:,.1f}M/año", va="center", fontsize=9.5, color=INK)
    _estilo(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rentabilidad anual total del cluster ($ millones)", color=INK_2)
    fig.text(
        0.06,
        0.945,
        "Rentabilidad anual en riesgo, por cluster",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.905,
        "Lo que se pierde por año si se van todos los clientes del cluster.",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def dispersion_valor_actividad(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    out_png: Path,
    muestra_por_grupo: int = 600,
) -> None:
    """Un punto por cliente: actividad transaccional vs. rentabilidad
    mensual. Muestra para no saturar (>=600 por grupo alcanza para ver la
    forma de la nube sin ploteos de 100k+ puntos)."""
    baja2 = df_combinado[df_combinado["grupo"] == 1].merge(
        labels.rename(CLUSTER_COL), left_on=ID_COL, right_index=True
    )
    fiel = df_combinado[df_combinado["grupo"] == 0]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.09, 0.12, 0.68, 0.76])
    muestra_fiel = fiel.sample(min(muestra_por_grupo, len(fiel)), random_state=214363)
    ax.scatter(
        muestra_fiel["ctrx_quarter"],
        muestra_fiel["mrentabilidad"],
        s=10,
        color=INK,
        alpha=0.25,
        label="CONTINÚA (fiel)",
    )
    for c in sorted(labels.unique()):
        sub = baja2[baja2[CLUSTER_COL] == c]
        m = (
            sub.sample(min(muestra_por_grupo, len(sub)), random_state=214363)
            if len(sub)
            else sub
        )
        ax.scatter(
            m["ctrx_quarter"],
            m["mrentabilidad"],
            s=14,
            color=_color_cluster(c),
            alpha=0.55,
            label=f"cluster_{c}",
        )
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.30, 1.0))
    ax.set_xlabel(etiqueta("ctrx_quarter"), color=INK_2)
    ax.set_ylabel(etiqueta("mrentabilidad"), color=INK_2)
    fig.text(
        0.06,
        0.945,
        "Actividad transaccional vs. rentabilidad, por cliente",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.905,
        f"Muestra de hasta {muestra_por_grupo} clientes por grupo.",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Distribuciones y composición
# --------------------------------------------------------------------------- #


def caja_variable(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    variable: str,
    out_png: Path,
) -> None:
    """Boxplot por grupo (3 clusters + fiel) para una variable."""
    baja2 = df_combinado[df_combinado["grupo"] == 1].merge(
        labels.rename(CLUSTER_COL), left_on=ID_COL, right_index=True
    )
    fiel = df_combinado[df_combinado["grupo"] == 0]
    clusters = sorted(labels.unique())
    datos = [fiel[variable].dropna().values] + [
        baja2.loc[baja2[CLUSTER_COL] == c, variable].dropna().values for c in clusters
    ]
    etiquetas_x = ["CONTINÚA\n(fiel)"] + [f"Cluster {c}" for c in clusters]
    colores = [INK] + [_color_cluster(c) for c in clusters]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.68])
    bp = ax.boxplot(
        datos,
        tick_labels=etiquetas_x,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.8},
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for whisker in bp["whiskers"]:
        whisker.set_color(INK_2)
    for cap in bp["caps"]:
        cap.set_color(INK_2)
    _estilo(ax)
    ax.set_ylabel(etiqueta(variable), color=INK_2)
    fig.text(
        0.06,
        0.945,
        f"{etiqueta(variable)} — distribución por grupo",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.905,
        "Caja = Q25-Q75, línea = mediana. Se recortan valores extremos "
        "(bigotes) para legibilidad.",
        fontsize=10,
        color=INK_2,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def distribucion_demografia(
    df_combinado: pd.DataFrame,
    labels: pd.Series,
    out_png: Path,
) -> None:
    baja2 = df_combinado[df_combinado["grupo"] == 1].merge(
        labels.rename(CLUSTER_COL), left_on=ID_COL, right_index=True
    )
    fiel = df_combinado[df_combinado["grupo"] == 0]
    clusters = sorted(labels.unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    for var, ax in [("cliente_edad", ax1), ("cliente_antiguedad", ax2)]:
        bins = np.linspace(
            df_combinado[var].quantile(0.01), df_combinado[var].quantile(0.99), 30
        )
        ax.hist(
            fiel[var].dropna(),
            bins=bins,
            density=True,
            color=INK,
            alpha=0.25,
            label="CONTINÚA (fiel)",
        )
        for c in clusters:
            ax.hist(
                baja2.loc[baja2[CLUSTER_COL] == c, var].dropna(),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=_color_cluster(c),
                label=f"cluster_{c}",
            )
        _estilo(ax)
        ax.set_xlabel(etiqueta(var), color=INK_2)
        ax.set_ylabel("densidad", color=INK_2)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Demografía por grupo", fontsize=16, color=INK, weight="bold", x=0.06, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def mix_clusters_por_mes(df_churn: pd.DataFrame, out_png: Path) -> None:
    """Composición de clusters por mes CALENDARIO de la baja (no mes
    relativo): permite ver si el mix cambia de mes a mes (posible efecto de
    cohorte/panel) o es estable."""
    from z503_base_analisis import MES_COL, TARGET_COL

    baja2 = df_churn[df_churn[TARGET_COL] == "BAJA+2"]
    tab = baja2.groupby([MES_COL, CLUSTER_COL]).size().unstack(CLUSTER_COL).fillna(0)
    prop = tab.div(tab.sum(axis=1), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    x = np.arange(len(tab.index))
    bottom = np.zeros(len(tab.index))
    for c in tab.columns:
        ax1.bar(
            x,
            tab[c].values,
            bottom=bottom,
            color=_color_cluster(int(c)),
            label=f"cluster_{c}",
            width=0.6,
        )
        bottom += tab[c].values
    ax1.set_xticks(x, [str(m) for m in tab.index], fontsize=9)
    _estilo(ax1)
    ax1.set_ylabel("clientes BAJA+2", color=INK_2)
    ax1.set_title("Volumen por mes de baja", fontsize=11, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8)

    bottom = np.zeros(len(prop.index))
    for c in prop.columns:
        ax2.bar(
            x, prop[c].values, bottom=bottom, color=_color_cluster(int(c)), width=0.6
        )
        bottom += prop[c].values
    ax2.set_xticks(x, [str(m) for m in prop.index], fontsize=9)
    _estilo(ax2)
    ax2.set_ylabel("proporción", color=INK_2)
    ax2.set_title("Mix (%) por mes de baja", fontsize=11, color=INK, loc="left")

    fig.suptitle(
        "Composición de clusters por mes calendario de la baja",
        fontsize=16,
        color=INK,
        weight="bold",
        x=0.06,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Compilación de PDFs
# --------------------------------------------------------------------------- #


def pagina_texto(pdf: PdfPages, titulo: str, lineas: list[str]) -> None:
    """Página de texto plano (portada / separador de sección)."""
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.07, 0.90, titulo, fontsize=22, color=INK, weight="bold", va="top")
    fig.text(
        0.07,
        0.80,
        "\n".join(lineas),
        fontsize=11,
        color=INK_2,
        va="top",
        linespacing=1.7,
    )
    pdf.savefig(fig)
    plt.close(fig)


def compilar_pdf(paths: list[Path], out_pdf: Path) -> None:
    """Junta una lista de PNGs (en el orden dado) en un PDF apaisado, una
    imagen por página. Rutas inexistentes se saltean en silencio (permite
    armar la lista de forma declarativa sin chequear cada archivo antes)."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for img_path in paths:
            img_path = Path(img_path)
            if not img_path.exists():
                continue
            fig = plt.figure(figsize=FIGSIZE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plt.imread(img_path))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
