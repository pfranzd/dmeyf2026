"""Runner del análisis final de clusters: tablas, gráficos y hallazgos.txt.

No vuelve a correr el clustering (usa clusters_marcados.csv tal cual). Cruza
contra la población fiel completa (no la muestra 1:1 del pipeline de
exploración) y arma todo el material para la presentación en
clases/analisis_clustering/analisis_final/.

Uso:
    python z505_analisis_final.py                 # corre todo
    python z505_analisis_final.py --solo tablas    # solo tablas (rápido, sin gráficos)
    python z505_analisis_final.py --solo graficos  # gráficos a partir de tablas ya guardadas
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from z503_base_analisis import (
    CLUSTER_COL,
    DIR_BARRAS,
    DIR_DISTRIBUCIONES,
    DIR_RADARES,
    DIR_TENDENCIAS,
    DIR_VALOR,
    DOMINIOS,
    EXCLUIR_DIFERENCIADORAS,
    FEATURES_CURADAS,
    ID_COL,
    MES_COL,
    NOMBRES_CLUSTER,
    OUT_DIR,
    TARGET_COL,
    agregar_mes_relativo,
    calcular_medias_por_grupo,
    cargar_churners,
    cargar_fieles_curadas,
    cohen_d_completo,
    columnas_numericas_completas,
    conectar_fe,
    diferenciadoras_entre_clusters,
    etiqueta,
    etiquetas_por_cliente,
    log,
    regla_alerta,
    rent_fiel_anual,
    rentabilidad_por_cluster,
    senales_anticipacion,
    tasa_base_baja2,
    tendencias_por_mes_relativo,
    variables_binarias,
    variables_diferenciadoras,
)
from z504_graficos_final import (
    FIGSIZE,
    barras_cohen_d,
    barras_grupo_vs_fiel,
    barras_valor_anual,
    barras_valor_en_riesgo,
    caja_variable,
    descomposicion_margen,
    dispersion_valor_actividad,
    distribucion_demografia,
    heatmap_cohen_d,
    linea_tendencia,
    mix_clusters_por_mes,
    pagina_texto,
    panel_dominio,
    radar_cluster,
    radar_dominio,
    radar_panorama,
)

from dmeyf.metrics import COSTO_ESTIMULO

# mes_relativo está anclado al primer mes de AUSENCIA del cliente (ver
# agregar_mes_relativo en z503_base_analisis): t=-1 es su último mes real
# (fila BAJA+1), t=-2 la fila BAJA+2, t=-3 un mes CONTINUA anterior. No hace
# falta excluir nada: con ese ancla ninguna fila real cae en t>=0.
RANGO_MESES = [-3, -2, -1]

# Variables "siempre mostrar" en tendencias, además de las top diferenciadoras
# de cada cluster: dan el panorama de negocio aunque no encabecen el ranking.
CORE_TEMPORAL = [
    "ctrx_quarter",
    "mrentabilidad",
    "cproductos",
    "mcuentas_saldo",
    "mprestamos_personales",
    "ctarjeta_visa",
    "ctarjeta_master",
    "thomebanking",
    "cliente_antiguedad",
    "cpayroll_trx",
    "mpayroll",
    "f_sin_payroll",
]

# Grupos de variables para barras agrupadas, separados por orden de magnitud
# (nunca se mezcla una escala 0-1 con una de conteos grandes en el mismo eje).
GRUPO_TENENCIA_CHICA = [
    "ctarjeta_visa",
    "ctarjeta_master",
    "cplazo_fijo",
    "cinversion1",
    "cseguro_vida",
    "thomebanking",
    "tmobile_app",
    "internet",
]
GRUPO_CONTEOS = [
    "cproductos",
    "c_productos_contratados",
    "cprestamos_personales",
    "c_seguros_totales",
]
GRUPO_RIESGO = [
    "f_mora",
    "f_delinquency",
    "f_saldo_negativo",
    "f_sobregiro",
    "f_cheque_rechazado",
]

# Variables continuas (no zero-inflated) para boxplot.
VARS_CAJA = ["mrentabilidad", "mcuentas_saldo", "ctrx_quarter", "cliente_antiguedad"]


def _fmt_monto(x: float) -> str:
    return f"${x:,.0f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def calcular_tablas() -> dict:
    """Toda la aritmética: un solo lugar, sin gráficos. Devuelve un dict con
    todas las tablas ya calculadas, que tanto los gráficos como
    hallazgos.txt consumen."""
    t0 = time.time()
    log("cargando churners + FE")
    df = agregar_mes_relativo(cargar_churners())
    labels = etiquetas_por_cliente(df)
    n_por_cluster = labels.value_counts().sort_index()

    con = conectar_fe()
    rent_fiel = rent_fiel_anual(con)
    tasa_base = tasa_base_baja2(con)
    n_fieles = con.execute("SELECT COUNT(*) FROM fieles_ids").fetchone()[0]
    log(
        f"[{time.time() - t0:.1f}s] fiel: {n_fieles:,} clientes, ${rent_fiel:,.0f}/año, "
        f"tasa base BAJA+2 = {tasa_base * 100:.2f}%"
    )

    log("trayendo variables curadas de la población fiel")
    fieles = cargar_fieles_curadas(con, FEATURES_CURADAS)
    cols_combinado = [ID_COL, MES_COL, TARGET_COL, "grupo"] + FEATURES_CURADAS
    combinado = pd.concat(
        [df.assign(grupo=1)[cols_combinado], fieles], ignore_index=True
    )
    log(f"[{time.time() - t0:.1f}s] combinado: {combinado.shape}")

    feats_analisis = [f for f in FEATURES_CURADAS if f not in EXCLUIR_DIFERENCIADORAS]
    vb = variables_binarias(con, df, feats_analisis)

    log("diferenciadoras vs. fieles")
    dif_vs_fieles = variables_diferenciadoras(
        combinado, labels, feats_analisis, top_n=15
    )

    log("diferenciadoras entre clusters")
    dif_vs_otros = diferenciadoras_entre_clusters(
        df, labels, feats_analisis, vb, top_n=15
    )

    log("rentabilidad por cluster")
    tabla_rent, rent_fiel_sano = rentabilidad_por_cluster(combinado, labels)

    log("perfil por cluster (medias de las variables curadas)")
    medias = calcular_medias_por_grupo(combinado, labels, FEATURES_CURADAS)
    perfil = pd.DataFrame(medias).T
    perfil.insert(
        0,
        "n_clientes",
        [
            int(n_por_cluster.get(int(g.split("_")[1]), 0))
            if g != "fieles"
            else n_fieles
            for g in perfil.index
        ],
    )
    perfil.index.name = "grupo"

    log("tendencias por mes relativo (variables temporales)")
    top_vars_dif = (
        dif_vs_fieles.assign(abs_d=dif_vs_fieles["cohen_d"].abs())
        .sort_values(["cluster", "abs_d"], ascending=[True, False])
        .groupby("cluster")
        .head(6)["variable"]
        .tolist()
    )
    temporal_vars = list(
        dict.fromkeys(top_vars_dif + CORE_TEMPORAL)
    )  # dedup, conserva orden
    tend = tendencias_por_mes_relativo(df, temporal_vars, RANGO_MESES)

    log("señales de anticipación")
    sd_ref = fieles[temporal_vars].std()
    senales = senales_anticipacion(tend, temporal_vars, sd_ref=sd_ref)

    log("reglas de alerta (denominador poblacional real)")
    candidatas = (
        dif_vs_fieles.assign(abs_d=dif_vs_fieles["cohen_d"].abs())
        .sort_values(["cluster", "abs_d"], ascending=[True, False])
        .groupby("cluster")
        .head(3)[["cluster", "variable", "direccion"]]
    )
    filas_reglas = []
    for _, r in candidatas.drop_duplicates(["variable", "direccion"]).iterrows():
        try:
            res = regla_alerta(con, r["variable"], r["direccion"], tasa_base)
        except Exception as e:  # noqa: BLE001 — best-effort: se saltea, no se aborta el barrido
            log(f"  regla_alerta({r['variable']}) falló: {e}")
            continue
        res["cluster"] = int(r["cluster"])
        filas_reglas.append(res)
    reglas = pd.DataFrame(filas_reglas)

    log("ranking completo (633 variables, apéndice)")
    feats_completos = columnas_numericas_completas(df)
    dif_completo = cohen_d_completo(con, df, labels, feats_completos, min_effect=0.3)

    log("mix de clusters por mes calendario de la baja")
    mix = (
        df[df[TARGET_COL] == "BAJA+2"]
        .groupby([MES_COL, CLUSTER_COL])
        .size()
        .unstack(CLUSTER_COL)
        .fillna(0)
        .astype(int)
    )
    mix_prop = mix.div(mix.sum(axis=1), axis=0)

    log(f"[{time.time() - t0:.1f}s] tablas listas")
    return {
        "df": df,
        "labels": labels,
        "con": con,
        "combinado": combinado,
        "rent_fiel": rent_fiel,
        "rent_fiel_sano": rent_fiel_sano,
        "tasa_base": tasa_base,
        "n_fieles": n_fieles,
        "n_por_cluster": n_por_cluster,
        "feats_analisis": feats_analisis,
        "dif_vs_fieles": dif_vs_fieles,
        "dif_vs_otros": dif_vs_otros,
        "tabla_rent": tabla_rent,
        "perfil": perfil,
        "temporal_vars": temporal_vars,
        "tend": tend,
        "senales": senales,
        "reglas": reglas,
        "dif_completo": dif_completo,
        "mix": mix,
        "mix_prop": mix_prop,
        "fieles": fieles,
    }


def guardar_tablas(t: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t["perfil"].to_csv(OUT_DIR / "perfil_clusters.csv")
    t["dif_vs_fieles"].to_csv(OUT_DIR / "diferenciadoras_vs_fieles.csv", index=False)
    t["dif_vs_otros"].to_csv(
        OUT_DIR / "diferenciadoras_vs_otros_clusters.csv", index=False
    )
    t["tabla_rent"].to_csv(OUT_DIR / "valor_por_cluster.csv", index=False)
    t["tend"].to_csv(OUT_DIR / "tendencias_mes_relativo.csv")
    t["senales"].to_csv(OUT_DIR / "senales_anticipacion.csv", index=False)
    t["reglas"].to_csv(OUT_DIR / "reglas_alerta.csv", index=False)
    t["dif_completo"].to_csv(
        OUT_DIR / "apendice_diferenciadoras_completo.csv", index=False
    )
    log(f"tablas guardadas en {OUT_DIR}")


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #


def generar_graficos(t: dict) -> None:
    for d in (DIR_RADARES, DIR_TENDENCIAS, DIR_BARRAS, DIR_VALOR, DIR_DISTRIBUCIONES):
        d.mkdir(parents=True, exist_ok=True)

    combinado, labels, df = t["combinado"], t["labels"], t["df"]
    dif_vs_fieles, tend = t["dif_vs_fieles"], t["tend"]
    clusters = sorted(labels.unique())

    log("radares")
    for c in clusters:
        radar_cluster(
            combinado, labels, dif_vs_fieles, c, DIR_RADARES / f"radar_cluster_{c}.png"
        )
    for dominio, variables in DOMINIOS.items():
        vars_cap = [v for v, _, _ in variables][:8]
        slug = dominio.lower().replace(" ", "_").replace("ó", "o").replace("é", "e")
        radar_dominio(
            combinado,
            labels,
            dominio,
            vars_cap,
            DIR_RADARES / f"radar_dominio_{slug}.png",
        )
    top_panorama = (
        dif_vs_fieles.assign(abs_d=dif_vs_fieles["cohen_d"].abs())
        .sort_values("abs_d", ascending=False)["variable"]
        .drop_duplicates()
        .head(8)
        .tolist()
    )
    radar_panorama(combinado, labels, top_panorama, DIR_RADARES / "radar_panorama.png")

    log("tendencias")
    medias_fieles = t["fieles"][t["temporal_vars"]].mean()
    for var in t["temporal_vars"]:
        linea_tendencia(
            tend,
            var,
            RANGO_MESES,
            medias_fieles[var],
            DIR_TENDENCIAS / f"tendencia_{var}.png",
        )
    for dominio, variables in DOMINIOS.items():
        vars_cap = [v for v, _, _ in variables][:9]
        tend_dom = tendencias_por_mes_relativo(df, vars_cap, RANGO_MESES)
        medias_dom = t["fieles"][vars_cap].mean()
        slug = dominio.lower().replace(" ", "_").replace("ó", "o").replace("é", "e")
        panel_dominio(
            tend_dom,
            dominio,
            vars_cap,
            RANGO_MESES,
            medias_dom,
            DIR_TENDENCIAS / f"panel_{slug}.png",
        )

    log("barras")
    for c in clusters:
        barras_cohen_d(dif_vs_fieles, c, DIR_BARRAS / f"cohen_d_cluster_{c}.png")
    heatmap_cohen_d(dif_vs_fieles, DIR_BARRAS / "heatmap_cohen_d.png")
    barras_grupo_vs_fiel(
        combinado,
        labels,
        GRUPO_TENENCIA_CHICA,
        "Tenencia de producto (canales y tarjetas)",
        "Promedio por grupo.",
        DIR_BARRAS / "tenencia_chica.png",
    )
    barras_grupo_vs_fiel(
        combinado,
        labels,
        GRUPO_CONTEOS,
        "Cantidad de productos y préstamos",
        "Promedio por grupo.",
        DIR_BARRAS / "tenencia_conteos.png",
    )
    barras_grupo_vs_fiel(
        combinado,
        labels,
        GRUPO_RIESGO,
        "Señales de riesgo",
        "Proporción de clientes con el flag activo, por grupo.",
        DIR_BARRAS / "riesgo.png",
    )

    log("valor")
    barras_valor_anual(
        t["tabla_rent"], t["rent_fiel_sano"], DIR_VALOR / "valor_anual.png"
    )
    descomposicion_margen(t["tabla_rent"], DIR_VALOR / "descomposicion_margen.png")
    barras_valor_en_riesgo(t["tabla_rent"], DIR_VALOR / "valor_en_riesgo.png")
    dispersion_valor_actividad(
        combinado, labels, DIR_VALOR / "dispersion_valor_actividad.png"
    )

    log("distribuciones")
    for var in VARS_CAJA:
        caja_variable(combinado, labels, var, DIR_DISTRIBUCIONES / f"caja_{var}.png")
    distribucion_demografia(combinado, labels, DIR_DISTRIBUCIONES / "demografia.png")
    mix_clusters_por_mes(df, DIR_DISTRIBUCIONES / "mix_clusters_por_mes.png")

    log("apéndice: tendencias de las top-40 variables del ranking completo")
    top40 = (
        t["dif_completo"]
        .assign(abs_d=t["dif_completo"]["cohen_d"].abs())
        .sort_values("abs_d", ascending=False)["variable"]
        .drop_duplicates()
        .head(40)
        .tolist()
    )
    con = t["con"]
    exprs = ", ".join(f"AVG({v}) AS {v}" for v in top40)
    fila = con.execute(f"SELECT {exprs} FROM fieles").fetchone()
    medias_top40 = pd.Series({v: fila[i] for i, v in enumerate(top40)})
    tend_top40 = tendencias_por_mes_relativo(df, top40, RANGO_MESES)
    (OUT_DIR / "_apendice_png").mkdir(exist_ok=True)
    for var in top40:
        linea_tendencia(
            tend_top40,
            var,
            RANGO_MESES,
            medias_top40[var],
            OUT_DIR / "_apendice_png" / f"tendencia_{var}.png",
        )


# --------------------------------------------------------------------------- #
# Compilación de PDFs
# --------------------------------------------------------------------------- #


def compilar_pdfs(t: dict) -> None:
    clusters = sorted(t["labels"].unique())
    slugs = {
        d: d.lower().replace(" ", "_").replace("ó", "o").replace("é", "e")
        for d in DOMINIOS
    }

    with PdfPages(OUT_DIR / "presentacion.pdf") as pdf:
        pagina_texto(
            pdf,
            "Caracterización de clusters de bajas premium",
            [
                f"{len(t['df'][ID_COL].unique()):,} clientes BAJA+2, agrupados en 3 clusters.",
                f"Referencia: {t['n_fieles']:,} clientes fieles, ${t['rent_fiel']:,.0f}/año promedio.",
                "",
                "Contenido:",
                "  1. Panorama general",
                "  2-4. Ficha por cluster (radar, diferenciadoras, tendencias, valor)",
                "  5. Señales de anticipación y reglas de alerta",
                "  6. Composición y drift",
            ],
        )
        _pdf_img(pdf, DIR_RADARES / "radar_panorama.png")
        _pdf_img(pdf, DIR_VALOR / "valor_anual.png")
        _pdf_img(pdf, DIR_VALOR / "descomposicion_margen.png")
        _pdf_img(pdf, DIR_VALOR / "valor_en_riesgo.png")
        _pdf_img(pdf, DIR_BARRAS / "heatmap_cohen_d.png")

        for c in clusters:
            pagina_texto(
                pdf,
                f"Cluster {c} — {NOMBRES_CLUSTER.get(c, '')}",
                [
                    (
                        f"{int(t['n_por_cluster'][c]):,} clientes "
                        f"({t['n_por_cluster'][c] / t['n_por_cluster'].sum() * 100:.0f}% "
                        "de las bajas premium)."
                    ),
                ],
            )
            _pdf_img(pdf, DIR_RADARES / f"radar_cluster_{c}.png")
            _pdf_img(pdf, DIR_BARRAS / f"cohen_d_cluster_{c}.png")
            for var in [
                "ctrx_quarter",
                "mrentabilidad",
                "mprestamos_personales",
                "ctarjeta_visa",
            ]:
                _pdf_img(pdf, DIR_TENDENCIAS / f"tendencia_{var}.png")

        pagina_texto(
            pdf,
            "Composición y drift",
            [
                "Mix de clusters por mes calendario de la baja.",
            ],
        )
        _pdf_img(pdf, DIR_DISTRIBUCIONES / "mix_clusters_por_mes.png")

        pagina_texto(pdf, "Dominios de negocio — radares y tendencias", [])
        for dominio in DOMINIOS:
            _pdf_img(pdf, DIR_RADARES / f"radar_dominio_{slugs[dominio]}.png")
            _pdf_img(pdf, DIR_TENDENCIAS / f"panel_{slugs[dominio]}.png")

        pagina_texto(pdf, "Barras y distribuciones adicionales", [])
        for p in (
            [
                DIR_BARRAS / "tenencia_chica.png",
                DIR_BARRAS / "tenencia_conteos.png",
                DIR_BARRAS / "riesgo.png",
                DIR_VALOR / "dispersion_valor_actividad.png",
            ]
            + [DIR_DISTRIBUCIONES / f"caja_{v}.png" for v in VARS_CAJA]
            + [DIR_DISTRIBUCIONES / "demografia.png"]
        ):
            _pdf_img(pdf, p)
    log("presentacion.pdf listo")

    with PdfPages(OUT_DIR / "apendice_completo.pdf") as pdf:
        pagina_texto(
            pdf,
            "Apéndice — ranking completo (633 variables)",
            [
                "Cohen's d de cada cluster vs. el cliente fiel, sobre TODAS las",
                "variables del feature engineering (no solo el subconjunto curado).",
                "Excluye slope_3_* (pendientes con overflow numérico, ver límites).",
                "",
                "Tabla completa: apendice_diferenciadoras_completo.csv",
                "Acá: tendencia t-3..t-1 de las 40 variables con mayor |Cohen's d|.",
            ],
        )
        top40 = (
            t["dif_completo"]
            .assign(abs_d=t["dif_completo"]["cohen_d"].abs())
            .sort_values("abs_d", ascending=False)["variable"]
            .drop_duplicates()
            .head(40)
            .tolist()
        )
        for var in top40:
            _pdf_img(pdf, OUT_DIR / "_apendice_png" / f"tendencia_{var}.png")
    log("apendice_completo.pdf listo")


def _pdf_img(pdf, img_path) -> None:
    img_path = Path(img_path)
    if not img_path.exists():
        log(f"  (falta {img_path.name}, se saltea)")
        return
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(plt.imread(img_path))
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# hallazgos.txt
# --------------------------------------------------------------------------- #


ESTRATEGIAS = {
    1: {
        "nombre": "Ahorrista sin nómina",
        "diagnostico": (
            "Cliente activo (ctrx_quarter alto, usa homebanking, tiene tarjetas y saldo en "
            "cuenta) pero SIN nómina domiciliada acá: el banco es una cuenta secundaria, no "
            "la principal. Rentabilidad de margen pasivo (depósitos), no de crédito ni "
            "comisiones de uso."
        ),
        "medidas": [
            (
                "Oferta de domiciliación de sueldo con beneficio escalonado (ej. bonificación "
                "de comisiones los primeros 6 meses) apenas se detecta actividad transaccional "
                "alta sin acreditación de haberes — es la brecha más grande vs. el fiel."
            ),
            (
                "Programa de fidelización de plazo fijo / inversión: ya tienen el saldo, "
                "falta atarlo a un producto con costo de salida (plazo fijo, fondo común)."
            ),
            (
                "Alerta temprana: caída de ctrx_quarter sostenida en 2 meses consecutivos "
                "(variable con mayor Cohen's d del cluster) dispara contacto proactivo antes "
                "de que la baja de actividad se convierta en baja de cliente."
            ),
        ],
    },
    2: {
        "nombre": "Cliente dormido",
        "diagnostico": (
            "Tiene los productos contratados pero la cuenta prácticamente vacía y actividad "
            "mínima en todos los canales. Es probable que ya haya migrado su operatoria a "
            "otra entidad antes de la baja formal — la baja formal es tardía respecto del "
            "abandono real."
        ),
        "medidas": [
            (
                'Campaña de reactivación ("win-back") disparada por decil más bajo de '
                "ctrx_quarter sostenido, no por señales de mora: acá el problema es desuso, "
                "no riesgo crediticio."
            ),
            (
                "Revisar comisiones de mantenimiento de cuentas inactivas: si el cliente no "
                "usa ningún canal, un costo de mantener la cuenta puede ser el empujón final "
                "hacia la baja formal — bonificarlo mientras dura la campaña de reactivación."
            ),
            (
                "Este cluster tiene la MENOR rentabilidad de los tres: priorizar solo el "
                "subconjunto con mayor antigüedad/mayor cantidad de productos contratados, "
                "donde el costo de reactivación se paga con menos volumen."
            ),
        ],
    },
    3: {
        "nombre": "Tomador de crédito digital",
        "diagnostico": (
            "El cluster de MAYOR rentabilidad (préstamos personales, margen activo), pero "
            "sin tarjetas de crédito y con saldo de cuenta negativo — el vínculo es "
            "prácticamente monoproducto (el préstamo) y digital. La baja probablemente "
            "coincide con la cancelación/fin del préstamo, no con una decisión de abandonar "
            "el banco por descontento."
        ),
        "medidas": [
            (
                "Oferta de renovación/refinanciación PROACTIVA antes de la última cuota: el "
                "monto de préstamos cae fuerte en los meses previos a la baja (ver señales "
                "de anticipación), que es exactamente cuando hay que ofrecer el próximo "
                "crédito."
            ),
            (
                "Venta cruzada de tarjeta de crédito: es la ausencia más marcada del cluster "
                "(f_sin_visa, f_sin_master con los lift más altos de los tres clusters) y "
                "una segunda pata de vínculo que sobreviva al fin del préstamo."
            ),
            (
                "Como es el cluster de mayor valor por cliente, el costo de estímulo "
                f"({_fmt_monto(COSTO_ESTIMULO)} por cliente, referencia de la cátedra) se "
                "justifica con un recall relativamente bajo: alcanza con retener a una "
                "fracción chica del cluster para que la acción cierre económicamente."
            ),
        ],
    },
}


def escribir_hallazgos(t: dict) -> None:
    tabla_rent = t["tabla_rent"].sort_values("cluster")
    n_total = t["n_por_cluster"].sum()
    lineas = []

    lineas.append("=" * 78)
    lineas.append(
        "CARACTERIZACIÓN DE CLUSTERS DE BAJAS PREMIUM — HALLAZGOS PRINCIPALES"
    )
    lineas.append("=" * 78)
    lineas.append("")
    lineas.append(
        "Toda la base son clientes premium (si están en el dataset, son premium; si se "
        "fueron, se fue un premium). Este análisis caracteriza los 3 tipos de baja "
        "encontrados por el clustering (RF + PCA + KMeans, k=3, msl=30, ver "
        "clases/z502_marcar_clusters.py) contra la población COMPLETA de clientes fieles "
        f"({t['n_fieles']:,} clientes, no una muestra), para poder generalizar los números."
    )
    lineas.append("")

    lineas.append("-" * 78)
    lineas.append("1. RESUMEN EJECUTIVO")
    lineas.append("-" * 78)
    lineas.append(
        f"\n{n_total:,} clientes premium se dieron de baja (BAJA+2) en el panel analizado."
    )
    lineas.append(
        f"Cliente fiel promedio (referencia): {_fmt_monto(t['rent_fiel'])}/año.\n"
    )
    for _, r in tabla_rent.iterrows():
        c = int(r["cluster"])
        lineas.append(
            f"  Cluster {c} — {NOMBRES_CLUSTER.get(c, '')}: {int(r['n_clientes']):,} clientes "
            f"({r['n_clientes'] / tabla_rent['n_clientes'].sum() * 100:.0f}%), "
            f"{_fmt_monto(r['rent_anual_por_cliente'])}/año/cliente "
            f"({r['ratio_vs_fiel']:.2f}x el fiel, {r['categoria_valor']}), "
            f"{_fmt_monto(r['rent_anual_cluster_total'])}/año en riesgo total."
        )
    lineas.append(
        f"\nTasa base real de BAJA+2 en el panel: {_fmt_pct(t['tasa_base'])} mensual "
        "(NO 50%: ese era el sesgo del pipeline de exploración, que usaba una muestra 1:1 "
        "de fieles en vez de la población real — ver sección de límites)."
    )

    lineas.append("\n" + "-" * 78)
    lineas.append("2. FICHA POR CLUSTER")
    lineas.append("-" * 78)
    for c in sorted(t["labels"].unique()):
        lineas.append(f"\n{'=' * 40}")
        lineas.append(f"CLUSTER {c} — {NOMBRES_CLUSTER.get(c, '')}")
        lineas.append("=" * 40)
        r = tabla_rent[tabla_rent["cluster"] == c].iloc[0]
        lineas.append(
            f"{int(r['n_clientes']):,} clientes · {_fmt_monto(r['rent_anual_por_cliente'])}/año "
            f"({r['ratio_vs_fiel']:.2f}x fiel, {r['categoria_valor']})"
        )
        lineas.append("\nQué lo distingue del cliente FIEL (top 5, Cohen's d):")
        dv = t["dif_vs_fieles"]
        for _, rr in dv[dv["cluster"] == c].head(5).iterrows():
            lineas.append(
                f"  {etiqueta(rr['variable']):40s} {rr['direccion']:5s} d={rr['cohen_d']:+.2f}  "
                f"(cluster: {rr['media_cluster']:,.2f} | fiel: {rr['media_fiel']:,.2f})"
            )
        lineas.append("\nQué lo distingue de los OTROS DOS clusters (top 5):")
        do = t["dif_vs_otros"]
        for _, rr in do[do["cluster"] == c].head(5).iterrows():
            lineas.append(
                f"  {etiqueta(rr['variable']):40s} {rr['direccion']:5s} d={rr['cohen_d']:+.2f}"
            )
        lineas.append(f"\nDiagnóstico: {ESTRATEGIAS[c]['diagnostico']}")

    lineas.append("\n" + "-" * 78)
    lineas.append("3. SEÑALES DE ANTICIPACIÓN Y REGLAS DE ALERTA")
    lineas.append("-" * 78)
    lineas.append(
        "\nCambio estandarizado entre t-3 y t-1 (meses antes de la baja), en desvíos "
        "estándar del fiel. Un |cambio| grande es candidato a señal temprana."
    )
    sen = t["senales"]
    for c in sorted(t["labels"].unique()):
        lineas.append(f"\nCluster {c}:")
        for _, rr in sen[sen["cluster"] == c].head(4).iterrows():
            lineas.append(
                f"  {etiqueta(rr['variable']):40s} cambio_std={rr['cambio_std']:+.2f}  "
                f"(n_t-3={rr['n_t-3']}, n_t-1={rr['n_t-1']})"
            )

    lineas.append(
        "\nReglas de alerta evaluadas contra el denominador poblacional REAL "
        "(no la muestra 1:1 del triggers_candidatos.csv original):"
    )
    for _, rr in t["reglas"].iterrows():
        lineas.append(
            f"  {etiqueta(rr['variable']):40s} {rr['direccion']:5s} decil {rr['decil_evaluado']}: "
            f"{rr['n_disparos']:,} clientes/mes disparan · recall={_fmt_pct(rr['recall'])} · "
            f"precisión={_fmt_pct(rr['precision'])} · lift={rr['lift_vs_tasa_base']:.1f}x"
        )

    lineas.append("\n" + "-" * 78)
    lineas.append("4. ESTRATEGIAS DE RETENCIÓN POR CLUSTER")
    lineas.append("-" * 78)
    lineas.append(
        f"\nCosto de estímulo de referencia (cátedra DMEyF 2026): {_fmt_monto(COSTO_ESTIMULO)} "
        "por cliente contactado que no se iba a dar de baja. Usar para dimensionar si una "
        "campaña cierra económicamente: costo total = n_contactados x costo_estimulo, "
        "compararlo contra el valor anual en riesgo del cluster."
    )
    for c in sorted(t["labels"].unique()):
        e = ESTRATEGIAS[c]
        lineas.append(f"\nCluster {c} — {e['nombre']}:")
        for m in e["medidas"]:
            lineas.append(f"  • {m}")

    lineas.append("\n" + "-" * 78)
    lineas.append("5. COMPOSICIÓN Y DRIFT")
    lineas.append("-" * 78)
    mix_prop = t["mix_prop"]
    lineas.append("\nMix de clusters por mes calendario de la baja:")
    for mes, row in mix_prop.iterrows():
        partes = ", ".join(f"cluster_{c}={_fmt_pct(v)}" for c, v in row.items())
        lineas.append(f"  {mes}: {partes}")
    variacion = mix_prop.max() - mix_prop.min()
    cluster_mas_variable = variacion.idxmax()
    lineas.append(
        f"\nEl cluster con mix más inestable mes a mes es el cluster {cluster_mas_variable} "
        f"(varía {_fmt_pct(variacion[cluster_mas_variable])} entre el mes más bajo y el más "
        "alto). Con solo 4 meses de bajas observadas, no se puede distinguir un efecto de "
        "cohorte/estacionalidad real de ruido muestral — señalarlo en la presentación como "
        "hipótesis a validar con más meses de datos, no como tendencia confirmada."
    )

    n_prof_ctrx = t["tend"]["n"]["ctrx_quarter"].unstack("mes_relativo")
    n_lejano = n_prof_ctrx[RANGO_MESES[0]]
    n_cercano = n_prof_ctrx[RANGO_MESES[-1]]

    lineas.append("\n" + "-" * 78)
    lineas.append("6. LÍMITES DEL ANÁLISIS")
    lineas.append("-" * 78)
    lineas.append(
        f"""
  • mes_relativo está anclado al primer mes en que el cliente YA NO aparece
    en el panel (t=0, sin datos, nunca se grafica): t=-1 es su último mes
    real (la fila que clase_ternaria etiqueta BAJA+1), t=-2 es la fila
    BAJA+2, t=-3 y anteriores son meses CONTINUA previos. No se excluye
    ninguna fila real del cliente — con este ancla, toda fila cae en t<=-1
    de forma automática.
  • rentabilidad_por_cluster() (sección 1) es la excepción: usa su propio
    criterio (rank_desde_baja > 2 sobre foto_mes, no mes_relativo) para
    excluir los últimos 2 meses reales del cliente, porque el consumo/saldo
    se distorsiona cerca de la baja — heredado del pipeline original.
  • Panel de 6 meses (202103-202108); los BAJA+2 están en 202103-202106. La
    profundidad temporal hacia atrás es corta: en t={RANGO_MESES[0]} quedan
    entre {int(n_lejano.min()):,} y {int(n_lejano.max()):,} clientes por
    cluster, contra entre {int(n_cercano.min()):,} y {int(n_cercano.max()):,}
    en t={RANGO_MESES[-1]} (de un total de {int(t["n_por_cluster"].min()):,} a
    {int(t["n_por_cluster"].max()):,} clientes por cluster) — cada gráfico de
    tendencia muestra su `n` explícitamente por esto.
  • El clustering se hizo SOLO sobre clientes BAJA+2: no hay un modelo que
    prediga a qué cluster iría un cliente fiel si se fuera. Las reglas de
    alerta de la sección 3 son umbrales descriptivos evaluados sobre la
    población completa, no un clasificador entrenado.
  • El apéndice de 633 variables excluye las slope_3_* (pendientes de
    ventanas de 3 meses): un desborde numérico en el cálculo de varianza
    poblacional para esas columnas (valores extremos en la regresión sobre
    ventanas casi verticales) impide calcular Cohen's d de forma confiable
    sin un tratamiento de outliers aparte.
  • El mix de clusters por mes de baja (sección 5) tiene solo 4 puntos: no
    alcanza para separar estacionalidad real de ruido.
  • La rentabilidad fiel de referencia surge de TODA la población fiel
    (157.837 clientes), no de la muestra 1:1 del pipeline de exploración
    original (4.067 clientes): los números de valor por cluster de este
    archivo difieren levemente de los de `analisis_k3_msl30_filtrado_pca10 -
    final/resumen.txt` por esa razón (baseline distinto), no por un cambio
    en el clustering.
""".strip("\n")
    )

    (OUT_DIR / "hallazgos.txt").write_text("\n".join(lineas), encoding="utf-8")
    log(f"hallazgos.txt escrito ({len(lineas)} líneas)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--solo",
        choices=["tablas"],
        default=None,
        help="tablas: solo calcula y guarda los CSV, sin gráficos ni PDFs "
        "(para iterar rápido sobre la aritmética).",
    )
    args = ap.parse_args()

    t0 = time.time()
    t = calcular_tablas()
    guardar_tablas(t)

    if args.solo == "tablas":
        log(f"listo (solo tablas) en {time.time() - t0:.1f}s")
        return

    generar_graficos(t)
    compilar_pdfs(t)
    escribir_hallazgos(t)
    log(f"TODO listo en {time.time() - t0:.1f}s → {OUT_DIR}/")


if __name__ == "__main__":
    main()
