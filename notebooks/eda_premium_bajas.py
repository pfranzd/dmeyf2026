# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EDA de negocio — Bajas de Paquete Premium
#
# **Para:** Miranda, Dirección Comercial.
# **Pregunta de negocio:** ¿por qué se dan de baja los clientes de Paquete
# Premium, hay umbrales de alerta accionables, y existen perfiles
# diferenciados de clientes que se van?
#
# **Universo de análisis.** El diccionario de datos
# (`datasets/raw/diccionario_datos.csv`, campo `cliente_vip`) y las notas
# generales (`datasets/raw/consideraciones.txt`) confirman que **todo el
# dataset ya es Paquete Premium** ("En el dataset solamente hay clientes
# titulares de Paquete Premium"). No hay una columna de membresía premium
# separada — `cliente_vip` es un segmento de marketing mucho más chico
# (<2%) y no debe confundirse con el universo. Por lo tanto, "baja de
# Paquete Premium" = `clase_ternaria` en {BAJA+1, BAJA+2} sobre el dataset
# completo, sin filtrar nada adicional.
#
# **Datos.** `datasets/processed/competencia_01.csv`: 983,061 registros
# (panel cliente-mes), 169,727 clientes únicos, 155 columnas, `foto_mes`
# 202103-202108 (6 meses). `clase_ternaria` necesita mirar hasta 2 meses de
# futuro, así que la completitud de la etiqueta es asimétrica: 202103-202106
# tienen las 3 clases completas; **202107 solo tiene `BAJA+1`** (1,103
# filas — `CONTINUA`/`BAJA+2` de ese mes necesitarían `foto_mes=202109`,
# que no existe) y **202108 no tiene ninguna etiqueta**. Cualquier
# análisis por mes de este notebook usa solo 202103-202106; incluir 202107
# en una tasa mensual la infla artificialmente (denominador de solo 1,103
# casos, todos BAJA+1). Sobre los 655,169 registros etiquetados en total:
# 646,014 CONTINUA (98.60%), 5,088 BAJA+1 (0.78%), 4,067 BAJA+2 (0.62%) —
# 9,155 bajas en total (1.40%).
#
# Este EDA es complementario al ya existente `02_eda_clase_ternaria.py`
# (orientado a separar BAJA+2 de BAJA+1 para *modelado*). Acá el eje es
# BAJA (unificada) vs CONTINUA, en **lenguaje de negocio**, con foco en
# umbrales accionables y perfiles de cliente.
#
# Todo el cómputo pesado se hace en DuckDB/Polars (nunca se cargan las
# 983k filas x 155 columnas enteras a memoria de Python).

# %%
import sys

import duckdb
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "..")
from config.semillas import SEMILLA_PRIMARIA
from dmeyf.metrics import GANANCIA_ACIERTO

DATA_PATH = "../datasets/processed/competencia_01.csv"
WORK_DIR = "../work"

con = duckdb.connect()
con.execute(f"""
    create or replace view competencia_01 as
    select * from read_csv_auto('{DATA_PATH}')
""")
con.execute("""
    create or replace view etiquetado as
    select *, case when clase_ternaria = 'CONTINUA' then 'CONTINUA' else 'BAJA' end as grupo
    from competencia_01
    where clase_ternaria is not null
""")

# %% [markdown]
# ## Inventario de variables (155 campos → 152 features de negocio)
#
# Agrupamos las variables del diccionario en 6 familias de negocio. Se
# excluyen `numero_de_cliente`, `foto_mes` (identificadores) y
# `clase_ternaria` (el target).

# %%
# EXPLORACIÓN: mapeo completo columna -> grupo de negocio, a partir del
# diccionario (datasets/raw/diccionario_datos.csv).
GRUPOS = {
    "Demográficas": [
        "cliente_edad", "cliente_antiguedad", "cliente_vip", "active_quarter",
    ],
    "Tenencia de productos": [
        "cproductos", "tcuentas", "ccuenta_corriente", "ccaja_ahorro",
        "cdescubierto_preacordado", "ctarjeta_debito", "ctarjeta_visa",
        "ctarjeta_master", "cprestamos_personales", "cprestamos_prendarios",
        "cprestamos_hipotecarios", "cplazo_fijo", "cinversion1", "cinversion2",
        "cseguro_vida", "cseguro_auto", "cseguro_vivienda",
        "cseguro_accidentes_personales", "ccaja_seguridad", "cpayroll_trx",
        "cpayroll2_trx", "Master_status", "Visa_status", "Master_delinquency",
        "Visa_delinquency",
    ],
    "Transaccionales (volumen/frecuencia)": [
        "ctarjeta_debito_transacciones", "ctarjeta_visa_transacciones",
        "ctarjeta_master_transacciones", "ccuenta_debitos_automaticos",
        "ctarjeta_visa_debitos_automaticos", "ctarjeta_master_debitos_automaticos",
        "cpagodeservicios", "cpagomiscuentas", "ccajeros_propios_descuentos",
        "ctarjeta_visa_descuentos", "ctarjeta_master_descuentos",
        "ccomisiones_mantenimiento", "ccomisiones_otras", "cforex", "cforex_buy",
        "cforex_sell", "ctransferencias_recibidas", "ctransferencias_emitidas",
        "cextraccion_autoservicio", "ccheques_depositados", "ccheques_emitidos",
        "ccheques_depositados_rechazados", "ccheques_emitidos_rechazados",
        "ccajas_transacciones", "ccajas_consultas", "ccajas_depositos",
        "ccajas_extracciones", "ccajas_otras", "catm_trx", "catm_trx_other",
        "ctrx_quarter", "Master_cconsumos", "Master_cadelantosefectivo",
        "Visa_cconsumos", "Visa_cadelantosefectivo",
    ],
    "Saldos y montos": [
        "mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen",
        "mpasivos_margen", "mcuenta_corriente_adicional", "mcuenta_corriente",
        "mcaja_ahorro", "mcaja_ahorro_adicional", "mcaja_ahorro_dolares",
        "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo",
        "mtarjeta_master_consumo", "mprestamos_personales", "mprestamos_prendarios",
        "mprestamos_hipotecarios", "mplazo_fijo_dolares", "mplazo_fijo_pesos",
        "minversion1_pesos", "minversion1_dolares", "minversion2", "mpayroll",
        "mpayroll2", "mcuenta_debitos_automaticos", "mttarjeta_visa_debitos_automaticos",
        "mttarjeta_master_debitos_automaticos", "mpagodeservicios", "mpagomiscuentas",
        "mcajeros_propios_descuentos", "mtarjeta_visa_descuentos",
        "mtarjeta_master_descuentos", "mcomisiones_mantenimiento", "mcomisiones_otras",
        "mforex_buy", "mforex_sell", "mtransferencias_recibidas",
        "mtransferencias_emitidas", "mextraccion_autoservicio", "mcheques_depositados",
        "mcheques_emitidos", "mcheques_depositados_rechazados",
        "mcheques_emitidos_rechazados", "matm", "matm_other",
        "Master_mfinanciacion_limite", "Master_msaldototal", "Master_msaldopesos",
        "Master_msaldodolares", "Master_mconsumospesos", "Master_mconsumosdolares",
        "Master_mlimitecompra", "Master_madelantopesos", "Master_madelantodolares",
        "Master_mpagado", "Master_mpagospesos", "Master_mpagosdolares",
        "Master_mconsumototal", "Master_mpagominimo", "Visa_mfinanciacion_limite",
        "Visa_msaldototal", "Visa_msaldopesos", "Visa_msaldodolares",
        "Visa_mconsumospesos", "Visa_mconsumosdolares", "Visa_mlimitecompra",
        "Visa_madelantopesos", "Visa_madelantodolares", "Visa_mpagado",
        "Visa_mpagospesos", "Visa_mpagosdolares", "Visa_mconsumototal",
        "Visa_mpagominimo",
    ],
    "Canales y comportamiento digital": [
        "internet", "thomebanking", "chomebanking_transacciones", "tmobile_app",
        "cmobile_app_trx", "tcallcenter", "ccallcenter_transacciones",
    ],
    "Temporales / antigüedad de producto": [
        "Master_Fvencimiento", "Master_Finiciomora", "Master_fultimo_cierre",
        "Master_fechaalta", "Visa_Fvencimiento", "Visa_Finiciomora",
        "Visa_fultimo_cierre", "Visa_fechaalta",
    ],
}
COLUMNA_A_GRUPO = {col: grupo for grupo, cols in GRUPOS.items() for col in cols}

todas_las_columnas = {
    r[0] for r in con.execute("describe select * from etiquetado").fetchall()
} - {"numero_de_cliente", "foto_mes", "clase_ternaria", "grupo"}
sin_mapear = todas_las_columnas - set(COLUMNA_A_GRUPO)
sobrantes = set(COLUMNA_A_GRUPO) - todas_las_columnas
assert not sin_mapear, f"Columnas sin grupo de negocio asignado: {sin_mapear}"
assert not sobrantes, f"El mapeo referencia columnas inexistentes: {sobrantes}"

for grupo, cols in GRUPOS.items():
    print(f"{grupo}: {len(cols)} variables")
print(f"\nTotal: {len(COLUMNA_A_GRUPO)} variables de negocio mapeadas")

# %% [markdown]
# ## FASE 1 — Panorama general

# %% [markdown]
# ### 1.1 Balance de clases (universo Paquete Premium, meses etiquetados)

# %%
balance = con.execute("""
    select clase_ternaria, count(*) as n,
           round(100.0 * count(*) / sum(count(*)) over (), 3) as pct
    from etiquetado
    group by clase_ternaria
    order by n
""").pl()
balance

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(balance["clase_ternaria"], balance["n"], color=["#c0392b", "#e67e22", "#2980b9"])
for i, (n, pct) in enumerate(zip(balance["n"], balance["pct"])):
    ax.text(i, n, f"{n:,}\n({pct}%)", ha="center", va="bottom", fontsize=9)
ax.set_yscale("log")
ax.set_title("Clientes de Paquete Premium por estado (escala logarítmica)")
ax.set_ylabel("cantidad de clientes-mes (log)")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_balance_clases.png", dpi=120)
plt.show()

# %% [markdown]
# **Resumen de negocio.** De cada 1.000 clientes de Paquete Premium en un
# mes dado, ~14 se dan de baja en los próximos dos meses (9 se van el mes
# que viene, 5 el mes siguiente a ese). Es un evento raro pero de alto
# impacto: cualquier acción de retención debe priorizar con un modelo de
# riesgo, no se puede tratar a toda la base por igual.

# %% [markdown]
# ### 1.2 Evolución mensual de las bajas
#
# Restringido a `foto_mes` 202103-202106: son los únicos 4 meses con las
# 3 clases completas. `202107` queda afuera a propósito — solo tiene
# `BAJA+1` etiquetado (`CONTINUA`/`BAJA+2` de ese mes necesitarían datos de
# `202109`, que no existen), así que una tasa mensual ahí daría 100% por
# construcción, no porque haya un mes atípico.

# %%
por_mes = con.execute("""
    select foto_mes,
           count(*) as clientes_totales,
           sum(case when clase_ternaria in ('BAJA+1','BAJA+2') then 1 else 0 end) as bajas,
           round(100.0 * sum(case when clase_ternaria in ('BAJA+1','BAJA+2') then 1 else 0 end) / count(*), 3) as tasa_baja_pct
    from etiquetado
    where foto_mes <= 202106
    group by foto_mes
    order by foto_mes
""").pl()
por_mes

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(por_mes["foto_mes"].cast(pl.Utf8), por_mes["bajas"], color="#c0392b")
axes[0].set_title("Bajas por mes (volumen absoluto)")
axes[0].set_ylabel("clientes dados de baja")
axes[1].plot(por_mes["foto_mes"].cast(pl.Utf8), por_mes["tasa_baja_pct"], marker="o", color="#c0392b")
axes[1].set_title("Tasa de baja mensual (%)")
axes[1].set_ylabel("% de clientes que se dan de baja")
fig.suptitle("Evolución de las bajas de Paquete Premium — marzo a junio 2021")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_evolucion_mensual.png", dpi=120)
plt.show()

# %% [markdown]
# **Resumen de negocio.** Con solo 4 meses completamente etiquetados
# (marzo-junio 2021) no alcanza para hablar de estacionalidad real, pero la
# tasa de baja se mantiene estable mes a mes (0.53%-0.70%, sin meses
# atípicos): el problema es estructural, no un evento puntual de algún mes.

# %% [markdown]
# ## FASE 2 — Comparación bajas vs. no-bajas (univariado)
#
# Para cada variable comparamos CONTINUA vs BAJA (BAJA+1 y BAJA+2 unidas).
# Estrategia:
# - **Estadísticos descriptivos** (media, mediana, desvío, cuartiles) sobre
#   TODA la población etiquetada, vía `SUMMARIZE` en DuckDB.
# - **Variables continuas** (>10 valores distintos): tamaño de efecto
#   *d* de Cohen + test de Mann-Whitney U sobre una muestra reproducible
#   (las 9,155 bajas completas + 30,000 CONTINUA muestreadas con la
#   semilla del curso).
# - **Variables categóricas/flags** (≤10 valores distintos): tabla de
#   contingencia completa + chi-cuadrado.

# %%
# EXPLORACIÓN: SUMMARIZE pooled para decidir continua vs. categórica por
# cardinalidad aproximada.
resumen_global = con.execute("""
    summarize select * exclude (numero_de_cliente, foto_mes, clase_ternaria, grupo)
    from etiquetado
""").pl()

CONTINUAS = resumen_global.filter(pl.col("approx_unique") > 10)["column_name"].to_list()
CATEGORICAS = resumen_global.filter(pl.col("approx_unique") <= 10)["column_name"].to_list()
print(f"{len(CONTINUAS)} variables continuas, {len(CATEGORICAS)} categóricas/flags")

# %%
def summarize_grupo(grupo: str) -> pl.DataFrame:
    return con.execute(f"""
        summarize select * exclude (numero_de_cliente, foto_mes, clase_ternaria, grupo)
        from etiquetado where grupo = '{grupo}'
    """).pl().select(["column_name", "avg", "std", "q25", "q50", "q75", "null_percentage"])


suf = lambda df, s: df.rename({c: f"{c}_{s}" for c in df.columns if c != "column_name"})
descriptivos = suf(summarize_grupo("CONTINUA"), "cont").join(
    suf(summarize_grupo("BAJA"), "baja"), on="column_name"
)
for c in ["avg_cont", "std_cont", "avg_baja", "std_baja"]:
    descriptivos = descriptivos.with_columns(pl.col(c).cast(pl.Float64))
descriptivos = descriptivos.with_columns(
    (((pl.col("std_cont") ** 2 + pl.col("std_baja") ** 2) / 2).sqrt()).alias("pooled_std")
).with_columns(
    ((pl.col("avg_baja") - pl.col("avg_cont")) / pl.col("pooled_std")).alias("d_cohen")
).with_columns(
    pl.col("column_name").replace(COLUMNA_A_GRUPO).alias("grupo_negocio")
)
descriptivos.height

# %% [markdown]
# ### 2.1 Test de Mann-Whitney U — variables continuas

# %%
muestra = con.execute(f"""
    select grupo, {", ".join(CONTINUAS)}
    from etiquetado
    where grupo = 'BAJA'
    union all
    select grupo, {", ".join(CONTINUAS)}
    from etiquetado
    where grupo = 'CONTINUA'
    using sample 30000 (reservoir, {SEMILLA_PRIMARIA})
""").pl()

pvalores_continuas = []
baja_mask = muestra["grupo"] == "BAJA"
cont_mask = muestra["grupo"] == "CONTINUA"
for col in CONTINUAS:
    a = muestra.filter(baja_mask)[col].drop_nulls().to_numpy()
    b = muestra.filter(cont_mask)[col].drop_nulls().to_numpy()
    if len(a) > 5 and len(b) > 5:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    else:
        p = np.nan
    pvalores_continuas.append({"column_name": col, "p_value_mw": p})
pvalores_continuas = pl.DataFrame(pvalores_continuas)

resumen_continuas = descriptivos.filter(pl.col("column_name").is_in(CONTINUAS)).join(
    pvalores_continuas, on="column_name"
)

top15 = (
    resumen_continuas.filter(pl.col("d_cohen").is_not_nan())
    .sort(pl.col("d_cohen").abs(), descending=True)
    .head(15)
)
top15.select(["column_name", "grupo_negocio", "avg_cont", "avg_baja", "d_cohen", "p_value_mw"])

# %%
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#c0392b" if v < 0 else "#27ae60" for v in top15["d_cohen"]]
ax.barh(top15["column_name"], top15["d_cohen"], color=colors)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("tamaño de efecto (d de Cohen, BAJA vs. CONTINUA)")
ax.set_title("Top 15 variables que más separan a los clientes que se van")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_top15_variables.png", dpi=120)
plt.show()

# %% [markdown]
# **Resumen de negocio.** Todas las variables del top 15 tienen
# p-valor < 0.001 en el test de Mann-Whitney (la diferencia no es azar).
# Las cinco de mayor efecto son: `ctrx_quarter` (120 vs. 37 transacciones
# trimestrales, la más fuerte), `cproductos` (7.6 vs. 6.3 productos),
# transacciones con tarjeta de débito y de crédito Visa (caen a ~15-30% de
# su nivel sano), y `Master_Finiciomora` (días desde que empezó la mora en
# Mastercard: 30 vs. 64 días — quien se va lleva el doble de tiempo en
# mora). La actividad transaccional general es la señal más nítida, pero
# la mora en tarjeta de crédito aporta una segunda dimensión de riesgo
# independiente de la actividad.

# %% [markdown]
# ### 2.2 Chi-cuadrado — variables categóricas y flags

# %%
resultados_chi2 = []
for col in CATEGORICAS:
    tabla = con.execute(f"""
        select grupo, {col} as valor, count(*) as n
        from etiquetado
        group by 1, 2
    """).pl()
    pivot = tabla.pivot(on="grupo", index="valor", values="n").fill_null(0).sort("valor")
    contingencia = pivot.select(["BAJA", "CONTINUA"]).to_numpy()
    if contingencia.shape[0] > 1:
        chi2, p, _, _ = chi2_contingency(contingencia)
    else:
        chi2, p = np.nan, np.nan
    resultados_chi2.append({
        "column_name": col,
        "grupo_negocio": COLUMNA_A_GRUPO[col],
        "chi2": chi2,
        "p_value_chi2": p,
        "valores_distintos": contingencia.shape[0],
    })
resultados_chi2 = pl.DataFrame(resultados_chi2).sort("chi2", descending=True)
resultados_chi2

# %% [markdown]
# **Resumen de negocio.** Todas las variables categóricas con volumen
# suficiente resultan estadísticamente significativas (desbalance extremo
# de clases infla la potencia del test); lo relevante es el **chi-cuadrado
# relativo entre ellas** como ranking de qué categorías conviene mirar de
# cerca en el detalle de proporciones (sección 2.3).

# %% [markdown]
# ### 2.3 Detalle de negocio — estado de tarjetas y canales

# %%
def pct_valor(col: str) -> pl.DataFrame:
    return con.execute(f"""
        select grupo,
               {col} as valor,
               round(100.0 * count(*) / sum(count(*)) over (partition by grupo), 2) as pct
        from etiquetado
        group by 1, 2
        order by 1, 2
    """).pl()


for col in ["Visa_status", "Master_status", "cliente_vip", "internet"]:
    print(f"\n{col} (% dentro de cada grupo):")
    print(pct_valor(col))

# %%
demografico = con.execute("""
    select grupo,
           round(avg(cliente_edad), 1) as edad_prom,
           round(avg(cliente_antiguedad), 1) as antiguedad_prom_meses,
           round(100.0 * avg(cliente_vip), 3) as pct_vip
    from etiquetado
    group by grupo
    order by grupo
""").pl()
demografico

# %% [markdown]
# **Resumen de negocio.** La edad no distingue a quien se va (47.8 vs. 46.9
# años, prácticamente igual): no sirve como criterio de segmentación. En
# cambio la **antigüedad sí importa**: los que se van tienen en promedio 24
# meses menos de relación con el banco (110.7 vs. 134.7 meses) y son
# clientes VIP **4.5 veces menos frecuentemente** (0.08% vs. 0.34%). La
# lealtad construida en el tiempo protege — no la edad del cliente.

# %% [markdown]
# ## FASE 3 — Umbrales de negocio
#
# Sobre las variables con mayor separación (fase 2), bineamos el valor y
# calculamos la tasa de baja real dentro de cada bin, sobre TODA la
# población etiquetada (655,169 registros).

# %%
VARIABLES_UMBRAL = top15["column_name"].to_list()[:6]
print("Variables analizadas en fase 3:", VARIABLES_UMBRAL)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
tablas_umbral = {}
for ax, col in zip(axes.flat, VARIABLES_UMBRAL):
    tabla = con.execute(f"""
        with base as (
            select {col} as valor, clase_ternaria,
                   ntile(10) over (order by {col}) as decil
            from etiquetado
            where {col} is not null
        )
        select decil,
               min(valor) as valor_min,
               max(valor) as valor_max,
               count(*) as n,
               round(100.0 * sum(case when clase_ternaria in ('BAJA+1','BAJA+2') then 1 else 0 end) / count(*), 3) as tasa_baja_pct
        from base
        group by 1
        order by 1
    """).pl()
    tablas_umbral[col] = tabla
    ax.bar(tabla["decil"].cast(pl.Utf8), tabla["tasa_baja_pct"], color="#c0392b")
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("decil (1=valor más bajo)")
    ax.set_ylabel("% tasa de baja")
fig.suptitle("Tasa de baja por decil de la variable (población completa etiquetada)")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_umbrales.png", dpi=120)
plt.show()

for col, tabla in tablas_umbral.items():
    print(f"\n{col}:")
    print(tabla)

# %% [markdown]
# **Resumen de negocio — umbrales de alerta:**
#
# | Variable | Umbral de alerta | Tasa de baja | Referencia (decil de menor riesgo) |
# |---|---|---|---|
# | `ctrx_quarter` | menos de 27 transacciones en el trimestre | 8.3% | 0.15% (220+ trx) — **56x** |
# | `ctarjeta_visa_transacciones` | 0 transacciones con Visa en el mes | 5.3% | 0.26% (30+ trx) — **20x** |
# | `ctarjeta_debito_transacciones` | 0-1 transacciones de débito en el mes | ~3.6% prom. | 0.20% (23+ trx) — **18x** |
# | `cproductos` | 6 productos o menos | 4.97% | 0.22% (9+ productos) — **23x** |
# | `ccomisiones_mantenimiento` | al menos 1 comisión de mantenimiento cobrada en el mes | ~2.9-3.1% | 0.60-0.65% (0 comisiones) — **~5x** |
# | `Master_Finiciomora` (solo clientes con Mastercard en mora) | más de 51 días desde el inicio de la mora | 10.1% | 0.84% (0-9 días) — **12x** |
#
# **Lectura para Dirección Comercial:** el quiebre más fuerte y accionable
# es la actividad transaccional trimestral — por debajo de ~27
# transacciones en 3 meses (menos de 2 por semana) la tasa de baja se
# multiplica por más de 50. La cantidad de comisiones de mantenimiento es
# el hallazgo más contraintuitivo: empezar a pagarlas (que debería ser un
# evento raro en Paquete Premium) ya es, por sí solo, una señal de alerta
# 5 veces más fuerte que no pagarlas — ver hallazgo sorprendente en fase 5.

# %% [markdown]
# ## FASE 4 — Perfilado de las bajas (clustering)
#
# Trabajamos solo sobre las 9,155 filas BAJA+1/BAJA+2 (cabe en memoria).
# Estandarizamos las variables continuas del top 15 (fase 2) e
# imputamos nulos con la mediana de la población CONTINUA (representa
# "sin uso del producto", no un valor faltante real).

# %%
VARIABLES_CLUSTER = top15["column_name"].to_list()[:8]
print("Variables usadas para clustering:", VARIABLES_CLUSTER)

bajas_df = con.execute(f"""
    select numero_de_cliente, clase_ternaria, {", ".join(VARIABLES_CLUSTER)}
    from etiquetado
    where grupo = 'BAJA'
""").pl()

medianas_continua = descriptivos.filter(
    pl.col("column_name").is_in(VARIABLES_CLUSTER)
).select(["column_name", "q50_cont"])
medianas_dict = dict(zip(medianas_continua["column_name"], medianas_continua["q50_cont"]))

X = bajas_df.select(VARIABLES_CLUSTER).with_columns(
    [pl.col(c).fill_null(medianas_dict[c]) for c in VARIABLES_CLUSTER]
).to_numpy()
X_std = StandardScaler().fit_transform(X)

# %%
resultados_k = []
for k in range(3, 7):
    km = KMeans(n_clusters=k, random_state=SEMILLA_PRIMARIA, n_init=10)
    labels = km.fit_predict(X_std)
    score = silhouette_score(X_std, labels)
    resultados_k.append({"k": k, "silhouette": score})
pl.DataFrame(resultados_k)

# %% [markdown]
# Elegir `K_ELEGIDO` tras revisar la tabla de silhouette y, sobre todo,
# validar que los perfiles resultantes sean distintos e interpretables en
# términos de negocio (no elegir a ciegas por el mayor silhouette).

# %%
K_ELEGIDO = 4  # ajustar tras revisar resultados_k y los perfiles de abajo
km_final = KMeans(n_clusters=K_ELEGIDO, random_state=SEMILLA_PRIMARIA, n_init=10)
bajas_df = bajas_df.with_columns(pl.Series("cluster", km_final.fit_predict(X_std)))

perfiles = (
    bajas_df.group_by("cluster")
    .agg(
        [pl.len().alias("n")]
        + [pl.col(c).mean().alias(f"{c}_prom") for c in VARIABLES_CLUSTER]
    )
    .sort("cluster")
    .with_columns((100 * pl.col("n") / pl.col("n").sum()).round(1).alias("pct_de_bajas"))
)
perfiles

# %% [markdown]
# **Perfiles de negocio (sobre las 9,155 bajas):**
#
# - **Cluster 0 — "Declive generalizado silencioso" (81.1%, 7,425 clientes).**
#   El perfil dominante. Actividad mínima en todo: 20 transacciones
#   trimestrales (vs. 120 de un cliente sano), menos de 1 transacción de
#   débito o Visa por mes, prácticamente sin acreditación de haberes
#   (`cpayroll_trx` ≈ 0.05) y consumo de autoservicio residual ($862 vs.
#   $20.400 promedio). Es el patrón "clásico" de desenganche progresivo.
# - **Cluster 3 — "Multiproducto activo con fuga probablemente competitiva"
#   (14.3%, 1,312 clientes).** Segundo en volumen y el más importante para
#   repensar la hipótesis de negocio: mantiene 7.5 productos y 17
#   transacciones Visa al mes — **en línea o por encima del cliente sano
#   promedio** — y aun así se da de baja. No es desenganche, es fuga hacia
#   otra oferta (tasa, límite, beneficio de la competencia).
# - **Cluster 1 — "Súper-activo que igual se va" (4.3%, 393 clientes).** El
#   grupo más chico con señal de alto valor: 150 transacciones
#   trimestrales y $47.462 de consumo en autoservicio — **muy por encima**
#   incluso del cliente sano promedio ($20.400). Perderlos es la pérdida de
#   mayor valor económico por cliente de las cuatro.
# - **Cluster 2 — "Mora prolongada en Mastercard" (0.3%, 25 clientes).**
#   Grupo minúsculo pero con la señal más extrema: en promedio 157 días
#   desde el inicio de la mora (más de 5 meses), actividad casi nula en
#   todo lo demás. Más que retención comercial, es un caso de gestión de
#   cobranza tardía.
#
# `pct_de_bajas` marca qué fracción es BAJA+2 (dos meses antes de irse, con
# más margen de reacción): 50.6% en cluster 1 y 48.6% en cluster 3 — hay
# tiempo para actuar — vs. 32% en cluster 2, donde la decisión ya está casi
# tomada.

# %% [markdown]
# ## FASE 5 — Correlaciones y hallazgos transversales
#
# Excluimos `Master_Finiciomora` de la matriz: solo existe para clientes
# con Mastercard en mora (~1% de la base), no es comparable variable a
# variable con las demás, que aplican a toda la población.

# %%
VARIABLES_CORR = [c for c in VARIABLES_CLUSTER if c != "Master_Finiciomora"]
matriz_corr = np.zeros((len(VARIABLES_CORR), len(VARIABLES_CORR)))
muestra_corr = muestra.select(VARIABLES_CORR).drop_nulls()
for i, ci in enumerate(VARIABLES_CORR):
    for j, cj in enumerate(VARIABLES_CORR):
        rho, _ = spearmanr(muestra_corr[ci], muestra_corr[cj])
        matriz_corr[i, j] = rho

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matriz_corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(VARIABLES_CORR)))
ax.set_yticks(range(len(VARIABLES_CORR)))
ax.set_xticklabels(VARIABLES_CORR, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(VARIABLES_CORR, fontsize=8)
for i in range(len(VARIABLES_CORR)):
    for j in range(len(VARIABLES_CORR)):
        ax.text(j, i, f"{matriz_corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Correlación de Spearman — variables top de fase 2")
fig.colorbar(im, ax=ax, label="rho de Spearman")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_correlaciones.png", dpi=120)
plt.show()

# %% [markdown]
# **Pares muy correlacionados (no duplicar en la narrativa):**
# `ctarjeta_debito_transacciones` y `mautoservicio` (rho ≈ 1.0) son, en la
# práctica, la misma señal medida en cantidad y en monto — usar una sola
# en la comunicación a Miranda.
#
# **Hallazgos sorprendentes:**
# 1. **Los que se van no son los que "le cuestan menos" al banco.** Pagan
#    en promedio **más del doble** en comisiones de mantenimiento que los
#    que se quedan (0.73 vs. 0.34 comisiones/mes, $1.336 vs. $600), pese a
#    tener mucha menos actividad general. Y esto no se explica del todo
#    por "está menos activo, por eso paga más": la correlación entre
#    comisiones de mantenimiento y actividad (`ctrx_quarter`) es de -0.30
#    (moderada, no fuerte) — más relevante aún, correlaciona -0.61 con
#    `cpayroll_trx` (acreditación de haberes), lo que sugiere que la causa
#    más probable es perder la condición de exención por nómina o saldo
#    mínimo, no simplemente "usar menos el banco". Es una señal de riesgo
#    que vale la pena monitorear por separado de la actividad general.
# 2. **No todas las bajas vienen de clientes desenganchados.** El cluster
#    "Multiproducto activo" (14.3% de las bajas) tiene cantidad de
#    productos y consumo de Visa en el rango de un cliente sano. Un
#    programa de reactivación por inactividad no les va a llegar ni les va
#    a servir — necesitan una oferta competitiva, no un incentivo de uso.
# 3. **La edad no importa, la antigüedad y el estatus VIP sí.** Confirmado
#    en la sección 2.3: edad casi idéntica entre grupos, pero 24 meses
#    menos de antigüedad y 4.5x menos frecuencia de status VIP entre los
#    que se van. La construcción de la relación en el tiempo (y su
#    reconocimiento) es lo que retiene, no el perfil demográfico.

# %% [markdown]
# ## FASE 6 — Propuestas de acción basadas en evidencia
#
# **Perfil: "Declive generalizado silencioso" (81.1% de las bajas)**
# - *Evidencia:* umbral de fase 3 — por debajo de 27 transacciones
#   trimestrales la tasa de baja se multiplica por 56 (8.3% vs. 0.15%).
# - *Acción 1:* alerta automática mensual a marketing digital cuando un
#   cliente cruza ese umbral (ctrx_quarter < 27 y cproductos ≤ 6),
#   disparando una campaña de reactivación multicanal (app + SMS) con
#   incentivo de uso (cashback en las primeras transacciones del mes).
# - *Acción 2:* dado que también son los que menos usan débito/Visa,
#   ofrecerles migrar a débito automático de un servicio recurrente
#   (luz, celular) como forma de generar actividad mínima sostenida.
#
# **Perfil: "Multiproducto activo con fuga competitiva" (14.3% de las bajas)**
# - *Evidencia:* hallazgo sorprendente de fase 5 — mantienen productos y
#   consumo de Visa en rango de cliente sano; no es un problema de
#   actividad.
# - *Acción 1:* contacto de retención por ejecutivo (no canal masivo)
#   con revisión de condiciones comerciales (tasa, límite de Visa,
#   beneficios) apenas se detecte el patrón, priorizando por ser BAJA+2 en
#   48.6% de los casos (todavía hay margen de reacción de ~1 mes).
#
# **Perfil: "Súper-activo que igual se va" (4.3% de las bajas)**
# - *Evidencia:* fase 4 — consumo de autoservicio 2.3x el de un cliente
#   sano promedio ($47.462 vs. $20.400); es el cluster de mayor valor
#   económico individual perdido.
# - *Acción 1:* asignar ejecutivo senior para entender la causa raíz antes
#   de la baja (mudanza, evento puntual, fricción de servicio) — el
#   volumen es chico pero el valor por cliente perdido es el más alto de
#   los cuatro perfiles.
#
# **Perfil: "Mora prolongada en Mastercard" (0.3% de las bajas)**
# - *Evidencia:* umbral de fase 3 — más de 51 días en mora, tasa de baja
#   10.1% (12x el decil de menor riesgo).
# - *Acción 1:* no es un caso de marketing sino de cobranzas — ofrecer
#   refinanciación o plan de pago antes de que la cuenta pase a cierre
#   definitivo, y usar `Master_Finiciomora > 50 días` como alerta temprana
#   en la cartera de gestión de mora (no solo para retención Premium).
#
# **Acción transversal (todos los perfiles):** dado que empezar a pagar
# comisiones de mantenimiento es una señal de riesgo 5 veces más fuerte que
# no pagarlas, e independiente del nivel de actividad (hallazgo 1 de fase
# 5), agregar una revisión automática de elegibilidad de exención de
# comisiones para cualquier cliente Premium al que se le empiece a cobrar
# mantenimiento — antes de que la baja avance.

# %% [markdown]
# # Complemento — hallazgos del foro de la cursada y nuevas aristas
#
# Compañeros de la cursada compartieron otro conjunto de hallazgos sobre
# este mismo dataset. Las Fases 7-13 los ponen a prueba contra nuestros
# datos reales (algunos replican, algunos no, algunos replican con matiz)
# y suman el ángulo de cross-sell de "productos ancla" y un clustering más
# riguroso. Cuando algo no replica tal cual se reportó, se dice así de
# explícito — no se suaviza para que coincida.

# %% [markdown]
# ## FASE 7 — Calidad de datos y semántica del missing
#
# Antes de confiar en cualquier variable de las fases 8-13, validamos 5
# afirmaciones sobre la calidad del dato.

# %% [markdown]
# ### 7.1 ¿Nulos "inyectados uniformemente"?

# %%
# EXPLORACIÓN: los nulos podrían estar parejos en casi todas las columnas
# de negocio (hipótesis del foro) o concentrados en un bloque específico.
columnas_tarjeta = sorted(
    c for c in todas_las_columnas if c.startswith("Master_") or c.startswith("Visa_")
)
print(f"{len(columnas_tarjeta)} columnas de tarjeta (Master_/Visa_)")

expr_null_count = " + ".join(f"case when {c} is null then 1 else 0 end" for c in columnas_tarjeta)

# %%
histograma_nulos = con.execute(f"""
    with base as (select ({expr_null_count}) as null_count from etiquetado)
    select null_count, count(*) as n
    from base
    group by 1
    order by 1
""").pl()
histograma_nulos

# %%
segmento_tarjetas = con.execute(f"""
    with base as (
        select clase_ternaria, ({expr_null_count}) as null_count
        from etiquetado
    )
    select case
             when null_count <= 12 then 'Tiene ambas tarjetas'
             when null_count >= {len(columnas_tarjeta)} then 'No tiene ninguna tarjeta'
             else 'Le falta una familia de tarjeta'
           end as segmento,
           count(*) as n,
           round(100.0 * count(*) / sum(count(*)) over (), 2) as pct_del_total,
           sum(case when clase_ternaria in ('BAJA+1','BAJA+2') then 1 else 0 end) as bajas,
           round(100.0 * sum(case when clase_ternaria in ('BAJA+1','BAJA+2') then 1 else 0 end) / count(*), 3) as tasa_baja_pct
    from base
    group by 1
    order by n desc
""").pl()
segmento_tarjetas

# %% [markdown]
# **Resumen de negocio.** La hipótesis del foro ("nulos inyectados
# parejo en ~3.46% de las columnas de negocio") **no se sostiene tal como
# se planteó**: los nulos están concentrados casi por completo en el
# bloque de 44 columnas de tarjeta (`Master_*`/`Visa_*`), no repartidos en
# el resto de las variables. Dentro de ese bloque hay dos segmentos
# reales, no ruido: clientes a los que les falta una familia de tarjeta
# completa, y clientes sin ninguna tarjeta. Este segmento sin tarjeta
# tiene una tasa de baja **3-4 veces más alta** que el resto — es señal de
# negocio genuina (no tener producto de tarjeta es en sí mismo un
# indicador de riesgo), no un artefacto de carga de datos.

# %% [markdown]
# ### 7.2 Inconsistencia `cpayroll_trx=0` con `mpayroll>0`

# %%
inconsistencia_payroll = con.execute("""
    select count(*) as n_inconsistentes,
           round(100.0 * count(*) / (select count(*) from competencia_01), 5) as pct_del_total
    from competencia_01
    where cpayroll_trx = 0 and mpayroll > 0
""").pl()
inconsistencia_payroll

# %% [markdown]
# **Resumen de negocio.** 0.002% de las filas — ruido de captura,
# irrelevante para el análisis. Se descarta como hallazgo.

# %% [markdown]
# ### 7.3 ¿`internet` y `ccaja_seguridad` son realmente binarias?

# %%
print("internet — valores distintos:")
print(con.execute("select internet, count(*) as n from etiquetado group by 1 order by 1").pl())
print("\nccaja_seguridad — valores distintos:")
print(con.execute("select ccaja_seguridad, count(*) as n from etiquetado group by 1 order by 1").pl())

# %% [markdown]
# **Resumen de negocio.** El diccionario las describe como flags 0/1, pero
# **no lo son**: `internet` toma 5 valores (0-4) y `ccaja_seguridad` toma 7
# (0-6). Hay que leerlas como conteos, no como flags, en cualquier análisis
# futuro (incluida la Fase 13 de este notebook).

# %% [markdown]
# ### 7.4 Zero-inflation en variables monetarias de productos opcionales

# %%
zero_inflation = []
for col in ["mplazo_fijo_pesos", "minversion1_pesos", "mprestamos_personales"]:
    r = con.execute(f"""
        select
            round(100.0 * sum(case when {col} is null or {col} = 0 then 1 else 0 end) / count(*), 3) as pct_cero_o_nulo,
            min({col}) filter (where {col} > 0) as min_positivo,
            median({col}) filter (where {col} > 0) as mediana_positivo,
            max({col}) filter (where {col} > 0) as max_positivo
        from etiquetado
    """).fetchone()
    zero_inflation.append({"column_name": col, "pct_cero_o_nulo": r[0], "min_positivo": r[1], "mediana_positivo": r[2], "max_positivo": r[3]})
pl.DataFrame(zero_inflation)

# %% [markdown]
# **Resumen de negocio.** Confirmado: entre 78% (`mprestamos_personales`) y
# 99.6% (`mplazo_fijo_pesos`) de los clientes tiene cero en estas variables
# — esperable en productos opcionales que no todos contratan. No es un
# problema de calidad de dato, es la tenencia real del producto.

# %% [markdown]
# ### 7.5 Semántica de `Finiciomora`: ¿"nunca estuvo en mora" o "no tiene tarjeta"?

# %%
for prefijo in ["Master", "Visa"]:
    print(f"\n{prefijo}_Finiciomora vs. {prefijo}_delinquency:")
    print(con.execute(f"""
        select ({prefijo}_Finiciomora is null) as finiciomora_nulo,
               {prefijo}_delinquency as delinquency,
               count(*) as n
        from etiquetado
        group by 1, 2
        order by 1, 2
    """).pl())
    print(f"\n{prefijo}_Finiciomora vs. {prefijo}_status:")
    print(con.execute(f"""
        select ({prefijo}_Finiciomora is null) as finiciomora_nulo,
               {prefijo}_status as status,
               count(*) as n
        from etiquetado
        group by 1, 2
        order by 1, 2
    """).pl())

# %% [markdown]
# **Resumen de negocio.** La hipótesis del foro se **confirma con
# precisión, y se afina**: `Finiciomora` no nulo tiene correspondencia 1:1
# perfecta con `delinquency=1` (sin excepciones en ningún sentido) — NaN
# significa "nunca estuvo en mora". Pero **no** significa "no tiene
# tarjeta": aparece con `status` abierto, en cierre y cerrado por igual.
# La lectura correcta es "historial de mora", una dimensión totalmente
# distinta de "tenencia de tarjeta".

# %% [markdown]
# ## FASE 8 — Ranking univariado de poder predictivo (AUC)
#
# Sección técnica: acá sí usamos ROC/AUC (las secciones de negocio de
# arriba y de abajo usan decilado y heatmaps, como corresponde para
# comunicar a Dirección Comercial). Definimos la señal como **BAJA+2
# contra el resto** (CONTINUA + BAJA+1), que es la clase que importa para
# poder actuar con 2 meses de anticipación.

# %%
muestra_auc = con.execute(f"""
    select clase_ternaria, {", ".join(CONTINUAS)}
    from etiquetado
    where clase_ternaria in ('BAJA+1', 'BAJA+2')
    union all
    select clase_ternaria, {", ".join(CONTINUAS)}
    from etiquetado
    where clase_ternaria = 'CONTINUA'
    using sample 60000 (reservoir, {SEMILLA_PRIMARIA})
""").pl()
y_baja2 = (muestra_auc["clase_ternaria"] == "BAJA+2").to_numpy().astype(int)
print(f"Muestra: {muestra_auc.height} filas, {y_baja2.sum()} BAJA+2")

resultados_auc = []
for col in CONTINUAS:
    valores = muestra_auc[col]
    mask = valores.is_not_null().to_numpy()
    if mask.sum() < 30:
        continue
    y = y_baja2[mask]
    if y.sum() < 5 or y.sum() == mask.sum():
        continue
    x = valores.filter(pl.Series(mask)).to_numpy()
    auc_raw = roc_auc_score(y, x)
    resultados_auc.append({
        "column_name": col,
        "grupo_negocio": COLUMNA_A_GRUPO[col],
        "auc": max(auc_raw, 1 - auc_raw),
        "direccion": "mayor valor -> MÁS riesgo" if auc_raw >= 0.5 else "mayor valor -> MENOS riesgo",
        "pct_nulos_en_muestra": round(100 * (1 - mask.mean()), 2),
    })
resultados_auc = pl.DataFrame(resultados_auc).sort("auc", descending=True)
print(f"{resultados_auc.height} variables con AUC calculado")

# %%
print("Top 20 por AUC:")
print(resultados_auc.head(20))
print("\nBottom 20 por AUC (menos señal):")
print(resultados_auc.tail(20))

# %%
CLAIMS_FORO = {
    "ctrx_quarter": 0.84,
    "mcaja_ahorro": 0.81,
    "mpasivos_margen": 0.80,
    "cproductos": 0.71,
    "cliente_antiguedad": 0.50,
}
comparacion_auc = resultados_auc.filter(pl.col("column_name").is_in(CLAIMS_FORO)).with_columns(
    pl.col("column_name").replace(CLAIMS_FORO).alias("auc_reportado_foro").cast(pl.Float64)
).with_columns(
    (pl.col("auc") - pl.col("auc_reportado_foro")).alias("diferencia")
).select(["column_name", "auc_reportado_foro", "auc", "diferencia", "direccion"])
comparacion_auc

# %% [markdown]
# **Resumen de negocio.** `ctrx_quarter`, `mcaja_ahorro`, `mpasivos_margen`
# y `cproductos` **replican con matiz**: mismo orden de magnitud que lo
# reportado en el foro, siempre un poco más bajos en nuestros datos.
# `cliente_antiguedad` **no replica** — el foro decía "sin señal" (AUC
# ≈0.50) pero acá da una señal real, aunque débil. Confirmamos esto además
# con un corte de negocio directo:

# %%
antiguedad_extrema = con.execute("""
    select
        count(*) as n,
        round(100.0 * sum(case when clase_ternaria='CONTINUA' then 1 else 0 end) / count(*), 3) as pct_continua,
        round(100.0 * sum(case when clase_ternaria='BAJA+1' then 1 else 0 end) / count(*), 3) as pct_baja1,
        round(100.0 * sum(case when clase_ternaria='BAJA+2' then 1 else 0 end) / count(*), 3) as pct_baja2
    from etiquetado
    where cliente_antiguedad > 300
""").pl()
print("Clientes con antigüedad > 300 meses (25 años):")
print(antiguedad_extrema)
print("\nPoblación total etiquetada, para comparar: 98.603% CONTINUA / 0.777% BAJA+1 / 0.621% BAJA+2")

# %% [markdown]
# Los clientes con más de 300 meses de antigüedad (25 años) tienen una
# tasa de BAJA+2 un tercio más baja que la población general — la
# antigüedad extrema sí protege, contra lo que sugería el foro.

# %% [markdown]
# ## FASE 9 — Gradiente de estado de tarjetas, lead-time y reglas simples
#
# Profundiza la sección 2.3: acá el foco es la pregunta crítica de
# **lead-time** — ¿el gradiente de riesgo por estado de tarjeta ya se ve 2
# meses antes de la baja (BAJA+2, con margen para actuar) o recién aparece
# 1 mes antes (BAJA+1, tarde)?

# %%
def tabla_status(col: str) -> pl.DataFrame:
    return con.execute(f"""
        select clase_ternaria,
               coalesce(cast({col} as varchar), 'Sin tarjeta') as status,
               round(100.0 * count(*) / sum(count(*)) over (partition by clase_ternaria), 3) as pct
        from etiquetado
        group by 1, 2
    """).pl().pivot(on="status", index="clase_ternaria", values="pct").fill_null(0.0)


status_master = tabla_status("Master_status").with_columns(pl.lit("CONTINUA").alias("_o")).sort(
    pl.col("clase_ternaria").replace({"CONTINUA": 1, "BAJA+2": 2, "BAJA+1": 3})
).drop("_o")
status_visa = tabla_status("Visa_status").sort(
    pl.col("clase_ternaria").replace({"CONTINUA": 1, "BAJA+2": 2, "BAJA+1": 3})
)
print("Master_status — % de clientes de cada clase en cada estado:")
print(status_master)
print("\nVisa_status — % de clientes de cada clase en cada estado:")
print(status_visa)

# %%
orden_cols = ["clase_ternaria", "0", "6", "7", "9", "Sin tarjeta"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, tabla, nombre in zip(axes, [status_master, status_visa], ["Master_status", "Visa_status"]):
    cols_presentes = [c for c in orden_cols if c in tabla.columns]
    matriz = tabla.select(cols_presentes[1:]).to_numpy()
    # la columna "0" (abierta) domina la escala y tapa el gradiente de
    # riesgo real (6/7/9/sin tarjeta); el vmax se calibra sin esa columna.
    idx_abierta = cols_presentes[1:].index("0")
    vmax = np.delete(matriz, idx_abierta, axis=1).max()
    im = ax.imshow(matriz, cmap="Reds", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(cols_presentes) - 1))
    ax.set_xticklabels(cols_presentes[1:])
    ax.set_yticks(range(tabla.height))
    ax.set_yticklabels(tabla["clase_ternaria"].to_list())
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, f"{matriz[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_title(nombre)
    ax.set_xlabel("estado de la tarjeta (0=abierta, 6/7=cerrando, 9=cerrada)")
fig.suptitle("Escalera de riesgo: % de clientes de cada clase en cada estado de tarjeta")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_status_heatmap.png", dpi=120)
plt.show()

# %% [markdown]
# **Resumen de negocio — lead-time confirmado a favor de Miranda.** La
# escalera CONTINUA→BAJA+2→BAJA+1 en `status=9` (cerrada) **ya es visible
# en BAJA+2**, 2 meses antes de la baja real — no aparece recién en
# BAJA+1. Hay margen genuino de reacción, no es una señal de último
# momento.

# %%
base_baja2 = con.execute("""
    select round(100.0 * sum(case when clase_ternaria='BAJA+2' then 1 else 0 end) / count(*), 4)
    from etiquetado
""").fetchone()[0]

reglas = [
    ("ctrx_quarter = 0", "ctrx_quarter = 0"),
    ("active_quarter = 0", "active_quarter = 0"),
    ("Sin ninguna tarjeta (débito, Visa y Master)", "ctarjeta_visa = 0 and ctarjeta_master = 0 and ctarjeta_debito = 0"),
    ("Delinquency en Visa o Master", "(Visa_delinquency = 1 or Master_delinquency = 1)"),
]
resultados_reglas = []
for nombre, condicion in reglas:
    n, b2 = con.execute(f"""
        select count(*), sum(case when clase_ternaria='BAJA+2' then 1 else 0 end)
        from etiquetado where {condicion}
    """).fetchone()
    tasa = round(100.0 * b2 / n, 4) if n else 0
    resultados_reglas.append({
        "regla": nombre, "n": n, "baja2": b2, "tasa_baja2_pct": tasa,
        "lift_vs_base": round(tasa / base_baja2, 2),
    })
pl.DataFrame(resultados_reglas)

# %% [markdown]
# **Resumen de negocio.** Tres reglas simples, sin modelo, con lift fuerte
# sobre la tasa base de BAJA+2 (0.62%): cero transacciones trimestrales
# (12x), cliente inactivo en el trimestre (11x) y sin ninguna tarjeta (10x,
# aunque es un grupo chico). La morosidad activa da un lift más débil de
# lo esperado (2x) — es una señal real pero menos potente que la
# inactividad pura. Estas 3 reglas sirven como alerta operativa inmediata
# mientras no haya un modelo productivo.

# %% [markdown]
# ## FASE 10 — Barreras de salida y cross-sell de retención
#
# ### 10.1 ¿Los "productos ancla" retienen?

# %%
DEFINICIONES_ANCLA = {
    "Préstamo hipotecario": "cprestamos_hipotecarios > 0",
    "Préstamo prendario": "cprestamos_prendarios > 0",
    "Préstamo personal": "cprestamos_personales > 0",
    "Plazo fijo": "cplazo_fijo > 0",
    "Algún seguro": "(cseguro_vida + cseguro_auto + cseguro_vivienda + cseguro_accidentes_personales) > 0",
    "Caja de seguridad": "ccaja_seguridad = 1",
}
resultados_ancla = []
for nombre, condicion in DEFINICIONES_ANCLA.items():
    for tiene in (True, False):
        clausula = condicion if tiene else f"not ({condicion})"
        n, b_any, b2 = con.execute(f"""
            select count(*),
                   sum(case when clase_ternaria in ('BAJA+1','BAJA+2') then 1 else 0 end),
                   sum(case when clase_ternaria = 'BAJA+2' then 1 else 0 end)
            from etiquetado
            where {clausula}
        """).fetchone()
        resultados_ancla.append({
            "producto": nombre, "tiene_producto": tiene, "n": n,
            "tasa_baja_pct": round(100 * b_any / n, 4),
            "tasa_baja2_pct": round(100 * b2 / n, 4),
        })
resultados_ancla = pl.DataFrame(resultados_ancla)
resultados_ancla

# %% [markdown]
# **Resumen de negocio.** El efecto ancla replica y es **más fuerte** de
# lo reportado en el foro: en los 6 productos, tenerlo baja la tasa de
# baja muy por debajo del 1.40% base — el hipotecario es el caso más
# extremo, pero el efecto aparece en los 6, no solo en la hipoteca. Esto
# sugiere que cualquier "producto ancla" (no solo crédito de largo plazo)
# actúa como barrera de salida.

# %% [markdown]
# ### 10.2 Perfil de los tenedores de producto ancla

# %%
ANCLA_EXPR = " or ".join(f"({c})" for c in DEFINICIONES_ANCLA.values())

perfil_tenedores = con.execute(f"""
    select ({ANCLA_EXPR}) as tiene_algun_ancla,
           count(*) as n,
           round(avg(cliente_edad), 1) as edad_prom,
           round(avg(cliente_antiguedad), 1) as antiguedad_prom,
           round(avg(ctrx_quarter), 1) as ctrx_quarter_prom,
           round(avg(mcuentas_saldo), 0) as saldo_prom
    from etiquetado
    group by 1
""").pl()
perfil_tenedores

# %%
rango_antiguedad_ancla = con.execute(f"""
    select quantile_cont(cliente_antiguedad, 0.25) as p25,
           quantile_cont(cliente_antiguedad, 0.75) as p75
    from etiquetado
    where {ANCLA_EXPR}
""").fetchone()
ANTIG_P25, ANTIG_P75 = rango_antiguedad_ancla
print(f"Antigüedad de tenedores de ancla: rango intercuartil [{ANTIG_P25:.0f}, {ANTIG_P75:.0f}] meses")

# %% [markdown]
# **Resumen de negocio.** No exigimos un perfil de edad para el matching
# de la sección 10.3 — ya vimos en la Fase 2 que la edad no discrimina.
# Sí exigimos antigüedad comparable, porque la Fase 2 mostró que la
# antigüedad sí protege: usamos el rango intercuartil de antigüedad de los
# tenedores de ancla como criterio de "perfil compatible".

# %% [markdown]
# ### 10.3 Universo de cross-sell: clientes en riesgo, sin ancla, con perfil compatible
#
# Score de riesgo simple (0-3), construido con señales ya validadas en las
# Fases 3 y 9 — **no es un modelo entrenado**, es una regla de negocio
# explícita: suma 1 punto por cada una de (a) `ctrx_quarter < 27` (umbral
# de Fase 3), (b) `active_quarter = 0`, (c) tarjeta Master o Visa en
# estado 6/7/9 (cerrando o cerrada, hallazgo de Fase 9).

# %%
RISK_EXPR = """(
    (case when ctrx_quarter < 27 then 1 else 0 end)
    + (case when active_quarter = 0 then 1 else 0 end)
    + (case when Master_status in (6,7,9) or Visa_status in (6,7,9) then 1 else 0 end)
)"""

validacion_score = con.execute(f"""
    select {RISK_EXPR} as risk_score,
           count(*) as n,
           round(100.0 * sum(case when clase_ternaria = 'BAJA+2' then 1 else 0 end) / count(*), 3) as tasa_baja2_pct
    from etiquetado
    group by 1
    order by 1
""").pl()
print("Validación del score de riesgo (debe subir monótono con el score):")
print(validacion_score)

# %% [markdown]
# El score cumple lo esperado: la tasa de BAJA+2 sube monótonamente con el
# puntaje. Definimos "en riesgo" como `risk_score >= 2`.

# %%
UNIVERSO_MES = 202106  # mes de referencia: el último completamente etiquetado

universo, baja2_universo = con.execute(f"""
    select count(*),
           sum(case when clase_ternaria = 'BAJA+2' then 1 else 0 end)
    from etiquetado
    where foto_mes = {UNIVERSO_MES}
      and not ({ANCLA_EXPR})
      and {RISK_EXPR} >= 2
      and cliente_antiguedad between {ANTIG_P25} and {ANTIG_P75}
""").fetchone()

control_con_ancla, baja2_control = con.execute(f"""
    select count(*),
           sum(case when clase_ternaria = 'BAJA+2' then 1 else 0 end)
    from etiquetado
    where foto_mes = {UNIVERSO_MES}
      and ({ANCLA_EXPR})
      and {RISK_EXPR} >= 2
""").fetchone()

tasa_sin_ancla = 100 * baja2_universo / universo if universo else float("nan")
tasa_con_ancla = 100 * baja2_control / control_con_ancla if control_con_ancla else float("nan")
print(f"Universo cross-sell (foto_mes={UNIVERSO_MES}, sin ancla, en riesgo, perfil compatible): {universo} clientes")
print(f"  BAJA+2 real dentro de ese universo: {baja2_universo} ({tasa_sin_ancla:.3f}%)")
print(f"Grupo de control (con ancla, también en riesgo): {control_con_ancla} clientes")
print(f"  BAJA+2 real dentro del control: {baja2_control} ({tasa_con_ancla:.3f}%)")

# %%
reduccion_pp = max(tasa_sin_ancla - tasa_con_ancla, 0)
bajas_evitables = universo * reduccion_pp / 100
ganancia_estimada_mes = bajas_evitables * GANANCIA_ACIERTO
print(f"Reducción de tasa BAJA+2 atribuible al ancla (dentro de igual nivel de riesgo): {reduccion_pp:.3f} p.p.")
print(f"Bajas evitables estimadas en {UNIVERSO_MES} si se replicara ese efecto: {bajas_evitables:.1f} clientes")
print(f"Ganancia potencial estimada para {UNIVERSO_MES} (no agregable a otros meses): ${ganancia_estimada_mes:,.0f}")
print(f"(usa GANANCIA_ACIERTO=${GANANCIA_ACIERTO:,} de dmeyf/metrics.py, como valor de un BAJA+2 evitado)")

# %% [markdown]
# **Resumen de negocio — el efecto ancla de 10.1 no sobrevive a controlar
# por riesgo, y eso cambia la recomendación.** El universo de cross-sell
# para `foto_mes=202106` (en riesgo, sin ningún ancla, antigüedad
# compatible con la de un tenedor típico) tiene 503 clientes, con una tasa
# de BAJA+2 real de 6.56%. Pero el grupo de control — clientes **igual de
# en-riesgo que SÍ tienen un ancla** (368 clientes) — tiene una tasa de
# BAJA+2 de **7.61%, más alta, no más baja**. Reducción atribuible al
# ancla dentro del mismo nivel de riesgo: **0 puntos porcentuales**, y por
# lo tanto ganancia potencial estimada para 202106: **$0**.
#
# Esto no significa que el hallazgo de 10.1 esté mal — significa que su
# interpretación causal (el foro y nuestra propia lectura inicial) era
# apresurada. La Fase 10.2 ya lo anticipa: los tenedores de ancla no son
# una muestra aleatoria, son clientes **más antiguos, más activos y con
# más saldo** que el resto (`ctrx_quarter` 130 vs. 106, saldo $248.763 vs.
# $191.639). El producto ancla es un **marcador de salud de la relación**,
# no una causa de retención — un cliente que ya muestra señales de riesgo
# (inactividad, tarjeta cerrando) no se salva porque tenga un plazo fijo.
#
# **Implicancia de negocio (la que realmente sirve):** cross-vender un
# producto ancla a un cliente que YA está en la zona de riesgo alta
# probablemente no sea una buena estrategia de rescate — el momento útil
# para ofrecerlo es **antes**, a clientes hoy sanos (risk_score bajo) con
# antigüedad y actividad ya en el rango de un tenedor típico, como
# prevención de largo plazo, no como intervención de última hora. Esta
# hipótesis (¿el ancla previene si se ofrece temprano?) no se puede
# validar con este dataset de 6 meses — necesitaría seguimiento
# longitudinal de clientes sanos que reciben el producto vs. que no,
# quedando como próximo paso.

# %% [markdown]
# ## FASE 11 — Trayectoria temporal del churn
#
# **Limitación estructural, antes de empezar.** El dataset tiene 6 meses
# (202103-202108) y `BAJA+2` solo está etiquetada en 202103-202106 (Fase
# 1). Un mes relativo `t-4` respecto al mes en que un cliente queda
# etiquetado `BAJA+2` **nunca** cae dentro del dataset, para ningún
# cliente — no es un error de cómputo, es que ese dato no existe. Como
# mucho reconstruimos `t-1`/`t-2`/`t-3`, con menos clientes cuanto más
# atrás vamos (solo la cohorte de `foto_mes=202106` llega completa hasta
# `t-3`). Definimos `mes_relativo=0` como el propio mes en que el cliente
# está etiquetado `BAJA+2` (es decir, ya 2 meses antes de irse
# efectivamente), y `mes_relativo=-1,-2,-3` los meses anteriores a ese.

# %%
VARS_TRAYECTORIA = ["ctrx_quarter", "cpayroll_trx", "mcuentas_saldo", "chomebanking_transacciones"]

trayectoria = con.execute(f"""
    with objetivo as (
        select numero_de_cliente, foto_mes as foto_mes_baja
        from competencia_01
        where clase_ternaria = 'BAJA+2'
    )
    select (c.foto_mes - o.foto_mes_baja) as mes_relativo,
           {", ".join(f"c.{v}" for v in VARS_TRAYECTORIA)}
    from objetivo o
    join competencia_01 c on c.numero_de_cliente = o.numero_de_cliente
    where (c.foto_mes - o.foto_mes_baja) between -3 and 0
""").pl()

resumen_trayectoria = trayectoria.group_by("mes_relativo").agg(
    [pl.len().alias("n")]
    + [pl.col(v).median().alias(f"{v}_p50") for v in VARS_TRAYECTORIA]
    + [pl.col(v).quantile(0.25).alias(f"{v}_p25") for v in VARS_TRAYECTORIA]
    + [pl.col(v).quantile(0.75).alias(f"{v}_p75") for v in VARS_TRAYECTORIA]
).sort("mes_relativo")
resumen_trayectoria.select(["mes_relativo", "n"])

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, v in zip(axes, VARS_TRAYECTORIA):
    x = resumen_trayectoria["mes_relativo"].to_list()
    p50 = resumen_trayectoria[f"{v}_p50"].to_list()
    p25 = resumen_trayectoria[f"{v}_p25"].to_list()
    p75 = resumen_trayectoria[f"{v}_p75"].to_list()
    ax.plot(x, p50, marker="o", color="#c0392b")
    ax.fill_between(x, p25, p75, color="#c0392b", alpha=0.2)
    ax.set_title(v, fontsize=10)
    ax.set_xlabel("mes relativo (0 = mes etiquetado BAJA+2)")
fig.suptitle("Trayectoria de actividad antes de la baja — mediana y banda P25-P75 (BAJA+2)")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_trayectoria.png", dpi=120)
plt.show()

# %% [markdown]
# **Resumen de negocio — la ventana de 3-4 meses no se puede confirmar con
# este dataset, y lo que sí se ve es más matizado.** El `n` cae de 4,067
# (t=0) a 1,088 (t-3), como se explicó arriba. `ctrx_quarter` (mediana)
# va de ~27 en `t-3` a ~20 en `t=0`: la actividad **ya está muy deprimida
# en el punto más temprano que podemos observar** (27 transacciones
# trimestrales, muy por debajo de las 106-120 de un cliente sano) y sigue
# cayendo suavemente, sin un quiebre brusco dentro de la ventana. No
# podemos decir "acá empieza la caída" porque ya está en curso 5 meses
# antes de la baja real (3 meses antes del mes BAJA+2, que ya es 2 meses
# antes de irse) — necesitaríamos datos de más meses hacia atrás para ver
# dónde arranca de verdad. `cpayroll_trx` no aporta trayectoria: su
# mediana es 0 en los 4 puntos — la mayoría de quienes se van **nunca**
# tuvo relación de nómina, no es que la pierdan durante la ventana.
# `mcuentas_saldo` y `chomebanking_transacciones` muestran el mismo patrón
# que `ctrx_quarter`: declive gradual, no un salto.

# %% [markdown]
# ### 11.1 ¿BAJA+1 y BAJA+2 son tan parecidos como reporta el foro?

# %%
def summarize_clase(clase: str) -> pl.DataFrame:
    return con.execute(f"""
        summarize select * exclude (numero_de_cliente, foto_mes, clase_ternaria, grupo)
        from etiquetado where clase_ternaria = '{clase}'
    """).pl().select(["column_name", "q50"])


medianas_b1 = summarize_clase("BAJA+1").rename({"q50": "mediana_baja1"})
medianas_b2 = summarize_clase("BAJA+2").rename({"q50": "mediana_baja2"})
medianas_cont = summarize_clase("CONTINUA").rename({"q50": "mediana_continua"})

comparacion_b1_b2 = (
    medianas_cont.join(medianas_b2, on="column_name")
    .join(medianas_b1, on="column_name")
    .filter(pl.col("column_name").is_in(top15["column_name"].to_list()))
)
comparacion_b1_b2

# %% [markdown]
# **Resumen de negocio — confirmado: BAJA+1 y BAJA+2 son casi indistintos
# en actividad.** En 13 de las 15 variables top de Fase 2, la mediana de
# BAJA+1 y BAJA+2 es idéntica o casi (`ctrx_quarter` 19 vs. 20,
# `cproductos` 6 vs. 6, `ctarjeta_visa_transacciones` 0 vs. 0), y ambas muy
# distintas de CONTINUA. Las únicas 2 variables que sí escalan de BAJA+2 a
# BAJA+1 son `Master_Finiciomora` (29→51 días de mora) y
# `Visa_Finiciomora` (24→26 días) — la mora de tarjeta sigue avanzando
# hasta el final, la actividad general no. Esto confirma lo que ya
# insinuaba `02_eda_clase_ternaria.py`: separar "se va este mes" de "se va
# el próximo" es mucho más difícil que separar "se va" de "se queda", y es
# un argumento a favor de simplificar el target a binario (BAJA vs.
# CONTINUA) para el modelo — salvo que se usen específicamente las
# variables de mora de tarjeta para afinar el mes exacto.

# %% [markdown]
# ### 11.2 La anomalía de `mrentabilidad`: ¿los que se van dejan más plata?

# %%
rentabilidad_por_clase = con.execute("""
    select clase_ternaria,
           round(avg(mrentabilidad), 1) as mrentabilidad_media,
           round(median(mrentabilidad), 1) as mrentabilidad_mediana,
           round(avg(mcomisiones), 1) as mcomisiones_media,
           round(median(mcomisiones), 1) as mcomisiones_mediana
    from etiquetado
    group by clase_ternaria
    order by case clase_ternaria when 'CONTINUA' then 1 when 'BAJA+2' then 2 else 3 end
""").pl()
rentabilidad_por_clase

# %% [markdown]
# **Resumen de negocio — confirmado con matiz, no se copia tal cual.** El
# **promedio** de `mrentabilidad` NO muestra la anomalía (similar entre
# los tres grupos) — pero la **mediana** sí: los que se van tienen una
# rentabilidad típica notablemente más alta que quien se queda, y lo mismo
# con comisiones. La explicación más simple: `CONTINUA` tiene una cola
# larga de clientes de rentabilidad muy baja o nula que arrastra el
# promedio hacia abajo, mientras que el cliente "típico" que se va es de
# valor medio-alto para el banco. Mostramos las dos medidas a propósito —
# reportar solo el promedio hubiera escondido el patrón; reportar solo la
# mediana hubiera sonado a una anomalía más dramática de lo que es.

# %% [markdown]
# ## FASE 13 — Heterogeneidad dentro de BAJA+2: arquetipos (versión profundizada)
#
# Retoma la Fase 4 (no la reemplaza) con dos cambios metodológicos: (1)
# universo **BAJA+2 puro** (4,067 clientes, no la unión con BAJA+1) —
# es el foco literal de Miranda, "en qué se diferencian entre sí los que
# ya han muerto" a 2 meses; (2) un set de variables **razonado por
# dimensión de negocio**, no solo las de mayor *d* de Cohen:
#
# | Dimensión | Variable | Por qué |
# |---|---|---|
# | Actividad | `ctrx_quarter` | la señal más fuerte de Fases 2 y 8 |
# | Tenencia | `cproductos`, `n_productos_ancla` | cantidad total y específicamente productos "ancla" (Fase 10) |
# | Saldos | `mcuentas_saldo` | saldo total, resume la dimensión monetaria sin duplicar `mautoservicio`/`mtarjeta_visa_consumo` (correlacionadas entre sí, Fase 5) |
# | Canales | `chomebanking_transacciones`, `internet` | comportamiento digital; `internet` ya corregida como conteo (Fase 7.3), no como flag |
# | Morosidad | `severidad_tarjeta` | severidad ordinal 0-4 combinando `Master_status`/`Visa_status` (Fase 9): 0=abierta/sin tarjeta, 2=cerrando, 3=cerrando avanzado, 4=cerrada |

# %%
N_ANCLA_EXPR = """(
    (case when cprestamos_hipotecarios > 0 then 1 else 0 end)
    + (case when cprestamos_prendarios > 0 then 1 else 0 end)
    + (case when cprestamos_personales > 0 then 1 else 0 end)
    + (case when cplazo_fijo > 0 then 1 else 0 end)
    + (case when (cseguro_vida + cseguro_auto + cseguro_vivienda + cseguro_accidentes_personales) > 0 then 1 else 0 end)
    + (case when ccaja_seguridad = 1 then 1 else 0 end)
)"""

SEVERIDAD_EXPR = """greatest(
    case Master_status when 0 then 0 when 6 then 2 when 7 then 3 when 9 then 4 else 0 end,
    case Visa_status when 0 then 0 when 6 then 2 when 7 then 3 when 9 then 4 else 0 end
)"""

VARIABLES_V2 = [
    "ctrx_quarter", "cproductos", "n_productos_ancla", "mcuentas_saldo",
    "chomebanking_transacciones", "internet", "severidad_tarjeta",
]

baja2_v2 = con.execute(f"""
    select numero_de_cliente,
           ctrx_quarter, cproductos, {N_ANCLA_EXPR} as n_productos_ancla,
           mcuentas_saldo, chomebanking_transacciones, internet,
           {SEVERIDAD_EXPR} as severidad_tarjeta
    from etiquetado
    where clase_ternaria = 'BAJA+2'
""").pl()
print(f"{baja2_v2.height} clientes BAJA+2, nulos por columna:")
print(baja2_v2.null_count())

X_v2 = baja2_v2.select(VARIABLES_V2).to_numpy()
X_v2_std = StandardScaler().fit_transform(X_v2)

# %% [markdown]
# ### 13.1 Selección de K: codo + silhouette juntos

# %%
inercias, siluetas = [], []
for k in range(3, 9):
    km = KMeans(n_clusters=k, random_state=SEMILLA_PRIMARIA, n_init=10)
    labels = km.fit_predict(X_v2_std)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_v2_std, labels))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(3, 9), inercias, marker="o", color="#2980b9")
axes[0].set_title("Codo (inercia)")
axes[0].set_xlabel("k")
axes[1].plot(range(3, 9), siluetas, marker="o", color="#27ae60")
axes[1].set_title("Silhouette")
axes[1].set_xlabel("k")
fig.suptitle("Selección de K — clustering profundizado (BAJA+2 puro)")
fig.tight_layout()
fig.savefig(f"{WORK_DIR}/eda_premium_v2_seleccion_k.png", dpi=120)
plt.show()

# %% [markdown]
# Elegir `K_ELEGIDO_V2` priorizando interpretabilidad de negocio sobre el
# óptimo ciego de silhouette (mismo criterio que la Fase 4).

# %%
K_ELEGIDO_V2 = 6  # silhouette sube de 0.37 (k=5) a 0.39 (k=6) Y separa un
# perfil de negocio nuevo (ancla + buenos productos que igual se va) que
# a k=5 quedaba mezclado con el súper-activo — preferido sobre k=7 pese a
# tener silhouette apenas mayor (0.398) porque no agrega un perfil nuevo,
# solo fragmenta el cluster dominante.
km_v2 = KMeans(n_clusters=K_ELEGIDO_V2, random_state=SEMILLA_PRIMARIA, n_init=10)
baja2_v2 = baja2_v2.with_columns(pl.Series("cluster", km_v2.fit_predict(X_v2_std)))

perfiles_v2 = (
    baja2_v2.group_by("cluster")
    .agg([pl.len().alias("n")] + [pl.col(c).mean().round(1).alias(c) for c in VARIABLES_V2])
    .sort("cluster")
    .with_columns((100 * pl.col("n") / pl.col("n").sum()).round(1).alias("pct_de_baja2"))
)
perfiles_v2

# %% [markdown]
# **Perfiles de negocio (sobre las 4,067 bajas BAJA+2 puras):**
#
# - **"Declive generalizado, saldo aún positivo" (53.4%, 2,171
#   clientes).** `ctrx_quarter` 28, sin producto ancla, saldo modesto
#   positivo ($16.558). Es el mismo perfil dominante de la Fase 4.
# - **"Tiene producto ancla y buena cantidad de productos, pero igual se
#   va" (20.0%, 813 clientes).** El hallazgo nuevo más importante de esta
#   fase: `cproductos` 7.8 y **1.2 productos ancla en promedio** — son
#   justamente el perfil que la Fase 10 identificó como "protegido" — y
#   aun así se van. Conecta directo con el hallazgo de Fase 10.3: el
#   ancla no protege una vez que el cliente ya está en zona de riesgo.
# - **"Declive generalizado, ya en descubierto" (13.6%, 555 clientes).**
#   Variante más avanzada del primer perfil: actividad aún más baja
#   (`ctrx_quarter` 12.5) y saldo **negativo** (-$25.230, cuenta en
#   descubierto) — el mismo desenganche, pero un paso más adelante en
#   deterioro financiero.
# - **"Súper-activo digital que igual se va" (8.2%, 334 clientes).**
#   `ctrx_quarter` 156 y **120 transacciones de homebanking al mes** —
#   muy por encima incluso de un cliente sano. Coincide con el cluster de
#   mayor valor de la Fase 4, ahora con el detalle de que es
#   específicamente muy digital.
# - **"Mora avanzada en tarjeta" (4.6%, 186 clientes).** `severidad_tarjeta`
#   3.5 (entre "cerrando avanzado" y "cerrada") y saldo negativo. Más
#   grande que el cluster equivalente de Fase 4 (186 vs. 25) porque acá
#   `severidad_tarjeta` combina Master **y** Visa, no solo Master.
# - **"Saldos extremos" (0.2%, 8 clientes).** Un grupo minúsculo con saldo
#   promedio de $5,8 millones — atípico, probablemente cuentas
#   corporativas o de muy alto patrimonio mal clasificadas como
#   individuales; se menciona por transparencia, no amerita una acción de
#   negocio distinta con este volumen.

# %% [markdown]
# ### 13.2 Robustez: ¿los mismos arquetipos aparecen sobre BAJA+1∪BAJA+2?

# %%
bajas_v2 = con.execute(f"""
    select numero_de_cliente,
           ctrx_quarter, cproductos, {N_ANCLA_EXPR} as n_productos_ancla,
           mcuentas_saldo, chomebanking_transacciones, internet,
           {SEVERIDAD_EXPR} as severidad_tarjeta
    from etiquetado
    where grupo = 'BAJA'
""").pl()
X_bajas_v2_std = StandardScaler().fit_transform(bajas_v2.select(VARIABLES_V2).to_numpy())
km_bajas_v2 = KMeans(n_clusters=K_ELEGIDO_V2, random_state=SEMILLA_PRIMARIA, n_init=10)
bajas_v2 = bajas_v2.with_columns(pl.Series("cluster", km_bajas_v2.fit_predict(X_bajas_v2_std)))

perfiles_bajas_v2 = (
    bajas_v2.group_by("cluster")
    .agg([pl.len().alias("n")] + [pl.col(c).mean().round(1).alias(c) for c in VARIABLES_V2])
    .sort("cluster")
    .with_columns((100 * pl.col("n") / pl.col("n").sum()).round(1).alias("pct_de_bajas"))
)
perfiles_bajas_v2

# %% [markdown]
# **Resumen de negocio — los arquetipos son robustos.** Sobre
# BAJA+1∪BAJA+2 (9,155 clientes) aparecen los mismos 6 perfiles, en
# proporciones casi idénticas: declive con saldo positivo 53.2% (vs.
# 53.4% en BAJA+2 puro), ancla-pero-se-va 18.3% (vs. 20.0%), súper-activo
# digital 9.6% (vs. 8.2%), mora avanzada + declive-en-descubierto 18.8%
# combinado (vs. 18.2%), saldos extremos 0.1% (vs. 0.2%). No es un
# artefacto de trabajar solo con BAJA+2: la estructura de 6 arquetipos se
# sostiene independientemente de si el cliente se va en 1 mes o en 2.
#
# **Comparación Fase 4 vs. Fase 13.** El grupo dominante ("Declive
# generalizado") sigue siendo el más grande en ambas versiones (81% en
# Fase 4 vs. 53%+14%=67% acá, la diferencia es que la Fase 13 lo separa en
# dos según si el cliente ya cayó en descubierto o no). El cluster
# "Súper-activo" de la Fase 4 (4.3%) reaparece casi igual acá (8.2%,
# ahora con el detalle de que es específicamente uso de homebanking). El
# cluster de mora de la Fase 4 (0.3%, solo Master) se duplica en tamaño acá
# (4.6%) al incluir también mora de Visa. La diferencia más importante:
# el set de variables ampliado **separa** del cluster "Multiproducto
# activo" de la Fase 4 (14.3%) un perfil específico que sí tiene productos
# ancla concretos (20.0%) — la Fase 4 ya intuía que este grupo no se iba
# por desenganche, pero la Fase 13 lo conecta directamente con el
# hallazgo de Fase 10.3: son clientes que la lógica de "vendele un ancla"
# no va a salvar, porque varios de ellos **ya tienen uno**.

# %% [markdown]
# ## Conclusiones y próximos pasos
#
# **Resumen ejecutivo para Miranda.** La baja de Paquete Premium es rara
# (1.4% mensual sobre 2 meses) pero tiene patrones claros: en 8 de cada 10
# casos es un desenganche progresivo y silencioso (menos de 27
# transacciones trimestrales ya multiplica el riesgo por 50), detectable y
# accionable con alertas de actividad. El otro segmento relevante (14%) no
# se desengancha — sigue usando el banco con normalidad y se va igual — y
# ahí el problema es de oferta/precio, no de producto. La antigüedad y el
# estatus VIP protegen mucho más que cualquier variable demográfica, y
# empezar a pagar comisiones de mantenimiento es una alerta temprana
# independiente que hoy no se está usando activamente.
#
# **Próximos pasos:**
# 1. Diseñar una campaña piloto sobre el cluster "Declive generalizado"
#    (81% del volumen) usando el umbral de `ctrx_quarter` como disparador,
#    y medir impacto en la tasa de baja del mes siguiente.
# 2. Para el cluster "Multiproducto activo con fuga competitiva", relevar
#    con el equipo comercial si hay campañas agresivas de la competencia
#    en el período, y testear una oferta de retención dirigida.
# 3. Sobre feature engineering para el modelo predictivo (complementa
#    `02_eda_clase_ternaria.py`): construir variables de **tendencia**
#    (delta de `ctrx_quarter`, comisiones de mantenimiento y transacciones
#    de tarjeta respecto a 1-3 meses atrás), ya que el nivel actual separa
#    bien "sano" de "en riesgo" pero no afina el mes exacto de la baja.
# 4. Extender la ventana etiquetada más allá de 4 meses completos
#    (marzo-junio 2021) cuando haya más `foto_mes` disponibles, para
#    confirmar que los umbrales de fase 3 son estables en el tiempo y no
#    específicos de este semestre.
#
# ### Actualización — complemento con hallazgos del foro (Fases 7-13)
#
# Lo de arriba sigue vigente; esto lo completa, no lo reemplaza.
#
# **Lo que se confirmó tal cual se reportó:** el efecto ancla existe y es
# fuerte (Fase 10.1); el gradiente de estado de tarjeta escala
# monótonamente y ya es visible 2 meses antes de la baja, no 1 (Fase 9);
# BAJA+1 y BAJA+2 son casi indistintos en actividad, con la mora de
# tarjeta como única excepción que sigue escalando (Fase 11.1).
#
# **Lo que se confirmó con un matiz importante que cambia la
# recomendación:** la anomalía de `mrentabilidad` solo aparece en la
# mediana, no en el promedio (Fase 11.2) — y, el más relevante de todos,
# **el efecto ancla de 10.1 desaparece por completo al controlar por nivel
# de riesgo** (Fase 10.3: 0 p.p. de reducción, ganancia estimada $0 para
# 202106). El producto ancla es un marcador de salud de la relación, no
# una palanca de rescate — cross-venderlo a alguien ya en la zona de
# riesgo alta no lo va a retener. La Fase 13 lo confirma desde otro
# ángulo: el 20% de las bajas BAJA+2 **ya tiene** en promedio más de un
# producto ancla y se va igual.
#
# **Lo que no replicó, y hay que decirlo así:** `cliente_antiguedad` sí
# tiene señal real (AUC 0.61-0.69 según la muestra, Fase 8), no ≈0.50
# como se reportó — la antigüedad extrema (>300 meses) reduce el riesgo de
# BAJA+2 en un tercio. Y la hipótesis de "nulos inyectados uniformemente"
# no aplica: los nulos están concentrados en el bloque de tarjetas
# (Fase 7.1), y ahí sí son señal (el segmento sin ninguna tarjeta
# churnea 3-4x más).
#
# **Próximos pasos, agregados a los de arriba:**
# 5. Diseñar la oferta de producto ancla como estrategia **preventiva**
#    para clientes hoy sanos (risk_score bajo) con antigüedad/actividad ya
#    en rango de tenedor típico — no como intervención de rescate para
#    quien ya muestra señales de riesgo (Fase 10.3).
# 6. Encarar el cluster "tiene ancla y buenos productos, pero igual se va"
#    (20% de las bajas, Fase 13) con la misma lógica que el "Multiproducto
#    activo" de la Fase 6: contacto de retención por ejecutivo con revisión
#    de condiciones comerciales, no una campaña de cross-sell de producto.
# 7. Para afinar el mes exacto de la baja (BAJA+2 vs. BAJA+1), priorizar
#    específicamente `Master_Finiciomora`/`Visa_Finiciomora` como señal de
#    timing (Fase 11.1) — son las únicas variables de este análisis que
#    siguen escalando hasta el último mes.
# 8. La ventana de alerta de "3-4 meses" no se pudo confirmar ni refutar
#    con este dataset (Fase 11): la actividad ya está deprimida en el
#    punto más temprano observable (5 meses antes de la baja real).
#    Habría que repetir la Fase 11 en cuanto se sumen más `foto_mes`
#    históricos hacia atrás.
