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
# # EDA orientado al target — `clase_ternaria` (competencia_01)
#
# Dataset: `datasets/processed/competencia_01.csv` (con `clase_ternaria`, ver
# `exp/z101_target_sql/target_sql.py`). Diccionario de datos:
# `datasets/raw/diccionario_datos.csv`.
#
# Objetivo: a diferencia del EDA general (`01_eda_competencia_01_crudo.py`),
# acá buscamos patrones que **separen BAJA+2 de BAJA+1 y de CONTINUA**,
# insumo directo para feature engineering. `BAJA+2` es la clase que importa
# para la ganancia del negocio (se puede actuar dos meses antes de la baja),
# así que la pregunta relevante no es "quién se da de baja" sino "quién se
# da de baja en 2 meses en vez de en 1, o de quién sigue".
#
# Se usa DuckDB para toda la agregación pesada (655k filas etiquetadas x 155
# columnas) y solo se traen a Python resultados ya agregados o muestras
# chicas para graficar.

# %%
import duckdb
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "..")
from config.semillas import SEMILLA_PRIMARIA

DATA_PATH = "../datasets/processed/competencia_01.csv"

con = duckdb.connect()
con.execute(f"""
    create or replace view competencia_01 as
    select * from read_csv_auto('{DATA_PATH}')
""")
con.execute("""
    create or replace view labeled as
    select * from competencia_01
    where clase_ternaria is not null
""")

# %% [markdown]
# ## 1. Balance de clases
#
# Los últimos 2 `foto_mes` no tienen `clase_ternaria` (no hay suficiente
# futuro para decidirla, ver `exp/z101_target_sql/target_sql.py`) y se
# excluyen de todo el análisis (`labeled`).

# %%
balance = con.execute("""
    select clase_ternaria, count(*) as n,
           round(100.0 * count(*) / sum(count(*)) over (), 3) as pct
    from labeled
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
ax.set_title("Balance de clases — clase_ternaria (escala log)")
ax.set_ylabel("cantidad de registros (log)")
fig.tight_layout()
fig.savefig("../work/eda_target_balance.png", dpi=120)
plt.show()

# %% [markdown]
# **Hallazgo 1 — desbalance extremo.** `BAJA+2` es el 0.62% de los registros
# etiquetados (4,067 de 655,169), `BAJA+1` el 0.78%. Cualquier modelo debe
# manejar este desbalance explícitamente (pesos de clase, undersampling de
# `CONTINUA`, o directamente optimizar la métrica de ganancia del negocio en
# vez de accuracy/logloss estándar).

# %%
por_mes = con.execute("""
    PIVOT labeled
    ON clase_ternaria
    USING count(numero_de_cliente)
    GROUP BY foto_mes
    ORDER BY foto_mes
""").pl()
por_mes

# %% [markdown]
# El volumen de `BAJA+1`/`BAJA+2` es estable mes a mes (~900-1100 clientes),
# sin outliers temporales fuertes dentro de la ventana etiquetada
# (202103-202107).

# %% [markdown]
# ## 2. Metodología: ¿qué variables separan mejor cada par de clases?
#
# Para cada variable numérica calculamos el promedio y desvío por clase
# (`SUMMARIZE ... WHERE clase_ternaria = ...`) y con eso el **d de Cohen**
# (diferencia de medias / desvío combinado) para dos comparaciones:
#
# - `d_b2_vs_cont`: BAJA+2 contra CONTINUA (clientes sanos).
# - `d_b2_vs_b1`: BAJA+2 contra BAJA+1 (la comparación difícil: ambos ya
#   están en proceso de baja, la diferencia es *cuándo*).
#
# `min_abs_d` = el menor de los dos en valor absoluto: una variable con
# `min_abs_d` alto separa a BAJA+2 de **ambas** clases, no solo de una.

# %%
NUMERIC_TYPES = {"BIGINT", "DOUBLE", "INTEGER"}
EXCLUDE_COLS = {"numero_de_cliente", "foto_mes", "clase_ternaria"}


def summarize_clase(clase: str) -> pl.DataFrame:
    return con.execute(f"""
        summarize select * exclude ({", ".join(EXCLUDE_COLS)})
        from labeled where clase_ternaria = '{clase}'
    """).pl().select(["column_name", "column_type", "avg", "std", "null_percentage"])


suf = lambda df, s: df.rename({c: f"{c}_{s}" for c in df.columns if c != "column_name"})

resumen = (
    suf(summarize_clase("CONTINUA"), "cont")
    .join(suf(summarize_clase("BAJA+1"), "b1"), on="column_name")
    .join(suf(summarize_clase("BAJA+2"), "b2"), on="column_name")
    .filter(pl.col("column_type_cont").is_in(NUMERIC_TYPES))
)
for c in ["avg_cont", "std_cont", "avg_b1", "std_b1", "avg_b2", "std_b2"]:
    resumen = resumen.with_columns(pl.col(c).cast(pl.Float64))

resumen = resumen.with_columns([
    (((pl.col("std_cont") ** 2 + pl.col("std_b2") ** 2) / 2).sqrt()).alias("pooled_std_cont_b2"),
    (((pl.col("std_b1") ** 2 + pl.col("std_b2") ** 2) / 2).sqrt()).alias("pooled_std_b1_b2"),
]).with_columns([
    ((pl.col("avg_b2") - pl.col("avg_cont")) / pl.col("pooled_std_cont_b2")).alias("d_b2_vs_cont"),
    ((pl.col("avg_b2") - pl.col("avg_b1")) / pl.col("pooled_std_b1_b2")).alias("d_b2_vs_b1"),
]).with_columns(
    pl.min_horizontal(pl.col("d_b2_vs_cont").abs(), pl.col("d_b2_vs_b1").abs()).alias("min_abs_d")
)

print(f"{resumen.height} variables numéricas analizadas")

# %% [markdown]
# ### 2.1 BAJA+2 vs CONTINUA — separar churners de clientes sanos

# %%
top_cont = (
    resumen.filter(pl.col("d_b2_vs_cont").is_not_nan())
    .sort(pl.col("d_b2_vs_cont").abs(), descending=True)
    .head(15)
)
top_cont.select(["column_name", "avg_cont", "avg_b1", "avg_b2", "d_b2_vs_cont"])

# %%
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#c0392b" if v < 0 else "#27ae60" for v in top_cont["d_b2_vs_cont"]]
ax.barh(top_cont["column_name"], top_cont["d_b2_vs_cont"], color=colors)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("d de Cohen (BAJA+2 - CONTINUA)")
ax.set_title("Top variables: BAJA+2 vs CONTINUA")
fig.tight_layout()
fig.savefig("../work/eda_top_d_b2_vs_continua.png", dpi=120)
plt.show()

# %% [markdown]
# ### 2.2 BAJA+2 vs BAJA+1 — el problema difícil
#
# Distinguir "se va en 2 meses" de "se va el mes que viene" es la parte que
# realmente vale para poder actuar a tiempo.

# %%
top_b1 = (
    resumen.filter(pl.col("d_b2_vs_b1").is_not_nan())
    .sort(pl.col("d_b2_vs_b1").abs(), descending=True)
    .head(15)
)
top_b1.select(["column_name", "avg_cont", "avg_b1", "avg_b2", "d_b2_vs_b1"])

# %%
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#c0392b" if v < 0 else "#27ae60" for v in top_b1["d_b2_vs_b1"]]
ax.barh(top_b1["column_name"], top_b1["d_b2_vs_b1"], color=colors)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("d de Cohen (BAJA+2 - BAJA+1)")
ax.set_title("Top variables: BAJA+2 vs BAJA+1 (la comparación difícil)")
fig.tight_layout()
fig.savefig("../work/eda_top_d_b2_vs_baja1.png", dpi=120)
plt.show()

# %% [markdown]
# **Hallazgo 2 — los efectos son un orden de magnitud más chicos acá.** Los
# `d` de la sección 2.1 llegan a ~1.1 (`ctrx_quarter`); acá el máximo es
# ~0.31 (`Master_Finiciomora`). Confirma que "está en baja" es fácil de ver
# (actividad transaccional se desploma), pero "en qué mes exacto se da de
# baja" es mucho más sutil y depende de variables de **estado de tarjeta**
# (`Master_status`, `Visa_status`, `_Finiciomora`) más que de actividad
# bruta.

# %% [markdown]
# ## 3. La historia común a ambas comparaciones: declive gradual
#
# En casi todas las variables de actividad, el orden es
# **CONTINUA > BAJA+2 > BAJA+1** (o el espejo, para variables "malas" como
# mora). Es decir: BAJA+2 no se comporta como CONTINUA — ya muestra la
# caída de actividad **dos meses antes de irse**, en un punto intermedio
# entre un cliente sano y uno que se va el mes próximo.

# %%
variables_declive = [
    "ctrx_quarter", "cproductos", "ctarjeta_visa_transacciones",
    "mtarjeta_visa_consumo", "ccomisiones_mantenimiento", "cdescubierto_preacordado",
]
tabla_declive = resumen.filter(pl.col("column_name").is_in(variables_declive)).select(
    ["column_name", "avg_cont", "avg_b2", "avg_b1"]
)
tabla_declive

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, var in zip(axes.flat, variables_declive):
    row = tabla_declive.filter(pl.col("column_name") == var)
    valores = [row["avg_cont"][0], row["avg_b2"][0], row["avg_b1"][0]]
    ax.bar(["CONTINUA", "BAJA+2\n(t-2)", "BAJA+1\n(t-1)"], valores,
           color=["#2980b9", "#e67e22", "#c0392b"])
    ax.set_title(var, fontsize=10)
fig.suptitle("Declive gradual: CONTINUA → BAJA+2 → BAJA+1", fontsize=13)
fig.tight_layout()
fig.savefig("../work/eda_declive_gradual.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Distribuciones: `ctrx_quarter` y consumo de tarjeta por clase
#
# Boxplots sobre una muestra (todas las filas de `BAJA+1`/`BAJA+2` —son
# pocas— más una muestra aleatoria reproducible de `CONTINUA`) para no traer
# 655k filas a memoria.

# %%
muestra = con.execute(f"""
    select clase_ternaria, ctrx_quarter, mtarjeta_visa_consumo, cproductos
    from labeled
    where clase_ternaria in ('BAJA+1', 'BAJA+2')
    union all
    select clase_ternaria, ctrx_quarter, mtarjeta_visa_consumo, cproductos
    from labeled
    where clase_ternaria = 'CONTINUA'
    using sample 20000 (reservoir, {SEMILLA_PRIMARIA})
""").pl()

orden = ["CONTINUA", "BAJA+2", "BAJA+1"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

datos_ctrx = [muestra.filter(pl.col("clase_ternaria") == c)["ctrx_quarter"].to_list() for c in orden]
axes[0].boxplot(datos_ctrx, tick_labels=orden, showfliers=False)
axes[0].set_title("ctrx_quarter")

datos_visa = [
    muestra.filter((pl.col("clase_ternaria") == c) & (pl.col("mtarjeta_visa_consumo") > 0))["mtarjeta_visa_consumo"].to_list()
    for c in orden
]
axes[1].boxplot(datos_visa, tick_labels=orden, showfliers=False)
axes[1].set_title("mtarjeta_visa_consumo (>0)")

datos_prod = [muestra.filter(pl.col("clase_ternaria") == c)["cproductos"].to_list() for c in orden]
axes[2].boxplot(datos_prod, tick_labels=orden, showfliers=False)
axes[2].set_title("cproductos")

fig.suptitle("Distribuciones por clase (outliers ocultos, muestra reproducible)")
fig.tight_layout()
fig.savefig("../work/eda_boxplots_actividad.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Estado de las tarjetas (`Master_status` / `Visa_status`)
#
# Del diccionario de datos: `0` abierta, `6`/`7` en proceso de cierre, `9`
# cerrada (y puede reabrirse). Miramos la distribución de estados dentro de
# cada clase.

# %%
def pct_status(col: str) -> pl.DataFrame:
    return con.execute(f"""
        select clase_ternaria,
               round(100.0*sum(case when {col}=0 then 1 else 0 end)/count(*), 2) as abierta,
               round(100.0*sum(case when {col} in (6,7) then 1 else 0 end)/count(*), 2) as en_proceso_cierre,
               round(100.0*sum(case when {col}=9 then 1 else 0 end)/count(*), 2) as cerrada,
               round(100.0*sum(case when {col} is null then 1 else 0 end)/count(*), 2) as sin_tarjeta
        from labeled
        group by clase_ternaria
        order by case clase_ternaria when 'CONTINUA' then 1 when 'BAJA+2' then 2 else 3 end
    """).pl()

status_master = pct_status("Master_status")
status_visa = pct_status("Visa_status")
print("Master_status (%):")
print(status_master)
print("\nVisa_status (%):")
print(status_visa)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, tabla, nombre in zip(axes, [status_master, status_visa], ["Master_status", "Visa_status"]):
    categorias = ["abierta", "en_proceso_cierre", "cerrada", "sin_tarjeta"]
    bottom = np.zeros(tabla.height)
    for cat, color in zip(categorias, ["#2980b9", "#f1c40f", "#c0392b", "#95a5a6"]):
        valores = tabla[cat].to_numpy()
        ax.bar(tabla["clase_ternaria"], valores, bottom=bottom, label=cat, color=color)
        bottom += valores
    ax.set_title(nombre)
    ax.set_ylabel("% de clientes")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
fig.suptitle("Estado de la tarjeta por clase")
fig.tight_layout()
fig.savefig("../work/eda_status_tarjetas.png", dpi=120)
plt.show()

# %% [markdown]
# **Hallazgo 3 — la tarjeta cerrada es la señal más nítida de "se va este
# mes".** `% cerrada` en Visa: 0.10% (CONTINUA) → 2.09% (BAJA+2) → 5.92%
# (BAJA+1). Es monotónica y casi se **triplica** entre BAJA+2 y BAJA+1: es
# de las pocas variables que sí distingue bien *cuándo* se va, no solo *si*
# se va. En cambio "no tener tarjeta activa" (`sin_tarjeta`, incluye a
# quienes nunca tuvieron) ya sube fuerte en BAJA+2 y casi no cambia hacia
# BAJA+1 — ese salto ocurre más temprano, es señal de "en riesgo" más que
# de "se va ya".

# %% [markdown]
# ## 6. Missingness de campos Visa_/Master_ como señal
#
# Los campos de consumo/pagos de tarjeta (`Visa_mconsumospesos`,
# `Visa_cconsumos`, etc.) son NULL cuando el cliente no tiene ese producto o
# no lo usó ese mes. El % de nulos por clase es, en sí mismo, una feature.

# %%
campos_visa_consumo = [c for c in resumen["column_name"] if c.startswith("Visa_") and "consumo" in c.lower()]
null_visa = resumen.filter(pl.col("column_name").is_in(campos_visa_consumo)).select(
    ["column_name", "null_percentage_cont", "null_percentage_b2", "null_percentage_b1"]
)
null_visa

# %% [markdown]
# `Visa_mconsumospesos`: 12.8% nulos en CONTINUA vs ~47% en BAJA+1 **y**
# BAJA+2 (casi idéntico entre ambas). Mismo patrón que la actividad
# transaccional: el salto ya ocurrió para cuando faltan 2 meses, y no crece
# más hacia el mes de la baja. Es una buena variable "churner sí/no", floja
# para el timing.

# %% [markdown]
# ## 7. Perfil demográfico y de antigüedad

# %%
demograf = con.execute("""
    select clase_ternaria,
           round(avg(cliente_edad), 1) as edad_prom,
           round(avg(cliente_antiguedad), 1) as antiguedad_prom_meses,
           round(100.0 * avg(cliente_vip), 2) as pct_vip,
           round(100.0 * avg(active_quarter), 2) as pct_active_quarter
    from labeled
    group by clase_ternaria
    order by case clase_ternaria when 'CONTINUA' then 1 when 'BAJA+2' then 2 else 3 end
""").pl()
demograf

# %% [markdown]
# **Hallazgo 4 — la edad no discrimina, la antigüedad y el status VIP sí.**
# `cliente_edad` es prácticamente igual en las 3 clases (~47 años): no
# sirve como feature. En cambio los clientes que se dan de baja tienen en
# promedio ~24 meses menos de antigüedad (111 vs 134.7) y son VIP 4-5 veces
# menos frecuentemente (0.07-0.08% vs 0.34%) — clientes más nuevos y no-VIP
# son más propensos a la baja, pero de nuevo sin diferenciar bien BAJA+2 de
# BAJA+1 entre sí.

# %% [markdown]
# ## 8. Conclusiones
#
# **Sobre la dificultad del problema:**
# - El desbalance es extremo: BAJA+2 es 0.62% del dataset etiquetado. Hay
#   que optimizar para la métrica de negocio (ganancia), no accuracy.
# - Separar "churner" de "cliente sano" es relativamente fácil (`d` de
#   Cohen hasta ~1.1 en `ctrx_quarter`). Separar BAJA+2 de BAJA+1 —lo que
#   realmente importa para poder actuar a tiempo— es mucho más difícil
#   (`d` máximo ~0.31): son clientes que **ya** están en proceso de baja,
#   la diferencia es solo el mes exacto.
#
# **Patrón dominante — declive gradual, no un quiebre repentino:**
# - En variables de actividad (`ctrx_quarter`, transacciones con tarjeta,
#   `cproductos`, débitos automáticos, missingness de consumo de tarjeta)
#   el orden es CONTINUA ≫ BAJA+2 > BAJA+1, pero la brecha grande está entre
#   CONTINUA y los dos grupos de baja — BAJA+2 y BAJA+1 son casi iguales
#   entre sí. Conclusión práctica: **el nivel de actividad en el mes actual
#   ya cayó dos meses antes de la baja**, así que como feature nivel-actual
#   sirve para detectar riesgo temprano pero no para afinar el mes exacto.
# - En variables de **estado de tarjeta** (`Master_status`, `Visa_status`,
#   `Finiciomora`) sí aparece una escalada monotónica más marcada hacia el
#   mes de la baja (p. ej. `% Visa cerrada`: 0.10% → 2.09% → 5.92%): estas
#   variables son las que mejor aportan a distinguir BAJA+2 de BAJA+1.
#
# **Implicancia para feature engineering (el paso siguiente):**
# - Como las variables de *nivel* actual no separan bien BAJA+2 de BAJA+1,
#   conviene construir features de **tendencia/delta** (variación de
#   `ctrx_quarter`, consumo de tarjeta, etc. respecto a 1-3 meses atrás) en
#   vez de solo usar el valor del mes corriente: la hipótesis es que la
#   *velocidad* de la caída, no el nivel, es lo que separa a quién se va ya
#   de quién se va el mes que viene.
#   Estas features no se pudieron explorar en este EDA porque requieren
#   traer el histórico por cliente (lags), fuera del alcance de esta
#   corrida — es el próximo paso natural.
# - Priorizar en el modelo: variables de estado de tarjeta y mora
#   (`Master_status`, `Visa_status`, `_Finiciomora`), variables de actividad
#   agregada (`ctrx_quarter`, `cproductos`) y antigüedad/VIP como señal de
#   riesgo de base, más que variables demográficas (`cliente_edad` no
#   aporta).
