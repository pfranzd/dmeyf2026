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
# # Feature importance (Random Forest) y EDA dirigido — BAJA+2
#
# Continuación de `02_eda_clase_ternaria.py`. Ahí el enfoque fue manual:
# elegir variables "a mano" y medir separación con d de Cohen. Acá se
# invierte el orden: se entrena un Random Forest para **identificar
# clientes BAJA+2** (el problema real de la cátedra — actuar 2 meses antes
# de la baja), se extrae el feature importance sobre las ~150 variables, y
# el EDA de las secciones siguientes profundiza específicamente sobre las
# variables que el modelo marcó como relevantes — incluyendo cruces
# bivariados que un ranking univariado no puede mostrar.
#
# Target: `es_baja2` = 1 si `clase_ternaria == 'BAJA+2'`, 0 en caso
# contrario (agrupa CONTINUA y BAJA+1 como negativo). Es la formulación
# binaria estándar del problema: importa identificar quién se da de baja en
# 2 meses, no clasificar los 3 estados por igual.

# %%
import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")
from config.semillas import SEMILLA_PRIMARIA

DATA_PATH = "../datasets/processed/competencia_01.csv"

con = duckdb.connect()
con.execute(f"""
    create or replace view labeled as
    select * exclude (numero_de_cliente, foto_mes)
    from read_csv_auto('{DATA_PATH}')
    where clase_ternaria is not null
""")

# %% [markdown]
# ## 1. Preparación de datos
#
# Imputación simple por mediana de columna (suficiente para un ranking
# exploratorio de importancia; no es el pipeline del modelo final de la
# competencia). `RandomForestClassifier` de scikit-learn no acepta nulos.

# %%
df = con.execute("select * from labeled").pl()
print(f"filas: {df.height:,}  columnas: {df.width}")

feature_cols = [c for c in df.columns if c != "clase_ternaria"]
X = df.select(feature_cols)
y = (df["clase_ternaria"] == "BAJA+2").cast(pl.Int8).to_numpy()

medianas = {c: X[c].median() for c in feature_cols}
X = X.with_columns([pl.col(c).fill_null(medianas[c]) for c in feature_cols])
X_np = X.to_numpy().astype(np.float32)

print(f"tasa de BAJA+2: {y.mean() * 100:.3f}%")

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y, test_size=0.3, random_state=SEMILLA_PRIMARIA,
    stratify=df["clase_ternaria"].to_numpy(),
)
print(f"train: {X_train.shape}   test: {X_test.shape}")

# %% [markdown]
# ## 2. Entrenamiento del Random Forest
#
# `class_weight="balanced"` compensa el desbalance extremo (0.6% positivos)
# sin necesidad de undersamplear. Profundidad y hoja mínima moderadas para
# no sobreajustar sobre una clase minoritaria tan chica.

# %%
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEMILLA_PRIMARIA,
)
rf.fit(X_train, y_train)

# %% [markdown]
# ### 2.1 Validación — ¿el modelo aprendió algo real?
#
# Antes de confiar en el feature importance hay que confirmar que el
# modelo separa señal de ruido. Con 0.6% de positivos, accuracy no dice
# nada: se mira AUC-ROC, AP (área bajo precision-recall) y precision en el
# top-K (K = cantidad real de BAJA+2 en test), que es la métrica que
# importa en la práctica: de los K clientes con score más alto, cuántos
# son realmente BAJA+2.

# %%
proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)

k = int(y_test.sum())
top_idx = np.argsort(-proba)[:k]
precision_at_k = y_test[top_idx].mean()
baseline = y_test.mean()

print(f"AUC-ROC:        {auc:.4f}")
print(f"AP (PR-AUC):    {ap:.4f}")
print(f"precision@K:    {precision_at_k:.4f}  (K={k:,})")
print(f"baseline (azar):{baseline:.4f}")
print(f"lift@K:         {precision_at_k / baseline:.1f}x")

# %% [markdown]
# **AUC ~0.89 y lift@K ~20x sobre el azar**: el modelo separa bien, aunque
# está lejos de un clasificador perfecto (AP bajo es esperable con 0.6% de
# positivos). Es una base sólida para confiar en el feature importance —
# esto es un RF exploratorio para ranking de variables, no el modelo final
# de la competencia (que debería ajustarse con Optuna/LightGBM y series de
# tiempo, no un solo `foto_mes` de test).

# %% [markdown]
# ## 3. Feature importance — listado completo
#
# Importancia de **las 152 variables**, sin recortar, ordenada de mayor a
# menor. Se guarda también a `work/rf_feature_importance_baja2.csv`.

# %%
importancia = pl.DataFrame({
    "variable": feature_cols,
    "importancia": rf.feature_importances_,
}).sort("importancia", descending=True).with_columns(
    (pl.col("importancia").cum_sum()).alias("importancia_acumulada")
)

importancia.write_csv("../work/rf_feature_importance_baja2.csv")

with pl.Config(tbl_rows=200, fmt_str_lengths=40):
    print(importancia)

# %% [markdown]
# Las 12 variables más importantes explican una fracción grande de la
# importancia total del modelo — el resto de las ~140 variables restantes
# aporta cada vez menos. Confirma que, aunque el dataset tiene muchas
# columnas, la señal está concentrada.

# %%
top_n = 30
top_importancia = importancia.head(top_n)
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(top_importancia["variable"], top_importancia["importancia"], color="#2980b9")
ax.invert_yaxis()
ax.set_xlabel("importancia (Random Forest)")
ax.set_title(f"Top {top_n} variables — feature importance para BAJA+2")
fig.tight_layout()
fig.savefig("../work/eda_rf_feature_importance_top30.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Por qué el ranking cambia respecto al EDA por d de Cohen
#
# En `02_eda_clase_ternaria.py`, variables monetarias como `mpayroll` o
# `mcaja_ahorro` **no** aparecían entre las más separadoras. El Random
# Forest las pone primeras. La razón es metodológica: el d de Cohen usa
# media y desvío, que en variables de monto muy asimétricas (con outliers
# de clientes de alto poder adquisitivo) quedan dominados por esos
# outliers — el desvío se infla y el d se aplasta. El Random Forest separa
# por umbrales (parecido a comparar medianas/percentiles), que son
# robustos a esa asimetría.

# %%
top_12 = importancia.head(12)["variable"].to_list()
print("Top 12 variables (Random Forest):")
for i, v in enumerate(top_12, 1):
    print(f"{i:2d}. {v}")

# %%
comparacion = con.execute(f"""
    select clase_ternaria,
           round(avg(mpayroll), 1) as mpayroll_media,
           round(median(mpayroll), 1) as mpayroll_mediana,
           round(100.0 * sum(case when mpayroll = 0 then 1 else 0 end) / count(*), 1) as mpayroll_pct_cero
    from labeled
    group by clase_ternaria
    order by case clase_ternaria when 'CONTINUA' then 1 when 'BAJA+2' then 2 else 3 end
""").pl()
comparacion

# %% [markdown]
# **Hallazgo nuevo — `mpayroll` (acreditación de haberes) es la variable
# más importante y el EDA anterior no la había detectado.** El 45.4% de
# `CONTINUA` tiene sueldo acreditado en el banco, contra apenas 7.8%
# (BAJA+2) y 5.1% (BAJA+1). La mediana es 0 en ambas clases de baja y
# $37,536 en `CONTINUA`. Perder o no tener la acreditación de sueldo en el
# banco es de las señales más fuertes de riesgo de baja — coherente con la
# intuición de negocio (cambiar de banco de nómina suele ser la antesala
# de mover el resto de los productos).

# %% [markdown]
# ## 5. Deep dive univariado — top 12 variables del Random Forest
#
# Medias, medianas y % de ceros por clase para las 12 variables más
# importantes (todas las que no vimos ya en el EDA anterior, más
# `ctrx_quarter` como referencia conocida).

# %%
def perfil_variable(col: str) -> pl.DataFrame:
    return con.execute(f"""
        select clase_ternaria,
               round(avg({col}), 1) as media,
               round(median({col}), 1) as mediana,
               round(100.0 * sum(case when {col} = 0 then 1 else 0 end) / count(*), 1) as pct_cero
        from labeled
        group by clase_ternaria
        order by case clase_ternaria when 'CONTINUA' then 1 when 'BAJA+2' then 2 else 3 end
    """).pl().with_columns(pl.lit(col).alias("variable"))


perfiles = pl.concat([perfil_variable(c) for c in top_12])
with pl.Config(tbl_rows=50):
    print(perfiles.select(["variable", "clase_ternaria", "media", "mediana", "pct_cero"]))

# %%
muestra = con.execute(f"""
    select clase_ternaria, {", ".join(top_12)}
    from labeled
    where clase_ternaria in ('BAJA+1', 'BAJA+2')
    union all
    select clase_ternaria, {", ".join(top_12)}
    from labeled
    where clase_ternaria = 'CONTINUA'
    using sample 15000 (reservoir, {SEMILLA_PRIMARIA})
""").pl()

orden_clases = ["CONTINUA", "BAJA+2", "BAJA+1"]
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, var in zip(axes.flat, top_12):
    datos = [muestra.filter(pl.col("clase_ternaria") == c)[var].to_list() for c in orden_clases]
    ax.boxplot(datos, tick_labels=orden_clases, showfliers=False)
    ax.set_title(var, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Top 12 variables por feature importance — distribución por clase (sin outliers)", fontsize=13)
fig.tight_layout()
fig.savefig("../work/eda_rf_top12_boxplots.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Casos bivariados
#
# Un ranking univariado (d de Cohen o feature importance por separado) no
# muestra interacciones: dos variables mediocres por separado pueden
# separar bien combinadas. Se grafican 6 cruces entre las variables más
# importantes, con una muestra reproducible (todas las bajas + una muestra
# de `CONTINUA`) para no saturar el gráfico con 646k puntos. Ejes de monto
# en escala `symlog` (los saldos/márgenes pueden ser negativos).

# %%
pares = [
    ("mpayroll", "ctrx_quarter"),
    ("mcaja_ahorro", "mcuentas_saldo"),
    ("ctarjeta_visa_transacciones", "mtarjeta_visa_consumo"),
    ("ctarjeta_debito_transacciones", "mautoservicio"),
    ("cpayroll_trx", "mpasivos_margen"),
    ("mtransferencias_recibidas", "ccomisiones_otras"),
]
variables_pares = sorted({v for par in pares for v in par})

muestra_biv = con.execute(f"""
    select clase_ternaria, {", ".join(variables_pares)}
    from labeled
    where clase_ternaria in ('BAJA+1', 'BAJA+2')
    union all
    select clase_ternaria, {", ".join(variables_pares)}
    from labeled
    where clase_ternaria = 'CONTINUA'
    using sample 8000 (reservoir, {SEMILLA_PRIMARIA})
""").pl()

MONTO_PREFIX = ("m",)  # variables de monto -> escala symlog

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, (vx, vy) in zip(axes.flat, pares):
    for clase, color, alpha, size, z in [
        ("CONTINUA", "#95a5a6", 0.25, 8, 1),
        ("BAJA+1", "#e67e22", 0.5, 12, 2),
        ("BAJA+2", "#c0392b", 0.8, 16, 3),
    ]:
        sub = muestra_biv.filter(pl.col("clase_ternaria") == clase)
        ax.scatter(sub[vx], sub[vy], c=color, alpha=alpha, s=size, label=clase, zorder=z, linewidths=0)
    if vx.startswith(MONTO_PREFIX):
        ax.set_xscale("symlog", linthresh=1000)
    if vy.startswith(MONTO_PREFIX):
        ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel(vx, fontsize=9)
    ax.set_ylabel(vy, fontsize=9)

axes.flat[0].legend(loc="upper right", fontsize=8, markerscale=2)
fig.suptitle("Casos bivariados — top variables del Random Forest", fontsize=13)
fig.tight_layout()
fig.savefig("../work/eda_rf_bivariados.png", dpi=120)
plt.show()

# %% [markdown]
# **Hallazgo — las tres clases forman capas, no clusters separados.** En
# los 6 cruces, `BAJA+2` (rojo) ocupa consistentemente una región
# intermedia entre `CONTINUA` (gris, arriba a la derecha en casi todos los
# pares — más actividad y más monto) y `BAJA+1` (naranja, en el extremo de
# baja actividad/monto). No hay una combinación de 2 variables que aísle a
# `BAJA+2` en un cluster propio: confirma, con otra técnica, la misma
# conclusión central del EDA anterior — **BAJA+2 es un estado de tránsito
# gradual entre cliente sano y baja consumada, no un perfil cualitativamente
# distinto**. El cruce `ctarjeta_visa_transacciones` x `mtarjeta_visa_consumo`
# es el más "limpio": ambas crecen juntas y CONTINUA se despega claramente,
# pero BAJA+2 y BAJA+1 siguen mezclados entre sí — de nuevo, separar el
# *cuándo* exacto de la baja necesita algo más que el nivel de estas
# variables en el mes actual.

# %% [markdown]
# ## 7. Conclusiones
#
# **Sobre el Random Forest como herramienta de exploración:**
# - AUC-ROC 0.89 y lift@K ~20x confirman que hay señal real y que el
#   ranking de importancia es confiable como guía de EDA (no como modelo
#   final: falta ajuste de hiperparámetros, y la validación es sobre un
#   solo split, no walk-forward temporal).
# - El ranking de importancia y el ranking por d de Cohen del EDA anterior
#   coinciden parcialmente (`ctrx_quarter`, `ctarjeta_visa_transacciones`,
#   `ctarjeta_debito_transacciones` aparecen en ambos) pero difieren en las
#   variables de monto: el RF es robusto a la asimetría de las variables
#   `m*` y el d de Cohen no. **Conclusión metodológica: para EDA sobre
#   variables de monto muy asimétricas conviene mirar medianas y feature
#   importance basado en árboles, no solo medias/d de Cohen.**
#
# **Hallazgo nuevo más importante — `mpayroll`/`cpayroll_trx` (nómina):**
# - Tener sueldo acreditado en el banco es la señal individual más potente
#   para identificar BAJA+2: 45.4% de `CONTINUA` la tiene vs. 7.8% de
#   `BAJA+2` y 5.1% de `BAJA+1`. No había aparecido en el EDA anterior por
#   ser una variable dominada por ceros y outliers altos.
# - Como feature de negocio es además accionable: la pérdida de
#   acreditación de haberes es visible y puede disparar una alerta temprana
#   independiente del score del modelo.
#
# **Confirma y refuerza la conclusión central del EDA anterior:**
# - Tanto en el análisis univariado (boxplots por variable) como en los
#   6 cruces bivariados, `BAJA+2` se ubica sistemáticamente **entre**
#   `CONTINUA` y `BAJA+1`, nunca en una región propia. No es un perfil de
#   cliente distinto, es un punto intermedio en una trayectoria de
#   deterioro. Ninguna combinación de nivel-actual de 1 o 2 variables
#   (incluidas las que el Random Forest más valora) separa nítidamente
#   BAJA+2 de BAJA+1.
# - Esto reafirma la recomendación de feature engineering del EDA anterior:
#   la variable que probablemente falta es la **velocidad de caída**
#   (deltas/tendencias de `mpayroll`, `ctrx_quarter`, `mcaja_ahorro`, etc.
#   respecto a 1-3 meses atrás), no más variables de nivel — que es
#   justamente lo que este Random Forest, entrenado solo con el `foto_mes`
#   actual, no puede capturar.
