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
# # EDA básico — competencia_01_crudo.csv
#
# Dataset: `datasets/raw/competencia_01_crudo.csv` (descargado desde la URL
# especificada en `monday/z101_target_sql.ipynb`, ver `datasets/NOTAS.md`).
# Diccionario de datos: `datasets/raw/diccionario_datos.csv`
# (`datasets/raw/DiccionarioDatos_2026.ods`).
#
# Todavía NO tiene `clase_ternaria` (se construye a partir de este crudo,
# ver `clases/z101_target_sql.ipynb`). Este EDA es exclusivamente sobre las
# variables de entrada.
#
# Se usa DuckDB para todo el trabajo pesado (el CSV pesa ~493 MB / ~983k
# filas) y solo se traen a Python resultados ya agregados.

# %%
import duckdb
import polars as pl
import matplotlib.pyplot as plt

DATA_PATH = "../datasets/raw/competencia_01_crudo.csv"
DICT_PATH = "../datasets/raw/diccionario_datos.csv"

con = duckdb.connect()
con.execute(f"""
    create or replace view competencia_01_crudo as
    select * from read_csv_auto('{DATA_PATH}')
""")

# %% [markdown]
# ## 1. Tamaño y estructura

# %%
n_rows, n_cols = con.execute("select count(*), (select count(*) from (describe competencia_01_crudo)) from competencia_01_crudo").fetchone()
n_clientes = con.execute("select count(distinct numero_de_cliente) from competencia_01_crudo").fetchone()[0]
n_periodos = con.execute("select count(distinct foto_mes) from competencia_01_crudo").fetchone()[0]
print(f"filas: {n_rows:,}")
print(f"columnas: {n_cols}")
print(f"clientes unicos: {n_clientes:,}")
print(f"periodos (foto_mes) unicos: {n_periodos}")

# %%
schema = con.execute("describe competencia_01_crudo").pl()
schema.head(20)

# %% [markdown]
# ## 2. Periodos (`foto_mes`)
#
# Cuántos clientes hay registrados en cada foto_mes. Permite ver altas/bajas
# de cartera y detectar meses incompletos o anómalos.

# %%
por_mes = con.execute("""
    select foto_mes, count(*) as cantidad_clientes
    from competencia_01_crudo
    group by foto_mes
    order by foto_mes
""").pl()
por_mes

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(por_mes["foto_mes"].cast(pl.Utf8), por_mes["cantidad_clientes"])
ax.set_title("Clientes por foto_mes — competencia_01_crudo")
ax.set_xlabel("foto_mes")
ax.set_ylabel("cantidad de clientes")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig("../work/eda_clientes_por_mes.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Valores faltantes por columna
#
# % de nulls por columna, ordenado de mayor a menor. Clave para decidir
# imputación / descarte de variables en feature engineering.

# %%
columnas = schema["column_name"].to_list()
exprs = ", ".join(
    f'round(100.0 * sum(case when "{c}" is null then 1 else 0 end) / count(*), 2) as "{c}"'
    for c in columnas
)
nulls_wide = con.execute(f"select {exprs} from competencia_01_crudo").pl()

nulls_pct = (
    nulls_wide
    .transpose(include_header=True, header_name="campo", column_names=["pct_nulls"])
    .sort("pct_nulls", descending=True)
)
nulls_pct.head(20)

# %%
top_nulls = nulls_pct.filter(pl.col("pct_nulls") > 0).head(25)
if top_nulls.height > 0:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * top_nulls.height)))
    ax.barh(top_nulls["campo"], top_nulls["pct_nulls"])
    ax.invert_yaxis()
    ax.set_xlabel("% nulls")
    ax.set_title("Top columnas con valores faltantes")
    fig.tight_layout()
    fig.savefig("../work/eda_top_nulls.png", dpi=120)
    plt.show()
else:
    print("No hay columnas con valores nulos.")

# %% [markdown]
# ## 4. Estadísticos descriptivos — variables monetarias clave
#
# Cruzamos con el diccionario de datos para elegir algunas variables `m*`
# (montos) representativas.

# %%
diccionario = pl.read_csv(DICT_PATH)
diccionario.filter(pl.col("campo").str.starts_with("m")).head(10)

# %%
vars_monto = ["mrentabilidad", "mcuentas_saldo", "mcaja_ahorro", "mtarjeta_visa_consumo", "mprestamos_personales"]
describe_monto = con.execute(f"""
    summarize select {", ".join(vars_monto)} from competencia_01_crudo
""").pl()
describe_monto.select(["column_name", "min", "max", "avg", "std", "q25", "q50", "q75"])

# %% [markdown]
# ## 5. Antigüedad y edad de los clientes

# %%
demo = con.execute("""
    select cliente_edad, cliente_antiguedad
    from competencia_01_crudo
    where foto_mes = (select max(foto_mes) from competencia_01_crudo)
""").pl()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(demo["cliente_edad"].drop_nulls(), bins=40)
axes[0].set_title("cliente_edad (último foto_mes)")
axes[1].hist(demo["cliente_antiguedad"].drop_nulls(), bins=40)
axes[1].set_title("cliente_antiguedad — meses (último foto_mes)")
fig.tight_layout()
fig.savefig("../work/eda_edad_antiguedad.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Presencia intermitente de clientes
#
# El notebook oficial de la cátedra (`clases/z101_target_sql.ipynb`) señala
# que no todos los clientes aparecen en todos los `foto_mes` (algunos entran
# y salen). Confirmamos la magnitud del fenómeno: cuántos "meses presente"
# tiene cada cliente sobre el total de períodos.

# %%
presencia = con.execute("""
    select meses_presente, count(*) as cantidad_clientes
    from (
        select numero_de_cliente, count(distinct foto_mes) as meses_presente
        from competencia_01_crudo
        group by numero_de_cliente
    )
    group by meses_presente
    order by meses_presente
""").pl()
presencia

# %% [markdown]
# ## 7. Resumen
#
# - Dataset de panel cliente-mes (`numero_de_cliente` x `foto_mes`), sin
#   `clase_ternaria` todavía.
# - Ver `../datasets/raw/diccionario_datos.csv` para el significado de cada
#   campo y `../datasets/raw/consideraciones.txt` para las notas generales
#   de la cátedra (solo clientes Paquete Premium, montos en pesos
#   argentinos, fechas relativas, etc).
# - `clase_ternaria` se construye en `exp/z101_target_sql/target_sql.py`,
#   que genera `datasets/processed/competencia_01.csv`.
