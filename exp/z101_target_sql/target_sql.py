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
# # Construcción de `clase_ternaria` — competencia_01
#
# Implementación de la tarea planteada en `monday/z101_target_sql.ipynb`
# (cátedra DMEyF 2026, upstream `dmecoyfin/dmeyf2026`, commit `21402a7`).
#
# Genera `datasets/processed/competencia_01.csv` a partir de
# `datasets/raw/competencia_01_crudo.csv`, agregando la columna
# `clase_ternaria` con las categorías **CONTINUA**, **BAJA+1** y **BAJA+2**.

# %%
import duckdb
import polars as pl

RAW_PATH = "../../datasets/raw/competencia_01_crudo.csv"
OUT_PATH = "../../datasets/processed/competencia_01.csv"

con = duckdb.connect()
con.execute(f"""
    create or replace table competencia_01_crudo as
    select * from read_csv_auto('{RAW_PATH}')
""")

# %% [markdown]
# ## Target
#
# Para cada cliente y `foto_mes` se arma el panel completo cliente x período
# (`cross join`) y se mira si el cliente aparece en el mes actual (`mes_0`),
# en el siguiente (`mes_1`) y a los dos meses (`mes_2`) usando `lead` sobre
# la ventana particionada por cliente:
#
# - **CONTINUA**: sigue presente dentro de los próximos 2 meses.
# - **BAJA+2**: está el mes siguiente pero se da de baja al segundo mes.
# - **BAJA+1**: ya no está el mes siguiente.
#
# Los últimos `foto_mes` del dataset quedan con `clase_ternaria` NULL: no hay
# suficiente "futuro" en los datos para decidir la clase.

# %%
con.execute("""
    create or replace table competencia_01 as
    with periodos as (
        select distinct foto_mes from competencia_01_crudo
    ), clientes as (
        select distinct numero_de_cliente from competencia_01_crudo
    ), todo as (
        select numero_de_cliente, foto_mes from clientes cross join periodos
    ), clase_ternaria as (
        select
            c.*
            , if(c.numero_de_cliente is null, 0, 1) as mes_0
            , lead(mes_0, 1) over (partition by t.numero_de_cliente order by foto_mes) as mes_1
            , lead(mes_0, 2) over (partition by t.numero_de_cliente order by foto_mes) as mes_2
            , case
                when mes_2 = 1 then 'CONTINUA'
                when mes_1 = 1 and mes_2 = 0 then 'BAJA+2'
                when mes_1 = 0 then 'BAJA+1'
                else null
              end as clase_ternaria
        from todo t
        left join competencia_01_crudo c using (numero_de_cliente, foto_mes)
    )
    select * EXCLUDE (mes_0, mes_1, mes_2)
    from clase_ternaria
    where mes_0 = 1
""")

# %% [markdown]
# ## Verificación

# %%
n_total = con.execute("select count(*) from competencia_01").fetchone()[0]
print(f"filas en competencia_01: {n_total:,}")

# %%
pivot = con.execute("""
    PIVOT competencia_01
    ON clase_ternaria
    USING count(numero_de_cliente)
    GROUP BY foto_mes
    ORDER BY foto_mes
""").pl()
pivot

# %% [markdown]
# ## Guardar

# %%
con.execute(f"COPY competencia_01 TO '{OUT_PATH}' (FORMAT CSV, HEADER)")
print(f"Guardado: {OUT_PATH}")
