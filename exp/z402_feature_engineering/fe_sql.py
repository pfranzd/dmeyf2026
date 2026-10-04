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
# # Feature engineering en SQL sobre `competencia_01`
#
# Implementación de las técnicas presentadas en
# `monday/z402_Feature_Engineering_en_SQL.ipynb`
# (cátedra DMEyF 2026, upstream `dmecoyfin/dmeyf2026`, commit `c9ff668`).
#
# El notebook de cátedra muestra el catálogo de técnicas con queries
# exploratorias (`select ... limit 10`) y usa magics `%%sql` de jupysql. Acá se
# aplican de verdad, armando la query como **string de Python** y ejecutándola
# con `con.execute()`, igual que en `exp/z101_target_sql/target_sql.py`.
#
# Entrada : `datasets/processed/competencia_01.csv` (155 cols, ya con target)
# Salidas : `datasets/processed/competencia_01_fe.{parquet,csv}`
#           `work/z402_fe_catalogo.csv` (una fila por variable creada)
#
# ## Bloques de features
#
# | Bloque | Familia | Idea |
# |---|---|---|
# | 0 | Macros | `suma_segura`, `ratio_seguro`, `delta_pct` (null-safe, sin div/0) |
# | 1 | **Creadas a partir de otras** (intra-fila) | consolidado Master+Visa, fechas relativas, ratios de dominio, agregados, flags de riesgo |
# | 2 | **Rankings intra-mes** | `percent_rank`/`ntile` con `partition by foto_mes`: anti data-drifting |
# | 3 | **Lags** | `lag(x, 1)`, `lag(x, 2)` por cliente |
# | 4 | **Deltas** | `x - lag_n`, y delta porcentual |
# | 5 | **Ventanas móviles** | `avg`/`max`/`min`/`stddev` sobre t-3..t, y `x / avg_3` |
# | 6 | **Tendencia** | `regr_slope(x, cliente_antiguedad)` sobre la ventana |
# | 7 | **Historia en el panel** | meses observados, antigüedad relativa |

# %%
import sys
from pathlib import Path

import duckdb
import polars as pl

# La consola de Windows por defecto usa cp1252, que no puede imprimir los
# caracteres de caja (─, │, ┌...) que usa `print()` sobre un DataFrame de
# polars. Se fuerza UTF-8 en stdout/stderr para que el script corra igual
# interactivo (VS Code) o como proceso batch (`python fe_sql.py > log.txt`).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# "curado"   -> series temporales solo sobre CAMPOS_SERIE_CURADO (~36 vars)
# "completo" -> series temporales sobre todas las columnas numéricas
MODO = "curado"

RAIZ = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path("../..")
IN_PATH = RAIZ / "datasets" / "processed" / "competencia_01.csv"
OUT_PARQUET = RAIZ / "datasets" / "processed" / "competencia_01_fe.parquet"
OUT_CSV = RAIZ / "datasets" / "processed" / "competencia_01_fe.csv"
OUT_CATALOGO = RAIZ / "work" / "z402_fe_catalogo.csv"

VENTANA = 3  # rows between VENTANA preceding and current row  (t-3 .. t)

OUT_CATALOGO.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Catálogo
#
# Cada feature se registra cuando se genera su fragmento de SQL, así la query y
# su documentación no pueden desincronizarse. `_reg` devuelve la línea SQL y de
# paso la anota en `CATALOGO`.

# %%
CATALOGO: list[dict] = []


def _reg(bloque: str, feature: str, expresion: str, origen: str = "") -> str:
    """Registra la feature en el catálogo y devuelve su línea de SELECT."""
    CATALOGO.append(
        {
            "bloque": bloque,
            "feature": feature,
            "expresion": expresion,
            "origen": origen,
        }
    )
    return f"\n    , {expresion} as {feature}"


def _suma(campos: list[str]) -> str:
    """Suma null-safe de N campos (la macro `suma_segura` es binaria)."""
    return " + ".join(f"ifnull({c}, 0)" for c in campos)


def _cuenta_positivos(campos: list[str]) -> str:
    """Cuenta cuántos de los campos son > 0 (tratando NULL como 0)."""
    return " + ".join(f"if(ifnull({c}, 0) > 0, 1, 0)" for c in campos)


# %% [markdown]
# ## Conexión y carga
#
# `read_csv` con `sample_size = -1` en vez de `read_csv_auto`: escanea el
# archivo completo para inferir tipos y evita que una columna con muchos NULL
# al principio quede mal tipada.

# %%
con = duckdb.connect()
con.execute(f"""
    create or replace table competencia_01 as
    select * from read_csv('{IN_PATH.as_posix()}', sample_size = -1)
""")

n_in, n_cols_in = con.execute("""
    select count(*), count(*) from competencia_01, (select 1) limit 1
""").fetchone()
n_in = con.execute("select count(*) from competencia_01").fetchone()[0]
n_cols_in = len(con.execute("describe competencia_01").fetchall())
print(f"entrada: {n_in:,} filas x {n_cols_in} columnas")

# %% [markdown]
# ## Bloque 0 — Macros
#
# `ratio_seguro` resuelve la tarea planteada en el notebook (celda 14): un
# ratio que sobreviva tanto a los NULL como a la división por cero. `nullif`
# manda el denominador cero a NULL, y NULL propaga sin romper la query.

# %%
con.execute("""
    CREATE OR REPLACE MACRO suma_segura(a, b) AS ifnull(a, 0) + ifnull(b, 0);
""")
con.execute("""
    CREATE OR REPLACE MACRO ratio_seguro(a, b) AS ifnull(a, 0) / nullif(b, 0);
""")
con.execute("""
    CREATE OR REPLACE MACRO delta_pct(act, prev) AS (act - prev) / nullif(abs(prev), 0);
""")

# %% [markdown]
# ## Panel denso
#
# **El panel no es denso**: ~93% de las combinaciones cliente x mes existen. Si
# se aplica `lag()` sobre la tabla tal cual, un cliente que falta en 202104
# recibe en 202105 el valor de 202103 *como si fuera t-1*. Se reconstruye
# entonces el panel completo con `cross join clientes x periodos` (mismo patrón
# que `z101_target_sql`), se calculan las ventanas sobre él, y recién al final
# se filtran las filas fantasma con `mes_0 = 1`.
#
# `pk_cliente` / `pk_mes` vienen del lado izquierdo del join: sobreviven a las
# filas fantasma, donde `numero_de_cliente` y `foto_mes` son NULL. Todas las
# ventanas particionan y ordenan por esas dos.

# %%
SQL_PANEL = """
    with periodos as (
        select distinct foto_mes from competencia_01
    ), clientes as (
        select distinct numero_de_cliente from competencia_01
    ), todo as (
        select numero_de_cliente, foto_mes from clientes cross join periodos
    ), panel as (
        select
            t.numero_de_cliente as pk_cliente
            , t.foto_mes as pk_mes
            , if(c.numero_de_cliente is null, 0, 1) as mes_0
            , c.*
        from todo t
        left join competencia_01 c
          on c.numero_de_cliente = t.numero_de_cliente
         and c.foto_mes = t.foto_mes
    )
"""

# %% [markdown]
# ## Bloque 1 — Variables creadas a partir de otras
#
# Todo lo que se calcula dentro de una misma fila, sin mirar el pasado.
#
# **1a. Consolidado de tarjetas.** El cliente puede tener Master, Visa, ambas o
# ninguna, y "ninguna" se codifica como NULL. Una suma cruda `Master + Visa`
# devuelve NULL cuando falta una de las dos (el notebook lo muestra
# explícitamente), así que se usa `suma_segura`.
#
# **1b. Fechas relativas.** Según `datasets/raw/consideraciones.txt` (nota 8),
# las fechas están en **días relativos** al `foto_mes` (`fecha_foto − f`), no en
# formato absoluto. Por eso `least`/`greatest` entre Master y Visa tienen
# sentido: menor `Fvencimiento` = la tarjeta que vence antes; mayor `fechaalta`
# = la relación más antigua.
#
# **1c. Ratios de dominio.** Todos los montos están en pesos nominales y el
# dataset cubre 2021, así que los valores absolutos sufren drift
# inflacionario. Un ratio entre dos montos del mismo mes es inmune a eso.
# `r_tc_uso_limite` (saldo / límite) es el indicador clásico de estrés
# financiero previo a la baja.
#
# **1d. Agregados de negocio.** Consolidan familias de productos y canales.
# `r_engagement_digital` (digital / total) captura el traslado de canal, que en
# churn bancario suele preceder al abandono.
#
# **1e. Flags de riesgo.** De la sección "Ideas de feature engineering" de
# `docs/Hallazgos EDA Zulip.txt`: flag de missing en `Finiciomora`, agrupar
# `*_status` en normal vs anormal (6/7/9), saldo negativo, sin acreditación de
# sueldo. Los `f_sin_master` / `f_sin_visa` marcan el NULL **estructural** (no
# tiene el producto), que no es lo mismo que un dato faltante.

# %%
PARES_TC = [
    "mfinanciacion_limite",
    "msaldototal",
    "msaldopesos",
    "msaldodolares",
    "mconsumospesos",
    "mconsumosdolares",
    "mlimitecompra",
    "madelantopesos",
    "madelantodolares",
    "mpagado",
    "mpagospesos",
    "mpagosdolares",
    "mconsumototal",
    "cconsumos",
    "cadelantosefectivo",
    "mpagominimo",
]

PRODUCTOS_FLAG = [
    "ccuenta_corriente",
    "ccaja_ahorro",
    "ctarjeta_debito",
    "ctarjeta_visa",
    "ctarjeta_master",
    "cprestamos_personales",
    "cprestamos_prendarios",
    "cprestamos_hipotecarios",
    "cplazo_fijo",
    "cinversion1",
    "cinversion2",
    "cseguro_vida",
    "cseguro_auto",
    "cseguro_vivienda",
    "cseguro_accidentes_personales",
    "ccaja_seguridad",
]


def sql_bloque_1() -> str:
    frag = ""

    # -- 1a. consolidado Master + Visa -------------------------------------
    for suf in PARES_TC:
        frag += _reg(
            "1a_tc_consolidado",
            f"tc_{suf}",
            f"suma_segura(Master_{suf}, Visa_{suf})",
            f"Master_{suf} + Visa_{suf}",
        )

    # -- 1b. fechas relativas y estado de las tarjetas ---------------------
    frag += _reg(
        "1b_tc_fechas",
        "tc_fvencimiento_menor",
        "least(Master_Fvencimiento, Visa_Fvencimiento)",
        "Master_Fvencimiento, Visa_Fvencimiento",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_fvencimiento_mayor",
        "greatest(Master_Fvencimiento, Visa_Fvencimiento)",
        "Master_Fvencimiento, Visa_Fvencimiento",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_finiciomora_menor",
        "least(Master_Finiciomora, Visa_Finiciomora)",
        "Master_Finiciomora, Visa_Finiciomora",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_fultimo_cierre_menor",
        "least(Master_fultimo_cierre, Visa_fultimo_cierre)",
        "Master_fultimo_cierre, Visa_fultimo_cierre",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_fechaalta_mayor",
        "greatest(Master_fechaalta, Visa_fechaalta)",
        "Master_fechaalta, Visa_fechaalta",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_fechaalta_menor",
        "least(Master_fechaalta, Visa_fechaalta)",
        "Master_fechaalta, Visa_fechaalta",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_delinquency_max",
        "greatest(ifnull(Master_delinquency, 0), ifnull(Visa_delinquency, 0))",
        "Master_delinquency, Visa_delinquency",
    )
    frag += _reg(
        "1b_tc_fechas",
        "tc_status_max",
        "greatest(ifnull(Master_status, 0), ifnull(Visa_status, 0))",
        "Master_status, Visa_status",
    )

    # -- 1d. agregados de negocio (antes que los ratios que los usan) ------
    frag += _reg(
        "1d_agregados",
        "m_activos_totales",
        _suma(
            [
                "mcuentas_saldo",
                "mplazo_fijo_pesos",
                "mplazo_fijo_dolares",
                "minversion1_pesos",
                "minversion1_dolares",
                "minversion2",
            ]
        ),
        "saldos + plazos fijos + inversiones",
    )
    frag += _reg(
        "1d_agregados",
        "m_prestamos_totales",
        _suma(
            [
                "mprestamos_personales",
                "mprestamos_prendarios",
                "mprestamos_hipotecarios",
            ]
        ),
        "mprestamos_*",
    )
    frag += _reg(
        "1d_agregados",
        "c_prestamos_totales",
        _suma(
            [
                "cprestamos_personales",
                "cprestamos_prendarios",
                "cprestamos_hipotecarios",
            ]
        ),
        "cprestamos_*",
    )
    frag += _reg(
        "1d_agregados",
        "c_seguros_totales",
        _suma(
            [
                "cseguro_vida",
                "cseguro_auto",
                "cseguro_vivienda",
                "cseguro_accidentes_personales",
            ]
        ),
        "cseguro_*",
    )
    frag += _reg(
        "1d_agregados",
        "c_trx_digitales",
        _suma(["chomebanking_transacciones", "cmobile_app_trx"]),
        "homebanking + mobile app",
    )
    frag += _reg(
        "1d_agregados",
        "c_trx_presenciales",
        _suma(
            [
                "ccajas_transacciones",
                "catm_trx",
                "catm_trx_other",
                "cextraccion_autoservicio",
            ]
        ),
        "cajas + ATM + autoservicio",
    )
    frag += _reg(
        "1d_agregados",
        "c_trx_total",
        _suma(
            [
                "chomebanking_transacciones",
                "cmobile_app_trx",
                "ccajas_transacciones",
                "catm_trx",
                "catm_trx_other",
                "cextraccion_autoservicio",
                "ctarjeta_debito_transacciones",
                "ctarjeta_visa_transacciones",
                "ctarjeta_master_transacciones",
            ]
        ),
        "todos los canales",
    )
    frag += _reg(
        "1d_agregados",
        "m_transferencias_neto",
        "ifnull(mtransferencias_recibidas, 0) - ifnull(mtransferencias_emitidas, 0)",
        "mtransferencias_recibidas, mtransferencias_emitidas",
    )
    frag += _reg(
        "1d_agregados",
        "c_cheques_rechazados",
        _suma(["ccheques_depositados_rechazados", "ccheques_emitidos_rechazados"]),
        "ccheques_*_rechazados",
    )
    frag += _reg(
        "1d_agregados",
        "m_descuentos_total",
        _suma(
            [
                "mcajeros_propios_descuentos",
                "mtarjeta_visa_descuentos",
                "mtarjeta_master_descuentos",
            ]
        ),
        "m*_descuentos",
    )
    frag += _reg(
        "1d_agregados",
        "c_productos_contratados",
        _cuenta_positivos(PRODUCTOS_FLAG),
        "conteo de familias de producto con tenencia > 0",
    )

    # -- 1c. ratios de dominio (usan 1a y 1d) ------------------------------
    ratios = [
        ("r_tc_uso_limite", "tc_msaldototal", "tc_mlimitecompra", "utilización de TC"),
        (
            "r_tc_consumo_limite",
            "tc_mconsumototal",
            "tc_mlimitecompra",
            "presión de consumo",
        ),
        ("r_tc_pago_saldo", "tc_mpagado", "tc_msaldototal", "capacidad de repago"),
        ("r_tc_pagominimo", "tc_mpagominimo", "tc_msaldototal", "exigencia mínima"),
        (
            "r_tc_dolarizacion",
            "tc_msaldodolares",
            "tc_msaldototal",
            "exposición en dólares",
        ),
        ("r_payroll_saldo", "mpayroll", "mcuentas_saldo", "sueldo vs saldo"),
        (
            "r_comisiones_rentabilidad",
            "mcomisiones",
            "mrentabilidad",
            "peso de comisiones",
        ),
        (
            "r_rentabilidad_producto",
            "mrentabilidad",
            "cproductos",
            "rentabilidad unitaria",
        ),
        ("r_activos_pasivos", "mactivos_margen", "mpasivos_margen", "mix de margen"),
        (
            "r_prestamos_saldo",
            "m_prestamos_totales",
            "m_activos_totales",
            "apalancamiento",
        ),
        ("r_trx_producto", "ctrx_quarter", "cproductos", "intensidad de uso"),
        (
            "r_antiguedad_edad",
            "cliente_antiguedad",
            "cliente_edad * 12",
            "fracción de vida como cliente",
        ),
    ]
    for nombre, num, den, nota in ratios:
        frag += _reg(
            "1c_ratios_dominio",
            nombre,
            f"ratio_seguro({num}, {den})",
            f"{num} / {den} — {nota}",
        )
    frag += _reg(
        "1c_ratios_dominio",
        "r_engagement_digital",
        "ratio_seguro(c_trx_digitales, c_trx_digitales + c_trx_presenciales)",
        "digital / (digital + presencial) — traslado de canal",
    )

    # -- 1e. flags de riesgo -----------------------------------------------
    flags = [
        (
            "f_sin_payroll",
            "if(ifnull(cpayroll_trx, 0) + ifnull(cpayroll2_trx, 0) = 0, 1, 0)",
            "sin acreditación de sueldo",
        ),
        (
            "f_saldo_negativo",
            "if(ifnull(mcuentas_saldo, 0) < 0, 1, 0)",
            "saldo consolidado en rojo",
        ),
        (
            "f_caja_ahorro_negativa",
            "if(ifnull(mcaja_ahorro, 0) < 0, 1, 0)",
            "comisiones impagas (consideraciones, nota 5)",
        ),
        ("f_sin_trx", "if(ifnull(ctrx_quarter, 0) = 0, 1, 0)", "cliente inactivo"),
        (
            "f_mora",
            "if(Master_Finiciomora is not null or Visa_Finiciomora is not null, 1, 0)",
            "entró en mora en alguna TC",
        ),
        (
            "f_tc_status_anormal",
            "if(ifnull(Master_status, 0) in (6, 7, 9) or ifnull(Visa_status, 0) in (6, 7, 9), 1, 0)",
            "status de TC anormal",
        ),
        (
            "f_delinquency",
            "if(greatest(ifnull(Master_delinquency, 0), ifnull(Visa_delinquency, 0)) > 0, 1, 0)",
            "atraso registrado",
        ),
        (
            "f_sin_master",
            "if(Master_msaldototal is null, 1, 0)",
            "NULL estructural: no tiene Master",
        ),
        (
            "f_sin_visa",
            "if(Visa_msaldototal is null, 1, 0)",
            "NULL estructural: no tiene Visa",
        ),
        (
            "f_callcenter",
            "if(ifnull(ccallcenter_transacciones, 0) > 0, 1, 0)",
            "contactó al call center",
        ),
        (
            "f_cheque_rechazado",
            "if(ifnull(ccheques_depositados_rechazados, 0) + ifnull(ccheques_emitidos_rechazados, 0) > 0, 1, 0)",
            "rebote de cheques",
        ),
        (
            "f_sobregiro",
            "if(ifnull(cdescubierto_preacordado, 0) > 0 and ifnull(mcuentas_saldo, 0) < 0, 1, 0)",
            "usando el descubierto",
        ),
    ]
    for nombre, expr, nota in flags:
        frag += _reg("1e_flags_riesgo", nombre, expr, nota)

    return frag


BLOQUE_1 = sql_bloque_1()
print(f"bloque 1: {len(CATALOGO)} features intra-fila")

# %% [markdown]
# ## Campos sobre los que se calculan las series temporales
#
# En modo `curado` se toman las variables con mayor AUC / importancia según el
# EDA (`ctrx_quarter` 0,841 · `mcaja_ahorro` 0,813 · `mpasivos_margen` 0,803 ·
# `cpayroll_trx` · `chomebanking_transacciones` · `cproductos` 0,709) más las
# derivadas del bloque 1 que condensan señal.
#
# En modo `completo` la lista sale por introspección del esquema de `base`, sin
# hardcodear 150 nombres.

# %%
CAMPOS_SERIE_CURADO = [
    # actividad transaccional
    "ctrx_quarter",
    "chomebanking_transacciones",
    "cmobile_app_trx",
    "ccajas_transacciones",
    "catm_trx",
    "ctarjeta_debito_transacciones",
    "ccallcenter_transacciones",
    # saldos y márgenes
    "mcuentas_saldo",
    "mcaja_ahorro",
    "mcuenta_corriente",
    "mpasivos_margen",
    "mactivos_margen",
    "mrentabilidad",
    "mcomisiones",
    "mcomisiones_mantenimiento",
    # sueldo y productos
    "cpayroll_trx",
    "mpayroll",
    "cproductos",
    "mautoservicio",
    "matm",
    # consumo y crédito
    "mtarjeta_visa_consumo",
    "mtarjeta_master_consumo",
    "mprestamos_personales",
    "mtransferencias_recibidas",
    "mtransferencias_emitidas",
    "cdescubierto_preacordado",
    # derivadas del bloque 1
    "tc_msaldototal",
    "tc_mconsumototal",
    "tc_mlimitecompra",
    "tc_mpagado",
    "r_tc_uso_limite",
    "c_trx_digitales",
    "c_trx_presenciales",
    "r_engagement_digital",
    "m_activos_totales",
    "c_productos_contratados",
]

# Nunca entran a las series temporales: claves, target y las propias derivadas
# temporales.
NO_SERIE = {"pk_cliente", "pk_mes", "mes_0", "numero_de_cliente", "foto_mes", "clase_ternaria"}

# %% [markdown]
# ## `base` como VIEW
#
# Se define `base` (panel denso + bloque 1) como **vista**, no como tabla: da
# el esquema para la introspección del modo `completo` sin materializar ~1 GB
# extra en memoria.

# %%
con.execute(f"""
    create or replace view base as
    {SQL_PANEL}
    select *{BLOQUE_1}
    from panel
""")

esquema_base = con.execute("describe base").pl()
TIPOS_NUM = ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT")

if MODO == "completo":
    CAMPOS_SERIE = [
        r["column_name"]
        for r in esquema_base.iter_rows(named=True)
        if r["column_name"] not in NO_SERIE
        and r["column_type"].upper().startswith(TIPOS_NUM)
    ]
    TRANSFORMACIONES = ["lag_1", "delta_1", "delta_2", "avg_3", "max_3", "min_3", "slope_3"]
else:
    CAMPOS_SERIE = CAMPOS_SERIE_CURADO
    TRANSFORMACIONES = [
        "lag_1",
        "lag_2",
        "delta_1",
        "delta_2",
        "deltapct_1",
        "avg_3",
        "max_3",
        "min_3",
        "std_3",
        "ratioavg_3",
        "slope_3",
    ]

cols_base = set(esquema_base["column_name"].to_list())
faltantes = [c for c in CAMPOS_SERIE if c not in cols_base]
if faltantes:
    raise ValueError(f"campos de serie inexistentes en `base`: {faltantes}")

print(f"modo={MODO}: {len(CAMPOS_SERIE)} campos x {len(TRANSFORMACIONES)} transformaciones")

# %% [markdown]
# ## Bloques 3, 5, 6 y 7 — se calculan sobre el panel denso
#
# **Bloque 3 (lags).** `lag(x, n) over ventana_orden`, particionado por cliente
# y ordenado por mes. Con 6 meses de historia, `lag_2` recién existe desde
# 202105: los primeros meses quedan en NULL, que es lo correcto.
#
# **Bloque 5 (ventanas móviles).** `rows between 3 preceding and current row`
# (4 filas: t-3 .. t), el rango que usa el notebook. `avg`/`max`/`min` resumen
# el nivel habitual del cliente, `stddev` su volatilidad.
#
# **Bloque 6 (tendencia).** `regr_slope(x, cliente_antiguedad)`: la pendiente
# de la recta ajustada sobre la ventana. El eje X es `cliente_antiguedad`
# (meses, equiespaciado) y **no** `foto_mes`, que como entero `YYYYMM` no es
# lineal (202112 → 202201 salta 89).
#
# **Bloque 7 (historia).** Cuántos meses lleva observado el cliente dentro del
# panel; distingue altas recientes de clientes de siempre.

# %%
VENTANAS_SQL = f"""
    window ventana_orden as (partition by pk_cliente order by pk_mes)
         , ventana as (partition by pk_cliente order by pk_mes
                       rows between {VENTANA} preceding and current row)
         , historia as (partition by pk_cliente order by pk_mes
                        rows between unbounded preceding and current row)
"""


def sql_series(campos: list[str], transf: list[str]) -> str:
    frag = ""
    for c in campos:
        if "lag_1" in transf:
            frag += _reg("3_lags", f"lag_1_{c}", f"lag({c}, 1) over ventana_orden", c)
        if "lag_2" in transf:
            frag += _reg("3_lags", f"lag_2_{c}", f"lag({c}, 2) over ventana_orden", c)
        if "avg_3" in transf:
            frag += _reg(
                "5_ventanas", f"avg_{VENTANA}_{c}", f"avg({c}) over ventana", c
            )
        if "max_3" in transf:
            frag += _reg(
                "5_ventanas", f"max_{VENTANA}_{c}", f"max({c}) over ventana", c
            )
        if "min_3" in transf:
            frag += _reg(
                "5_ventanas", f"min_{VENTANA}_{c}", f"min({c}) over ventana", c
            )
        if "std_3" in transf:
            frag += _reg(
                "5_ventanas", f"std_{VENTANA}_{c}", f"stddev({c}) over ventana", c
            )
        if "slope_3" in transf:
            frag += _reg(
                "6_tendencia",
                f"slope_{VENTANA}_{c}",
                f"regr_slope({c}, cliente_antiguedad) over ventana",
                f"{c} vs cliente_antiguedad",
            )
    return frag


def sql_bloque_7() -> str:
    frag = _reg(
        "7_historia",
        "meses_en_panel",
        "sum(mes_0) over historia",
        "presencias acumuladas del cliente",
    )
    frag += _reg(
        "7_historia",
        "antiguedad_panel",
        "row_number() over ventana_orden",
        "posición del mes dentro del panel",
    )
    return frag


BLOQUE_SERIES = sql_series(CAMPOS_SERIE, TRANSFORMACIONES)
BLOQUE_7 = sql_bloque_7()

# %% [markdown]
# ## Bloques 2 y 4 — se calculan después de descartar las filas fantasma
#
# **Bloque 2 (rankings intra-mes).** `partition by foto_mes` como en el
# notebook: para cada mes se ordena a los clientes y se calcula su posición
# relativa. Es la defensa contra el data drifting — la inflación mueve todos
# los montos nominales, pero no la posición del cliente respecto de sus
# contemporáneos. **Tiene que calcularse sobre las filas reales**: si se hiciera
# sobre el panel denso, las ~7% de filas fantasma (con todos los campos en
# NULL) entrarían al denominador de `percent_rank` y comprimirían la escala.
#
# **Bloque 4 (deltas).** Dependen de los `lag_*` del bloque 3, por eso van en
# una CTE posterior. El delta porcentual es la versión drift-robusta de la
# regla "caída de `ctrx_quarter` ≥ 40%" que aparece en el EDA.
#
# **Bloque 5b (`ratioavg`).** `x / avg_3(x)`: normaliza al cliente contra su
# propia historia reciente. Inmune al drift agregado *y* a la escala individual.

# %%
CAMPOS_RANK = [
    "mrentabilidad",
    "mrentabilidad_annual",
    "mcomisiones",
    "mactivos_margen",
    "mpasivos_margen",
    "mcuentas_saldo",
    "mcaja_ahorro",
    "mcuenta_corriente",
    "mpayroll",
    "mprestamos_personales",
    "mtarjeta_visa_consumo",
    "mtarjeta_master_consumo",
    "tc_msaldototal",
    "tc_mlimitecompra",
    "ctrx_quarter",
    "cproductos",
    "cliente_antiguedad",
    "cliente_edad",
    "chomebanking_transacciones",
]

# deciles solo para las de mayor AUC según el EDA
CAMPOS_DECIL = ["ctrx_quarter", "mcaja_ahorro", "mpasivos_margen", "mcuentas_saldo"]


def sql_bloque_2() -> str:
    frag = ""
    for c in CAMPOS_RANK:
        frag += _reg(
            "2_rank_intrames",
            f"pr_{c}",
            f"percent_rank() over (partition by foto_mes order by {c})",
            f"{c} — posición relativa dentro del mes",
        )
    for c in CAMPOS_DECIL:
        frag += _reg(
            "2_rank_intrames",
            f"d10_{c}",
            f"ntile(10) over (partition by foto_mes order by {c})",
            f"{c} — decil dentro del mes",
        )
    return frag


def sql_bloque_4(campos: list[str], transf: list[str]) -> str:
    frag = ""
    for c in campos:
        if "delta_1" in transf:
            frag += _reg(
                "4_deltas", f"delta_1_{c}", f"{c} - lag_1_{c}", f"{c}, lag_1_{c}"
            )
        if "delta_2" in transf:
            # en modo completo no se materializa lag_2; se calcula el delta
            # contra t-2 directamente sobre la ventana ya resuelta
            origen = f"lag_2_{c}" if "lag_2" in transf else "lag(x,2)"
            expr = f"{c} - lag_2_{c}" if "lag_2" in transf else None
            if expr is None:
                continue
            frag += _reg("4_deltas", f"delta_2_{c}", expr, f"{c}, {origen}")
        if "deltapct_1" in transf:
            frag += _reg(
                "4_deltas",
                f"deltapct_1_{c}",
                f"delta_pct({c}, lag_1_{c})",
                f"variación % contra t-1",
            )
        if "ratioavg_3" in transf:
            frag += _reg(
                "5_ventanas",
                f"ratioavg_{VENTANA}_{c}",
                f"ratio_seguro({c}, avg_{VENTANA}_{c})",
                f"{c} contra su media móvil",
            )
    return frag


BLOQUE_2 = sql_bloque_2()
BLOQUE_4 = sql_bloque_4(CAMPOS_SERIE, TRANSFORMACIONES)

# %% [markdown]
# ### Nota sobre `delta_2` en modo `completo`
#
# `TRANSF_COMPLETO` pide `delta_2` pero no materializa `lag_2`, así que hay que
# agregar el lag auxiliar. Se lo suma a la lista de transformaciones de series
# y se lo marca para excluir de la salida final.

# %%
LAGS_AUXILIARES: list[str] = []
if "delta_2" in TRANSFORMACIONES and "lag_2" not in TRANSFORMACIONES:
    aux = ""
    for c in CAMPOS_SERIE:
        LAGS_AUXILIARES.append(f"lag_2_{c}")
        aux += f"\n    , lag({c}, 2) over ventana_orden as lag_2_{c}"
    BLOQUE_SERIES += aux
    BLOQUE_4 += "".join(
        _reg("4_deltas", f"delta_2_{c}", f"{c} - lag_2_{c}", f"{c}, lag_2_{c}")
        for c in CAMPOS_SERIE
    )

EXCLUDE_FINAL = ", ".join(["pk_cliente", "pk_mes", "mes_0", *LAGS_AUXILIARES])

# %% [markdown]
# ## Query final
#
# Se arma por f-string y se ejecuta con `con.execute()`. El `COPY (query) TO
# parquet` deja que DuckDB **streamee** el resultado a disco en vez de
# materializar ~600 columnas x 983k filas en memoria.

# %%
QUERY_FE = f"""
{SQL_PANEL}
   , base as (
        select *{BLOQUE_1}
        from panel
     )
   , series as (
        select *{BLOQUE_SERIES}{BLOQUE_7}
        from base
        {VENTANAS_SQL}
     )
   , reales as (
        select *
        from series
        where mes_0 = 1
     )
select * exclude ({EXCLUDE_FINAL}){BLOQUE_2}{BLOQUE_4}
from reales
"""

print(f"query: {len(QUERY_FE):,} caracteres, {QUERY_FE.count(chr(10)):,} líneas")
print(f"features nuevas registradas: {len(CATALOGO)}")

# %%
con.execute(f"""
    COPY ({QUERY_FE}) TO '{OUT_PARQUET.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")
print(f"escrito: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1e6:,.0f} MB)")

# %%
con.execute(f"""
    create or replace view fe as
    select * from read_parquet('{OUT_PARQUET.as_posix()}')
""")

# %% [markdown]
# ## Catálogo de variables creadas
#
# Una fila por feature nueva, con el bloque al que pertenece, la expresión SQL
# exacta y las columnas de origen.

# %%
catalogo = pl.DataFrame(CATALOGO)

resumen = (
    catalogo.group_by("bloque")
    .len(name="n_features")
    .sort("bloque")
)
print(resumen)
print(f"\nTOTAL features creadas : {catalogo.height}")
print(f"columnas de entrada    : {n_cols_in}")
print(f"columnas de salida     : {len(con.execute('describe fe').fetchall())}")

# %%
catalogo.write_csv(OUT_CATALOGO)
print(f"catálogo: {OUT_CATALOGO}")

# %%
# Muestra de cada bloque
for b in resumen["bloque"]:
    ejemplos = catalogo.filter(pl.col("bloque") == b).head(3)
    print(f"\n--- {b} ---")
    for r in ejemplos.iter_rows(named=True):
        print(f"  {r['feature']:<38} = {r['expresion'][:80]}")

# %% [markdown]
# ## Verificación
#
# ### 1. El panel denso no agregó ni perdió filas

# %%
n_out = con.execute("select count(*) from fe").fetchone()[0]
print(f"entrada {n_in:,} -> salida {n_out:,}  {'OK' if n_in == n_out else 'ERROR'}")
assert n_in == n_out, "el panel denso alteró la cantidad de filas"

# %% [markdown]
# ### 2. El target quedó intacto

# %%
pivot_in = con.execute("""
    PIVOT competencia_01 ON clase_ternaria USING count(numero_de_cliente)
    GROUP BY foto_mes ORDER BY foto_mes
""").pl()
pivot_out = con.execute("""
    PIVOT fe ON clase_ternaria USING count(numero_de_cliente)
    GROUP BY foto_mes ORDER BY foto_mes
""").pl()
print(pivot_out)
assert pivot_in.equals(pivot_out), "cambió la distribución del target"
print("target intacto: OK")

# %% [markdown]
# ### 3. El panel denso corrige los lags de clientes con huecos
#
# Se busca un cliente al que le falte algún mes intermedio y se compara el
# `lag_1` correcto (con panel denso) contra el que devolvería un `lag()` sobre
# la tabla cruda. Sin panel denso, el mes posterior al hueco recibiría el valor
# de t-2 disfrazado de t-1.

# %%
cliente_con_hueco = con.execute("""
    with idx as (
        select foto_mes, dense_rank() over (order by foto_mes) as i
        from (select distinct foto_mes from competencia_01)
    ), x as (
        select c.numero_de_cliente, min(i) as mn, max(i) as mx, count(*) as n
        from competencia_01 c join idx using (foto_mes)
        group by 1
    )
    select numero_de_cliente from x where n <> mx - mn + 1 order by 1 limit 1
""").fetchone()

if cliente_con_hueco is None:
    print("no hay clientes con meses faltantes intermedios")
else:
    cli = cliente_con_hueco[0]
    print(f"cliente con hueco: {cli}\n")
    comparacion = con.execute(f"""
        select
            f.foto_mes
            , f.ctrx_quarter
            , f.lag_1_ctrx_quarter as lag_1_correcto
            , n.lag_1_naive
        from fe f
        join (
            select numero_de_cliente, foto_mes
                 , lag(ctrx_quarter, 1) over (
                       partition by numero_de_cliente order by foto_mes) as lag_1_naive
            from competencia_01
            where numero_de_cliente = {cli}
        ) n using (numero_de_cliente, foto_mes)
        where f.numero_de_cliente = {cli}
        order by f.foto_mes
    """).pl()
    print(comparacion)
    n_difs = con.execute(f"""
        select count(*) from (
            select f.lag_1_ctrx_quarter as a, n.lag_1_naive as b
            from fe f
            join (
                select numero_de_cliente, foto_mes
                     , lag(ctrx_quarter, 1) over (
                           partition by numero_de_cliente order by foto_mes) as lag_1_naive
                from competencia_01
            ) n using (numero_de_cliente, foto_mes)
        ) where a is distinct from b
    """).fetchone()[0]
    print(f"\nfilas donde el lag naive difiere del correcto: {n_difs:,}")

# %% [markdown]
# ### 4. Spot check aritmético sobre un cliente con los 6 meses

# %%
cliente_completo = con.execute("""
    select numero_de_cliente from competencia_01
    group by 1 having count(*) = 6 and sum(ctrx_quarter) > 100
    order by 1 limit 1
""").fetchone()[0]

con.execute(f"""
    select foto_mes, ctrx_quarter, lag_1_ctrx_quarter, lag_2_ctrx_quarter
         , delta_1_ctrx_quarter, deltapct_1_ctrx_quarter
         , avg_3_ctrx_quarter, min_3_ctrx_quarter, max_3_ctrx_quarter
         , slope_3_ctrx_quarter, ratioavg_3_ctrx_quarter, meses_en_panel
    from fe where numero_de_cliente = {cliente_completo} order by foto_mes
""").pl()

# %% [markdown]
# ### 5. Cobertura de NULLs por bloque
#
# Se esperan NULLs altos en `lag_2`/`std_3` durante 202103-202104: no hay
# historia suficiente. Los rankings no deberían tener NULLs.

# %%
muestras = {
    "3_lags": "lag_2_ctrx_quarter",
    "4_deltas": "delta_1_ctrx_quarter",
    "5_ventanas": f"std_{VENTANA}_ctrx_quarter",
    "6_tendencia": f"slope_{VENTANA}_ctrx_quarter",
    "2_rank_intrames": "pr_ctrx_quarter",
    "1c_ratios_dominio": "r_tc_uso_limite",
}
cols = ", ".join(
    f"round(100.0 * count(*) filter ({c} is null) / count(*), 1) as pct_null_{b}"
    for b, c in muestras.items()
)
con.execute(f"select foto_mes, {cols} from fe group by 1 order by 1").pl()

# %% [markdown]
# ### 6. Ningún ratio devolvió infinito
#
# El `nullif` del denominador en `ratio_seguro` debería haber evitado toda
# división por cero.

# %%
cols_ratio = [r["feature"] for r in CATALOGO if r["feature"].startswith(("r_", "ratioavg_", "deltapct_"))]
chk = " + ".join(f"count(*) filter (isinf({c}))" for c in cols_ratio)
n_inf = con.execute(f"select {chk} from fe").fetchone()[0]
print(f"valores infinitos en {len(cols_ratio)} columnas de ratio: {n_inf}")
assert n_inf == 0, "hay divisiones por cero sin cubrir"

# %% [markdown]
# ## Export a CSV
#
# Se lee del Parquet ya escrito en vez de recalcular la query.

# %%
con.execute(f"""
    COPY (select * from read_parquet('{OUT_PARQUET.as_posix()}'))
    TO '{OUT_CSV.as_posix()}' (FORMAT CSV, HEADER)
""")
print(f"escrito: {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:,.0f} MB)")

# %%
con.close()
