"""Base de datos y utilidades compartidas para el análisis final de clusters.

Cruza `clusters_marcados.csv` (los 4.067 clientes BAJA+2 con su cluster,
generado por z502_marcar_clusters.py) con la población completa de clientes
fieles del FE (`competencia_01_fe.parquet`), para caracterizar cada cluster
contra el universo real de clientes retenidos — no contra la muestra 1:1
que usó el pipeline de exploración (z501_*).

No vuelve a correr el clustering: reusa las etiquetas ya asignadas.

Uso: módulo de importación, sin `main`. Ver z505_analisis_final.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Repo root en sys.path (para dmeyf.metrics / config.semillas) y este directorio
# (para los z501_*.py, que ya son import-safe: solo definen funciones/constantes).
_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

from z501_analisis_video_pca_sin_rent import (
    EXCLUIR_DIFERENCIADORAS,
    a_percentil_global,
    calcular_lift,
    calcular_medias_por_grupo,
    dibujar_radar,
    es_binaria,
    rentabilidad_por_cluster,
    variables_diferenciadoras,
)
from z501_cluster_rf import (
    BANDAS,
    CLUSTER_COL,
    COLORES_CLUSTER,
    GRID,
    GRUPO_COL,
    ID_COL,
    INK,
    INK_2,
    MES_COL,
    TARGET_COL,
    log,
)

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #

CLUSTERS_CSV = _AQUI / "analisis_clustering" / "clusters_marcados.csv"
FE_PARQUET = _RAIZ / "datasets" / "processed" / "competencia_01_fe.parquet"
OUT_DIR = _AQUI / "analisis_clustering" / "analisis_final"
DIR_RADARES = OUT_DIR / "01_radares"
DIR_TENDENCIAS = OUT_DIR / "02_tendencias"
DIR_BARRAS = OUT_DIR / "03_barras"
DIR_VALOR = OUT_DIR / "04_valor"
DIR_DISTRIBUCIONES = OUT_DIR / "05_distribuciones"

# Semilla del clustering ganador (z502_marcar_clusters.py). NO es una de las 5
# semillas del curso (config/semillas.SEMILLAS): es la que reproduce los
# clusters ya asignados en clusters_marcados.csv, así que se mantiene fija acá
# en vez de tomarla de config/semillas.py.
SEED = 214363

NOMBRES_CLUSTER = {
    1: "Ahorrista sin nómina",
    2: "Cliente dormido",
    3: "Tomador de crédito digital",
}

# --------------------------------------------------------------------------- #
# Diccionario de variables curado, por dominio de negocio
# --------------------------------------------------------------------------- #
# (variable, etiqueta legible, tipo) — tipo solo se usa para formatear ejes/textos:
#   monto = $ ; conteo = número ; flag = proporción 0-1 ; ratio = cociente ; edad = años

DOMINIOS: dict[str, list[tuple[str, str, str]]] = {
    "Valor y margen": [
        ("mrentabilidad", "Rentabilidad mensual", "monto"),
        ("mrentabilidad_annual", "Rentabilidad anualizada", "monto"),
        ("mcomisiones", "Comisiones cobradas", "monto"),
        ("mactivos_margen", "Margen por activos (créditos)", "monto"),
        ("mpasivos_margen", "Margen por pasivos (depósitos)", "monto"),
        ("mcomisiones_mantenimiento", "Comisión de mantenimiento", "monto"),
        ("mcomisiones_otras", "Otras comisiones", "monto"),
    ],
    "Tarjetas de crédito": [
        ("ctarjeta_visa", "Tarjetas Visa", "conteo"),
        ("ctarjeta_master", "Tarjetas Master", "conteo"),
        ("f_sin_visa", "Sin tarjeta Visa (flag)", "flag"),
        ("f_sin_master", "Sin tarjeta Master (flag)", "flag"),
        ("tc_msaldototal", "Saldo total tarjetas", "monto"),
        ("tc_mconsumototal", "Consumo total tarjetas", "monto"),
        ("tc_mlimitecompra", "Límite de compra", "monto"),
        ("tc_mpagado", "Pagado tarjetas", "monto"),
        ("r_tc_uso_limite", "Uso del límite (ratio)", "ratio"),
        ("r_tc_consumo_limite", "Consumo / límite (ratio)", "ratio"),
        ("r_tc_pago_saldo", "Pago / saldo (ratio)", "ratio"),
        ("r_tc_pagominimo", "Pago mínimo (ratio)", "ratio"),
        ("tc_cconsumos", "Cantidad de consumos", "conteo"),
        ("tc_delinquency_max", "Mora tarjetas (máx.)", "flag"),
    ],
    "Actividad transaccional": [
        ("ctrx_quarter", "Transacciones último trimestre", "conteo"),
        ("c_trx_total", "Transacciones totales", "conteo"),
        ("c_trx_digitales", "Transacciones digitales", "conteo"),
        ("c_trx_presenciales", "Transacciones presenciales", "conteo"),
        ("ctarjeta_debito_transacciones", "Transacciones débito", "conteo"),
        ("catm_trx", "Transacciones en cajero", "conteo"),
        ("mautoservicio", "Consumo autoservicio", "monto"),
        ("ccajas_transacciones", "Transacciones en caja", "conteo"),
    ],
    "Canales digitales": [
        ("thomebanking", "Tiene homebanking", "flag"),
        ("chomebanking_transacciones", "Transacciones homebanking", "conteo"),
        ("tmobile_app", "Tiene app móvil", "flag"),
        ("cmobile_app_trx", "Transacciones app móvil", "conteo"),
        ("internet", "Cliente internet", "flag"),
        ("r_engagement_digital", "Engagement digital (ratio)", "ratio"),
        ("tcallcenter", "Usa call center", "flag"),
        ("ccallcenter_transacciones", "Transacciones call center", "conteo"),
    ],
    "Vínculo y nómina": [
        ("cpayroll_trx", "Transacciones de nómina", "conteo"),
        ("mpayroll", "Monto de nómina", "monto"),
        ("f_sin_payroll", "Sin nómina (flag)", "flag"),
        ("ccuenta_debitos_automaticos", "Débitos automáticos", "conteo"),
        ("cpagodeservicios", "Pago de servicios", "conteo"),
        ("cpagomiscuentas", "Pago de tarjetas propias", "conteo"),
        ("r_payroll_saldo", "Nómina / saldo (ratio)", "ratio"),
    ],
    "Saldos e inversión": [
        ("mcuentas_saldo", "Saldo en cuentas", "monto"),
        ("mcaja_ahorro", "Saldo en caja de ahorro", "monto"),
        ("mcuenta_corriente", "Saldo en cuenta corriente", "monto"),
        ("cplazo_fijo", "Plazos fijos", "conteo"),
        ("mplazo_fijo_pesos", "Monto en plazo fijo", "monto"),
        ("cinversion1", "Inversión tipo 1", "conteo"),
        ("minversion1_pesos", "Monto inversión tipo 1", "monto"),
        ("cinversion2", "Inversión tipo 2", "conteo"),
        ("m_activos_totales", "Activos totales", "monto"),
    ],
    "Crédito": [
        ("cprestamos_personales", "Préstamos personales", "conteo"),
        ("mprestamos_personales", "Monto préstamos personales", "monto"),
        ("c_prestamos_totales", "Préstamos totales", "conteo"),
        ("m_prestamos_totales", "Monto préstamos totales", "monto"),
        ("r_prestamos_saldo", "Préstamos / saldo (ratio)", "ratio"),
    ],
    "Productos y seguros": [
        ("cproductos", "Cantidad de productos", "conteo"),
        ("c_productos_contratados", "Productos contratados (score)", "conteo"),
        ("c_seguros_totales", "Seguros totales", "conteo"),
        ("cseguro_vida", "Seguro de vida", "conteo"),
        ("cseguro_auto", "Seguro de auto", "conteo"),
        ("ccaja_seguridad", "Caja de seguridad", "conteo"),
    ],
    "Señales de riesgo": [
        ("f_mora", "Mora (flag)", "flag"),
        ("f_delinquency", "Delincuencia (flag)", "flag"),
        ("f_saldo_negativo", "Saldo negativo (flag)", "flag"),
        ("f_sobregiro", "Sobregiro (flag)", "flag"),
        ("cdescubierto_preacordado", "Descubierto preacordado", "conteo"),
        ("tc_status_max", "Status tarjeta (máx.)", "ratio"),
        ("f_cheque_rechazado", "Cheque rechazado (flag)", "flag"),
    ],
    "Demografía": [
        ("cliente_edad", "Edad", "edad"),
        ("cliente_antiguedad", "Antigüedad (meses)", "conteo"),
        ("cliente_vip", "Cliente VIP", "flag"),
        ("r_antiguedad_edad", "Antigüedad / edad (ratio)", "ratio"),
    ],
}

FEATURES_CURADAS: list[str] = [v for grupo in DOMINIOS.values() for v, _, _ in grupo]
ETIQUETAS: dict[str, str] = {
    v: lbl for grupo in DOMINIOS.values() for v, lbl, _ in grupo
}
TIPOS: dict[str, str] = {v: t for grupo in DOMINIOS.values() for v, _, t in grupo}
DOMINIO_DE: dict[str, str] = {
    v: d for d, grupo in DOMINIOS.items() for v, _, _ in grupo
}


def etiqueta(var: str) -> str:
    return ETIQUETAS.get(var, var)


# --------------------------------------------------------------------------- #
# Carga de churners (clusters_marcados.csv)
# --------------------------------------------------------------------------- #


def cargar_churners() -> pd.DataFrame:
    """Historia completa cliente-mes de los 4.067 BAJA+2, con su cluster."""
    if not CLUSTERS_CSV.exists():
        raise SystemExit(f"no encuentro {CLUSTERS_CSV}")
    df = (
        duckdb.connect()
        .execute(f"SELECT * FROM read_csv('{CLUSTERS_CSV.as_posix()}', sample_size=-1)")
        .df()
    )
    df[TARGET_COL] = df[TARGET_COL].astype("string")
    return df


def etiquetas_por_cliente(df_churn: pd.DataFrame) -> pd.Series:
    """cluster por numero_de_cliente (constante dentro de cada cliente)."""
    return df_churn.drop_duplicates(ID_COL).set_index(ID_COL)[CLUSTER_COL]


def agregar_mes_relativo(df_churn: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'mes_relativo', anclado en el mes SIGUIENTE al último mes en
    que el cliente aparece en el panel: t=0 es ese mes de ausencia (nunca
    tiene datos, es solo el ancla del evento), t=-1 es la última fila real
    del cliente (la que la convención clase_ternaria etiqueta BAJA+1),
    t=-2 es la fila BAJA+2, t=-3 y anteriores son meses CONTINUA previos.
    Se usa índice de mes lineal (año*12 + mes) porque foto_mes es YYYYMM y
    no es lineal.

    OJO: el ancla NO es la fecha de la fila BAJA+2 (eso pondría en t=0 una
    fila con dato real, dos meses antes de que el cliente deje de aparecer
    de verdad) — es la fecha del último mes presente del cliente, más uno.
    Con este ancla, TODA fila real del cliente cae en t<=-1 sin necesidad de
    filtrar nada aparte."""

    def idx(mes: pd.Series) -> pd.Series:
        return (mes // 100) * 12 + (mes % 100)

    ultimo = df_churn.groupby(ID_COL)[MES_COL].max().rename("_mes_ultimo").reset_index()
    ultimo["_anchor_idx"] = idx(ultimo["_mes_ultimo"]) + 1
    out = df_churn.merge(ultimo[[ID_COL, "_anchor_idx"]], on=ID_COL, how="left")
    out["mes_relativo"] = idx(out[MES_COL]) - out["_anchor_idx"]
    return out.drop(columns="_anchor_idx")


# --------------------------------------------------------------------------- #
# Conexión a la población completa de fieles (FE parquet)
# --------------------------------------------------------------------------- #


def conectar_fe(parquet_path: Path = FE_PARQUET) -> duckdb.DuckDBPyConnection:
    """Conexión DuckDB con:
    - raw_fe: vista sobre TODO el FE (983.061 filas, todos los clientes-mes)
    - fieles_ids: tabla de los clientes fieles (presentes en los 6 meses,
      nunca BAJA+1/BAJA+2) — población completa, NO la muestra 1:1 del
      pipeline de exploración.
    - fieles: vista de raw_fe filtrada a esos clientes.
    """
    if not parquet_path.exists():
        raise SystemExit(f"no encuentro {parquet_path}")
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW raw_fe AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
    )
    con.execute(f"""
        CREATE TEMP TABLE fieles_ids AS
        WITH n_meses AS (SELECT COUNT(DISTINCT {MES_COL}) AS n FROM raw_fe)
        SELECT {ID_COL}
        FROM raw_fe
        GROUP BY {ID_COL}
        HAVING COUNT(*) = (SELECT n FROM n_meses)
           AND SUM(CASE WHEN {TARGET_COL} IN ('BAJA+1', 'BAJA+2') THEN 1 ELSE 0 END) = 0
    """)
    con.execute(
        f"CREATE VIEW fieles AS SELECT r.* FROM raw_fe r JOIN fieles_ids f USING ({ID_COL})"
    )
    n_fieles = con.execute("SELECT COUNT(*) FROM fieles_ids").fetchone()[0]
    log(f"población fiel completa: {n_fieles:,} clientes")
    return con


def cargar_fieles_curadas(
    con: duckdb.DuckDBPyConnection, variables: list[str] | None = None
) -> pd.DataFrame:
    """Trae las columnas curadas de TODA la población fiel a pandas
    (float32: ~90 columnas x 947k filas ~ 300MB). Se usa para poder reutilizar
    variables_diferenciadoras() / rentabilidad_por_cluster() de z501 tal cual,
    contra la población real en lugar de la muestra 1:1."""
    variables = variables or FEATURES_CURADAS
    cols = ", ".join(variables)
    df = con.execute(
        f"SELECT {ID_COL}, {MES_COL}, {TARGET_COL}, {cols} FROM fieles"
    ).df()
    for v in variables:
        if pd.api.types.is_numeric_dtype(df[v]):
            df[v] = df[v].astype("float32")
    df[TARGET_COL] = df[TARGET_COL].astype("string")
    df[GRUPO_COL] = 0
    return df


def rent_fiel_anual(con: duckdb.DuckDBPyConnection) -> float:
    return float(
        con.execute("SELECT AVG(mrentabilidad) * 12 FROM fieles").fetchone()[0]
    )


def tasa_base_baja2(con: duckdb.DuckDBPyConnection) -> float:
    """Tasa real de BAJA+2 sobre TODO el panel (excluye meses sin etiqueta,
    202107 parcial y 202108 completo, que no tienen 2 meses de lookahead)."""
    fila = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE {TARGET_COL} = 'BAJA+2') AS n_baja2,
            COUNT(*) FILTER (WHERE {TARGET_COL} IS NOT NULL) AS n_total
        FROM raw_fe
    """).fetchone()
    return fila[0] / fila[1]


def variables_binarias(
    con: duckdb.DuckDBPyConnection, df_churn: pd.DataFrame, variables: list[str]
) -> dict[str, bool]:
    """Una variable es binaria si TODO valor no nulo, en fieles ∪ churners,
    es 0 o 1. Un solo pase por la población fiel (COUNT...FILTER)."""
    filtros = ", ".join(
        f"COUNT(*) FILTER (WHERE {v} IS NOT NULL AND {v} NOT IN (0,1)) AS {v}"
        for v in variables
    )
    fila = con.execute(f"SELECT {filtros} FROM fieles").fetchone()
    out = {}
    for i, v in enumerate(variables):
        no_binaria_fieles = fila[i] > 0
        vals_churn = pd.unique(df_churn[v].dropna())
        no_binaria_churn = not set(vals_churn.astype(float)).issubset({0.0, 1.0})
        out[v] = not (no_binaria_fieles or no_binaria_churn)
    return out


# --------------------------------------------------------------------------- #
# Diferenciadoras entre clusters (no existía en el pipeline de exploración,
# que solo comparaba cada cluster contra los fieles)
# --------------------------------------------------------------------------- #


def diferenciadoras_entre_clusters(
    df_churn: pd.DataFrame,
    labels: pd.Series,
    features: list[str],
    var_binaria: dict[str, bool],
    top_n: int = 10,
    min_effect: float = 0.3,
) -> pd.DataFrame:
    """Para cada cluster, ranking de variables por Cohen's d contra los OTROS
    dos clusters de BAJA+2 (no contra los fieles). Responde "qué hace a este
    cluster distinto de los otros tipos de baja", complementario a
    variables_diferenciadoras() (que responde "por qué se va" comparando
    contra el fiel)."""
    baja2 = df_churn.copy()
    if CLUSTER_COL not in baja2.columns:
        baja2 = baja2.merge(
            labels.rename(CLUSTER_COL), left_on=ID_COL, right_index=True
        )

    resultados = []
    for c in sorted(labels.unique()):
        cli = baja2[baja2[CLUSTER_COL] == c]
        otros = baja2[baja2[CLUSTER_COL] != c]
        cli_media, cli_var = cli[features].mean(), cli[features].var()
        otros_media, otros_var = otros[features].mean(), otros[features].var()
        sd_pool = np.sqrt((cli_var + otros_var) / 2).replace(0, np.nan)
        cohen_d = (cli_media - otros_media) / sd_pool

        for var in features:
            d = cohen_d[var]
            if pd.isna(d) or abs(d) < min_effect:
                continue
            direccion = "MAYOR" if d > 0 else "MENOR"
            lift = calcular_lift(cli[var], otros[var], var_binaria[var], direccion)
            resultados.append(
                {
                    "cluster": c,
                    "variable": var,
                    "media_cluster": cli_media[var],
                    "media_otros_clusters": otros_media[var],
                    "diferencia": cli_media[var] - otros_media[var],
                    "cohen_d": d,
                    "lift": lift,
                    "abs_d": abs(d),
                    "direccion": direccion,
                    "magnitud": (
                        "GRANDE"
                        if abs(d) >= 0.8
                        else "MEDIO"
                        if abs(d) >= 0.5
                        else "CHICO"
                    ),
                }
            )

    if not resultados:
        return pd.DataFrame()
    out = pd.DataFrame(resultados)
    out = out.sort_values(["cluster", "abs_d"], ascending=[True, False])
    out = out.groupby("cluster").head(top_n).reset_index(drop=True)
    return out.drop(columns="abs_d")


# --------------------------------------------------------------------------- #
# Tendencias alineadas al evento (mes_relativo), no al calendario
# --------------------------------------------------------------------------- #


def tendencias_por_mes_relativo(
    df_churn: pd.DataFrame, atributos: list[str], rango: list[int], banda: str = "ic95"
) -> pd.DataFrame:
    """Igual que tendencias_por_cluster() de z501_cluster_rf, pero agrupando
    por 'mes_relativo' (0 = mes del BAJA+2) en lugar de por foto_mes: los
    clientes se van en meses calendario distintos, así que alinear al evento
    es lo que permite leer "esta variable se mueve antes de la baja"."""
    d = df_churn[df_churn["mes_relativo"].isin(rango)]
    g = d.groupby([CLUSTER_COL, "mes_relativo"])[atributos]
    n = g.count()
    if banda == "iqr":
        centro, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    else:
        centro, sd = g.mean(), g.std()
        ancho = sd if banda == "desvio" else 1.96 * sd / np.sqrt(n)
        lo, hi = centro - ancho, centro + ancho
    lo = lo.clip(lower=d[atributos].min(), axis=1)
    hi = hi.clip(upper=d[atributos].max(), axis=1)
    return pd.concat({"centro": centro, "lo": lo, "hi": hi, "n": n}, axis=1)


def senales_anticipacion(
    tend: pd.DataFrame,
    variables: list[str],
    t_lejano: int = -3,
    t_cercano: int = -1,
    min_n: int = 80,
    sd_ref: pd.Series | None = None,
) -> pd.DataFrame:
    """Para cada (cluster, variable), cambio estandarizado entre t_lejano y
    t_cercano: (media[t_cercano] - media[t_lejano]) / sd_ref[variable]. Un
    |cambio| grande y con `n` suficiente en ambos extremos es candidato a
    señal de anticipación (se mueve ANTES de la baja, no solo en el mes de
    la baja). Heurística de ranking, no un test de monotonía formal."""
    filas = []
    clusters = sorted(tend.index.get_level_values(CLUSTER_COL).unique())
    for c in clusters:
        s = tend.xs(c, level=CLUSTER_COL)
        for var in variables:
            if t_lejano not in s.index or t_cercano not in s.index:
                continue
            n_lejano = s.loc[t_lejano, ("n", var)]
            n_cercano = s.loc[t_cercano, ("n", var)]
            if n_lejano < min_n or n_cercano < min_n:
                continue
            m_lejano = s.loc[t_lejano, ("centro", var)]
            m_cercano = s.loc[t_cercano, ("centro", var)]
            escala = (
                sd_ref[var] if sd_ref is not None and sd_ref.get(var, 0) else np.nan
            )
            cambio_std = (
                (m_cercano - m_lejano) / escala
                if escala and not pd.isna(escala)
                else np.nan
            )
            filas.append(
                {
                    "cluster": c,
                    "variable": var,
                    f"media_t{t_lejano}": m_lejano,
                    f"media_t{t_cercano}": m_cercano,
                    "cambio": m_cercano - m_lejano,
                    "cambio_std": cambio_std,
                    f"n_t{t_lejano}": int(n_lejano),
                    f"n_t{t_cercano}": int(n_cercano),
                }
            )
    out = pd.DataFrame(filas)
    if out.empty:
        return out
    out["abs_cambio_std"] = out["cambio_std"].abs()
    out = out.sort_values(["cluster", "abs_cambio_std"], ascending=[True, False])
    return out.drop(columns="abs_cambio_std").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Reglas de alerta con el denominador poblacional real (corrige
# triggers_candidatos.csv, que usaba la muestra 1:1 con tasa base ~50%)
# --------------------------------------------------------------------------- #


def decil_poblacional(con: duckdb.DuckDBPyConnection, variable: str) -> pd.DataFrame:
    """Tasa de BAJA+2 por decil de `variable`, con NTILE(10) partido por
    foto_mes (igual criterio que los d10_*/pr_* del FE: evita que un mes con
    otra escala de la variable contamine el corte) sobre TODO el panel con
    etiqueta conocida (excluye 202107 parcial y 202108, sin lookahead)."""
    q = f"""
        WITH base AS (
            SELECT {TARGET_COL},
                   NTILE(10) OVER (PARTITION BY {MES_COL} ORDER BY {variable}) AS decil
            FROM raw_fe
            WHERE {TARGET_COL} IS NOT NULL AND {variable} IS NOT NULL
        )
        SELECT decil,
               COUNT(*) AS n_total,
               COUNT(*) FILTER (WHERE {TARGET_COL} = 'BAJA+2') AS n_baja2
        FROM base
        GROUP BY decil ORDER BY decil
    """
    df = con.execute(q).df()
    df["tasa_baja2"] = df["n_baja2"] / df["n_total"]
    return df


def regla_alerta(
    con: duckdb.DuckDBPyConnection, variable: str, direccion: str, tasa_base: float
) -> dict:
    """Evalúa la regla "variable en el decil extremo (1 si MENOR, 10 si MAYOR)"
    contra el denominador poblacional real: volumen, recall sobre el total de
    BAJA+2 del panel, precisión (tasa del decil) y lift vs tasa_base real."""
    dec = decil_poblacional(con, variable)
    decil_obj = 1 if direccion == "MENOR" else 10
    fila = dec[dec["decil"] == decil_obj].iloc[0]
    total_baja2 = dec["n_baja2"].sum()
    return {
        "variable": variable,
        "direccion": direccion,
        "decil_evaluado": decil_obj,
        "n_disparos": int(fila["n_total"]),
        "n_baja2_en_disparo": int(fila["n_baja2"]),
        "recall": fila["n_baja2"] / total_baja2 if total_baja2 else np.nan,
        "precision": fila["tasa_baja2"],
        "lift_vs_tasa_base": fila["tasa_baja2"] / tasa_base if tasa_base else np.nan,
    }


# --------------------------------------------------------------------------- #
# Ranking completo (todas las variables del FE, no solo las curadas) —
# para el apéndice. Solo Cohen's d (sin lift): evita traer a pandas los
# valores crudos de 633 columnas x 947k filas.
# --------------------------------------------------------------------------- #

# regr_slope() sobre ventanas de 3 meses casi verticales da pendientes
# extremas que desbordan VAR_POP en DuckDB. Se excluyen del ranking completo;
# quedan fuera del apéndice (ver hallazgos.txt, sección de límites).
EXCLUIR_APENDICE_PREFIJOS = ("slope_3_",)


def columnas_numericas_completas(df_churn: pd.DataFrame) -> list[str]:
    excluir = {ID_COL, MES_COL, TARGET_COL, CLUSTER_COL, "mes_relativo"}
    return [
        c
        for c in df_churn.columns
        if c not in excluir
        and pd.api.types.is_numeric_dtype(df_churn[c])
        and not c.startswith(EXCLUIR_APENDICE_PREFIJOS)
    ]


def cohen_d_completo(
    con: duckdb.DuckDBPyConnection,
    df_churn: pd.DataFrame,
    labels: pd.Series,
    variables: list[str],
    min_effect: float = 0.3,
) -> pd.DataFrame:
    """Cohen's d de las 633 variables del FE (menos slope_3_*), cada cluster
    vs. el fiel. Media/varianza del fiel en un solo pase DuckDB sobre toda la
    población; media/varianza del cluster en pandas sobre clusters_marcados.csv
    (ya tiene las 633 columnas, no hace falta traer nada más)."""
    exprs = ", ".join(f"AVG({v}) AS {v}__m, VAR_POP({v}) AS {v}__v" for v in variables)
    fila = con.execute(f"SELECT {exprs} FROM fieles").fetchone()
    fiel_media = pd.Series({v: fila[2 * i] for i, v in enumerate(variables)})
    fiel_var = pd.Series({v: fila[2 * i + 1] for i, v in enumerate(variables)})

    baja2 = df_churn.copy()
    if CLUSTER_COL not in baja2.columns:
        baja2 = baja2.merge(
            labels.rename(CLUSTER_COL), left_on=ID_COL, right_index=True
        )

    resultados = []
    for c in sorted(labels.unique()):
        cli = baja2[baja2[CLUSTER_COL] == c]
        cli_media, cli_var = cli[variables].mean(), cli[variables].var()
        sd_pool = np.sqrt((cli_var + fiel_var) / 2).replace(0, np.nan)
        cohen_d = (cli_media - fiel_media) / sd_pool
        for var in variables:
            d = cohen_d[var]
            if pd.isna(d) or abs(d) < min_effect:
                continue
            resultados.append(
                {
                    "cluster": c,
                    "variable": var,
                    "media_cluster": cli_media[var],
                    "media_fiel": fiel_media[var],
                    "cohen_d": d,
                    "magnitud": (
                        "GRANDE"
                        if abs(d) >= 0.8
                        else "MEDIO"
                        if abs(d) >= 0.5
                        else "CHICO"
                    ),
                }
            )
    out = pd.DataFrame(resultados)
    if out.empty:
        return out
    out["abs_d"] = out["cohen_d"].abs()
    return out.sort_values(["cluster", "abs_d"], ascending=[True, False]).reset_index(
        drop=True
    )


__all__ = [
    "BANDAS",
    "CLUSTERS_CSV",
    "CLUSTER_COL",
    "COLORES_CLUSTER",
    "DIR_BARRAS",
    "DIR_DISTRIBUCIONES",
    "DIR_RADARES",
    "DIR_TENDENCIAS",
    "DIR_VALOR",
    "DOMINIOS",
    "DOMINIO_DE",
    "ETIQUETAS",
    "EXCLUIR_APENDICE_PREFIJOS",
    "EXCLUIR_DIFERENCIADORAS",
    "FEATURES_CURADAS",
    "FE_PARQUET",
    "GRID",
    "GRUPO_COL",
    "ID_COL",
    "INK",
    "INK_2",
    "MES_COL",
    "NOMBRES_CLUSTER",
    "OUT_DIR",
    "SEED",
    "TARGET_COL",
    "TIPOS",
    "a_percentil_global",
    "agregar_mes_relativo",
    "calcular_lift",
    "calcular_medias_por_grupo",
    "cargar_churners",
    "cargar_fieles_curadas",
    "cohen_d_completo",
    "columnas_numericas_completas",
    "conectar_fe",
    "decil_poblacional",
    "dibujar_radar",
    "diferenciadoras_entre_clusters",
    "es_binaria",
    "etiqueta",
    "etiquetas_por_cliente",
    "log",
    "regla_alerta",
    "rent_fiel_anual",
    "rentabilidad_por_cluster",
    "senales_anticipacion",
    "tasa_base_baja2",
    "tendencias_por_mes_relativo",
    "variables_binarias",
    "variables_diferenciadoras",
]
