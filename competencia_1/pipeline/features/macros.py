"""Macros SQL null-safe (portadas de fe_sql.py, con `ratio_seguro` corregido)."""

import duckdb

MACROS = [
    # suma que trata NULL como 0 (el NULL de una tarjeta = "no tiene el producto")
    "CREATE OR REPLACE MACRO suma_segura(a, b) AS ifnull(a, 0) + ifnull(b, 0)",
    # a / b con b = 0 -> NULL. Un numerador NULL sigue siendo NULL (en z402 pasaba a 0)
    "CREATE OR REPLACE MACRO ratio_seguro(a, b) AS a / nullif(b, 0)",
    "CREATE OR REPLACE MACRO delta_pct(act, prev) AS (act - prev) / nullif(abs(prev), 0)",
]


def crear_macros(con: duckdb.DuckDBPyConnection) -> None:
    for m in MACROS:
        con.execute(m)
