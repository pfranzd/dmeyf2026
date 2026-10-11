"""Familia 9: señales de comportamiento previas a la baja (literatura de churn bancario).

Eventos discretos que un árbol solo aproxima con deltas: deja de entrar el sueldo, se pierde
un producto, sube el status de una tarjeta, se vacía la cuenta, cae la actividad. Recency,
frecuencia y monto (RFM) como en Van den Poel & Larivière (2004) y Verbeke et al. (2012).

Todo mira como máximo 2 meses atrás (ventana de 3 puntos t-2..t), igual que el FE de e032:
así una fila de train (202105-202106) y una de 202108 ven la misma profundidad de historia.
Si la ventana no está completa (202103, 202104) la recency queda NULL salvo que el evento
aparezca dentro de lo disponible.

Se calcula sobre el panel denso (filas fantasma incluidas: sus montos son NULL y los
agregados null-safe de la familia 1 valen 0). Requiere fe.intrafila.tc_consolidado y agregados.
"""

from competencia_1.pipeline.features.catalogo import Catalogo

BLOQUE = "9_comportamiento"
# Índice de mes continuo: con el panel denso, diferencias de índice = meses transcurridos.
IDX = "(pk_mes // 100 * 12 + pk_mes % 100)"
VENTANAS = {
    "comp_orden": "",
    "comp_previos": "rows between 2 preceding and 1 preceding",
    "comp_3": "rows between 2 preceding and current row",
}
# (nombre, condición del evento "hay actividad" en el mes)
RECENCY = [
    ("meses_desde_payroll", "ifnull(cpayroll_trx, 0) + ifnull(cpayroll2_trx, 0) > 0"),
    ("meses_desde_trx", "c_trx_total > 0"),
    ("meses_desde_consumo_tc", "tc_mconsumototal > 0"),
]


def ventanas_comportamiento() -> list[str]:
    return [
        f"{n} as (partition by pk_cliente order by pk_mes {marco})"
        for n, marco in VENTANAS.items()
    ]


def _recency(cond: str) -> str:
    """Meses desde el último mes con `cond` en t-2..t; 3 si no ocurrió con la ventana completa."""
    ultimo = f"max(case when {cond} then {IDX} end) over comp_3"
    return (
        f"case when {ultimo} is not null then {IDX} - {ultimo} "
        f"when count(*) over comp_3 = 3 then 3 end"
    )


def sql_comportamiento_series(cat: Catalogo) -> str:
    """Fragmento del SELECT de `series` (ventanas sobre el panel)."""
    payroll = "if(ifnull(cpayroll_trx, 0) + ifnull(cpayroll2_trx, 0) > 0, 1, 0)"
    frag = cat.reg(
        BLOQUE,
        "f_perdio_payroll",
        f"if(max({payroll}) over comp_previos = 1 and {payroll} = 0, 1, 0)",
        "tenía sueldo en t-1 o t-2 y en t no",
    )
    frag += cat.reg(
        BLOQUE,
        "n_productos_perdidos",
        "greatest(max(c_productos_contratados) over comp_previos - c_productos_contratados, 0)",
        "familias de producto perdidas contra el máximo de t-1/t-2",
    )
    for tc in ("Visa", "Master"):
        frag += cat.reg(
            BLOQUE,
            f"f_{tc.lower()}_status_sube",
            f"if(ifnull({tc}_status, 0) > max(ifnull({tc}_status, 0)) over comp_previos, 1, 0)",
            f"{tc}_status empeora contra t-1/t-2 (camino al cierre)",
        )
    for nombre, cond in RECENCY:
        frag += cat.reg(BLOQUE, nombre, _recency(cond), f"recency: {cond}")
    frag += cat.reg(
        BLOQUE,
        "r_saldo_vs_max",
        "ratio_seguro(mcuentas_saldo, nullif(greatest(max(mcuentas_saldo) over comp_3, 0), 0))",
        "drenaje: saldo / máximo de t-2..t",
    )
    frag += cat.reg(
        BLOQUE,
        "r_activos_vs_max",
        "ratio_seguro(m_activos_totales, nullif(max(m_activos_totales) over comp_3, 0))",
        "drenaje: activos / máximo de t-2..t",
    )
    frag += cat.reg(
        BLOQUE,
        "racha_caida_ctrx",
        "if(ctrx_quarter < lag(ctrx_quarter, 1) over comp_orden, 1, 0)"
        " + if(ctrx_quarter < lag(ctrx_quarter, 1) over comp_orden"
        " and lag(ctrx_quarter, 1) over comp_orden < lag(ctrx_quarter, 2) over comp_orden, 1, 0)",
        "meses consecutivos con caída de ctrx_quarter (0-2)",
    )
    return frag


def sql_comportamiento_mes(cat: Catalogo) -> str:
    """Fragmento intra-fila (no necesita historia)."""
    return cat.reg(
        BLOQUE,
        "r_transf_emitidas_recibidas",
        "ratio_seguro(mtransferencias_emitidas, mtransferencias_recibidas)",
        "salida de fondos por transferencias",
    )


def sql_comportamiento_final(cat: Catalogo) -> str:
    """RFM por quintiles dentro del mes (sobre filas reales, en el SELECT final)."""
    q = "least(1 + floor(5 * percent_rank() over (partition by foto_mes order by {e})), 5)"
    r = q.format(e="ifnull(meses_desde_trx, 3) desc")
    f = q.format(e="ifnull(ctrx_quarter, 0)")
    m = q.format(e="ifnull(mcuentas_saldo, 0)")
    return cat.reg(
        BLOQUE,
        "rfm_score",
        f"({r} + {f} + {m})::INTEGER",
        "recency + frecuencia (ctrx_quarter) + monto (mcuentas_saldo), quintiles del mes",
    )
