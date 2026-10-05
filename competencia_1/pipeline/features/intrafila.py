"""Familia 1: variables creadas a partir de otras, dentro de la misma fila.

Portado de exp/z402_feature_engineering/fe_sql.py (bloque 1; upstream
monday/z402, commit c9ff668 de dmecoyfin/dmeyf2026). Cada sub-familia se
activa desde `fe.intrafila.*`. Orden de generación (importa porque el SELECT
referencia alias previos): 1a tc -> 1b fechas -> 1d agregados -> 1c ratios -> 1e flags.
"""

from competencia_1.pipeline.config import IntraFila
from competencia_1.pipeline.features.catalogo import Catalogo

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

# (feature, expresión, columnas de origen). Si alguna origen está en `drop`, se omite.
FECHAS_TC = [
    (
        "tc_fvencimiento_menor",
        "least(Master_Fvencimiento, Visa_Fvencimiento)",
        ["Master_Fvencimiento", "Visa_Fvencimiento"],
    ),
    (
        "tc_fvencimiento_mayor",
        "greatest(Master_Fvencimiento, Visa_Fvencimiento)",
        ["Master_Fvencimiento", "Visa_Fvencimiento"],
    ),
    (
        "tc_finiciomora_menor",
        "least(Master_Finiciomora, Visa_Finiciomora)",
        ["Master_Finiciomora", "Visa_Finiciomora"],
    ),
    (
        "tc_fultimo_cierre_menor",
        "least(Master_fultimo_cierre, Visa_fultimo_cierre)",
        ["Master_fultimo_cierre", "Visa_fultimo_cierre"],
    ),
    (
        "tc_fechaalta_mayor",
        "greatest(Master_fechaalta, Visa_fechaalta)",
        ["Master_fechaalta", "Visa_fechaalta"],
    ),
    (
        "tc_fechaalta_menor",
        "least(Master_fechaalta, Visa_fechaalta)",
        ["Master_fechaalta", "Visa_fechaalta"],
    ),
    (
        "tc_delinquency_max",
        "greatest(ifnull(Master_delinquency, 0), ifnull(Visa_delinquency, 0))",
        ["Master_delinquency", "Visa_delinquency"],
    ),
    (
        "tc_status_max",
        "greatest(ifnull(Master_status, 0), ifnull(Visa_status, 0))",
        ["Master_status", "Visa_status"],
    ),
]

RATIOS = [
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
    ("r_comisiones_rentabilidad", "mcomisiones", "mrentabilidad", "peso de comisiones"),
    ("r_rentabilidad_producto", "mrentabilidad", "cproductos", "rentabilidad unitaria"),
    ("r_activos_pasivos", "mactivos_margen", "mpasivos_margen", "mix de margen"),
    ("r_prestamos_saldo", "m_prestamos_totales", "m_activos_totales", "apalancamiento"),
    ("r_trx_producto", "ctrx_quarter", "cproductos", "intensidad de uso"),
    (
        "r_antiguedad_edad",
        "cliente_antiguedad",
        "cliente_edad * 12",
        "fracción de vida como cliente",
    ),
]

FLAGS = [
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


def _suma(campos: list[str]) -> str:
    """Suma null-safe de N campos (la macro `suma_segura` es binaria)."""
    return " + ".join(f"ifnull({c}, 0)" for c in campos)


def _cuenta_positivos(campos: list[str]) -> str:
    return " + ".join(f"if(ifnull({c}, 0) > 0, 1, 0)" for c in campos)


def _agregados() -> list[tuple[str, str, str]]:
    return [
        (
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
        ),
        (
            "m_prestamos_totales",
            _suma(
                [
                    "mprestamos_personales",
                    "mprestamos_prendarios",
                    "mprestamos_hipotecarios",
                ]
            ),
            "mprestamos_*",
        ),
        (
            "c_prestamos_totales",
            _suma(
                [
                    "cprestamos_personales",
                    "cprestamos_prendarios",
                    "cprestamos_hipotecarios",
                ]
            ),
            "cprestamos_*",
        ),
        (
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
        ),
        (
            "c_trx_digitales",
            _suma(["chomebanking_transacciones", "cmobile_app_trx"]),
            "homebanking + mobile app",
        ),
        (
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
        ),
        (
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
        ),
        (
            "m_transferencias_neto",
            "ifnull(mtransferencias_recibidas, 0) - ifnull(mtransferencias_emitidas, 0)",
            "mtransferencias_recibidas, mtransferencias_emitidas",
        ),
        (
            "c_cheques_rechazados",
            _suma(["ccheques_depositados_rechazados", "ccheques_emitidos_rechazados"]),
            "ccheques_*_rechazados",
        ),
        (
            "m_descuentos_total",
            _suma(
                [
                    "mcajeros_propios_descuentos",
                    "mtarjeta_visa_descuentos",
                    "mtarjeta_master_descuentos",
                ]
            ),
            "m*_descuentos",
        ),
        (
            "c_productos_contratados",
            _cuenta_positivos(PRODUCTOS_FLAG),
            "conteo de familias de producto con tenencia > 0",
        ),
    ]


def sql_intrafila(cfg: IntraFila, cat: Catalogo, drop: set[str]) -> str:
    """Fragmento de SELECT con las sub-familias activas de la familia 1."""
    frag = ""

    if cfg.tc_consolidado:  # 1a
        for suf in PARES_TC:
            frag += cat.reg(
                "1a_tc_consolidado",
                f"tc_{suf}",
                f"suma_segura(Master_{suf}, Visa_{suf})",
                f"Master_{suf} + Visa_{suf}",
            )

    if cfg.tc_fechas:  # 1b
        for nombre, expr, origen in FECHAS_TC:
            if drop & set(origen):
                continue  # derivada de una columna con drift: no se reintroduce
            frag += cat.reg("1b_tc_fechas", nombre, expr, ", ".join(origen))

    if cfg.agregados:  # 1d (antes que los ratios que los usan)
        for nombre, expr, nota in _agregados():
            frag += cat.reg("1d_agregados", nombre, expr, nota)

    if cfg.ratios:  # 1c (usan 1a y 1d)
        for nombre, num, den, nota in RATIOS:
            frag += cat.reg(
                "1c_ratios_dominio",
                nombre,
                f"ratio_seguro({num}, {den})",
                f"{num} / {den} — {nota}",
            )
        frag += cat.reg(
            "1c_ratios_dominio",
            "r_engagement_digital",
            "ratio_seguro(c_trx_digitales, c_trx_digitales + c_trx_presenciales)",
            "digital / (digital + presencial) — traslado de canal",
        )

    if cfg.flags:  # 1e
        for nombre, expr, nota in FLAGS:
            frag += cat.reg("1e_flags_riesgo", nombre, expr, nota)

    return frag
