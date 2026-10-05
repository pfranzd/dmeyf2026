"""Listas de campos "curados" y resolución de `campos_serie` / `rankings.campos`."""

import logging

log = logging.getLogger("competencia_1.fe")

# Variables con mayor AUC / importancia según el EDA (docs/Hallazgos EDA Zulip.txt)
# más derivadas de la familia 1 (fe_sql.py, CAMPOS_SERIE_CURADO).
CAMPOS_SERIE_CURADO = [
    "ctrx_quarter",
    "chomebanking_transacciones",
    "cmobile_app_trx",
    "ccajas_transacciones",
    "catm_trx",
    "ctarjeta_debito_transacciones",
    "ccallcenter_transacciones",
    "mcuentas_saldo",
    "mcaja_ahorro",
    "mcuenta_corriente",
    "mpasivos_margen",
    "mactivos_margen",
    "mrentabilidad",
    "mcomisiones",
    "mcomisiones_mantenimiento",
    "cpayroll_trx",
    "mpayroll",
    "cproductos",
    "mautoservicio",
    "matm",
    "mtarjeta_visa_consumo",
    "mtarjeta_master_consumo",
    "mprestamos_personales",
    "mtransferencias_recibidas",
    "mtransferencias_emitidas",
    "cdescubierto_preacordado",
    # derivadas de la familia 1
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

CAMPOS_RANK_CURADO = [
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

# Nunca entran a series ni rankings: claves, target y columnas auxiliares del panel.
NO_SERIE = {
    "pk_cliente",
    "pk_mes",
    "mes_0",
    "numero_de_cliente",
    "foto_mes",
    "clase_ternaria",
}
TIPOS_NUM = (
    "BIGINT",
    "INTEGER",
    "DOUBLE",
    "FLOAT",
    "DECIMAL",
    "HUGEINT",
    "SMALLINT",
    "TINYINT",
)


def resolver_campos(
    spec: str | list[str],
    curado: list[str],
    esquema: dict[str, str],
    excluidos: set[str],
    etiqueta: str,
) -> list[str]:
    """Resuelve una especificación de campos contra el esquema de `base`.

    - "curado": la lista curada, salvo columnas inexistentes (familia apagada) o excluidas.
    - "todos": todas las numéricas, salvo claves y excluidas (p. ej. drop_drift).
    - lista explícita: se respeta tal cual; un nombre inexistente es error.
    """
    if isinstance(spec, list):
        faltan = [c for c in spec if c not in esquema]
        if faltan:
            raise ValueError(f"{etiqueta}: campos inexistentes: {faltan}")
        return list(spec)
    if spec == "curado":
        usables = [c for c in curado if c in esquema and c not in excluidos]
        omitidos = [c for c in curado if c not in usables]
        if omitidos:
            log.info(
                "%s curado: se omiten %d campos no disponibles/excluidos: %s",
                etiqueta,
                len(omitidos),
                omitidos,
            )
        return usables
    if spec == "todos":
        return [
            c
            for c, t in esquema.items()
            if c not in NO_SERIE
            and c not in excluidos
            and t.upper().startswith(TIPOS_NUM)
        ]
    raise ValueError(f"{etiqueta}: valor inválido {spec!r} (curado | todos | lista)")
