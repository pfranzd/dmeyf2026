# z402 — Feature engineering en SQL (DuckDB)

Implementación de las técnicas presentadas en
`monday/z402_Feature_Engineering_en_SQL.ipynb` (cátedra DMEyF 2026, upstream
`dmecoyfin/dmeyf2026`, commit `c9ff668`). El notebook de cátedra queda en modo
tutorial: construye el catálogo de técnicas con `select ... limit 10`
exploratorios y el `COPY` final exporta `competencia_01` sin tocarla
(verificado: `competencia_01_fe.csv` era byte-idéntico al original, md5
`1ebb0f2d67b6f987c36d1111788ee9ed`). Este experimento las aplica de verdad.

## Hipótesis

El churn en este dominio no se explica bien con valores nominales de un solo
mes: importa (a) cómo se combina la tenencia de Master/Visa, (b) la posición
relativa del cliente contra sus contemporáneos del mismo mes (los montos en
pesos sufren drift inflacionario a lo largo de 2021) y (c) la trayectoria
reciente — caídas de actividad, cambios de signo en el saldo, aceleración o
freno de la relación con el banco. `ctrx_quarter`, `mcaja_ahorro` y
`mpasivos_margen` ya mostraron el AUC más alto en el EDA univariado; la
apuesta es que sus lags, deltas y pendientes lo mejoran aún más.

## Cómo se construyó la query

Igual que `exp/z101_target_sql/target_sql.py`: la query se arma como **string
de Python** (f-strings) y se ejecuta con `con.execute()`, no con magics
`%%sql`. Cada fragmento de SELECT se genera junto con su registro en un
catálogo (`CATALOGO: list[dict]`), así el SQL y su documentación no pueden
desincronizarse — ver `_reg()` en `fe_sql.py`.

### Panel denso

El dataset **no es un panel denso**: ~93% de las combinaciones cliente×mes
existen. Un `lag()` directo sobre la tabla cruda saltea los meses donde el
cliente no aparece y le atribuye al mes siguiente el valor de hace 2 o 3
meses. Se reconstruye el panel completo con `cross join clientes × períodos`
(mismo patrón que `z101_target_sql`), se calculan todas las ventanas sobre
él, y solo al final se descartan las filas fantasma (`where mes_0 = 1`). El
script verifica esto explícitamente comparando, para un cliente con huecos,
el lag correcto contra el que devolvería la tabla cruda sin panel denso.

### Macros

```sql
CREATE OR REPLACE MACRO suma_segura(a, b) AS ifnull(a, 0) + ifnull(b, 0);
CREATE OR REPLACE MACRO ratio_seguro(a, b) AS ifnull(a, 0) / nullif(b, 0);
CREATE OR REPLACE MACRO delta_pct(act, prev) AS (act - prev) / nullif(abs(prev), 0);
```

`ratio_seguro` resuelve la tarea que el notebook deja planteada en la celda
14: un ratio que sobrevive tanto a NULL como a división por cero.

## Catálogo de variables creadas

| Bloque | Familia | Qué es | Ejemplos |
|---|---|---|---|
| `1a_tc_consolidado` | Creadas a partir de otras | `suma_segura(Master_x, Visa_x)` sobre 16 pares Master/Visa | `tc_msaldototal`, `tc_mlimitecompra` |
| `1b_tc_fechas` | Creadas a partir de otras | `least`/`greatest` sobre fechas relativas y status de TC | `tc_fvencimiento_menor`, `tc_status_max` |
| `1c_ratios_dominio` | Creadas a partir de otras | ratios null/zero-safe entre dos montos del mismo mes | `r_tc_uso_limite`, `r_engagement_digital` |
| `1d_agregados` | Creadas a partir de otras | sumas de familias de producto/canal | `m_activos_totales`, `c_trx_digitales` |
| `1e_flags_riesgo` | Creadas a partir de otras | flags binarios de señales de churn conocidas | `f_sin_payroll`, `f_saldo_negativo`, `f_sin_master` |
| `2_rank_intrames` | Ranking anti-drift | `percent_rank`/`ntile(10)` con `partition by foto_mes` | `pr_ctrx_quarter`, `d10_mcaja_ahorro` |
| `3_lags` | Lags | `lag(x, 1)`, `lag(x, 2)` por cliente sobre el panel denso | `lag_1_ctrx_quarter` |
| `4_deltas` | Deltas | `x - lag_n`, y delta porcentual | `delta_1_ctrx_quarter`, `deltapct_1_ctrx_quarter` |
| `5_ventanas` | Ventanas móviles | `avg`/`max`/`min`/`stddev` sobre t-3..t, y `x / avg_3(x)` | `avg_3_ctrx_quarter`, `ratioavg_3_mcaja_ahorro` |
| `6_tendencia` | Tendencia | `regr_slope(x, cliente_antiguedad)` sobre la ventana | `slope_3_ctrx_quarter` |
| `7_historia` | Historia en el panel | meses observados, posición dentro del panel | `meses_en_panel`, `antiguedad_panel` |

El detalle fila por fila (feature, expresión SQL exacta, columnas de origen)
queda en `work/z402_fe_catalogo.csv`, generado por el mismo script.

`cliente_antiguedad` se usa como eje X de las pendientes en vez de `foto_mes`
porque este último es un entero `YYYYMM` no lineal (202112 → 202201 salta 89).
Las fechas relativas (`Master_Fvencimiento`, etc.) son días respecto al
`foto_mes` (`consideraciones.txt`, nota 8), no fechas absolutas, por eso
`least`/`greatest` entre Master y Visa tienen sentido directo.

## Decisiones

- **Modo `curado` por defecto** (`MODO` en `fe_sql.py`): las series temporales
  (lags/deltas/ventanas/slope) se calculan sobre ~36 variables elegidas por
  AUC/importancia del EDA más las derivadas del bloque 1, no sobre las 150
  columnas numéricas. Cambiar a `MODO = "completo"` genera esas mismas
  transformaciones (un subconjunto algo más chico: sin `lag_2`/`std_3`/
  `ratioavg_3` para no explotar el tamaño) sobre **todas** las columnas
  numéricas, obtenidas por introspección de `describe base` — no hardcodeadas.
- **Ventana de 3** (`rows between 3 preceding and current row`, t-3..t): con
  solo 6 meses de historia es el rango que deja señal usable en la mayoría
  del panel; una ventana más larga dejaría los primeros meses casi vacíos.
- **`clase_ternaria` no se toca**: se preserva tal cual viene de
  `competencia_01.csv` (NULL parcial en 202107, NULL total en 202108).
- **Salida en Parquet y CSV**: el `COPY` escribe primero a Parquet (streaming,
  compresión ZSTD) y el CSV se genera leyendo ese Parquet, para no volver a
  materializar la query completa.
- `polars` se agregó a `pyproject.toml` (ya se usaba en otros scripts del
  repo sin estar declarado).

## Verificación

El propio `fe_sql.py` corre, al final, estos chequeos:

1. La cantidad de filas de salida es igual a la de entrada (el panel denso no
   agrega ni pierde filas).
2. La distribución de `clase_ternaria` por `foto_mes` es idéntica antes y
   después del FE.
3. Comparación explícita, para un cliente con meses faltantes, entre el
   `lag_1` correcto (panel denso) y el que devolvería un `lag()` ingenuo sobre
   la tabla cruda — y conteo global de filas donde difieren.
4. Spot check aritmético de lag/delta/avg/slope sobre un cliente con los 6
   meses completos.
5. % de NULL por bloque y por mes (se espera alto en `lag_2`/`std_3` durante
   202103-202104, por falta de historia).
6. Verificación de que ningún ratio produjo `inf`.

## Próximos pasos

- Entrenar LightGBM sobre `competencia_01_fe.parquet` con split temporal con
  gap de 2 meses (regla del proyecto) y medir con `ganancia_prob` de
  `dmeyf/metrics.py`, comparando contra un baseline sin FE.
- Si el modelo lo justifica, correr `MODO = "completo"` y comparar ganancia
  vs. tiempo de entrenamiento.
- Revisar `feature_importance` para podar el catálogo curado en la próxima
  iteración (`exp/z402_feature_engineering_v2/` si hace falta un rediseño).
