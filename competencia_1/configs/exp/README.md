# Experimentos de competencia_1

Cada `eNNN_*.yaml` es una hipótesis: hereda de `../base.yaml` y declara solo lo que cambia.
Los resultados (runs, cachés, estudios de Optuna) viven en `work/` y `db/`, que no se
versionan. Acá quedan las configuraciones y este registro.

La versión **definitiva** que debe reproducirse no se corre desde acá: está congelada en
`competencia_1/definitiva/` (ver el README principal, sección «Reproducir la entrega»).

## Experimentos

| Config | Propósito |
|---|---|
| `e001_baseline_catedra` | Baseline de cátedra (z470): originales, target BAJA+2, sin undersampling |
| `e002_fe_completo` | Igual, con todo el FE (aísla el efecto del FE) |
| `e003_optuna` | Optuna sobre el FE completo (50 trials) y estabilidad |
| `e004_ablacion_fe` | Ablación de familias de FE (14 variantes × 5 semillas) |
| `e005_drift_canaritos` | Caché con las columnas de drift y 20 canaritos |
| `e006_drift` | ¿Sirve conservar las columnas con drift? |
| `e007_canaritos`, `e008_seleccion_canaritos` | Importancia contra canaritos y selección top-K |
| `e009_optuna_orig_deltas`, `e010_optuna_orig_deltas_ext` | Optuna sobre originales + deltas (e010 amplía el espacio) |
| `e011_entrega` | Orig + deltas curados (292 features), params estables de e010, 20 semillas, final con 202106 |
| `e012_fold_unico_1mes`, `e014_e012_final1m` | Un solo fold (train 202104) con lag_1/delta_1 únicamente; final con 3 meses (e012) y con 1 mes (e014) |
| `e013_e011_final3m` | e011 con el final entrenado en 202104–202106 |
| `e015_fe_completo_foldA` | FE completo tuneado solo en el fold A (no se corrió) |
| `e016_orig_deltas_us025`, `e017_noche_us05` | Re-tuning con undersampling 0,25 / 0,5 y espacio ampliado (e017: 300 trials, hereda de e016; e016 sola no se corrió) |
| `e018_e011_final4m` | e013 con 4 meses de final (202103–202106) |
| `e019_e011_final3m_mindata` | e013 con `min_data_in_leaf` ×3 (`e019_params.json`) |
| `e020_orig_deltas_rank`, `e021_orig_deltas_rank_final` | Orig + deltas + rankings intra-mes (e021 no se corrió) |
| `e022_fe_deltas_todos`, `e023_fe_deltas_todos_final` | Deltas 1 y 2 sobre **todas** las variables (740 features); e023 es el final al estilo e019 (`e023_params.json`) |
| `e024_fe_deltas_lags_todos`, `e025_…_final` | e022 + lags 1 y 2 (1036 features); solo se corrió la validación |
| `e026_fe_deltas1_todos`, `e027_…_final` | Deltas de 1 mes sobre todas las variables (444 features) |

Archivos auxiliares: `e012_excluir.txt` (columnas a excluir), `e019_params.json` y
`e023_params.json` (hiperparámetros con `min_data_in_leaf` ya multiplicado por 3).

## Hallazgos (validación en 2 folds temporales: valid 202106 y 202105, ganancia meseta)

- **Tunear importa más que el FE:** 4 trials al azar sobre el FE completo ya rinden ~318 M
  contra 291 M con los parámetros de z470.
- **Promediar semillas aporta ~+20 M** sobre un modelo suelto en validación. Ojo: allí cada
  semilla ve otra muestra de undersampling; el modelo final entrena sin undersampling, así
  que el beneficio real puede ser menor.
- **FE completo vs solo originales: +19 M**, casi todo en 202106 (nulo en 202105). Quitar una
  sola familia queda dentro del ruido; `orig + deltas` (292 features) casi iguala al
  completo, y re-tuneado lo supera por ~18 M (354 M contra 336 M).
- **Piso de ruido ~±6 M:** sumar 20 columnas aleatorias dio +5,9 M en 5 de 5 semillas.
  Diferencias menores entre conjuntos de features no son interpretables.
- **Drift:** quitar las columnas con drift (z701) no pierde nada (−2,2 ± 1,4 M a favor de quitarlas).
- **Canaritos con importancia por ganancia no sirven para podar aquí:** absorben el 10 % de
  la ganancia y marcan el 97 % de las features como ruido; quedarse con el top 25 / 50 / 100
  pierde 76 / 38 / 22 M.
- **El leaderboard público engaña:** público y privado parten el mismo mes, así que con
  modelos de ganancia total parecida el ganador del público pierde el privado casi siempre
  (la desviación del privado es de ~10 M). No elegir entre candidatos por el público.
- **El modelo final con 3 meses (202104–202106) y `min_data_in_leaf` ×3 rinde mejor en el
  público** (e011 100,6 → e013 102,6 → e019 103,7 M). Un cuarto mes (202103, sin historia para
  los deltas) no suma (e018: 100,6 M).
- **Los deltas de 2 meses importan:** las variantes que los quitan salieron peores en el
  público (e012 contra e013: −3,9 M; e014 contra e011: −9,1 M; e027 contra e023: −7,4 M; las
  dos primeras cambian además los folds y los campos) aunque la validación casi no lo notaba.
  Agregar lags 1 y 2 sobre los deltas empeora la validación pareada 16,5 M (e024).
- **Más Optuna no mejora:** 300 trials, undersampling 0,5 y un espacio con `lambda_l2`,
  `min_gain_to_split` y bagging (e017) dan la misma mediana de validación (~360 M) y un
  público peor (99,1 M): la búsqueda llegó a una meseta de ~355–360 M.
- **Ensamblar no suma:** promediar rankings de e019 y e017 dio 101,9 M y de e019 y e023,
  102,3 M, cerca del promedio de sus partes.
- Los valores de validación se midieron en los mismos meses con los que se eligieron los
  parámetros: están algo inflados. No son una estimación del rendimiento en 202108.

## Bitácora de envíos (puntaje público, 11 000 envíos salvo que se indique)

El público tiene un ruido de ±3 M en la diferencia entre dos modelos parecidos; no usar
diferencias menores como criterio.

| Run | Config | Público | Notas |
|---|---|---|---|
| `20261004-235154_e011_entrega` | `e011_entrega` | 100,60 M | 10 000 y 12 000 no se subieron; 13 000: 96,41 M |
| `20261005-231845_e013_e011_final3m` | `e013_e011_final3m` | 102,63 M | |
| `20261005-223258_e012_fold_unico_1mes` | `e012_fold_unico_1mes` | 98,75 M | |
| `20261005-234740_e014_e012_final1m` | `e014_e012_final1m` | 91,54 M | |
| `20261006-001840_e018_e011_final4m` | `e018_e011_final4m` | 100,60 M | |
| `20261006-005701_e019_e011_final3m_mindata` | `e019_e011_final3m_mindata` | 103,73 M | 10 000 envíos: 104,50 M |
| `20261006-012932_e017_noche_us05` | `e017_noche_us05` | 99,10 M | 10 000 envíos: 97,50 M |
| ensamble `ens01_e019_e017` | `python -m competencia_1.ensamble` | 101,94 M (10 000) / 101,89 M (11 000) | |
| `20261007-075254_e023_fe_deltas_todos_final` | `e023_fe_deltas_todos_final` | **105,60 M** | 10 000 envíos: 104,03 M. **Entrega definitiva actual** |
| ensamble `ens02_e019_e023` | `python -m competencia_1.ensamble` | 102,27 M (10 500) | |
| `20261007-193742_e027_fe_deltas1_todos_final` | `e027_fe_deltas1_todos_final` | 98,18 M | 10 000 envíos: 96,11 M |

Los CSV y los runs están en `work/` (no se versionan). El sha256 de la entrega definitiva está
en `competencia_1/definitiva/entrega.json`.
