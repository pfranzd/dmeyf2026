# competencia_1 — Primera Competencia DMEyF 2026

Pipeline reproducible (solo scripts Python, sin notebooks) que genera los CSV con los
`numero_de_cliente` a estimular en un período objetivo (hoy `202108`, parametrizable en
`periodos.target`). Modelo: LightGBM, hiperparámetros con Optuna, semillerío con promedio.

La métrica de negocio es la ganancia: +1.072.500 por cada BAJA+2 estimulado y −27.500 por cada
otro cliente estimulado (`dmeyf/metrics.py`). Siempre se mide sobre BAJA+2, aunque el modelo
se entrene con BAJA+1 + BAJA+2.

## Uso rápido

Desde la raíz del repo, con el `.venv` activo:

```bash
# correr un experimento completo
python -m competencia_1.main --config competencia_1/configs/exp/e011_entrega.yaml

# cambiar parámetros sin tocar archivos (notación con puntos, valor en YAML)
python -m competencia_1.main --config competencia_1/configs/exp/e011_entrega.yaml \
    --set final.n_semillas=5 --set "salida.envios=[9000,10000]"

# elegir etapas desde la CLI (reemplaza etapas.* del config)
python -m competencia_1.main --config <exp>.yaml --etapas datos,features,validacion

# repetir un run previo con la configuración congelada
python -m competencia_1.main --reproducir work/competencia_1/runs/<run_id>

# tests y estilo
python -m pytest competencia_1/tests
ruff format competencia_1 && ruff check competencia_1
```

**Datos de entrada** (no se versionan): `datasets/processed/competencia_01.csv` (con
`clase_ternaria`). Para rehacer el target desde el crudo: `datos.reconstruir_target=true` con
`datasets/raw/competencia_01_crudo.csv`.

## Etapas

Se ejecutan en este orden; cada una se activa con `etapas.<nombre>` en el config.

| Etapa | Módulo | Qué hace |
|---|---|---|
| `datos` | `pipeline/datos.py` | CSV → parquet tipado (IDs `BIGINT`), valida unicidad, clases y meses con target completo |
| `features` | `pipeline/features/` | Feature engineering en DuckDB, configurable por familias; caché por `fe_hash` |
| `optuna` | `pipeline/optimizacion.py` | Búsqueda de hiperparámetros con validación temporal (SQLite, reanudable) |
| `estabilidad` | `pipeline/estabilidad.py` | Top-k trials × N semillas: elige por mediana, mide optimismo y simula public/private |
| `ablacion` | `pipeline/ablacion.py` | Compara conjuntos de features (pareado por semilla) |
| `canaritos` | `pipeline/canaritos.py` | Importancia contra columnas aleatorias |
| `validacion` | `pipeline/evaluacion.py` | Evalúa los hiperparámetros finales en los folds; deja la curva de ganancia |
| `final` | `pipeline/final.py` | Semillerío: N modelos con la misma config y distinta semilla |
| `salida` | `pipeline/salida.py` | Escribe y valida los CSV de entrega |

## Configuración

Todo vive en `configs/base.yaml` (única fuente de defaults); un experimento hereda con
`hereda: ../base.yaml` y declara solo lo que cambia (`configs/exp/`). Claves desconocidas o
faltantes son un error, así que un typo no se ignora en silencio. Lo más usado:

| Clave | Para qué |
|---|---|
| `periodos.target` | Mes a predecir. Los folds y el mes final se derivan con `periodos.gap` (2 meses) |
| `target.positivos` | Clases con y=1 al entrenar (la ganancia siempre se mide sobre BAJA+2) |
| `fe.*` | Familias de FE: intra-fila, lags, deltas, ventanas, tendencia, rankings, historia, canaritos |
| `fe.drop_drift` | Columnas con drift que no entran (monday/z701) |
| `dataset.excluir_bloques` / `excluir_features` | Selección de columnas sobre la caché, sin reconstruir el FE |
| `dataset.undersampling` | Fracción de CONTINUA en entrenamiento y validación (el final entrena con todo) |
| `optuna.n_trials`, `optuna.espacio` | Búsqueda: `n_trials` es el total deseado; el estudio se reanuda |
| `final.params_desde` | `manual`, `optuna:<estudio\|auto>`, `estable:<estudio\|auto>` o `archivo:<ruta>` |
| `final.n_semillas` (1–20) / `final.semillas` | Cantidad de modelos del semillerío |
| `salida.envios` | `auto`, un número o una lista de cortes |

Bloques para `excluir_bloques`: `1a_tc_consolidado`, `1b_tc_fechas`, `1c_ratios_dominio`,
`1d_agregados`, `1e_flags_riesgo`, `2_rank_intrames`, `3_lags`, `4_deltas`, `5_ventanas`,
`6_tendencia`, `7_historia`, `8_canaritos`, y los seudo-bloques `originales` y `derivadas`.

## Reproducibilidad

- **Semillas:** las 5 del curso (`config/semillas.py`) van primero; hasta 20 se derivan de
  forma determinística de `semilla_maestra`. Un semillerío de 20 extiende al de 5 sin cambiar
  los primeros modelos. Las semillas explícitas deben ser primos entre 100003 y 999983.
- **Determinismo:** LightGBM con `seed`, `deterministic=true` y `num_threads` fijo; el
  undersampling usa un hash propio estable; la salida de FE se ordena por (`foto_mes`,
  `numero_de_cliente`). Dos corridas del mismo config dan modelos, probabilidades y CSV
  idénticos byte a byte (verificado por sha256).
- **Cachés:** `work/competencia_1/features/<fe_hash>/` se reutiliza mientras no cambie la
  sección `fe`, `FE_VERSION` ni la base. Construir el FE por defecto tarda ~10 min y usa
  ~10 GB de RAM; solo se paga una vez.
- **Un run = una carpeta** `work/competencia_1/runs/<run_id>/`: `config_resuelta.yaml`,
  `run.log`, `meta.json` (commit y estado `dirty` de git, versiones, parámetros, métricas,
  sha256 de cada archivo generado), `probas/`, `modelos/`, `submits/`. `work/competencia_1/runs.csv`
  resume todos los runs. Los estudios de Optuna están en `db/competencia_1.db`.
- **Hiperparámetros ↔ features:** cada `params/<estudio>.json` guarda el `fe_hash` con el que
  se optimizó y `final` se niega a usarlo con otras features (salvo
  `final.params_ignorar_fe_hash=true`, con warning).
- Antes de reproducir una entrega, comprobar que `meta.json` indique `dirty: false`.

## Formato del CSV de entrega

Solo `numero_de_cliente`, un entero por línea, **sin encabezado**, sin notación científica ni
decimales. `salida.py` relee cada archivo como texto y exige que toda línea cumpla `^\d+$`,
que no haya duplicados, que el largo sea el esperado y que todos los IDs sean clientes del
período objetivo. Si algo falla, borra el archivo y lanza el error. Nombre:
`<run_id>_<modelo>_e<envios>.csv`, donde `<modelo>` es `s<semilla>` o `promedio`.

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
| `e011_entrega` | Entrega: `orig + deltas`, parámetros estables de e010, 20 semillas |

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
- Los valores de validación se midieron en los mismos meses con los que se eligieron los
  parámetros: están algo inflados. No son una estimación del rendimiento en 202108.

## Bitácora de entregas

| Fecha | Run | Config | Código | Archivo | sha256 | Resultado |
|---|---|---|---|---|---|---|
| 2026-10-05 | `20261004-235154_e011_entrega` | `configs/exp/e011_entrega.yaml` | `aa6f34c` | `…_promedio_e11000.csv` (principal) | `cebde444ffac…f4a6` | _completar_ |
| 2026-10-05 | ídem | ídem | ídem | `…_promedio_e10000.csv` | `f86168bc66a5…04d0` | _completar_ |
| 2026-10-05 | ídem | ídem | ídem | `…_promedio_e12000.csv` | `6d9621b1dea0…9864` | _completar_ |

Los CSV están en `work/competencia_1/runs/20261004-235154_e011_entrega/submits/` (no se
versionan). Con el mismo commit y la misma base, `python -m competencia_1.main --config
competencia_1/configs/exp/e011_entrega.yaml` regenera los mismos archivos (mismo sha256).
Cada nueva entrega se agrega a esta tabla junto con su run y su resultado.
