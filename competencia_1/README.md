# competencia_1 — Primera Competencia DMEyF 2026

Pipeline reproducible (solo scripts Python, sin notebooks) que genera los CSV con los
`numero_de_cliente` a estimular en un período objetivo (hoy `202108`, parametrizable en
`periodos.target`). Modelo: LightGBM, hiperparámetros con Optuna, semillerío con promedio.

La métrica de negocio es la ganancia: +1.072.500 por cada BAJA+2 estimulado y −27.500 por cada
otro cliente estimulado (`dmeyf/metrics.py`). Siempre se mide sobre BAJA+2, aunque el modelo
se entrene con BAJA+1 + BAJA+2.

## Reproducir la entrega

La entrega definitiva está **congelada** en [`definitiva/`](definitiva/): una configuración
autocontenida (`config.yaml`), los hiperparámetros finales (`params.json`), los **24 modelos ya
entrenados** (`modelos/`, ~140 MB) y el origen y el sha256 esperado del CSV (`entrega.json`). No
depende de `configs/exp/`, de `work/` ni de Optuna: solo del dataset crudo de la cátedra.

- **Reproducción rápida (por defecto, ~15 min, ~10 GB de RAM):** reconstruye los datos y las
  features desde el crudo y **solo predice** con los modelos guardados. Un modelo guardado y
  vuelto a cargar predice idéntico bit a bit, así que el CSV es el mismo.
- **Reproducción completa (`--entrenar`, ~80 min):** ignora los modelos guardados y reentrena
  los 24 desde cero (12 conjuntos de hiperparámetros × 2 semillas). Da el mismo CSV.

Ambos caminos terminan con `RESULTADO: OK, coincide` si el sha256 es el de `entrega.json`.

**Linux / VM de GCP** (crea el entorno, descarga el crudo y compara el sha256):

```bash
git clone <este repo> && cd <repo>
bash competencia_1/scripts/reproducir_entrega.sh
```

**Windows (PowerShell)**, con Python ≥ 3.11:

```powershell
python -m venv .venv; .venv\Scripts\python -m pip install -r competencia_1/requirements.txt
# descargar a datasets/raw/competencia_01_crudo.csv:
#   https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv
.venv\Scripts\python -m competencia_1.entrega reproducir
```

Termina con `RESULTADO: OK, coincide` si el CSV generado tiene el mismo sha256 que el entregado.
Para reentrenar todo: `... reproducir --entrenar` (en Linux, `bash .../reproducir_entrega.sh --entrenar`).
Todo se escribe en `work/entrega/` (no pisa los experimentos); `DMEYF_WORK=<dir>` lo cambia, por
ejemplo a un disco montado en GCP. El CSV queda en `work/entrega/runs/<run>/submits/`.

La igualdad byte a byte requiere las mismas versiones de las librerías (`requirements.txt`,
también registradas en `entrega.json`) y `lgbm.fijos.num_threads` sin cambios. Si el sha256 no
coincide pero el resto del log es igual, lo primero a revisar es la versión de LightGBM/DuckDB.

### Cambiar la entrega definitiva

Se elige un run ya ejecutado (por ejemplo desde `work/competencia_1/runs.csv`) y se congela:

1. `python -m competencia_1.entrega promover --run work/competencia_1/runs/<run_id> --envios 11000`
   (reescribe `definitiva/`; `--csv-procesado` usa el CSV procesado local en vez de rehacer el
   target desde el crudo; `--modelo s<semilla>` promueve un modelo suelto en vez del promedio;
   `--con-modelos` copia los modelos entrenados del run a `definitiva/modelos/`, así la
   reproducción solo predice).
2. Revisar el diff de `competencia_1/definitiva/`.
3. Opcional: `python -m competencia_1.entrega reproducir` debe dar `OK, coincide`.
4. Commit.

`promover` falla, sin escribir nada, si el run no terminó en `ok`, no corrió la etapa `final` o
no tiene un CSV con ese corte de envíos. Anotar el puntaje público en `entrega.json`
(`resultado_publico`) y en la bitácora de [`configs/exp/README.md`](configs/exp/README.md).

### Publicar la entrega oficial (repo de entregas)

La cátedra pide un link a GitHub con una carpeta que permita replicar exactamente el CSV
entregado. Para eso hay un repositorio aparte, [`dmeyf2026-entregas`](https://github.com/pfranzd/dmeyf2026-entregas)
(clonado en `../dmeyf2026-entregas`), que contiene solo lo entregable: código, entorno con
versiones fijas (`requirements-lock.txt`), hiperparámetros, modelos entrenados y la optimización
de Optuna que los originó. Nunca se suben los CSV de la competencia.

Elegir qué experimento es el oficial es un solo comando (o decirle a Claude "pasá <experimento> a
oficial": la skill `publicar-entrega` lo hace con todas las validaciones):

```powershell
python -m competencia_1.publicar --run work/competencia_1/runs/<run_id> --envios 10500 `
  --crudo-local datasets/raw/competencia_01_crudo.csv
```

Congela el run (`promover`), exporta la carpeta, corre la entrega exportada en un venv nuevo
instalado desde el lock y exige que el CSV sea **idéntico** (sha256 y bytes) al del run; recién
entonces hace commit y `push` y verifica con `ls-remote` que el remoto quedó en ese commit. Si
algo falla no se sube nada. `--validar entrenar|completo` valida reentrenando / rehaciendo
también Optuna; `--inicializar` es solo para el primer commit del repo vacío.
La entrega exportada se replica con `bash reproducir.sh [predecir|entrenar|completo]`
(`predecir` ~15 min, `entrenar` ~80 min, `completo` ~2,5 h) o con `entrega.ipynb`.

## Estructura

```
competencia_1/
├── README.md, requirements.txt
├── main.py            # pipeline por etapas (experimentos)
├── entrega.py         # promover / reproducir la entrega definitiva
├── ensamble.py        # promedia las probabilidades de varios runs
├── definitiva/        # entrega congelada: config.yaml, params.json, entrega.json
├── configs/base.yaml  # defaults; configs/exp/ = un YAML por experimento + su README
├── scripts/           # preparar_entorno.sh y reproducir_entrega.sh (Linux / GCP)
├── pipeline/          # código; tests/ = pytest
```

No se versionan (ver `.gitignore`): `work/` (runs, cachés, CSV), `db/` (estudios de Optuna),
`datasets/`.

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

## Cómo experimentar

**Ciclo:** una hipótesis = un YAML nuevo en `configs/exp/` con `hereda: ../base.yaml`, un
`experimento:` propio y solo lo que cambia. Filtro rápido con `ablacion` (varias semillas, sin
Optuna); confirmación con `optuna` + `estabilidad`; la entrega sale de un `eNNN_entrega.yaml`
nuevo que se commitea y se anota en la bitácora (`configs/exp/README.md`); para dejarla como
entrega definitiva, `python -m competencia_1.entrega promover`.

```bash
python -m competencia_1.main --config competencia_1/configs/exp/eNNN_mi_idea.yaml
python -m competencia_1.main --config <yaml> --set optuna.n_trials=20
python -m competencia_1.main --config <yaml> --etapas optuna,estabilidad
```

### Dónde cambiar cada cosa

| Quiero cambiar… | Archivo | Qué tocar |
|---|---|---|
| Features que usa el modelo (sin reconstruir el FE) | `configs/exp/<tu>.yaml` | `dataset.excluir_bloques`, `dataset.excluir_features` |
| Comparar conjuntos de features | `configs/exp/<tu>.yaml` | `ablacion.variantes` y `etapas.ablacion: true` (ver `e004`) |
| Variables sobre las que se calculan lags, deltas y ventanas | `configs/exp/<tu>.yaml` | `fe.campos_serie`: `curado`, `todos` o lista (reconstruye el FE, ~10 min) |
| Activar o apagar familias de FE | `configs/exp/<tu>.yaml` | `fe.lags`, `fe.deltas`, `fe.ventanas`, `fe.tendencia`, `fe.rankings`, `fe.intrafila.*` |
| Un ratio o flag nuevo | `pipeline/features/intrafila.py` | Fila nueva en `RATIOS` o `FLAGS`; subir `FE_VERSION` en `pipeline/features/__init__.py` |
| Variables de las series o rankings curados | `pipeline/features/campos.py` | `CAMPOS_SERIE_CURADO`, `CAMPOS_RANK_CURADO` |
| Una familia de features nueva | `pipeline/features/` | `sql_<familia>()` nuevo, enchufado en `query.py`, flag en `FE` de `pipeline/config.py` y en `configs/base.yaml`; subir `FE_VERSION` |
| Espacio de búsqueda de Optuna | `configs/exp/<tu>.yaml` | `optuna.espacio` (`tipo`, `low`, `high`, `log`), `optuna.n_trials` |
| Parámetros fijos de LightGBM | `configs/base.yaml` o tu YAML | `lgbm.fijos` |
| Target | `configs/exp/<tu>.yaml` | `target.positivos` |
| Undersampling | `configs/exp/<tu>.yaml` | `dataset.undersampling` |
| Meses del modelo final / mes objetivo | `configs/exp/<tu>.yaml` | `periodos.meses_final` / `periodos.target` |
| Semillerío y cortes de envíos | `configs/exp/<tu>.yaml` | `final.n_semillas`, `final.promedio`, `salida.envios` |
| Hiperparámetros del modelo final | `configs/exp/<tu>.yaml` | `final.params_desde: estable:<estudio>` (nombres en `work/competencia_1/params/`) |
| Algoritmo o código de entrenamiento | `pipeline/modelo.py`, `pipeline/final.py` | Entrenamiento y predicción |
| Una clave de config nueva | `pipeline/config.py` **y** `configs/base.yaml` | Si falta en uno, es un error |
| Una etapa nueva | `pipeline/<etapa>.py` y `main.py` | Agregar a `ORDEN_ETAPAS` y registrar en `ETAPAS` |
| La métrica de ganancia | `dmeyf/metrics.py` | `ganancia_prob`, `curva_ganancia`, `ganancia_meseta` |
| Cambiar la entrega definitiva | `python -m competencia_1.entrega promover` | `--run`, `--envios` (ver «Reproducir la entrega») |

Cuidados: no poner `max_bin` ni `feature_pre_filter` en `optuna.espacio` (el Dataset se arma
una vez con `lgbm.fijos`); si cambia el SQL de FE, subir `FE_VERSION` o se reutiliza la caché
vieja; para quitar una variante o clave heredada, ponerla en `null`.

### Dónde mirar los resultados

| Qué | Dónde |
|---|---|
| Todos los runs | `work/competencia_1/runs.csv` |
| Ablación | `runs/<run_id>/ablacion.csv` (`delta_media`, `delta_por_fold`, `veredicto`) |
| Estabilidad de trials | `runs/<run_id>/estabilidad_trials.csv` (`semillas_mediana`) y `estabilidad.json` |
| Log, config y metadatos | `runs/<run_id>/run.log`, `config_resuelta.yaml`, `meta.json` |
| Hiperparámetros ganadores | `work/competencia_1/params/<estudio>__estable.json` |

### Cómo decidir si algo mejora

- Comparar la **mediana de 5 semillas**, no el valor de Optuna. Referencia actual: **357,8 M**
  (trial 53 de e010), mismo protocolo.
- Piso de ruido ~±6 M: adoptar un cambio solo si supera ~+10 M **y** es positivo en los dos folds.
- Un conjunto de features nuevo se **re-tunea**; los parámetros de otro conjunto sesgan la comparación.
- No elegir por el leaderboard público.
- Hay 4 meses con target (202103–202106): con `gap=2`, `n_meses_train=1` y `n_folds=2` ya están
  todos usados; subir `n_folds` o `n_meses_train` falla por mes sin target completo.
- Antes de commitear: `pytest` y `ruff`. No commitear `work/`, datos ni credenciales.

### Ideas pendientes (hipótesis, no promesas)

1. Deltas sobre más variables: `fe.campos_serie: todos` con ventanas, tendencia, rankings e
   intrafila apagados, y luego `dataset.excluir_bloques` como en e009.
2. Re-tunear `orig + deltas` con `target.positivos: [BAJA+2]` (todo lo tuneado usó BAJA+1 + BAJA+2).
3. Sumar `lambda_l2` y `min_gain_to_split` a `optuna.espacio`.
4. `periodos.meses_final` con más meses: no validable con los datos actuales y exige revisar
   `num_iterations` y `min_data_in_leaf`.
5. Más ratios en `intrafila.py` (`ctrx_quarter`, transacciones, saldos).

Tiempos de referencia: FE ~10 min; Optuna de 80 trials sobre 292 features ~25 min;
estabilidad ~7 min; final de 20 semillas ~12 min.

## Experimentos y bitácora

Las configuraciones de cada experimento (`configs/exp/eNNN_*.yaml`), los hallazgos de
validación y la bitácora de puntajes públicos están en
[`configs/exp/README.md`](configs/exp/README.md). Los resultados (runs, cachés, estudios de
Optuna) viven en `work/` y `db/`, que no se versionan.
