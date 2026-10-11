# Competencia 01 — entrega final: ensamble @@EXPERIMENTO@@ con @@ENVIOS@@ envíos

Esta carpeta contiene todo lo necesario para replicar **exactamente** el archivo que se subió
como entrega final de la competencia. La entrega es un **ensamble**: el promedio de los rankings
de @@N_COMPONENTES@@ modelos (cada uno, a su vez, un ensamble de LightGBM con varios conjuntos de
hiperparámetros y semillas). Cada componente está en su propia subcarpeta con su configuración,
sus hiperparámetros y sus modelos entrenados.

| | |
|---|---|
| Archivo entregado | `@@CSV_ORIGINAL@@` (solo `numero_de_cliente`, sin encabezado, @@ENVIOS@@ filas) |
| sha256 | `@@SHA256@@` |
| Puntaje público | @@PUBLICO@@ |
| Período objetivo | 202108 |

El dataset de la competencia **no está en el repositorio**: el script lo descarga solo desde la
URL pública de la cátedra (`competencia_01_crudo.csv`).

## Componentes del ensamble

@@COMPONENTES@@

Combinación: cada componente predice la probabilidad de BAJA+2 en 202108; se convierte en su
rank normalizado dentro del mes (`competencia_1/pipeline/ensamble.py`, método `@@METODO@@`), se
promedian los rankings y se envían los @@ENVIOS@@ clientes de mayor valor.

## Cómo replicarla

Hace falta Python 3.11 y unos 16 GB de RAM. Hay tres niveles; en todos se reproduce cada
componente (y se verifica el sha256 de su propio CSV), se arma el ensamble y se compara el sha256
del CSV final con el de arriba (`[ensamble] RESULTADO: OK, coincide`):

| Modo | Qué rehace | Tiempo aprox. |
|---|---|---|
| `predecir` (default) | Datos y features desde el crudo, y **predice** con los @@N_MODELOS@@ modelos ya entrenados | @@T_PREDECIR@@ |
| `entrenar` | Lo anterior, pero **reentrena** los @@N_MODELOS@@ modelos con los hiperparámetros de cada componente | @@T_ENTRENAR@@ |
| `completo` | **Todo, incluida la optimización de hiperparámetros** de cada componente (Optuna, @@N_TRIALS@@ trials en total) | @@T_COMPLETO@@ |

**Linux o VM de GCP:**

```bash
git clone @@URL_REPO@@ && cd dmeyf2026-entregas/@@CARPETA@@
bash reproducir.sh              # o: bash reproducir.sh entrenar | bash reproducir.sh completo
```

**Windows (PowerShell):**

```powershell
git clone @@URL_REPO@@; cd dmeyf2026-entregas\@@CARPETA@@
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-lock.txt
$env:ENTREGA_MODO = "predecir"      # o "entrenar" / "completo"
.venv\Scripts\python entrega.py
```

También se puede abrir `entrega.ipynb` (es el mismo código que `entrega.py`) y ejecutarlo
completo; el modo se elige en la segunda celda.

El CSV del ensamble queda en `work/entrega/runs/<fecha>_ensamble/submits/` y los de cada
componente en `work/entrega/runs/<fecha>_entrega/submits/`.

## Entorno

- `requirements-lock.txt`: **versiones exactas** de todas las librerías (incluidas las
  transitivas) con las que se generó la entrega. `requirements.txt` lista solo las directas.
- Python @@PYTHON@@ (`.python-version`). El script avisa si alguna versión instalada difiere.
- La igualdad byte a byte depende de las versiones de LightGBM y DuckDB y de
  `lgbm.fijos.num_threads` (8) en `competencia_1/configs/base.yaml`; no los cambies.

## Qué hace cada modelo

@@RESUMEN@@

## De dónde salen los hiperparámetros (optimización)

@@OPTIMIZACION@@

## Contenido de la carpeta

```
entrega.py / entrega.ipynb     punto de entrada (mismo código)
reproducir.sh                  atajo para Linux / GCP
requirements*.txt, .python-version
competencia_1/                 pipeline (datos → features → Optuna → modelos → CSV)
  configs/base.yaml            defaults
  configs/exp/*.yaml           configuración de cada experimento (optimización + final)
  definitiva/entrega.json      el ensamble: componentes, método, pesos y sha256 esperado
  definitiva/componentes/<i>_<experimento>/   config.yaml, params.json, entrega.json y modelos/
dmeyf/, config/                métricas y semillas
optimizacion/<i>_<experimento>/  evidencia de la optimización de cada componente (Optuna)
```

## Trazabilidad

Generada el @@FECHA@@ desde el repositorio de trabajo, commit `@@COMMIT@@`, ensamble
`@@RUN_ID@@`. El CSV de ese ensamble coincide byte a byte con el que reproduce esta carpeta.
