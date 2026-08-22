# Datasets

Esta carpeta está en `.gitignore` (los datos NO se suben al repo público). Esta nota
sí se versiona (excepción `!**/NOTAS.md`) para dejar registro de la procedencia.

## Estructura
- `raw/` → datos originales, tal cual se descargan de la fuente. No editar a mano.
- `processed/` → datos derivados (ej. `competencia_01.csv` con `clase_ternaria`),
  generados por notebooks/scripts de `exp/` o `clases/`.

## raw/competencia_01_crudo.csv
- Fuente: https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv
- Especificado en: `monday/z101_target_sql.ipynb` (cátedra DMEyF 2026)
- Descargado: 2026-08-22
- Ignorado por git (dataset grande, ~493 MB). No commitear.

## raw/DiccionarioDatos_2026.ods
- Diccionario de datos de la cátedra (155 campos + notas generales).
- Copiado desde Descargas el 2026-08-22.
- `diccionario_datos.csv` y `consideraciones.txt` son exports en UTF-8 de las
  hojas "Diccionario" y "Consideraciones" del .ods, para poder leerlos sin
  depender de `odfpy`/pandas.
- Es documentación liviana (no dataset), por eso SÍ está excluida del
  `.gitignore` y se versiona junto al resto del repo.
