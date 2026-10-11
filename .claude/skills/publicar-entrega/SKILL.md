---
name: publicar-entrega
description: Pasa un experimento de competencia_1 a ser la entrega oficial. Exporta el código, el entorno fijo, los hiperparámetros, los modelos y la optimización al repo de entregas (pfranzd/dmeyf2026-entregas), valida que el CSV generado allí sea idéntico al del experimento, y hace commit y push verificados. Usar cuando el usuario diga "pasá/promové/publicá <experimento> como entrega oficial/final/definitiva", "actualizá el repo de entregas" o similar.
---

# Publicar la entrega oficial

El repo de entregas (`C:\dev\dmeyf2026-entregas`, remoto `https://github.com/pfranzd/dmeyf2026-entregas`)
es lo que la cátedra clona para replicar el CSV final. Solo contiene lo entregable: nunca CSV ni
parquet de la competencia. Todo el flujo vive en `python -m competencia_1.publicar`; esta skill dice
cómo usarlo sin saltarse ninguna validación.

## Pasos

1. **Identificar el run y el corte de envíos.** Buscar el run del experimento en
   `work/competencia_1/runs.csv` y en la bitácora (`competencia_1/configs/exp/README.md`).
   El run debe estar en estado `ok`, con la etapa `final`, y tener el CSV de ese corte en `submits/`
   (si el corte se generó aparte, el CSV debe estar en el mismo `submits/`). Si hay ambigüedad
   (varios runs del mismo experimento, corte distinto del anotado), **preguntar al usuario**.
2. **Código commiteado.** El comando exige que `competencia_1/` (pipeline, configs, plantillas) y
   `dmeyf/`, `config/` no tengan cambios sin commitear: el commit queda como origen de la entrega.
   Si hay cambios, mostrarlos y proponer commitearlos (commit solo si el usuario lo pide).
3. **Repo de entregas listo.** Debe existir `C:\dev\dmeyf2026-entregas` (clonado con
   `git -c http.sslBackend=schannel clone https://github.com/pfranzd/dmeyf2026-entregas.git`),
   limpio, en `main` y al día con `origin`. Si el remoto está vacío, agregar `--inicializar`.
4. **Publicar** (corre en segundo plano; tarda ~25 min con la validación `predecir`, que es el
   mínimo aceptable; `--validar completo` suma ~1,5 h):

   ```powershell
   python -m competencia_1.publicar --run work/competencia_1/runs/<run_id> --envios <N> `
     --crudo-local datasets/raw/competencia_01_crudo.csv [--inicializar]
   ```

   El comando, en orden: verifica precondiciones → congela el run en `competencia_1/definitiva`
   (`promover`, con los modelos) → exporta la carpeta → chequea que no entren datos ni archivos
   >50 MB → corre la entrega exportada en un **venv nuevo** instalado desde
   `requirements-lock.txt` y exige que el CSV tenga el **mismo sha256 y los mismos bytes** que el
   del run → commit → `git push origin main` → verifica con `ls-remote` que el remoto quedó en el
   mismo commit.
5. **Si algo falla, parar.** El comando deja el repo de entregas como estaba y no sube nada.
   Reportar el motivo exacto (p. ej. sha256 esperado vs obtenido). **Nunca** usar `--force`, saltear
   la validación ni editar a mano la carpeta publicada para que coincida.
6. **Al terminar:**
   - Confirmar con `git -C C:\dev\dmeyf2026-entregas ls-remote origin main` y `git ... rev-parse HEAD`.
   - Verificar en el repo publicado: `git ls-files | grep -E "\.(csv|parquet)$"` debe estar vacío.
   - Marcar la entrega vigente en la bitácora de `competencia_1/configs/exp/README.md`, anotar
     `resultado_publico` en `competencia_1/definitiva/entrega.json` (y volver a publicar si cambió)
     y actualizar la memoria del proyecto.
   - `competencia_1/definitiva/` del repo principal queda modificada por `promover`: proponer el
     commit (y el push del fork solo si el usuario lo pide).
   - Dar el link: `https://github.com/pfranzd/dmeyf2026-entregas/tree/main/competencia_01` y el sha256.
7. **Recordatorio para el usuario:** el archivo a subir al bot es exactamente `csv_original` de
   `entrega.json` (el último submit antes de las 23:59:59 de Buenos Aires es el que cuenta), y el
   link va en el tópico de la entrega.

## Notas

- Lo que se valida es la carpeta tal como queda en git (con `.gitattributes` que evita conversión
  de saltos de línea en los modelos). La reproducción rápida (`predecir`) usa los modelos
  guardados; `entrenar` reentrena y `completo` rehace también la optimización de Optuna (solo
  automático si el experimento es autocontenido: `completo_automatico` en `entrega.json`).
- Si el experimento toma sus hiperparámetros de otra corrida (p. ej. `params_desde: archivo:`),
  la cátedra pide subir también esa optimización: el README generado lista los configs
  involucrados; completar a mano los pasos antes de dar la entrega por buena.
- Más detalle: sección "Publicar la entrega oficial" de `competencia_1/README.md`.
