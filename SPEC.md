# SPEC — Corrida final del sábado 2026-10-03 (rama `adrian/final-rag`)

Este documento registra lo que se hizo el sábado sobre las 992 preguntas ciegas, desde que llegaron
(09:00) hasta la entrega (15:00). Complementa el `SPEC.md` del repositorio principal
(`Ai-Week-LasNewJean/main-repo`, secciones 0–18), que documenta la semana de desarrollo. Aquí van
las decisiones del día, con sus números, para la bitácora de corpus y la verificación en vivo.

> Estado: **en curso**. Se actualiza en cada hito (ver sección 9 para la línea de tiempo).

## 0. Restricciones que gobiernan todo lo de abajo

- **Verificación en vivo (15:00–17:00):** el jurado re-ejecuta 2–3 ítems ya entregados con el
  sistema entregado y deben salir idénticos. Por eso la entrega sale de **un solo sistema**
  (un `indice/` congelado). Mezclar ítems de dos índices solo es válido donde se pueda probar que
  ambos producen exactamente la misma salida (sección 4).
- Reglas del reto: nada de modelos cerrados; temperatura 0; índice congelado y reconstruible por
  script; **no se indexa el banco de preguntas ni material con respuestas**; no se editan
  respuestas a mano.
- Máquina canónica: Turing (RTX 4090, 24 GB). Dos procesos con el decoder no caben juntos en la
  GPU; un proceso de solo recuperación (e5 + reranker, ~3–6 GB) sí cabe junto a la corrida e2e
  si esta ya creó su contexto de llama.cpp.
- `data/test_992.jsonl` está en `.gitignore`: no se versiona y no entra al corpus.

## 1. Corrida base (respaldo)

- 09:39: `python -m src.main --split test --out out/final_test.jsonl` con el índice congelado del
  viernes (`indice/`, 106 237 fragmentos; mejor puntaje en `sample_50`: 34.12/50 automáticos).
- Es el **respaldo**: si cualquier paso de abajo falla, se entrega esta corrida con este índice.

## 2. Análisis de huecos del corpus frente a las 992 preguntas

Solo se miraron **referencias normativas explícitas en los enunciados** (regex de normas +
`scripts/citations.extract`), contra el `corpus_manifest.json` y los `chunks.jsonl`. No se miraron
respuestas (no las hay) ni se copió nada del banco al corpus.

| Hallazgo | Ítems |
|---|---|
| **Resolución 368 de 2014 (MinAmbiente, relleno sanitario El Carrasco)**: bloque de preguntas de derecho administrativo sobre ese acto ("habiendo hecho la lectura previa de la Resolución No. 368 de 2014…"); el corpus no tenía ninguna resolución | 23 (20 de selección múltiple) |
| Decisión Andina 486, artículos 65/241/243 (el corpus congelado la tenía incompleta) | 5 |
| Ley 2437 de 2024 (reorganización abreviada), ausente | 3 |
| Decreto 175 de 2025 (timbre, conmoción interior), ausente | 2 |
| Decreto 4302 de 2008, Decreto 1382 de 2000 / 333 de 2021 (reparto de tutela), Acuerdo 01 de 2025 de la Corte Constitucional, Ley 600 de 2000 art. 536 | 1 c/u |
| Decretos Únicos Reglamentarios 1625/2016 y 2555/2010 (enormes; 1 mención c/u, sin necesidad real del texto) | descartados |

Cobertura a nivel de artículo citado: 79 citas con artículo en los enunciados; el índice congelado
tiene 65, la variante A (sección 3) 72.

**Decisión del equipo (Adrian, 09:50):** añadir solo los textos oficiales de las normas citadas que
faltan, y declararlo aquí. Se eligieron después de ver las preguntas; eso no viola la regla (se
indexan normas públicas, no preguntas ni respuestas), pero se deja escrito para el jurado.

## 3. Normas añadidas y cómo se obtuvieron

La red de Turing vuelve a salir a internet el sábado (el viernes `secretariasenado.gov.co` daba
timeout). Todo queda en `data/corpus_extra/` con el mismo esquema JSON que
`rescrape_senado.py`, y entra a la ingesta por `RESCRAPE_DIR` sin tocar `ingest_raw_sources.py`.

| doc_id | Fuente oficial | Método | Fragmentos |
|---|---|---|---|
| `resolucion-368-2014` | minambiente.gov.co (PDF escaneado de 22 hojas) | OCR (`src/ingest/ocr_rapid.py`) + reparación (`fetch_extra_norms.repair_ocr`) | 55 (arts. 1–10 + 45 ventanas de considerandos) |
| `decreto-175-2025` | Función Pública, Gestor Normativo | HTML → artículos | 31 |
| `decreto-4302-2008` | SIC (Diario Oficial 47.172, PDF con texto) | PDF → artículos | 10 |
| `decreto-1382-2000` | Normograma Cancillería | HTML → artículos | 8 |
| `decreto-333-2021` | Normograma Cancillería | HTML → artículos | 15 |
| `acuerdo-cc-1-2025` | Normograma Cancillería | HTML → artículos (ventanas de 300 palabras) | 129 |

Script: `python -m src.ingest.fetch_extra_norms` (descarga con caché en `data/corpus_extra/raw/`,
`--offline` para reconstruir solo desde la caché).

**OCR de la Resolución 368.** El PDF no tiene capa de texto y Turing no tiene tesseract. Se usó
RapidOCR (ONNX, solo CPU) en un venv aparte (para no tocar el venv CUDA de la solución). El
detector de RapidOCR perdía renglones justificados enteros (p. ej. la mitad del artículo 1);
se reemplazó por segmentación de renglones por perfil de proyección horizontal + solo el
reconocedor por franja, que recupera todos los renglones. Después, `repair_ocr`:
- quita el encabezado repetido de cada hoja;
- corrige `6n→ón` y `0→o` entre letras;
- quita la raya del margen pegada al renglón ("Ique" → "que");
- separa palabras pegadas ("rellenosanitarioel" → "relleno sanitario el") con programación
  dinámica de unigramas del propio corpus, aceptando la partición solo si todas las piezas son
  palabras conocidas (los nombres propios no se trocean);
- restituye tildes con la forma más frecuente en el corpus.
El texto OCR queda versionado (`data/corpus_extra/raw/resolucion-368-2014.ocr.txt`) para que la
reconstrucción no dependa del venv de OCR.

**Variante A: probada y descartada otra vez.** Primero se construyó F = variante A del viernes (267
leyes/decretos truncados completados por `rescrape_senado --completar` + Decisión 486) + las normas
nuevas. Con las 992 había evidencia de cobertura (9 ítems ganan artículos citados explícitamente),
pero el contexto cambiaba en el **59%** de los ítems y la compuerta en `sample_50` dio **31.91**
(cerradas 10/15, citación 11.84, abstención 6.74) frente a **34.12** del congelado (y 33.49 de A el
viernes). Dos mediciones de A por debajo del congelado: se descarta la completitud masiva.

**Resolución 368: probada y descartada por inundación.** Todas las preguntas del bloque empiezan
con el mismo preámbulo ("Habiendo hecho la lectura previa de la Resolución No. 368 de 2014…") y
cada fragmento de la resolución lleva ese mismo título, así que la resolución ocupa 5–6 de los 6
pasajes en los 23 ítems del bloque y desplaza a la doctrina que la pregunta jurídica necesita
(tipos de acto, vicios, recursos, silencio — su `legal_basis` en la muestra es el CPACA). En
`sample_50` hay un ítem del bloque (748): con el índice congelado responde bien (A, "falsa
motivación", con la T-294/14); con la resolución en el índice, sus 6 pasajes son de la resolución y
responde mal (D). Se deja fuera del índice; el JSON y el OCR quedan versionados en
`data/corpus_extra/` por si se quiere reutilizar con un tope de pasajes por documento.

**Índice entregado: F1 = congelado + normas puntuales.** Los 106 237 fragmentos del índice
congelado quedan idénticos (mismo texto, mismo vector) y se añaden 497 fragmentos:

| doc_id | Fragmentos | Ítems de las 992 donde entra al contexto (con F) |
|---|---|---|
| `decision-andina-486` (Decisión 486 completa, `parse_decision486.py` del viernes) | 283 | 31, todos de propiedad industrial (diseño industrial, patentes, signos notorios, licencias obligatorias) |
| `ley-2437-2024` (secretariasenado, extraída de `leyes_decretos_completos.json`) | 21 | 6 (insolvencia/reorganización abreviada) |
| `decreto-175-2025` | 31 | 9 (tributario: timbre, conmoción interior) |
| `decreto-4302-2008` | 10 | 10 (licencias obligatorias) |
| `decreto-1382-2000`, `decreto-333-2021` | 8 + 15 | 1 (reparto de tutela) |
| `acuerdo-cc-1-2025` | 129 | 6, con 1 de 6 pasajes salvo el de reparto de tutela |

En `sample_50`, F1 solo cambia el contexto de 1 ítem (352), así que su puntaje es el del congelado
(34.12) salvo ese ítem (sección 6).

## 4. Índice F1 y cómo reconstruirlo

```bash
F=../exp_index/F1
mkdir -p $F/rescrape
for f in data/corpus_rescrape/*.json; do [ "$(basename $f)" = leyes_decretos_completos.json ] || ln -s $PWD/$f $F/rescrape/; done
for f in data/corpus_extra/*.json;   do [ "$(basename $f)" = resolucion-368-2014.json ] || ln -s $PWD/$f $F/rescrape/; done
export RESCRAPE_DIR=$F/rescrape CHUNK_HEADINGS=0 RAG_CORPUS_DIR=$F/corpus \
       RAG_CORPUS_MANIFEST=$F/corpus_manifest.json RAG_INDEX_DIR=$F/indice
.venv/bin/python -m src.ingest.ingest_raw_sources      # 34 387 documentos, 106 734 fragmentos (~50 s)
.venv/bin/python -m src.index.build_bm25               # ~30 s
.venv/bin/python -m src.index.build_faiss              # desde cero (~15 min en la 4090), o:
.venv/bin/python -m src.index.build_faiss --reuse indice   # reutiliza los vectores del congelado
```

(`data/corpus_rescrape/` sin `leyes_decretos_completos.json` es exactamente el `RESCRAPE_DIR` con el
que se construyó el índice congelado del viernes: 106 237 fragmentos.)

`build_faiss --reuse <índice previo>` toma el vector del índice previo para cada fragmento con
texto idéntico y solo embebe los nuevos. Así F se construyó en CPU sin interrumpir la corrida base,
y F0/F1 en segundos en GPU. `IndexFlatIP` es posicional, así que el resultado equivale a una
reconstrucción completa salvo ruido de punto flotante por lote; sin `--reuse` se reconstruye
todo. Comprobado: `index.ntotal == líneas de chunks.jsonl` (122 134 en F, 106 734 en F1), y en F1
los 106 237 textos del congelado están todos presentes.

## 5. Regenerar solo lo que cambia (sin perder reproducibilidad)

Volver a correr las 992 con F tomaría ~2 h, lo cual no cabía con seguridad antes de las 15:00.
Pero el prompt del decoder depende solo del ítem y de la lista final de pasajes (`doc_id` + texto,
en orden; `prompts/_common.py`), y la abstención solo del score del mejor pasaje (`sufficiency`).
El decoder es greedy y determinista, así que **si con F la recuperación final de un ítem es
idéntica a la de la corrida base, la respuesta también lo es**, y la línea de la corrida base es
exactamente la que produce el sistema F (en la verificación en vivo, F reproduce esa línea).

- `answer_one.final_retrieval(item)`: se extrajo de `answer()` la recuperación completa (incluido
  el reintento ampliado), sin cambiar el comportamiento, para que el diff use la misma lógica.
- `python -m src.validate.retrieval_diff dump` guarda la recuperación final por ítem con el índice
  activo; `compare` la contrasta con una corrida e2e (mismos doc_id y textos en orden, scores a
  1e-3) y escribe los ids a regenerar. Las abstenciones previas siempre se regeneran.
- **Validación del método:** el `dump` con el índice congelado reprodujo exactamente los pasajes y
  scores de la corrida base en **104/104** ítems (determinismo de la recuperación entre procesos
  en GPU).
- `RAG_IDS_FILE=<ids> python -m src.main --split test --out out/rerun_F.jsonl` corre solo esos
  ítems con F; `python -m src.pipeline.splice` ensambla la entrega (ítems cambiados desde la
  corrida F, el resto desde la base).

Resultado: con F1, de los 321 ítems que la corrida base alcanzó a responder, **302 tienen
recuperación idéntica** y 19 cambian (14 porque entra una norma nueva; 5 por pequeños cambios de
orden: el IDF de BM25 cambia al crecer el corpus). Se regeneran con F1 esos 19 más los 671 que la
corrida base no alcanzó (se detuvo a propósito en el ítem 321, ver sección 9).

## 6. Compuerta (go / no-go)

_Pendiente_: `sample_50` e2e con F frente a 34.12 (congelado) y 33.49 (A). Criterio: no peor que
~1 punto (ruido de 50 ítems: ±1 cerrada = 1.33 pts). Además, los ítems de huecos deben recuperar
los documentos nuevos.

## 7. Entrega

_Pendiente_: archivo entregado, sha256, índice entregado (`indice/` ← F, respaldo del congelado en
`../backup_v4_pre_final/` con `SHA256SUMS`) y `determinism_check.py` sobre 3 ítems.

## 8. Lo que NO se hizo, y por qué

- Decretos Únicos Reglamentarios completos (1072/2015, 1625/2016, 2555/2010…): decenas de miles de
  fragmentos para 1–2 menciones; ya se midió el viernes que diluir el índice cuesta recuperación.
- Encabezados por artículo (`CHUNK_HEADINGS=1`, variante B): descartado el viernes (peor en todo).
- RAGAS: no se corre (el crédito de OpenRouter se reserva para la evaluación oficial).

## 9. Línea de tiempo

| Hora | Hito |
|---|---|
| 09:39 | Arranca la corrida base (índice congelado) |
| 09:50 | Análisis de huecos; decisión de añadir normas citadas faltantes |
| 10:00–10:12 | Descarga de normas; OCR de la Resolución 368 (CPU) |
| 10:13–10:17 | Índice F (ingesta + BM25 + FAISS con `--reuse`, todo en CPU) |
| 10:19 | Validación del diff de recuperación (104/104 idénticos con el índice congelado) |
| 10:22 | `dump` de recuperación con F en GPU, en paralelo a la corrida base |
| 10:25 | Corrida base detenida en 321/992 (la contención de GPU la tenía a ~2 ítems/min; `run_batch` reanuda por id) |
| 10:33 | F cambia el contexto del 59% de los ítems; compuerta `sample_50` con F = 31.91 → se descarta A |
| 10:41 | F0 = congelado + normas puntuales (incluida la Res. 368): solo 2/50 ítems de la muestra cambian |
| 10:43 | Ítem 748 (bloque Res. 368) pasa de bien a mal; la resolución inunda los 6 pasajes en 23/23 ítems → se quita |
| 10:46 | F1 (sin la Res. 368). Arranca la corrida F1 sobre los 671 ítems pendientes |
| 10:53 | Diff F1 vs base en los 321 hechos: 302 idénticos, 19 a regenerar |
