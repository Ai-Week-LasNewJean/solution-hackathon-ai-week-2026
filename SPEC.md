# SPEC — Sistema RAG de derecho colombiano (AI Week Hackathon 2026)

> Plan de arquitectura y desarrollo. No reemplaza el `enunciado.pdf` ni los templates en
> `entregables/` — es la hoja de ruta técnica del equipo para construirlos.

> **Mapa de ramas y procedencia de cada resultado (incluido el correo del viernes 17:00): [sección 17](#17-mapa-de-ramas-y-procedencia-de-los-resultados-2026-10-02-noche).**

> **Las secciones 0–9 son el plan original (escrito antes de correr nada contra datos reales).**
> Se dejan intactas como referencia del razonamiento de diseño. **Para saber dónde está el
> proyecto hoy y qué falta, ir directo a la [sección 10](#10-estado-actual-2026-10-01-y-próximos-pasos).**

> **CAMBIO DE INSTRUCCIONES (2026-10-02): Turing (GPU CUDA, RTX 4090) es ahora la máquina
> canónica; el Mac es solo auxiliar, para pruebas de determinismo cruzado.** Todo lo que en las
> secciones 2–9 y 10.1–10.6 diga "Mac canónico / solo el Mac" queda **superado por la
> [sección 11](#11-turing-como-máquina-canónica-y-resultados-2026-10-02)**.

## 0. Resumen del reto

Construir un sistema RAG que responda preguntas de derecho colombiano usando un **decoder abierto
≤8B parámetros** + un **encoder abierto** + un **corpus legal construido por el equipo**, sin
ningún modelo cerrado en el pipeline, determinista a temperatura 0. La hipótesis de trabajo del
enunciado es que **la calidad del corpus pesa más que el modelo** — por eso la construcción del
corpus es la actividad central de la semana, no un anexo.

**Restricciones duras (violar cualquiera = descalificación):**

- Ningún modelo cerrado (OpenAI/Anthropic/Google/Cohere/etc.) en generación, reformulación de
  consultas, reranking o generación de datos sintéticos.
- Temperatura 0 y determinismo en la ejecución final.
- El índice debe estar congelado al momento de la entrega y ser reconstruible por script.
- No indexar el banco de preguntas ni material con respuestas.
- No editar manualmente respuestas después de generadas.

**Puntaje (100 pts):** 80 automáticos sobre las 992 preguntas ciegas del sábado (Exactitud
cerradas 20 pts, baseline 0.905; Corrección texto libre 30 pts vía RAGAS `answer_correctness`,
juez `z-ai/glm-5.3-flash`, baseline 0.451; Calidad de citación 20 pts =
`recall_citas_ponderado − 2×tasa_sin_respaldo`, solo cuentan los primeros 10 pasajes recuperados
como respaldo; Abstención calibrada 10 pts = acierto 1.0 / abstención 0.5 / error 0.0) + 20
manuales (interfaz con identidad visual de Software Colombia 10 pts, bitácora de corpus 5 pts,
video ≤5min 3 pts, reproducibilidad desde contenedor limpio 2 pts). Confirmado leyendo
[`scripts/evaluate.py`](scripts/evaluate.py), [`scripts/citations.py`](scripts/citations.py) y
[`schema/submission.schema.json`](schema/submission.schema.json).

## 1. Realidad del cronograma

Hoy es miércoles. Viernes 13:00–17:00 son las charlas obligatorias de AI Week; viernes 17:00 es el
checkpoint (no calificado) del avance. Sábado 9:00 llegan las 992 preguntas reales; 9:00–15:00 es
ejecución ciega + pulido de interfaz en sitio; 15:00 cierra la entrega; 15:00–17:00 es verificación
en vivo (el jurado re-ejecuta 2–3 respuestas ya entregadas y deben regenerarse igual — divergencia o
no-reproducibilidad descalifica). Quedan realmente **~2.5 días de desarrollo** (hoy en la tarde,
jueves, viernes en la mañana) antes del checkpoint; el sábado es ejecución ciega/en vivo, no
desarrollo. Toda decisión de arquitectura abajo está dimensionada para esa realidad: simple y
robusto antes que sofisticado.

## 2. Infraestructura de cómputo: MacBook Pro M4 Pro + Sala Turing

> **SUPERADO (2026-10-02):** Turing es la máquina canónica y el Mac es auxiliar (solo determinismo
> cruzado). Donde esta sección diga "Mac canónico", léase Turing. Ver sección 11.

El equipo cuenta con **Sala Turing** (GPU CUDA, disponible toda la semana) y un **MacBook Pro M4
Pro** (Apple Silicon), que será la máquina presente el sábado para la ejecución y la verificación
en vivo. Esta combinación introduce un riesgo específico que hay que resolver explícitamente:

**Riesgo central:** con `temperatura=0`, la decodificación *greedy* es determinista dado un binario
y hardware fijos — pero CUDA y Metal usan kernels de punto flotante distintos. Los mismos pesos
GGUF, el mismo prompt y "temperatura 0" pueden producir tokens distintos en Turing (CUDA) que en el
Mac (Metal). Como la verificación en vivo del sábado se hace **sobre la máquina que el equipo lleve**,
y esa máquina es el Mac, **el Mac debe ser la fuente de verdad canónica para toda generación que
termine en `submissions.jsonl`.**

| Tarea                                                    | Máquina                                                                                    | Por qué                                                                                                                                                                                |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ingesta del corpus (fetch/parse/OCR)                     | Cualquiera                                                                                 | No hay riesgo de determinismo; es I/O y CPU.                                                                                                                                           |
| Construcción del índice de embeddings                    | **Turing** (rápido), validado re-ejecutando el mismo build en el **Mac** antes de congelar | Una vez escrito a disco, un vector es un vector — el índice congelado no tiene riesgo entre máquinas; solo el *script* de construcción debe ser reproducible en cualquiera de las dos. |
| Iteración de prompts contra `evaluate.py --split sample` | **Turing** (loop de feedback rápido)                                                       | Velocidad de desarrollo, no corrección final — los puntajes medidos aquí son orientativos.                                                                                             |
| **Pipeline final que genera `submissions.jsonl`**        | **Solo el Mac, desde el jueves**                                                           | Debe ser exactamente el software+hardware presente en la verificación en vivo.                                                                                                         |
| Chequeos de determinismo/reproducibilidad                | **Solo el Mac**                                                                            | Existen justamente para ensayar la verificación en vivo del sábado.                                                                                                                    |
| Stretch: fine-tuning LoRA/QLoRA (opcional, no crítico)   | Turing                                                                                     | Solo si el pipeline central ya está sólido el jueves en la noche; cualquier resultado debe re-validarse en el Mac antes del viernes.                                                   |

**Elección de motor de inferencia:** `llama-cpp-python` (GGUF) sobre MLX o vLLM, porque el mismo
código compila contra Metal y CUDA — minimiza el código divergente entre máquinas a una sola flag
de compilación, y tiene soporte maduro de gramáticas GBNF para forzar el JSON de salida (MLX aún no
tiene esto consolidado; vLLM es solo CUDA). Prueba de humo obligatoria **hoy**: compilar, cargar y
generar un token en ambas máquinas — problemas de compilador/drivers son el tipo de imprevisto que
descarrila un cronograma así de ajustado.

**Chequeo go/no-go, jueves:** correr las mismas 10–20 preguntas de muestra en ambas máquinas y
comparar la generación cruda. Si divergen (lo esperable), confirma que la regla "Mac canónico" debe
seguirse estrictamente: prototipar libremente en Turing, pero todo lo que se considere final se
re-valida en el Mac.

## 3. Stack técnico

| Capa                          | Elección                                                                                                                     | Por qué                                                                                                                                                                                                                                                          |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parseo HTML                   | `requests` + `beautifulsoup4` + `lxml`                                                                                       | Dos parsers específicos (`parse_senado_basedoc`, `parse_suin`) en vez de uno genérico — cubren casi todas las URLs de `donde_buscar` en `data/seed_targets.json`.                                                                                                |
| Extracción PDF                | `PyMuPDF` (`fitz`), fallback `pdfplumber`                                                                                    | Rápido, buen manejo de español, una sola dependencia para el caso común.                                                                                                                                                                                         |
| OCR                           | `pytesseract` + Tesseract `spa` + `pdf2image`                                                                                | Solo en páginas sin capa de texto; reservado para la jurisprudencia de mayor ROI (ver plan por días), nunca aplicado a todo el corpus.                                                                                                                           |
| Segmentación                  | Regex `ARTÍCULO\s+\d+` para códigos/leyes; ventana deslizante (~250 palabras, 1–2 oraciones de traslape) para jurisprudencia | Mismo método documentado en `Ejemplo de entrega/CORPUS.md`, la referencia de los organizadores.                                                                                                                                                                  |
| Encoder de embeddings         | `intfloat/multilingual-e5-large` vía `sentence-transformers`                                                                 | Es el mismo encoder que usa `evaluate.py` para el componente de similitud de RAGAS (`JUEZ_ENCODER`) — reutilizarlo evita un segundo stack y está validado por el corpus de ejemplo de los organizadores. Prefijos `query:`/`passage:`, vectores normalizados L2. |
| Índice vectorial              | `faiss-cpu`, `IndexFlatIP`                                                                                                   | Búsqueda exacta, sin aleatoriedad de entrenamiento (a diferencia de IVF/HNSW), trivialmente reconstruible — más importante que la velocidad a esta escala (miles de fragmentos, no millones).                                                                    |
| Recuperación dispersa         | `bm25s` (fallback `rank_bm25`)                                                                                               | NumPy puro, rápido, sin dependencia de JVM. Tokenización reutiliza `citations.norm()` para que BM25 y el matching de citas normalicen igual.                                                                                                                     |
| Fusión                        | Reciprocal Rank Fusion (RRF, k=60)                                                                                           | No requiere calibrar escalas entre BM25 y similitud coseno.                                                                                                                                                                                                      |
| Reranker (opcional)           | `BAAI/bge-reranker-v2-m3`, tras flag de configuración                                                                        | Construir el pipeline **sin** reranker primero; añadirlo solo si `evaluate.py --split sample` muestra una mejora medida.                                                                                                                                         |
| Decoder + motor de inferencia | `llama-cpp-python` (GGUF), un binario, dos backends (Metal en Mac, CUDA en Turing)                                           | Ver sección 2.                                                                                                                                                                                                                                                   |
| Salida JSON estructurada      | Gramática GBNF por `formato` vía `grammar=` de `llama-cpp-python`                                                            | Garantiza las llaves exactas del schema de forma determinista y suele ser más rápido (poda tokens inválidos). Fallback: prompting estricto + extracción de `{}` + un reintento determinista de reparación de JSON.                                               |
| Validación de schema          | `jsonschema` contra `schema/submission.schema.json`, más `validate()` de `evaluate.py`                                       | El segundo es literalmente lo que corre el jurado.                                                                                                                                                                                                               |
| Interfaz                      | **Streamlit**                                                                                                                | Camino más rápido a "pregunta → botón → respuesta formateada + tabla de pasajes + badges de respaldo de citas" con control real de layout (`st.columns`, `st.expander`, CSS inline para la identidad visual de Software Colombia).                               |
| Metadatos de citas            | [`scripts/citations.py`](scripts/citations.py) (importado, sin modificar)                                                    | Se reutiliza en ambos lados: para auto-verificar trazabilidad de fragmentos en la ingesta, y para auto-verificar citas sin respaldo en la generación, antes de tocar el evaluador.                                                                               |

## 4. Estructura del repositorio

Ajustada a la estructura final obligatoria ya especificada en
[`entregables/sabado/README.md`](entregables/sabado/README.md). `scripts/` se mantiene intacto —
se importa como librería, no se modifica.

```
<repo>/
├── README.md                     # plantilla README_EQUIPO.md llena, incl. "## Corpus e índice"
├── LICENSE
├── requirements.txt              # deps del pipeline (NO ragas — eso vive en scripts/requirements-evaluador.txt)
├── run.sh                        # punto de entrada de reproducibilidad en un comando
├── Dockerfile                    # reproducibilidad desde contenedor limpio (2 pts)
├── submissions.jsonl             # 992 respuestas finales
├── CORPUS.md
├── corpus_manifest.json
├── data/                         # existente: sample_50.jsonl, seed_targets.json, (sáb) test_992.jsonl
├── informe/
│   └── INFORME_TECNICO.pdf
├── interfaz/
│   └── app.py                    # GUI Streamlit, importa src.pipeline.answer_one directamente
├── scripts/                      # OFICIAL, sin modificar: common.py, citations.py, evaluate.py
└── src/
    ├── config.py                 # rutas, nombres de modelo/encoder, k's, umbrales, selector de backend (mac|turing)
    ├── ingest/
    │   ├── fetch.py               # fetch_url(), fetch_all(seed_targets_path)
    │   ├── parse_html.py          # parse_senado_basedoc(html), parse_suin(html) -> list[ArticleRecord]
    │   ├── parse_pdf.py           # extract_text(), needs_ocr(page_text), ocr_page(path, page_no, lang="spa")
    │   ├── segment.py             # segment_by_article(text), segment_paragraphs(text, target_words, overlap)
    │   ├── normalize.py           # clean_text(), prefix_with_norm_name(chunk, doc_meta), assert_traceable(chunk)
    │   └── build_corpus.py        # orquestador: fetch→parse→segment→normalize→corpus/<doc_id>.txt + manifest
    ├── index/
    │   ├── embed.py                # embed_passages(texts), embed_query(text) — prefijos e5 + normalización L2
    │   ├── build_faiss.py          # build(chunks_path, out_path) -> index.faiss
    │   └── build_bm25.py           # build(chunks_path) -> índice bm25
    ├── retrieve/
    │   ├── hybrid.py               # retrieve(query, k_bm25, k_dense) -> lista fusionada por RRF
    │   ├── rerank.py               # rerank(query, candidates, top_k) [opcional, tras flag]
    │   ├── query_expand.py         # expansión de alias basada en reglas usando citations.CODES — sin LLM
    │   └── sufficiency.py          # decide(scored) -> "sufficient" | "retry" | "abstain"
    ├── generate/
    │   ├── llm.py                  # class LLM: generate(prompt, max_tokens, grammar=None); backend=metal|cuda
    │   ├── grammars/*.gbnf         # una gramática por formato
    │   ├── prompts/{multiple_choice,semi_open,open_ended}.py
    │   ├── formatter.py            # parse(raw, formato) -> dict
    │   └── citation_check.py       # unsupported_citations(answer_text, pasajes) — espeja evaluate.citas_respaldadas
    ├── pipeline/
    │   ├── answer_one.py           # answer(item) -> dict con forma de entrega para un item
    │   └── run_batch.py            # main(split, out_path, resume=True) — escritura JSONL incremental, resiliente a fallos
    ├── validate/
    │   ├── schema_check.py
    │   ├── traceability_check.py   # cada fragmento produce citations.extract() no vacío cerca de su inicio
    │   ├── manifest_check.py       # cada doc_id en chunks.jsonl ∈ corpus_manifest.json, sha256 coincide
    │   ├── leakage_check.py        # sin solapamiento entre corpus/ y los campos de pregunta/respuesta de sample_50.jsonl
    │   ├── determinism_check.py    # solo Mac: reejecuta N items dos veces, compara byte a byte
    │   ├── latency_check.py        # mide sample_50 en el hardware real, extrapola a 992, valida margen
    │   └── run_all.py              # agregador único, <2 min, gate antes de cada checkpoint
    └── main.py                     # CLI: python src/main.py --split sample|test [--rebuild-index]
```

## 5. Diseño del pipeline RAG

### Flujo por pregunta (una pasada por defecto, reintento selectivo — no RAG agéntico uniforme)

```
pregunta ──► construir query (pregunta [+ opciones si es multiple_choice])
          ──► expansión de alias basada en reglas (citations.CODES, sin LLM)
          ──► recuperación híbrida: BM25 top-30 ∪ denso top-30 → fusión RRF
          ──► [opcional] rerank del top-20 fusionado → top-6
          ──► sufficiency.decide(scored_top_k)
                 ├─ "sufficient" ──► generar (1 llamada al LLM, temp=0, con gramática)
                 ├─ "retry"      ──► UNA recuperación ampliada/reformulada ──► redecidir
                 │                      ├─ ahora suficiente ──► generar
                 │                      └─ sigue débil     ──► abstenerse
                 └─ "abstain"    ──► saltar la generación (ahorra tiempo y una llamada al LLM)
          ──► si se generó: citation_check.unsupported_citations(answer_text, pasajes)
          ──► ensamblar el dict de entrega (todas las llaves requeridas por el schema)
```

Esto acota el peor caso a 2 pasadas de recuperación + máximo 1 llamada de generación por pregunta —
respetando la advertencia del propio enunciado (Anexo B.5) de que un diseño agéntico uniforme de
4–5 pasadas probablemente no cabe en el presupuesto de ~22 seg/pregunta del sábado.

### Decisiones de diseño derivadas directamente de leer `evaluate.py`

- **La decisión de abstenerse/responder se toma solo con señales de recuperación — nunca pidiéndole
  al LLM que se autoevalúe.** La confianza autorreportada es poco confiable y añade una fuente de
  no-determinismo. El umbral se ajusta empíricamente barriendo contra `evaluate.py --split sample`,
  porque eso optimiza directamente lo que se califica. Como `score_abstention` da 0.5 por abstenerse
  sin importar si la respuesta hubiera sido correcta, y los componentes de `cerradas`/`citas` solo
  pagan cuando efectivamente se responde, el umbral debe sesgarse hacia responder cuando la
  recuperación encontró algo plausible, y reservar la abstención para temas genuinamente sin
  cobertura en el corpus.
- **Auto-verificar citas sin respaldo antes de entregar, usando la misma lógica del evaluador.** La
  tasa de citas sin respaldo (`tasa_sin_respaldo`) se resta con peso **2×** en el puntaje de
  citación — y es completamente auto-computable en tiempo de generación llamando a
  `citations.extract()`/espejando `evaluate.citas_respaldadas()` (solo cuentan como respaldo los
  primeros `MAX_PASAJES_EVIDENCIA=10` pasajes), sin necesitar el `legal_basis` oculto. Usar esta
  señal para disparar una regeneración correctiva determinista o abstención.
- **Campos que importan por formato** (de `evaluate.answer_text()` / `ragas_text()`):
    - `multiple_choice`: solo `justificacion` se evalúa para citas — ahí debe ir la cita que sustenta.
    - `semi_open`: las citas se revisan en `respuesta` + `referencia_legal` combinados, pero **el
      juez de RAGAS (30 pts) solo ve `respuesta`** — debe ser una respuesta completa, correcta y
      autocontenida de 3–5 oraciones (≤150 palabras) por sí sola; no delegar el contenido sustantivo a
      `referencia_legal`.
    - `open_ended`: los cuatro campos cuentan tanto para citas como para RAGAS.
- **Los ajustes deterministas post-generación son válidos** (recorte al límite de palabras, quitar
  una cita sin respaldo, una regeneración correctiva scripted) — la prohibición de "edición manual
  posterior" se refiere a que un humano edite la salida a mano, no a una etapa reproducible del
  pipeline.
- **El máximo de tokens de salida se fija por formato según los límites del propio schema**
  (semi_open ≤150 palabras, open_ended `analisis` 5–8 oraciones) — esto refuerza el cumplimiento del
  formato y acota directamente la latencia por pregunta.

## 6. Validaciones — qué, dónde, cómo

| Chequeo                                                           | Corre en                              | Implementación                                                                                                                                 | Frecuencia                                                                                                                              |
|-------------------------------------------------------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Validación de schema                                              | generación (por item) + pre-entrega   | `jsonschema` + `validate()` de `evaluate.py`                                                                                                   | cada corrida; gate antes de cada checkpoint                                                                                             |
| Trazabilidad de fragmentos (cada chunk indica su norma de origen) | construcción del índice               | `traceability_check.py`: `citations.extract(texto[:200])` no vacío (o el nombre canónico de la norma presente) para ≥~98% de los chunks        | cada reconstrucción del corpus — construir esto **el jueves en la mañana**, es el requisito de corrección más consecuente del enunciado |
| Auto-verificación de respaldo de citas                            | generación, por item, antes de emitir | `citation_check.py`, espeja `evaluate.citas_respaldadas` (`MAX_PASAJES_EVIDENCIA=10`)                                                          | cada corrida                                                                                                                            |
| Validación del manifest del corpus                                | construcción del índice               | `manifest_check.py`: cada `doc_id` en `chunks.jsonl` ∈ `corpus_manifest.json`; campos requeridos presentes; sha256 coincide                    | cada reconstrucción                                                                                                                     |
| Sin fuga del banco de respuestas                                  | ingesta + construcción del índice     | `leakage_check.py`: sin solapamiento de subcadenas/n-gramas entre `corpus/*.txt` y los campos de pregunta/respuesta de `sample_50.jsonl`       | cada reconstrucción; reverificar justo al recibir `test_992.jsonl` el sábado                                                            |
| Determinismo/reproducibilidad                                     | pre-entrega, **solo Mac**             | `determinism_check.py`: reejecutar 3–5 items dos veces en procesos frescos, comparar byte a byte — ensaya literalmente la verificación en vivo | diario en el Mac; obligatorio el viernes en la noche e inmediatamente antes de entregar el sábado 15:00                                 |
| Presupuesto de tiempo/latencia                                    | pre-entrega, periódico                | `latency_check.py`: medir `sample_50.jsonl` **en el Mac**, extrapolar a 992, validar margen contra ~6h                                         | línea base el miércoles, diario, chequeo final el viernes en la noche                                                                   |
| Reconstructibilidad del índice                                    | antes de empaquetar el zip del corpus | borrar el índice construido, reejecutar `build_faiss.py`/`build_bm25.py` desde `corpus/` + manifest, comparar hashes                           | una vez, justo antes de congelar                                                                                                        |

Todo lo anterior corre tras `python src/validate/run_all.py` (objetivo: <2 min), usado como gate
antes del correo del viernes 17:00 y de la entrega del sábado 15:00.

## 7. Plan de trabajo por días

### Miércoles (hoy), tarde — pipeline naive de punta a punta, corpus mínimo, ambas máquinas probadas

Meta: que `scripts/evaluate.py --submission out/dev.jsonl --split sample` imprima un reporte real
esta noche.

- Esqueleto: `src/`, `interfaz/`, `informe/`, `requirements.txt`, ajustes de `.gitignore` para
  descargas crudas, pesos GGUF, artefactos de índice.
- **Riesgo oculto de hoy**: compilar/instalar `llama-cpp-python` con Metal en el Mac *y* con CUDA en
  Turing; cargar un GGUF de Llama-3.1-8B-Instruct (Q4_K_M) y generar un token en cada una. Confirmar
  ambas antes de construir nada encima.
- Corpus: solo la **Constitución** (90 items del banco, un parser HTML limpio) — fetch, parse,
  segmentar por artículo, escribir `corpus/constitucion.txt` + una fila del manifest.
- Conectar `build_faiss.py` + `build_bm25.py` (corpus de 1 doc) → `hybrid.py` (RRF, sin rerank aún)
  → `llm.py` + prompts reales por formato → `answer_one.py` → `run_batch.py`.
- Correr `src/main.py --split sample` → `scripts/evaluate.py --split sample`. **Este es el hito**:
  prueba que todo el cableado (recuperación→generación→schema→evaluador) funciona de punta a punta;
  todo lo siguiente es iteración, no riesgo de integración.
- Stretch: dejar `interfaz/app.py` llamando a `answer_one` directamente, sin estilo.

### Jueves — amplitud de corpus (Turing) + robustez del pipeline (validado en Mac)

**Mañana:** Expandir el corpus siguiendo el orden de prioridad de `seed_targets.json` — Código
General del Proceso (65 items_del_banco), CST (37), Estatuto Tributario (35), Decisión Andina 486 (23), Estatuto del
Consumidor (13), Ley 80/1993 (12), Ley 2220/2022 (11), Decreto 2153/1992 (10) —
todo HTML, mismos dos parsers, cero OCR. Esto solo ya cubre ~73% de la cobertura ponderada de
`seed_targets.json`. Construir `normalize.prefix_with_norm_name()` + `traceability_check.py` ahora.
Añadir `sufficiency.py` con un umbral real. Construir el índice de embeddings en Turing por
velocidad, luego **reejecutar la misma construcción en el Mac y comparar** antes de congelar. **Tarde:** Finalizar
prompts por formato con ejemplos few-shot estructurales (nunca copiar el
contenido real de "Ejemplo de entrega" como si fuera real). Añadir decodificación con gramática GBNF (o reparación
robusta de JSON como fallback). Añadir ingesta de PDF para la jurisprudencia de mayor
ROI (Sentencia C-355/2006 = 7 items es el mejor objetivo individual; luego el nivel ≥2 items,
ordenado directamente desde `seed_targets.json`). Construir `run_all.py`. **Correr el chequeo
go/no-go Mac-vs-Turing** (sección 2). Reejecutar `evaluate.py --split sample`, registrar el 2º dato
para la tabla "Evolución del puntaje" de `CORPUS.md`. **Noche (si hay tiempo):** Reranker tras flag, comparado A/B.
Estilizar la interfaz. Probar el
Dockerfile. Corrida cronometrada de las 50 muestras en el Mac, extrapolar latencia a 992.

### Viernes en la mañana — endurecer + checkpoint

- Seguir enriqueciendo el corpus hacia las áreas menos cubiertas (cruzar la tabla por área de
  `CORPUS.md` contra `seed_targets.json`).
- Correr `determinism_check.py` **en el Mac**; corregir cualquier no-determinismo (semillas sin
  fijar, deriva en el número de threads, orden de diccionarios filtrándose a los prompts).
- Finalizar `CORPUS.md` / `corpus_manifest.json` / `README.md` / `INFORME_TECNICO.md` con números
  reales.
- Correr `evaluate.py --split sample --ragas` una vez (requiere `OPENROUTER_API_KEY`) para un
  número real de RAGAS.
- Enviar `REPORTE_AVANCE.pdf` antes de las 17:00 a `rf.manrique@uniandes.edu.co`, asunto
  `[Hackathon 2026] Avance — <equipo>`.
- 13:00–17:00 charlas obligatorias — sin tiempo de desarrollo, por eso lo crítico del checkpoint
  debe quedar resuelto en la mañana.

### Viernes en la noche / sábado temprano (colchón) — congelar, en el Mac

- Congelar corpus + índice, calcular sha256 finales, empaquetar el zip `corpus_<equipo>.zip` con la
  estructura requerida, subirlo, verificar que el link abre en una sesión de navegación privada.
- Corrida final de determinismo + latencia **en el estado exacto de hardware/software del Mac**
  planeado para el sábado.

### Sábado 9:00–15:00 — ejecución ciega

- 9:00: recibir `test_992.jsonl`. Chequeo rápido de fuga (debería ser instantáneo ya que el índice
  está congelado).
- Lanzar `python src/main.py --split test` de inmediato como job por lotes desatendido, resiliente
  y con escritura incremental, en el Mac — luego trabajar en paralelo el pulido de la interfaz y el
  video mientras corre.
- Revisar puntualmente la salida parcial con `schema_check` a medida que se acumula.
- Grabar el video ≤5min y ensayar la consulta de demo en vivo de la interfaz.
- Al terminar (meta: bien antes de las 15:00): `run_all.py`, armar el repo final según el checklist
  del sábado, entregar.
- 15:00–17:00: verificación en vivo — mantener listo el entorno exacto (`requirements.txt`, mismo
  archivo GGUF, mismo Mac, misma config de threads) para reejecutar 2–3 items específicos a demanda.

## 8. Riesgos y mitigaciones

| Riesgo                                                                        | Mitigación                                                                                       |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Divergencia de generación entre Turing (CUDA) y Mac (Metal)                   | Mac como fuente canónica desde el jueves (sección 2); chequeo go/no-go explícito el jueves.      |
| Presupuesto de ~22 seg/pregunta insuficiente para un diseño agéntico          | Una pasada por defecto + reintento selectivo, nunca multi-pasada uniforme (sección 5).           |
| Fragmentos sin trazabilidad a su norma de origen (anula el respaldo de citas) | `traceability_check.py` construido el jueves en la mañana, antes de seguir ampliando el corpus.  |
| Citas sin respaldo penalizadas a 2×                                           | Auto-verificación con `citations.py` antes de emitir cada respuesta (sección 5).                 |
| No-reproducibilidad en la verificación en vivo                                | `determinism_check.py` diario en el Mac desde el jueves; ensayo final el viernes en la noche.    |
| OCR consume demasiado tiempo                                                  | Reservado solo para la jurisprudencia de mayor `items_del_banco`; todo lo demás es HTML sin OCR. |
| Instalación de `llama-cpp-python` con Metal/CUDA falla a último momento       | Prueba de humo obligatoria hoy miércoles, no el jueves.                                          |

## 9. Checklist pre-entrega

**Antes del correo del viernes 17:00:**

- [ ] `evaluate.py --split sample` (con y sin `--ragas`) corre limpio y sin errores de validación.
- [ ] `REPORTE_AVANCE.pdf` cubre: puntaje en sample_50 con fecha, estado del corpus, arquitectura
  actual, riesgos para el sábado.
- [ ] Enviado a `rf.manrique@uniandes.edu.co` con el asunto exacto `[Hackathon 2026] Avance — <equipo>`.

**Antes de la entrega del sábado 15:00:**

- [ ] `submissions.jsonl` valida contra `schema/submission.schema.json` y contra `validate()` de
  `evaluate.py`.
- [ ] Índice congelado, reconstruible desde `corpus/` + `corpus_manifest.json`.
- [ ] `determinism_check.py` pasó en el Mac en el estado exacto que se usará en vivo.
- [ ] `CORPUS.md` + `corpus_manifest.json` completos, con la tabla de evolución del puntaje.
- [ ] Zip `corpus_<equipo>.zip` subido, con `LICENSE`, abre en sesión privada, vigente 30 días.
- [ ] `README.md` tiene la sección `## Corpus e índice` con el link.
- [ ] Interfaz funcional, muestra pasajes recuperados y citas, con identidad visual de Software
  Colombia.
- [ ] `run.sh` reproduce desde un contenedor limpio en un solo comando.
- [ ] Video ≤5 min grabado.
- [ ] `INFORME_TECNICO.pdf` (≤3 páginas) con arquitectura, configuración de inferencia, resultados y
  limitaciones conocidas.

## 10. Estado actual (2026-10-01) y próximos pasos

### 10.1 Qué ya está corriendo de punta a punta

El pipeline completo — corpus → índice → recuperación → generación → evaluación — corre hoy sobre
el Mac y produce un puntaje real, no estimado:

- **Corpus:** 34 379 documentos / 99 067 fragmentos, 100% trazables (`traceability_check.py`), manifest válido
  (`manifest_check.py`). Construido con
  `src/ingest/ingest_raw_sources.py` — **no** con el flujo `fetch.py`→`parse_html.py`/`parse_pdf.py`
  descrito en la sección 4/7, porque el corpus que terminó usando el equipo llegó como volcado JSON
  ya pre-parseado o pre-troceado por fuente (`data/corpus/`), no como HTML/PDF crudo. El flujo
  original sigue existiendo en el repo (`build_corpus.py`) para fuentes que sí lleguen crudas, pero
  la fuente real del corpus de hoy es otra. Tres orígenes:
    - Scraping de secretariasenado.gov.co (constitución, códigos, leyes, decretos, actos
      legislativos) — ya venía segmentado por artículo; 2 627 documentos.
    - Metadatos de relatoría de la Corte Constitucional (número, tema, resumen, resuelve — **sin**
      texto íntegro de la providencia) — 31 413 documentos, uno por providencia.
    - Providencias de la Corte Suprema, ya trozadas por un compañero del equipo en el esquema que
      `build_faiss.py`/`build_bm25.py` esperan — 339 documentos.
- **Índices:** `indice/chunks.jsonl`, `indice/bm25.pkl` e `indice/index.faiss` (99 067 vectores ×
  1024 dim, `IndexFlatIP`) construidos y verificados — búsquedas reales devuelven evidencia
  correcta (ver ejemplos en 10.2).
- **Decoder:** `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (bartowski, HuggingFace) descargado a
  `models/`, carga en Metal (`mps`) en ~7s, genera con gramática GBNF.
- **Corrida completa:** `python -m src.main --split sample` sobre las 50 preguntas de muestra,
  seguido de `python scripts/evaluate.py --submission out/sample.jsonl --split sample`, produce:

  | Componente                                               |                          Puntos obtenidos | Posibles |
    |----------------------------------------------------------|------------------------------------------:|---------:|
  | Exactitud en cerradas (10/15 correctas)                  |                                     13.33 |       20 |
  | Calidad de citación (índice 0.367, 0 citas sin respaldo) |                                      7.35 |       20 |
  | Abstención calibrada                                     |                                      5.93 |       10 |
  | Corrección texto libre (RAGAS)                           | pendiente — requiere `OPENROUTER_API_KEY` |       30 |
  | **Total automático medido**                              |                            **26.61 / 50** |          |

  Esta es la primera corrida real contra el corpus definitivo — todavía no se ha ajustado ningún
  umbral ni prompt contra ella. Pendiente actualizar la tabla equivalente de `CORPUS.md` /
  `README.md` cada vez que se vuelva a correr.

- **Validaciones que SÍ corrieron:** `traceability_check` (100%, umbral 98%), `manifest_check`
  (limpio), `leakage_check` (hallazgos esperados contra `sample_50.jsonl` — se solapan por buenas
  razones, el corpus contiene la norma real que sustenta cada `legal_basis`; ver 10.3).
- **Validaciones que corrieron el 2026-10-02, en el Mac, y pasan:** `determinism_check.py` (0
  divergencias, 3 ítems re-ejecutados en procesos frescos) y `latency_check.py` (17.39 seg/ítem
  promedio sobre `sample_50.jsonl`, extrapola a 4.79h para las 992 preguntas del sábado contra el
  presupuesto de 6h, margen 1.21h). Ninguno de los dos corría con éxito antes de hoy: ambos scripts
  traían bugs nunca detectados porque nunca se habían ejecutado contra el estado real —
  `determinism_check.py` pasaba un `str` donde `common.read_jsonl` espera un `Path`;
  `latency_check.py` no atrapaba excepciones por ítem (a diferencia de `run_batch.py`, que sí es
  resiliente por diseño), así que un solo `FormatError` tumbaba toda la medición. Detalle de ambos
  fixes y de las corridas en
  [`../solution-hackathon-ai-week-2026/CORPUS.md` sección 4](../solution-hackathon-ai-week-2026/CORPUS.md).
- **Validación que sigue sin correr: el chequeo go/no-go Mac-vs-Turing de la sección 2** — ni
  siquiera se ha probado `llama-cpp-python` con CUDA en Turing todavía. Nada de lo de arriba lo
  reemplaza: determinismo y latencia se midieron *solo en el Mac*, que es la máquina canónica, pero
  el riesgo que la sección 2 señala (CUDA vs Metal pueden generar tokens distintos a
  `temperatura=0` con los mismos pesos) sigue sin descartarse ni confirmarse empíricamente. **Es lo
  más urgente de la lista de abajo** — ver [sección 10.6](#106-próximos-pasos-en-turing) para el
  plan concreto, pensado para quien tenga acceso real a Turing (esta sesión no lo tiene).

### 10.2 Diagnóstico de los 5/15 ítems de opción múltiple no acertados

Se diagnosticó cada fallo individualmente (pregunta, opciones, evidencia recuperada, justificación
generada) en vez de asumir una causa única. Los cuatro fallos reales se reparten en causas
completamente distintas — importa porque cada una se arregla distinto:

1. **Orden del esquema JSON invierte el razonamiento (2 de 4 fallos — ítems 290 y 308).** La
   gramática GBNF (`src/generate/grammars/multiple_choice.gbnf`) fuerza el orden
   `respuesta_correcta` → `justificacion` → `descarte_opciones`: el modelo debe comprometerse con
   una letra **antes** de escribir cualquier razonamiento. Ambos fallos citan el mismo artículo
   correctamente recuperado (Ley 1150 de 2007, Art. 11, que trae tres plazos distintos en tres
   oraciones consecutivas: 4 meses / 2 meses / 2 años para sub-cláusulas diferentes) y el modelo
   confunde cuál plazo responde la pregunta. En el ítem 308 la `justificacion` generada dice
   textualmente "dos (2) meses" — la respuesta correcta — pero `respuesta_correcta` ya quedó fijada
   en la letra equivocada antes de llegar ahí. No es un problema de recuperación ni de conocimiento:
   es que el campo final se genera antes que el razonamiento que lo informaría.
2. **Sin señal de vigencia (1 fallo — ítem 528).** Se recuperaron tanto la norma vigente (Ley 1564
   de 2012 / CGP, Art. 25) como una norma derogada que modificaba el antiguo Código de Procedimiento
   Civil (Ley 572 de 2000) — la derogada rankeó primero por un margen mínimo (0.0323 vs 0.0318) y el
   modelo la usó sin cuestionarla. Nada en la recuperación ni en el prompt distingue "vigente" de
   "derogado" cuando ambas coexisten en el corpus.
3. **Hueco real de cobertura del corpus (1 fallo — ítem 671).** El `legal_basis` de referencia es
   literalmente `"Doctrina"` — reglas de desempate de tratados tributarios internacionales, que no
   son derecho positivo colombiano codificado. Ningún pasaje recuperado es relevante, y
   reveladoramente los seis scores rondan el piso de suficiencia (0.0159–0.0164, umbral 0.014): el
   sistema debió reintentar/abstenerse en vez de responder con confianza.
4. **Truncamiento por presupuesto de tokens (1 ítem adicional — 128, cuenta como abstención con
   error, no como fallo de opción múltiple).** `MAX_TOKENS_BY_FORMAT["multiple_choice"] = 320` se
   agotó a mitad de la justificación, el JSON nunca cerró, `formatter.parse()` lanzó `FormatError`.
   `run_batch.py` lo atrapó y escribió una abstención de respaldo en vez de tumbar el lote completo
   — el manejo de errores funcionó como se diseñó, pero el ítem se perdió igual.

### 10.3 Por qué `leakage_check.py` marca "hallazgos" contra una corrida sana

Al correr `leakage_check.py` contra el corpus real aparecen decenas de coincidencias de n-gramas
entre `sample_50.jsonl` y `corpus/*.txt` — por diseño el check no distingue entre "el banco de
preguntas se filtró al corpus" (grave, descalifica) y "el corpus contiene la norma real que
sustenta la respuesta esperada" (exactamente lo que un corpus bien construido debe tener). Todos los
casos revisados manualmente son del segundo tipo (p.ej. `ley-984-2005.txt` para una pregunta sobre
la CEDAW). No se tocó la lógica del check (es oficial, sin modificar) — queda como limitación
conocida del check mismo, a tener en cuenta en vez de perseguir un "0 hallazgos" que no sería una
señal real. Re-verificar con criterio el sábado contra `test_992.jsonl`, que es donde sí importaría
una fuga real.

### 10.4 Próximos pasos, en orden de prioridad

**Actualizado 2026-10-02 tarde.** Items tachados con ~~…~~ ya se resolvieron hoy (en el Mac, sesión
sin acceso a Turing) — se dejan en la lista con el resultado medido en vez de borrarse, para que
quede registro de qué se intentó y qué no funcionó. Detalle completo de cada corrida en
[`../solution-hackathon-ai-week-2026/CORPUS.md` sección 4](../solution-hackathon-ai-week-2026/CORPUS.md).

**Bloqueantes de entrega (sección 6/9 — no son mejoras, son requisitos):**

1. ~~`determinism_check.py` en el Mac~~ — **hecho, pasa.** 0 divergencias en 3 ítems (procesos
   frescos). El script tenía un bug (`str` en vez de `Path`) que impedía que corriera con éxito
   antes de hoy — corregido.
2. ~~`latency_check.py` sobre las 50 muestras en el Mac~~ — **hecho, pasa.** 17.39 seg/ítem
   promedio, extrapola a 4.79h para las 992 preguntas contra el presupuesto de 6h (margen 1.21h).
   También tenía un bug (no atrapaba excepciones por ítem, a diferencia de `run_batch.py`) —
   corregido.
3. **Chequeo go/no-go Mac-vs-Turing (sección 2): sigue sin hacerse.** Ni siquiera se ha probado
   `llama-cpp-python` con CUDA en Turing todavía — esta sesión no tuvo acceso a Turing. **Ahora es
   el bloqueante más urgente de toda la lista.** Plan concreto en la nueva
   [sección 10.6](#106-próximos-pasos-en-turing).
4. Reconstructibilidad del índice desde cero (sección 6, última fila): borrar `indice/`, reconstruir
   con `build_bm25.py`/`build_faiss.py` desde `corpus/` + manifest, comparar hashes. Sigue pendiente
   (no depende de Turing, se puede hacer en cualquier máquina con el corpus local).

**Mejoras de precisión, de más a menos evidencia directa (sección 10.2):**

5. ~~Reordenar el esquema JSON de `multiple_choice`~~ — **probado y revertido: regresión medida, no
   mejora.** La hipótesis (comprometerse con `respuesta_correcta` antes de razonar causaba 290/308)
   era correcta para el ítem 308 — reordenarlo sí lo corrigió. Pero medido contra
   `evaluate.py --split sample` el cambio completo dio **6/15 en cerradas, no 10/15**: 5 ítems que
   antes acertaban pasaron a fallar, con contenido generado completamente distinto (otra ley, otro
   artículo), no solo la letra reordenada. A `temperatura=0`/greedy, reescribir el orden de llaves
   del few-shot y la gramática cambia la secuencia de tokens del prompt lo suficiente para que la
   decodificación diverja desde el primer token generado — no es un ajuste "seguro" aunque el
   diagnóstico detrás sea válido. Revertido a su forma original. Si se retoma: probar algo más
   quirúrgico (un campo de razonamiento *separado* antes del JSON final, fuera de la gramática
   forzada) y volver a medir antes de asumir que ayuda — no aplicar el mismo reorden a `semi_open`/
   `open_ended` sin medir cada uno por separado.
6. ~~Subir `MAX_TOKENS_BY_FORMAT["multiple_choice"]` por encima de 320~~ — **hecho.** Probado
   empíricamente (320→FAIL, 480→FAIL, 650→OK, 700 con margen) y confirmado aislado del punto 5 (sin
   el reorden, mismo resultado que el baseline en los 14 ítems no afectados por la truncación).
7. ~~Re-correr `src/main.py --split sample` + `evaluate.py`~~ — **hecho**, varias veces, para medir
   cada cambio de forma aislada antes de aceptarlo. Total automático actual: 26.49/50 (vs. 26.61
   baseline del 10-01 — la diferencia es solo el ítem que dejó de truncar y ahora falla
   "limpiamente" en vez de abstenerse por excepción).
8. Barrer `SUFFICIENCY_SCORE_THRESHOLD`/`RETRY_SCORE_THRESHOLD` — **sigue pendiente**, no se tocó
   esta sesión.
9. Señal de vigencia — **sigue pendiente**, no se tocó esta sesión.
10. Reranker opcional (`BAAI/bge-reranker-v2-m3`, `USE_RERANKER=1`) — **sigue pendiente**. Nota para
    quien lo retome: `src/retrieve/rerank.py` sustituye el score RRF del candidato por el score
    crudo del cross-encoder cuando `USE_RERANKER=1`, pero `sufficiency.py` sigue comparando ese
    score contra `SUFFICIENCY_SCORE_THRESHOLD`/`RETRY_SCORE_THRESHOLD`, calibrados en escala RRF
    (~0.0328 máx). Si se activa el reranker hay que recalibrar esos umbrales en la escala del
    cross-encoder, o la decisión de abstención queda comparando dos escalas incompatibles —
    detectado leyendo el código esta sesión, no verificado en una corrida real.

**Decoder (SPEC.md sección 10.5): implementado y A/B probado, resultado en el Mac — Turing es el
siguiente paso real, ver sección 10.6.**

**Cobertura del corpus:**

11. El Estatuto Orgánico del Sistema Financiero — **sigue pendiente**, no se tocó esta sesión.
12. Reglamento CNE y directivas presidenciales — **sigue pendiente**, probablemente no vale la pena
    (sin cambios en la evaluación de prioridad).
13. Preguntas de doctrina pura (ítem 671) — sigue siendo un no-respondible estructural, no un hueco
    de corpus; la mitigación sigue siendo el punto 8.

**Puntaje aún no medido:**

14. Correr `evaluate.py --split sample --ragas` apenas haya `OPENROUTER_API_KEY` — **sigue
    pendiente**, son 30 de los 50 puntos automáticos, completamente sin medir todavía.

**Interfaz y entregables:**

15. ~~`interfaz/app.py`: verificar pasajes recuperados, citas con badge de respaldo~~ — **citas con
    badge de respaldo, hecho** (pills verde/rojo reusando `citation_check.unsupported_citations`,
    sin tocar la lógica oficial). Identidad visual de Software Colombia ya estaba hecha desde antes
    de esta sesión. Pendiente: probar la interfaz corriendo de punta a punta con `streamlit run`
    contra una pregunta real (no solo revisión de código).
16. `INFORME_TECNICO.md`/`.pdf`, video ≤5 min, `Dockerfile` — **sigue pendiente**, no se tocó esta
    sesión.

### 10.5 Decoder intercambiable — implementado y comparado A/B (2026-10-02)

Lo que era una propuesta ahora está implementado y corrido contra `evaluate.py --split sample`, en
el Mac:

- `config.LLM_CANDIDATES`: registro de candidatos ≤8B probados por nombre corto → `{filename,
  chat_template}`. Hoy: `llama-3.1-8b-instruct` (activo) y `qwen3-8b`.
- `src/generate/llm.py`: `get_llm(model_name=None)` cachea instancias por ruta GGUF resuelta en vez
  de un singleton único — alternar entre candidatos ya descargados no recarga un GGUF de ~5GB desde
  cero. `LLM.generate()` envuelve el prompt en marcadores ChatML (`<|im_start|>`/`<|im_end|>`) solo
  para el candidato que lo declare (`qwen3-8b`) — Qwen3-Instruct se entrenó casi exclusivamente en
  ese formato y sin turno explícito sigue instrucciones peor; para Llama es un no-op, cero riesgo
  sobre el camino ya validado. La gramática GBNF por formato fuerza `"{"` como primer carácter en
  cualquier caso, así que el modo "thinking" de Qwen3 (bloque `<think>...</think>`) no puede
  filtrarse al output aunque no se desactive explícitamente — confirmado con un smoke test
  gramática-vs-sin-gramática.
- `answer()`/`run_batch.main()`/`src/main.py --model-name` y un selector en `interfaz/app.py` (bajo
  "Opciones de desarrollo", claramente marcado como no apto para la entrega final) lo exponen.

**Resultado A/B, Qwen3-8B vs. Llama-3.1-8B-Instruct, mismo corpus/índice/umbrales, sobre
`sample_50.jsonl`:**

| Métrica | Llama-3.1-8B (activo) | Qwen3-8B |
|---|---:|---:|
| Cerradas | 10/15 | 10/15 |
| Índice de citación | 0.367 | **0.5** |
| Abstención (pts /10) | 5.81 | **6.86** |
| **Total automático /50** | 26.49 | **30.19** |
| `determinism_check.py` | ✅ 0 divergencias | ✅ 0 divergencias |
| Latencia promedio/ítem | 17.39–17.82s | 19.97–20.87s |
| Margen proyectado (992, 6h) | 1.09–1.21h | **0.25–0.50h** (dos mediciones, bastante ruido) |

Qwen3 mide claramente mejor en lo que automatiza el puntaje — no es ruido de una sola corrida, la
ventaja en citación y abstención se repite. Se promovió a candidato activo brevemente el mismo día
y **se revirtió horas después**: una segunda medición de `latency_check.py` sobre el estado final
(sin overrides) dio un margen más delgado y más ruidoso que la primera corrida (0.25h, no 0.50h —
la varianza entre dos corridas idénticas en el mismo hardware ya es mayor que el margen reportado la
primera vez). Con los días de desarrollo que quedan y sin margen para resolver un desborde de
presupuesto en vivo el sábado, se decidió no aceptar ese riesgo todavía: `llama-3.1-8b-instruct`
sigue siendo el candidato activo. Detalle completo de ambas corridas en
[`../solution-hackathon-ai-week-2026/CORPUS.md` sección 4](../solution-hackathon-ai-week-2026/CORPUS.md).

**Restricción dura que no cambia:** el enunciado exige temperatura 0 y determinismo en la ejecución
final, y la sección 2 ya establece que **todo lo que termine en `submissions.jsonl` debe salir de
una única combinación fija de modelo + hardware (el Mac)**, validada por `determinism_check.py`.
"Intercambiable" sigue siendo una comodidad de desarrollo/demo, no una característica del pipeline
de entrega: el modelo ganador se congela antes de la corrida del sábado exactamente igual que hoy
se congela el índice. Alternar modelos por pregunta o por corrida en la entrega final violaría tanto
el requisito de determinismo como la regla de "Mac canónico".

**Por qué Qwen3 vale la pena recuperar, no descartar:** +3.7 puntos automáticos (~14% relativo) es
demasiado para abandonarlo solo porque la primera versión quedó ajustada de latencia. El camino más
prometedor es recortar `MAX_TOKENS_BY_FORMAT` específicamente para Qwen3 (hoy usa los mismos valores
que se calibraron para Llama, sin reajustar) — iterar eso rápido es exactamente el rol de Turing en
la sección 2. Plan concreto abajo.

### 10.6 Próximos pasos en Turing

Pensado para quien tenga acceso real a una sesión en Turing — esta sesión se hizo enteramente en el
Mac y no pudo tocar ninguno de estos puntos. En orden:

1. **Smoke test obligatorio de la sección 2, todavía sin hacer:** compilar/instalar
   `llama-cpp-python` con soporte CUDA en Turing, cargar cada GGUF que ya está en `models/` del Mac
   (`llama-3.1-8b-instruct-q4_k_m.gguf`, `Qwen3-8B-Q4_K_M.gguf` — copiarlos a Turing o
   re-descargarlos desde `Qwen/Qwen3-8B-GGUF` / el repo de Llama usado) y generar un token de cada
   uno. Si esto falla, es el tipo de imprevisto que hay que resolver *hoy*, no el viernes.
2. **Chequeo go/no-go Mac-vs-Turing (sección 2), el bloqueante de entrega más urgente de la
   sección 10.4 #3.** Correr las mismas 10–20 preguntas de `sample_50.jsonl` en Turing (CUDA) con
   `src/main.py --split sample --out out/sample_turing.jsonl` y comparar la generación cruda contra
   la ya producida en el Mac (`out/sample.jsonl` / `out/sample_v2.jsonl` en el repo de la solución).
   Es esperable que diverjan — eso *confirma* la regla de "Mac canónico" de la sección 2, no la
   contradice. Si en cambio son byte-idénticas, mejor todavía, pero no cambia el procedimiento: todo
   lo que termine en `submissions.jsonl` sigue saliendo solo del Mac.
3. **Recortar el presupuesto de latencia de Qwen3-8B, usando la velocidad de Turing para iterar
   rápido (sección 2: "Iteración de prompts... Turing, rápido... los puntajes medidos aquí son
   orientativos").** Hoy Qwen3 usa los mismos `MAX_TOKENS_BY_FORMAT` que se calibraron para Llama
   (700/420/900) sin reajustar — probar valores más bajos por formato específicamente para Qwen3
   (`LLM_MODEL_NAME=qwen3-8b src/main.py --split sample --model-name qwen3-8b`, o directamente
   `--model-name qwen3-8b`) contra `evaluate.py --split sample`, buscando el punto donde la latencia
   baja sin que la puntuación de citación/abstención se caiga de vuelta a los niveles de Llama.
   Objetivo: recuperar el margen de latencia (hoy 0.25–0.50h en el Mac) sin perder la ventaja de
   +3.7 puntos automáticos medida en la sección 10.5. **Cualquier configuración que parezca ganar en
   Turing se re-valida en el Mac** (`determinism_check.py` + `latency_check.py` con
   `LLM_MODEL_NAME=qwen3-8b` o el nuevo default si se decide promoverlo) antes de considerar
   promoverla a candidato activo — Turing nunca es la fuente de verdad para esta decisión, solo
   acelera la búsqueda.
4. Si el punto 3 libera tiempo: usar Turing para barrer `SUFFICIENCY_SCORE_THRESHOLD`/
   `RETRY_SCORE_THRESHOLD` (sección 10.4 #8) y comparar A/B el reranker opcional (sección 10.4 #10)
   — ambos son búsquedas empíricas contra `evaluate.py --split sample` que se benefician de
   iteración rápida antes de revalidar en el Mac. **Si se activa el reranker, recalibrar
   `SUFFICIENCY_SCORE_THRESHOLD`/`RETRY_SCORE_THRESHOLD` en la escala del cross-encoder** — hoy
   están calibrados en escala RRF (~0.0328 máx) y `sufficiency.py` compararía dos escalas
   incompatibles si el reranker queda prendido sin ese ajuste (ver nota en 10.4 #10).
5. Stretch, solo si el pipeline central ya está sólido y sobra tiempo (sección 2, última fila):
   fine-tuning LoRA/QLoRA sobre el candidato que gane. Cualquier resultado se re-valida en el Mac
   antes del viernes, igual que todo lo demás de esta lista.

**Esfuerzo estimado:** bajo-medio — el cambio en `llm.py`/`config.py` es pequeño y aditivo (no
rompe el comportamiento actual si no se pasa ningún override); el valor real está en tener
candidatos ya descargados y un protocolo de comparación, no en la mecánica del *switch* en sí.

## 11. Turing como máquina canónica y resultados (2026-10-02)

### 11.1 Cambio de instrucciones

Turing (Linux + RTX 4090 24 GB, `llama-cpp-python` compilado con CUDA, `torch` CUDA) es la **máquina
principal**: genera `submissions.jsonl` y corre los chequeos oficiales de determinismo y latencia. El
Mac queda como auxiliar, **solo** para pruebas de determinismo cruzado (`RAG_BACKEND=mac`).
Consecuencias aplicadas en código (`src/config.py`): `IS_CANONICAL = BACKEND == "turing"`;
`determinism_check.py` corre en Turing. Sigue valiendo que todo lo que termina en la entrega sale de
**una única combinación fija modelo + hardware** (ahora Turing), y que "intercambiable" es solo
comodidad de desarrollo.

La restricción de latencia deja de ser el cuello de botella: ~5–8 s/ítem en la 4090 vs ~17–20 s en el
Mac. Esto reabre candidatos que se habían descartado por latencia (Qwen3-8B) y permite el reranker.

### 11.2 Comparación de decoders ≤8B (GPU, mismo corpus/índice/umbrales, `sample_50`)

| Decoder (GGUF Q4_K_M) | Cerradas /15 | Índice citación | Abstención /10 | **Total auto /50** |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct (prompt crudo, anterior activo) | 7 | 0.395 | 4.88 | 22.10 |
| Llama-3.1-8B-Instruct (plantilla chat Llama-3) | 8 | 0.311 | 4.77 | 21.66 |
| Mistral-7B-Instruct-v0.3 | 8 | 0.418 | 6.40 | 25.44 |
| Qwen2.5-7B-Instruct | 10 | 0.429 | 6.28 | 28.20 |
| Aya-Expanse-8B (**CC-BY-NC**, verificar licencia) | 11 | 0.429 | 6.28 | 29.52 |
| **Qwen3-8B** | 11 | **0.500** | **6.98** | **31.65** |

Qwen3-8B gana en todos los componentes y pasa a **candidato activo** (`LLM_MODEL_NAME="qwen3-8b"`).
Caveat: 15 cerradas y 50 ítems en total — ±1 pregunta cerrada = ±1.33 pts; el orden de los extremos
es robusto, el de los intermedios no. Nota: la corrida de Llama en Turing (22.10) es menor que la del
Mac (26.49), consistente con la divergencia CUDA/Metal anticipada en la sección 2.
Los 29.5/31.7 usan umbrales RRF; el RAGAS (30 pts) sigue sin medirse (sin `OPENROUTER_API_KEY`).

### 11.3 Recuperación: reranker y número de pasajes (Qwen3-8B)

| Configuración | Cerradas | Citación | Abstención | Total /50 |
|---|---:|---:|---:|---:|
| Base (RRF, top-6) | 11 | 0.500 | 6.98 | 31.65 |
| RRF, top-10 (fusión 30) | 10 | 0.480 | 6.51 | 29.43 |
| **Reranker `bge-reranker-v2-m3`, top-6** | 10 | **0.633** | **7.21** | **33.19** |

Con Llama el reranker + top-10 midió *peor* (15.9–18.1) — una vez más el efecto depende del decoder y
la muestra es chica; se adoptó la combinación Qwen3 + reranker + top-6 por ser la mejor medida.
`recall@6` de las normas de referencia en recuperación pura (`src/validate/retrieval_eval.py`):
0.695 (RRF) → 0.724 (reranker); recall@20 = 0.79 — el techo de recuperación es ~0.8.
Más pasajes (top-10) no ayuda: el modelo no los aprovecha y diluye.

**Umbrales de abstención con reranker (cierra 10.4 #8 y #10):** el score pasa a ser la probabilidad
sigmoide del cross-encoder (0–1), no RRF. Calibrados: `SUFFICIENCY_SCORE_THRESHOLD=0.10`,
`RETRY_SCORE_THRESHOLD=0.02` (en el sample los únicos 2 ítems con score <0.1 fallaron). Sin reranker
siguen 0.014/0.006. Barrido en la escala RRF: el cubo de score mínimo (0.0164) acertó 3/5, así que
abstenerse (0.5) no habría ganado — no se movieron esos umbrales. `USE_RERANKER` ahora es `1` por
defecto.

### 11.4 Fix: desborde de contexto

Con pasajes muy largos el prompt superó `LLM_N_CTX=8192` (10 275 tokens en un ítem) y la excepción lo
convertía en abstención. `render_passages` ahora recorta cada pasaje a `18000 // n` caracteres (mín.
1200) **solo en el prompt**; `pasajes_recuperados` conserva el texto completo para el respaldo de citas.

### 11.5 Validaciones en Turing (config final: Qwen3-8B + reranker, top-6)

- `determinism_check.py` (ítems 51, 58, 290, 308, 528; procesos frescos): **0 divergencias**.
- `latency_check.py`: **7.64 s/ítem → 2.10 h para 992** (presupuesto 6 h, margen **3.90 h**).
- `evaluate.py --split sample`: **33.19 / 50** automáticos (cerradas 10/15 → 13.33; citación 0.633 →
  12.65; abstención 7.21; RAGAS pendiente).
- Reconstructibilidad del índice (10.4 #4): **pasa.** Reconstruido desde `chunks.jsonl` en Turing
  (GPU, ~15 min) en un directorio aparte: `index.faiss` (sha256 `8d52fcbd…`) y `bm25.pkl`
  (`37aaa33d…`) son **idénticos byte a byte** a los de `indice/`.

### 11.6 Estado de la lista 10.4 / próximos pasos

Hechos: #3 (smoke test CUDA + chequeo en Turing), #8 (umbrales), #10 (reranker + recalibración), y la
parte de decoder de 10.5/10.6 (Qwen3 recuperado, margen de latencia holgado). Siguen pendientes:
1. **Mac**: re-correr `determinism_check.py` como chequeo auxiliar cruzado (no bloqueante).
2. Medir RAGAS (`OPENROUTER_API_KEY`) — 30 de los 50 pts siguen sin medirse.
3. Señal de vigencia (10.4 #9); Estatuto Orgánico del Sistema Financiero (#11).
4. Recuperación: techo recall@20=0.79 — probar ampliar `FUSED_TOP_K`/`K_BM25` con reranker, o un
   encoder/reranker mejor; con 15 cerradas, validar cualquier cambio con `retrieval_eval.py` además
   del puntaje total.
5. Reordenar el esquema de `multiple_choice` con razonamiento previo (10.4 #5), ahora re-medible rápido.
6. Informe técnico, video, Dockerfile, interfaz de punta a punta.
7. Stretch: LoRA sobre Qwen3-8B en Turing; la licencia de Aya-Expanse (CC-BY-NC) lo descarta para entrega
   salvo confirmación de los organizadores.

## 12. Ronda de mejora con RAGAS (2026-10-02, tarde)

Presupuesto de RAGAS: **20 USD en total** (OpenRouter, juez `z-ai/glm-5.3-flash`); una corrida de 35
ítems cuesta ~0.05–0.09 USD, pero las primeras corridas con concurrencia por defecto (16 workers,
timeout 180 s) se llenaron de `TimeoutError` y gastaron ~2.7 USD sin producir puntaje. Herramientas
nuevas (rama `adrian/rag-ragas-tuning` del repo de la solución):
`src/validate/ragas_dev.py` (misma métrica/juez que `evaluate.py --ragas`, con `RunConfig` tolerante,
1 reintento, imprime el costo y se niega a correr con <8 USD restantes) y
`src/validate/freetext_proxy.py` (proxy gratuito: similitud e5 respuesta↔referencia).

### 12.1 Resultados (Qwen3-8B + reranker, sample_50, Turing)

| Configuración | Cerradas /15 | Citación | Abst. /10 | Auto /50 | RAGAS correctness | RAGAS /30 | **Total /80** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pool 30/30→20, prompt original | 10 | 0.633 | 7.21 | 33.19 | 0.418 | 12.54 | 45.73 |
| Pool 30/30→20, prompt conciso | 10 | 0.592 | 6.74 | 31.91 | 0.437 | 13.10 | 45.01 |
| Pool 100/100→50, prompt original | 11 | 0.571 | 6.74 | 32.84 | — | — | — |
| **Pool 100/100→50, prompt conciso (activo)** | 11 | 0.551 | 6.51 | 32.20 | **0.469** | **14.08** | **46.28** |

- **Pool ampliado** (`K_BM25=K_DENSE=100`, `FUSED_TOP_K=50`): recall@6 de las normas de referencia con
  reranker 0.724→0.764, recall@10 0.793→0.841 (`retrieval_eval.py`); satura en 50/100 (más no ayuda).
  Consulta con opciones incluidas supera a la consulta solo-pregunta (0.764 vs 0.715 @6). El reranker
  ahora trunca a `RERANKER_MAX_LEN=1024` tokens (con 8192 y pools grandes agotaba VRAM).
- **Prompt conciso de `semi_open`** (1–3 oraciones, ≤80 palabras, empieza por la respuesta, sin datos
  que la evidencia no respalde): las referencias son ~1–2 oraciones y las respuestas largas del modelo
  agregaban afirmaciones no respaldadas que penalizan la precisión factual del juez. RAGAS 0.418→0.469
  (+1.5 pts), a costa de algo de citación (menos normas mencionadas).
- **Encoder alternativo `BAAI/bge-m3`** (re-embedding de las 99 067 fragmentos en 13 min, mismo BM25 +
  reranker): recall@6 0.679 / @10 0.785, peor que `multilingual-e5-large` (0.764 / 0.841). Descartado.
- Recall por área (top-10, reranker): laboral, familia, penal, mercados, civil, tributario 1.00;
  constitucional 0.80; comercial 0.75; procesal 0.70; **administrativo 0.40** (el área más débil).
- Caveat de siempre: 15 cerradas / 50 ítems; diferencias de ±1 pregunta = ±1.33 pts. La configuración
  activa se eligió por el total /80, pero las diferencias entre filas son pequeñas.

### 12.2 Máquina de la verificación en vivo

**Confirmado (2026-10-02): la prueba/verificación en vivo del sábado se hace en Turing.** El MacBook Pro
M4 (48 GB) queda solo como respaldo/chequeo cruzado auxiliar; no hace falta regenerar nada allí. El
riesgo de divergencia CUDA vs Metal deja de ser bloqueante (la entrega y la verificación usan la misma
máquina y stack).

### 12.3 Próximos pasos

1. Cerradas (0.73 vs 0.905 de la línea base): Qwen3-8B en Q8_0, razonamiento previo fuera de la
   gramática, mejorar recuperación en administrativo/procesal.
2. Citación (11.0/20): el prompt conciso la bajó; probar pedir `referencia_legal` con todas las normas
   usadas y/o mostrar más normas respaldadas.
3. Vigencia (norma vigente vs derogada); Estatuto Orgánico del Sistema Financiero; texto íntegro de
   providencias de la Corte Constitucional (hoy solo relatoría).
4. Chequeo auxiliar en el Mac (determinismo), ya no bloqueante.

## 13. Ronda de mejora nocturna (2026-10-02, noche): problemas conocidos

Solo métricas gratuitas (sin RAGAS: el crédito de OpenRouter se reserva para el sábado, salvo una
corrida de confirmación de ~0.07 USD). Todo sobre `sample_50`, Turing, Qwen3-8B + reranker.
Ramas del repo de la solución: `adrian/rag-citation-prompt`, `adrian/rag-closed-merge`.

### 13.1 Corpus reescrapeado y BM25 con stemming (ya en la config activa)

Re-scrape completo de los códigos (Civil, Comercio, CST, CGP, Estatuto Tributario, CPACA, Leyes 80, 100,
472, 599, 906, 1098, 1480; `rescrape_senado.py`, 105 949 fragmentos) y BM25 con stemming y stopwords en
español (`BM25_STEM=1`, guardado en el índice): recall@10 de normas de referencia 0.841 → 0.854.
Medido con el reporte del viernes: **46.33/80** (31.91 automático + RAGAS 0.481).

### 13.2 Diagnóstico: dónde se pierden los puntos

- **Recuperación está cerca de su techo**: recall@6 0.833, @10 0.854 con reranker. De 49 normas de
  referencia, 41 están en el top-6; 8 no. Solo ~2–3 son huecos reales de corpus (Decreto-Ley 663 / Estatuto
  Orgánico: 26 artículos; la sentencia SU-277/25, solo en relatoría; ítem "Doctrina"). El resto (CPACA, Ley
  472, Ley 1581, CGP) están en el corpus y no suben al top-k. `python -m src.validate.retrieval_misses`
  lista cada fallo.
- **Pérdida principal en citación**: de las 41 normas de referencia recuperadas, solo 29 aparecían en la
  respuesta; el modelo cita providencias pero omite la Constitución o el código del que sale el artículo.
- **Ampliar el corpus rinde poco**: techo estimado de 2–3 normas de 49 (unos pocos % de citación) y esta
  máquina no tiene salida a secretariasenado.gov.co (cualquier scrape nuevo lo haría un compañero); más
  providencias empeorarían el desplazamiento de leyes por las ~31k de relatoría. Única excepción razonable:
  Estatuto Orgánico del Sistema Financiero, si alguien con internet lo baja.
- **Abstención casi sin margen**: abstenerse vale 0.5 sin importar el resultado, acertar 1, fallar 0. Un
  barrido de umbral ganaría ~0.3 pts en `sample_50`; no se cambia.

### 13.3 Experimentos

| Experimento | Resultado | Decisión |
|---|---|---|
| Bono a normas sobre providencias tras el reranker (`NORM_BOOST` 1–3 logits) | recall@6 0.833 → 0.821 / 0.813; recall@3 baja 0.805 → 0.744 | **Descartado** (revertido) |
| `referencia_legal` ilimitada y exhaustiva (v1) | citación 11.84 → 12.80, auto 31.91 → 33.46, pero RAGAS 0.481 → 0.426 (total 46.23 vs 46.33) y proxy de `respuesta` 0.885 → 0.858 | **Descartado**: neutro en el total y degrada la respuesta |
| `referencia_legal` con redacción suave (v2): "si la evidencia trae otras normas (Constitución, código) agrégalas aquí" | citación 11.84 → 12.24, auto 31.91 → 32.55, proxy de `respuesta` 0.884 (≈ igual), cerradas 10/15 | **Adoptado** |
| Cerradas: razonamiento previo (`MC_REASONING=1`), voto por permutación (`MC_VOTES=3`), ambos | 10/15 en las cuatro variantes; fallan siempre 128, 528, 671 | Sin mejora; flags apagados por defecto |
| Penalización de vigencia (pasajes del CPC derogado / Decreto 1 de 1984, −1/−2 logits) | recall sin cambio; ítem 528 sigue fallando | **Descartado** (revertido) |

Nota sobre el ítem 528: la causa no es solo vigencia. Aun con la Ley 572 de 2000 (derogada) por debajo del
CGP art. 25, hay que convertir "mínima cuantía = hasta 40 SMLMV" a pesos con el salario mínimo vigente,
dato que ningún pasaje recuperado trae. Es un hueco de conocimiento, no de ranking. Los otros fallos
estables (128: Decreto-Ley 663 sin cobertura; 671: "Doctrina") son huecos de corpus.

### 13.4 Estado y siguientes pasos

- Config activa para la entrega: la de la sección 12 + prompt `semi_open` v2. Pendiente: correr
  `determinism_check.py` y `latency_check.py` sobre el estado final exacto antes de congelar.
- Con 15 cerradas, ±1 pregunta = ±1.33 pts; cualquier cambio de ±1 se trata como ruido.
- Si alguien con internet puede bajar el Estatuto Orgánico (Decreto-Ley 663 de 1993) y reindexar
  (~15 min en la 4090), es la única ampliación de corpus que vale el costo.
- Stretch no explorado: dato del salario mínimo / UVT por año como pasaje fijo en el corpus (ítems de cuantía).

## 14. Indexación: experimentos (2026-10-02, noche)

Sin RAGAS; solo `retrieval_eval.py` (recall de normas de referencia, `sample_50`, n=41). Todos
revertidos: **el índice de la entrega no cambió** (`chunks.jsonl`, `index.faiss`, `bm25.pkl` idénticos).

**Diagnóstico del índice actual.** 4 825 fragmentos de normas y 11 481 de providencias superan ~350
palabras; `multilingual-e5-large` trunca a 512 tokens, así que su vector denso solo "ve" el inicio (el
resto solo lo cubre BM25). Además hay 333 artículos vacíos/derogados ("CÓDIGO CIVIL. ARTÍCULO 1761. .").

| Experimento | recall@6/10/20 con reranker (base 0.833 / 0.854 / 0.878) | Decisión |
|---|---|---|
| Vectores extra (multi-vector): ventanas de 250 palabras posteriores de cada fragmento largo, con las primeras 20 palabras del fragmento como encabezado, en un índice FAISS aparte (51 464 vectores, 35 754 fragmentos; ~7 min en la 4090); cada fragmento puntúa con su mejor vector | 0.801 / 0.821 / 0.854 (peor). Sin reranker mejora @6/@10 (0.756→0.772, 0.764→0.797) pero el pool de 50 que ve el reranker baja (recall@50 0.927→0.902): las providencias largas desplazan a las normas del top-100 denso | **Descartado** |
| Igual, pero solo ventanas de normas (no providencias) | 0.833 / 0.854 / 0.878 (idéntico a la base); pool@50 0.902 (algo menor) | Sin beneficio; descartado |
| Descartar los 333 artículos vacíos tras la fusión | idéntico (solo 3 de 300 pasajes finales eran vacíos) | Sin beneficio medible; no se agrega al pipeline congelado |

Conclusión: la indexación actual (artículo completo / ventana de 250 palabras + BM25 con stemming +
e5-large) no se mejora con estos cambios en esta muestra; las ganancias que quedan están en
generación/citación (sección 13) y en huecos de corpus.

## 15. Recuperación: consulta, corpus y reranker ajustado (2026-10-02, noche)

Problema a atacar: la norma correcta a veces no llega al top-6. En `sample_50` el pool de 50 candidatos
contiene el 92.7% de las normas de referencia, pero solo el 83.3% llega al top-6 tras el reranker:
~4 normas se pierden en el reordenamiento y ~3–4 nunca entran al pool (sobre todo huecos de corpus).

### 15.1 Consulta (rama `adrian/rag-query-rewrite`)

| Experimento | recall@3/6/10/20 con reranker (base 0.805/0.833/0.854/0.878) | Puntaje sample (base 32.55) | Decisión |
|---|---|---|---|
| Anteponer el `tema` del ítem a la consulta (`QUERY_TEMA=1`) | 0.793 / 0.825 / **0.890** / **0.915** | 29.13 (cerradas 9/15, citación 10.39) | **Descartado**; flag apagado |
| `tema` solo en la recuperación, no en el reranker | 0.793 / 0.813 / 0.854 / 0.902 | — | Descartado |
| Reformulación con Qwen3-8B (norma probable + términos clave) | 0.671 / 0.805 / 0.821 / 0.878 | — | **Descartado** (código eliminado) |

Además no hay garantía de que los ítems de prueba traigan `tema`. Nota de interpretación: el enunciado
(3.2) solo prohíbe modelos *cerrados* en la reformulación de consultas; el docstring de
`query_expand.py` es más estricto ("ningún modelo generativo"). Como la reformulación no ayudó, la
diferencia no bloquea nada.

### 15.2 Corpus: Estatuto Orgánico del Sistema Financiero (rama `adrian/rag-rescrape-eosf`)

La red volvió a estar disponible. `rescrape_senado.py estatuto_organico_sistema_financiero` bajó las 13
partes del Decreto 663 de 1993: **342 artículos** (antes 26, solo la parte 4). Re-ingesta con
`RESCRAPE_DIR=data/corpus_rescrape`: 34 380 documentos, **106 237 fragmentos** (+288; el Estatuto pasa de
26 a 314). Todos los demás fragmentos son idénticos byte a byte (sha256 de `chunks.jsonl` sin el
Estatuto: igual antes y después). BM25 y FAISS reconstruidos completos. Respaldo del índice anterior,
con sus sha256, en `../backup_v2_pre_eosf/` (fuera del repo).

### 15.3 Reranker ajustado con datos sintéticos (rama `adrian/rag-reranker-ft`)

Ajuste fino de `BAAI/bge-reranker-v2-m3` sin tocar el banco de preguntas (`sample_50` queda como test):

1. `src.train.synth_queries`: Qwen3-8B (abierto, permitido para datos sintéticos) redacta una pregunta
   jurídica por fragmento a partir de 4 000 fragmentos (70% normas, 30% providencias; ≥40 palabras),
   en tres estilos (directa, nombrando la norma, caso práctico). Se filtran fugas tipo "según el texto".
2. `src.train.mine_negatives`: negativos difíciles con la misma recuperación híbrida (top-30), descartando
   falsos negativos probables (otros fragmentos de la misma providencia o que citan el mismo artículo).
3. `src.train.train_reranker`: BCE con `pos_weight`, lr 1e-5, 1 época, bf16, `max_length` 384.
4. `src.train.eval_checkpoints`: recall@3/6/10 en `sample_50` del modelo base y de cada checkpoint.

**Checkpoints / reanudación.** Las etapas 1 y 2 escriben una línea por pregunta
(`data/train/synth_queries.jsonl`, `data/train/pairs.jsonl`, versionados; los fragmentos se identifican
por sha1 de `doc_id`+texto, así que sobreviven a reconstrucciones del índice). La etapa 3 guarda
checkpoints cada 200 pasos en `models/reranker-ft/` y reanuda desde el último. La etapa 4 registra cada
modelo evaluado en `data/train/eval_log.jsonl`. Para continuar: `bash src/train/run_all.sh [N]`.

**Uso.** `RERANKER_MODEL=models/reranker-ft/final` (o un checkpoint). Si se adopta: recalibrar los
umbrales de abstención (`SUFFICIENCY_SCORE_THRESHOLD`/`RETRY_SCORE_THRESHOLD`, hoy 0.10/0.02 en la
escala del modelo base), copiar el modelo a la máquina de la entrega y repetir `determinism_check.py`
y `latency_check.py`.

### 15.4 Resultados del reranker ajustado

4 000 preguntas sintéticas (Qwen3-8B, ~45 min), 3 997 con negativos minados (el fragmento de origen está en
el pool de 30 en el 96% de los casos y en el top-6 híbrido en el 87%: las preguntas sintéticas son más
fáciles que las reales porque comparten vocabulario con su texto), 29 939 pares, 1 872 pasos (~10 min).
Recall de normas de referencia en `sample_50` con el índice de 106 237 fragmentos:

| Modelo | recall@3 | recall@6 | recall@10 |
|---|---:|---:|---:|
| `BAAI/bge-reranker-v2-m3` (base) | **0.805** | 0.833 | 0.878 |
| checkpoint-1000 | 0.772 | 0.850 | 0.874 |
| checkpoint-1200 | 0.772 | **0.878** | 0.886 |
| checkpoint-1400 | 0.736 | 0.841 | 0.882 |
| checkpoint-1600 | 0.707 | 0.841 | 0.882 |
| checkpoint-1800 / final (1872) | 0.707 | 0.817 | 0.886 |

**No se adopta.** recall@3 cae de forma monótona con el entrenamiento: el modelo aprende el solapamiento
léxico entre pregunta sintética y fragmento, que las preguntas reales no tienen. El +0.045 de recall@6 del
paso 1200 son ~2 normas de 49, cuesta precisión en @3, y elegir ese checkpoint mirando `sample_50` sería
seleccionar sobre el test. La entrega sigue con el reranker base. Si se retoma: preguntas sintéticas menos
léxicas (parafraseo, estilo del banco), lr menor o menos pasos, y validar con un subconjunto separado. Los
checkpoints y los datos quedan en `models/reranker-ft/` y `data/train/` (reanudable con `run_all.sh`).

## 16. Cómo preparar una máquina de la sala Turing (sin sudo)

Procedimiento verificado el 2026-10-02 en la máquina canónica (RTX 4090 24 GB, 62 GB RAM, ~500 GB libres en
`$HOME`, Linux con driver NVIDIA del sistema). **No hay sudo**: nada se instala con `apt`; todo vive en
`$HOME` (uv, Python, venv, modelos, cachés). El repo de la solución está en
`~/ai-week/solution-hackathon-ai-week-2026`.

### 16.1 Driver: verificar, no instalar

El driver (módulo del kernel + librerías de usuario) lo administra la sala. Lo único que hay que comprobar es
que módulo y librerías coincidan:

```bash
cat /proc/driver/nvidia/version                                    # módulo del kernel (580.178.04 hoy)
nvidia-smi --query-gpu=name,driver_version --format=csv            # librerías de usuario: misma versión
echo "$LD_LIBRARY_PATH"                                            # no debe tener rutas de drivers propios
```

Si `nvidia-smi` dice `Failed to initialize NVML: Driver/library version mismatch`:
1. Revisar `LD_LIBRARY_PATH`: **nunca** poner ahí una copia de usuario del driver (p. ej.
   `~/nvidia_580_173_02`, que se usó como parche cuando el módulo era 580.173). Tras la actualización a
   580.178.04 esa copia quedaba por delante de `/usr/lib/x86_64-linux-gnu` y rompía NVML y
   `torch.cuda.is_available()`.
2. Si módulo y librerías del sistema realmente difieren (actualización de paquetes sin reinicio), hay que
   reiniciar la máquina; sin sudo eso lo hace el personal de la sala.

### 16.2 uv y Python sin sudo

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # instala en ~/.local/bin (uv 0.12.21 hoy)
export PATH="$HOME/.local/bin:$PATH"                   # agregar también a ~/.bashrc
uv python install 3.12                                 # Python administrado por uv (3.12.14), sin sudo
```

### 16.3 Acceso a GitHub

HTTPS no tiene credenciales guardadas y `gh` no está instalado: usar SSH.

```bash
ssh-keygen -t ed25519 -C "<correo>"     # registrar ~/.ssh/id_ed25519.pub en GitHub (Settings → SSH keys)
git clone git@github.com:Ai-Week-LasNewJean/solution-hackathon-ai-week-2026.git
git clone git@github.com:Ai-Week-LasNewJean/main-repo.git ai-week-hackathon-2026
# en clones existentes por HTTPS: git push git@github.com:Ai-Week-LasNewJean/<repo>.git <rama>
```

### 16.4 Entorno virtual y CUDA (el paso delicado)

El stack validado es **`torch 2.14.1+cu130` + librerías CUDA 13 en el venv (`nvidia-*-cu13`) +
`llama-cpp-python 0.3.36` precompilado para CUDA 13** (wheel `cu130` de abetlen). `uv.lock` fija
`llama-cpp-python==0.3.35` sin CUDA, así que el orden importa: **primero todas las sincronizaciones, al
final el reemplazo de llama-cpp, y después nunca más `uv sync`.**

```bash
cd ~/ai-week/solution-hackathon-ai-week-2026
uv sync --extra evaluador          # torch cu130 + libs CUDA 13 desde PyPI; llama-cpp queda en CPU (por ahora)
uv pip install --python .venv/bin/python --no-deps --reinstall "llama-cpp-python==0.3.36" \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
uv pip install --python .venv/bin/python --no-deps "requests==2.34.2"            # versión del venv validado
uv pip install --python .venv/bin/python --no-deps "accelerate>=1.1.0" psutil   # solo si se va a entrenar
```

Por qué no compilar llama-cpp desde fuente: el `nvcc` del sistema (`/usr/bin/nvcc`) es CUDA 11.5, que no
compila para la 4090 (sm_89); `/usr/local/cuda-12.2` enlazaría el runtime de CUDA 12, distinto del de
CUDA 13 que traen torch y el venv. El wheel `cu130` enlaza exactamente las mismas `libcudart.so.13` /
`libcublas.so.13` del venv. (La receta `CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv sync` del
`pyproject.toml` no sirve en esta sala por lo mismo.)

`LD_LIBRARY_PATH` debe contener **solo** las librerías CUDA 13 del venv (en cada shell, o en `~/.bashrc`):

```bash
export LD_LIBRARY_PATH=$HOME/ai-week/solution-hackathon-ai-week-2026/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
```

**Regla de oro:** después de este paso, **no correr `uv sync`, `uv add` ni `uv lock`** en esa máquina.
Cualquiera de ellos re-sincroniza con `uv.lock` y reemplaza en silencio llama-cpp CUDA por la versión CPU
(pasó el 2026-10-02 al agregar `accelerate` con `uv add`; también bajó `requests` a 2.32.5). Para agregar
un paquete: `uv pip install --python .venv/bin/python --no-deps <paquete>`, y repetir la verificación 16.5.

### 16.5 Verificación (correr tras cualquier instalación)

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
.venv/bin/python -c "import torch, llama_cpp; print(torch.__version__, torch.cuda.is_available(), llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
# esperado: 2.14.1+cu130 True 0.3.36 True
ldd .venv/lib/python3.12/site-packages/llama_cpp/lib/libggml-cuda.so | grep -E "cudart|cublas"
# esperado: libcudart.so.13 y libcublas*.so.13 resueltas dentro de .venv/.../nvidia/cu13/lib
```

### 16.6 Modelos y datos

- **Decoder** (GGUF, en `models/`, no versionado):
  `.venv/bin/hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models`
  sha256 esperado del activo: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- **Encoder y reranker** (`intfloat/multilingual-e5-large`, `BAAI/bge-reranker-v2-m3`): se descargan solos a
  `~/.cache/huggingface` en la primera corrida (~8.5 GB con `bge-m3`). El aviso "unauthenticated requests
  to the HF Hub" es inofensivo; `HF_TOKEN` solo sube el límite de descargas.
- **Corpus e índice**: copiar `corpus/`, `indice/` y `corpus_manifest.json` desde el almacenamiento del
  equipo, o reconstruir: datos crudos en `data/corpus/` (+ `data/corpus_rescrape/`), luego
  `RESCRAPE_DIR=$PWD/data/corpus_rescrape .venv/bin/python -m src.ingest.ingest_raw_sources`,
  `python -m src.index.build_bm25` y `python -m src.index.build_faiss` (~15 min en GPU). Verificar
  contra los sha256 del índice de referencia.
- **Secretos**: `scripts/.env` con `OPENROUTER_API_KEY` (gitignored), solo para RAGAS.

### 16.7 Validación antes de usar la máquina para la entrega

1. Regresión: correr `sample_50` y comparar ítem por ítem contra la salida de referencia del equipo
   (todas las llaves salvo `latencia_ms`). El 2026-10-02 el wheel `cu130` reinstalado reprodujo **50/50**
   respuestas idénticas a las de antes del incidente con `uv`.
2. `python -m src.validate.determinism_check` (0 divergencias) y `python -m src.validate.latency_check`.

### 16.8 Recursos y red

- VRAM 24 GB: Qwen3-8B Q4 (~7 GB) convive con e5/reranker o con la construcción de FAISS; **no** entrenar
  el reranker mientras corre una generación grande con el LLM cargado.
- La salida a algunos sitios (p. ej. secretariasenado.gov.co) fue intermitente: un `curl` agotó el tiempo
  y horas después respondió. Reintentar antes de concluir que no hay red.

## 17. Mapa de ramas y procedencia de los resultados (2026-10-02, noche)

**Estructura actual (desde el 2026-10-02 por la noche):**

| Repo | Ramas | Contenido |
|---|---|---|
| Solución (`solution-hackathon-ai-week-2026`) | **`adrian/main-rag`** | Todo el trabajo fusionado: pipeline, experimentos (flags apagados), Estatuto Orgánico, ajuste fino del reranker, documentación. Rama de la entrega. |
| | **`adrian/friday-email`** | Exactamente el commit `1cb7a89`: números y respuestas del correo del viernes 17:00 (46.33/80). |
| | `main` | Rama por defecto de GitHub, solo el README inicial (`b719da9`, contenido en las dos anteriores). Se conserva por ahora (decisión del equipo). |
| Principal (`main-repo`, este SPEC) | **`main`** | SPEC completo (secciones 0–17). |

Las ramas de trabajo de la noche se fusionaron y se borraron (local y remoto) después de verificar que la
punta de cada una es ancestro de `adrian/main-rag`. Las tablas siguientes se conservan como historial: los
nombres ya no existen, pero cada commit citado sigue accesible desde `adrian/main-rag` (o, para el SPEC,
desde `main`). Las secciones 13–16 nombran esas ramas; esta es la equivalencia.

### 17.1 Historial: ramas del repo de la solución (todas contenidas en `adrian/main-rag`)

| Rama | Último commit | Parte de | Contenido | Resultado medido (sample_50, Turing) |
|---|---|---|---|---|
| `main` | `b719da9` | — | README inicial | — |
| `adrian/rag-scaffolding` | `4aeae01` | `main` | Pipeline de punta a punta, interfaz con identidad de Software Colombia | — |
| `adrian/rag-dev` | `90096cd` | `rag-scaffolding` | Decoder intercambiable, Qwen3-8B activo, reranker activo con umbrales sigmoide, Turing canónico (secc. 10–11) | 33.19/50; **`determinism_check` 0 divergencias y `latency_check` 7.64 s/ítem (secc. 11.5)** |
| `adrian/rag-ragas-tuning` | `e17e654` | `rag-dev` | `ragas_dev`/proxy gratuito, pool 100/100→50, prompt conciso `semi_open`, primer reporte del viernes (secc. 12) | 32.20/50, 46.28/80 |
| `adrian/rag-closed-questions` | `bbfd72a` | `rag-ragas-tuning` | Experimentos de cerradas (`MC_VOTES`, `MC_REASONING`, Q8), apagados por defecto | 10/15 en todas las variantes (secc. 13.3) |
| **`adrian/rag-corpus-rescrape`** | `45990a5` (reporte en **`1cb7a89`**) | `rag-ragas-tuning` | Re-scrape de códigos, BM25 con stemming, **reporte enviado por correo a las 17:00** (`informe/viernes/`, `entrega_sample.jsonl`) | **31.91/50, RAGAS 0.481, 46.33/80** (índice de 105 949 fragmentos) |
| `adrian/rag-citation-prompt` | `d47e6f6` | `rag-corpus-rescrape` | `retrieval_misses.py`, prompt `semi_open` v2 (`referencia_legal` con normas de apoyo) | 32.55/50 (secc. 13.3) |
| `adrian/rag-closed-merge` | `75c1499` | `rag-citation-prompt` | + merge de `rag-closed-questions` (flags apagados) | igual a la anterior |
| `adrian/rag-docs-update` | `0fd24be` | `rag-closed-merge` | README/CORPUS con limitaciones y experimentos | — |
| `adrian/rag-query-rewrite` | `c789813` | `rag-docs-update` | Flag `QUERY_TEMA` (apagado); la reformulación con LLM se probó y se eliminó (secc. 15.1) | `QUERY_TEMA=1`: 29.13/50, descartado |
| `adrian/rag-rescrape-eosf` | `63b018d` | `rag-docs-update` | Estatuto Orgánico completo (342 art.), manifest de 106 237 fragmentos (secc. 15.2) | recall@10 0.854 → 0.878; sin corrida de punta a punta todavía |
| `adrian/rag-reranker-ft` | `17d8f53` | `rag-rescrape-eosf` | Ajuste fino del reranker con checkpoints (secc. 15.3), README con guía de reconstrucción y advertencia de `uv sync` | no adoptado (secc. 15.4) |
| → **`adrian/main-rag`** | `7c9e737`+ | `rag-reranker-ft` + merge de `rag-query-rewrite` | Resultados del reranker, README con la estructura de dos ramas | config de la entrega: prompt v2, reranker base, flags apagados |

Los experimentos descartados que no dejaron código (bono a normas, penalización de vigencia, vectores
extra para fragmentos largos, filtro de artículos vacíos) se midieron sobre `adrian/rag-citation-prompt` /
`adrian/rag-docs-update` y se revirtieron sin commit; están documentados en las secciones 13.3 y 14.

### 17.2 Historial: ramas del repo principal (fusionadas en `main`)

| Rama (borrada) | Último commit | Contenido |
|---|---|---|
| `adrian/rag-dev` | `db49af7` | SPEC secciones 0–12 |
| `adrian/rag-spec-s13` | `391eb4d` | + sección 13 |
| `adrian/rag-spec-s14` | `2787d7b` | + sección 14 |
| `adrian/rag-spec-s15` | `0e236a5` | + sección 15 |
| `adrian/rag-spec-turing-setup` | `225e7bc` | + sección 16 |
| `adrian/rag-spec-branch-map` | — | + sección 17; `main` avanzó a esta rama (fast-forward) |

### 17.3 Procedencia del correo de las 17:00 y alcance del chequeo de determinismo

- **Números del correo (46.33/80):** commit `1cb7a89` (hoy la rama **`adrian/friday-email`**; antes `adrian/rag-corpus-rescrape`), con el índice de
  105 949 fragmentos (corpus reescrapeado + BM25 con stemming, antes del Estatuto Orgánico). Ese índice está
  respaldado con sus sha256 en `../backup_v2_pre_eosf/` (`chunks.jsonl` `7a5e59c0…`, `index.faiss`
  `16bb2e18…`, `bm25.pkl` `ff879489…`). Respuestas del correo: `informe/viernes/entrega/entrega_sample.jsonl`.
- **Chequeo de determinismo citado en el reporte ("0 divergencias"):** se corrió en `adrian/rag-dev`
  (secc. 11.5: Qwen3-8B + reranker top-6, pool 30/30→20, prompt original, corpus de 99 067 fragmentos), **no**
  sobre la configuración exacta del correo. La configuración del correo comparte decoder, binario,
  temperatura 0 y gramática, pero cambió pool, prompt, corpus y BM25. Re-verificación sobre `1cb7a89`: ver 17.4.
- **Binario de llama-cpp:** el 2026-10-02 a las 19:27 un `uv add` reemplazó el wheel CUDA por uno CPU (secc. 16.4)
  y se restauró el wheel `cu130` 0.3.36. Las 50 respuestas de `sample_50` del prompt v2 con el índice de
  105 949 fragmentos salieron idénticas a las de antes del incidente (`out/regress_restored.jsonl` vs
  `out/exp_reflegal2.jsonl`).

### 17.4 Re-verificación del correo sobre `1cb7a89` (2026-10-02, ~20:30)

Hecha en un `git worktree` de `1cb7a89` (hoy `adrian/friday-email`) con el índice de 105 949 fragmentos
respaldado (sha256 verificados con `sha256sum -c`), el mismo venv (wheel `cu130` 0.3.36 restaurado) y Qwen3-8B Q4_K_M:

- **Reproducción:** `sample_50` regenerado → **50/50 respuestas idénticas** a
  `informe/viernes/entrega/entrega_sample.jsonl` (todas las llaves salvo `latencia_ms`); `evaluate.py` da
  los mismos **31.91/50** (cerradas 13.33, citación 11.84, abstención 6.74). Con el RAGAS ya medido
  (0.481 → 14.42) el total del correo, 46.33/80, queda respaldado por una corrida reproducible.
- **Determinismo:** `determinism_check.py --ids 51 58 290 308 528` (mismos ítems que 11.5, procesos frescos):
  **0 divergencias** (`"ok": true`).
- Latencia observada en la reproducción: ~7 s/ítem, coherente con 11.5.

Con esto el "determinismo verificado (0 divergencias)" del reporte aplica también a la configuración exacta
del correo, no solo a la de 11.5.

## 18. Huecos de corpus frente a `seed_targets.json` y encabezados de artículo (2026-10-02, noche)

### 18.1 Diagnóstico

`data/seed_targets.json` lista las 186 normas que citan los ítems del banco de 992 (con su conteo).
Contrastado con `indice/chunks.jsonl` (norma de origen del encabezado de cada fragmento):

- **Decisión Andina 486: 23 ítems del banco (5.ª norma más citada) y 0 fragmentos propios**; solo
  aparecía mencionada desde otras leyes (Ley 875 de 2004, Código de Comercio).
- **267 leyes/decretos venían truncados en el volcado** (`completo: False`, solo las primeras partes del
  HTML); el re-scrape de la sección 13.1 solo había completado 15 códigos. Entre los citados por el
  banco: Ley 2220 de 2022 (11 ítems, 105 → 153 artículos), Decreto 2153 de 1992 (10 ítems, 46 → 59),
  Ley 1116 de 2006 (7 ítems, **42 → 126**), Ley 1952 de 2019 (159 → 281), Ley 1801 de 2016 (182 → 249),
  Decreto 960 de 1970 (45 → 233), Ley 600 de 2000, Ley 769 de 2002, Ley 640 de 2001, Ley 820 de 2003.
- Normas del banco ausentes del volcado pero disponibles en secretariasenado: Ley 2452 de 2025 (nuevo
  Código Procesal del Trabajo, 331 art.), Ley 2437 de 2024, Ley 1700 de 2013, Ley 1692 de 2013, Decretos
  4334 de 2008 y 4886 de 2011.
- Siguen sin cubrir (~12 ítems): leyes anteriores a 1992 que secretariasenado no publica (Ley 54 de 1990,
  153 de 1887, 50 de 1990, 75 de 1968, 29 de 1982, 155 de 1959) y decretos únicos (1082 de 2015, 780 de
  2016). SUIN-Juriscol es una SPA en JavaScript y los sitios `.gov.co` fallan la verificación TLS desde
  Turing; no se intentó.
- **Encabezados descartados en la ingesta:** ~95% de los artículos de códigos/leyes traen epígrafe
  ("TERMINACION DEL CONTRATO POR JUSTA CAUSA") y ubicación (Libro/Título/Capítulo), pero el fragmento
  era solo `"<norma>. ARTÍCULO N <texto>"`. No se había probado en la sección 14.

### 18.2 Cambios (repo de la solución, `adrian/main-rag`, commit `3c705ed` y siguientes)

- `rescrape_senado.py --completar` → `data/corpus_rescrape/leyes_decretos_completos.json` (269 docs,
  0 fallos; si el re-scrape trae menos artículos que el volcado se conserva el volcado).
- `parse_decision486.py` → Decisión 486 desde el PDF oficial de la CAN (280 artículos + 3 transitorias),
  encabezado "Decisión Andina 486 (…)" que `citations.extract` reconoce como `decision_andina_486`.
- `CHUNK_HEADINGS=1` (default 0): `"<norma>. ARTÍCULO N <EPÍGRAFE>. [Libro > Título > Capítulo] <texto>"`.
  Con 0 la ingesta reproduce el `chunks.jsonl` congelado byte a byte (verificado).
- Overrides `RAG_CORPUS_DIR`/`RAG_CORPUS_MANIFEST`/`RAG_INDEX_DIR` para índices experimentales.
- `synth_retrieval_eval.py`: acierto@k sobre las 4 000 preguntas sintéticas (600 muestreadas) como
  segundo banco de recuperación.
- **Checkpoints:** `src/experiments/run_variant.sh` (marcador por paso, reanudable, commit + push del
  resultado al terminar) y `data/experiments/README.md` (estado y cómo reanudar).

### 18.3 Resultados

| Variante | recall@3/6/10/20 (sample_50) | sintéticas @3/@6 (mismo conjunto) | e2e auto /50 (cerradas · citación · abst.) |
|---|---|---|---|
| Índice congelado (106 237) | 0.805/0.833/0.878/0.902 | 0.915/0.945 | **34.12** (14.67 · 12.24 · 7.21) |
| A: corpus completado (121 886) | 0.805/0.833/0.841/0.902 | 0.901/0.931 | 33.49 (14.67 · 11.84 · 6.98) |
| B: A + encabezados | 0.780/0.789/0.829/0.878 | 0.894/0.931 | 32.55 (13.33 · 12.24 · 6.98) |

- **Encabezados de artículo (B): descartados**; no mejoran ninguna métrica (flag apagado por defecto).
- **Corpus completado (A):** en `sample_50` queda dentro del ruido del índice congelado (−0.63 pts; ±1 cerrada =
  1.33). `sample_50` casi no cita las normas completadas, así que no puede medir la ganancia de cobertura; el
  costo medible es ~1.4 pts de acierto sintético sobre artículos ya presentes (más fragmentos compiten). La
  justificación de A es `seed_targets.json` (≥ 60 ítems del banco citan normas antes ausentes o truncadas, 23 solo
  la Decisión 486). **Decisión pendiente del equipo**; si se adopta: reconstruir `indice/` sin overrides `RAG_*`
  y repetir `determinism_check.py` y `latency_check.py`.
- Incidente: la primera corrida e2e de A corrió con evaluaciones en paralelo en la GPU, llama.cpp no pudo crear el
  contexto y todo se abstuvo (0/15). Se descartó; `run_variant.sh` ahora exige ≥16 GB de VRAM libre y descarta
  corridas con errores por ítem. Nunca correr evaluaciones en GPU en paralelo con una corrida e2e.
- **El índice de la entrega no cambió**; respaldo en `../backup_v3_pre_fullscrape/`. Detalle y cómo reanudar:
  `data/experiments/README.md` del repo de la solución.
