# Las NewJeans — Hackathon 2026

**Universidad de los Andes**

Sistema de respuesta a preguntas de derecho colombiano con un decoder abierto
≤8B parametros, un encoder abierto y un corpus juridico propio. Arquitectura
completa documentada en [`SPEC.md`](../ai-week-hackathon-2026/SPEC.md) del
repositorio de la hackathon (no se versiona aqui).

> **Estado (2026-10-02, tarde):** **Turing (GPU CUDA, RTX 4090) es la máquina
> canónica**; el Mac es auxiliar (`SPEC.md` secciones 11–12). Configuración activa: **Qwen3-8B**
> (Q4_K_M) + reranker `bge-reranker-v2-m3` + pool BM25/denso top-100 → RRF 50 → top-6 + prompt
> conciso para `semi_open`. Medido en Turing sobre las 50 preguntas de muestra: **32.20/50**
> automáticos sin RAGAS + **14.08/30** RAGAS (correctness 0.469, línea base 0.451) =
> **46.28/80**; la corrida se reproduce idéntica (50/50 ítems) desde el código versionado.
> `determinism_check` 0 divergencias; 7–8 s/ítem (≈2.1 h para 992). Corpus: 34 379 documentos /
> 99 067 fragmentos, 100% trazables.

## Corpus e índice

<!-- OBLIGATORIO antes de la entrega del sabado. El jurado descarga desde
     aqui. Verificar el enlace desde una sesion privada del navegador antes
     de las 15:00. -->

| Recurso                             | Enlace            | Tamaño                                                     | Licencia  |
|-------------------------------------|-------------------|------------------------------------------------------------|-----------|
| Corpus procesado e índice vectorial | `<URL pendiente>` | ~685MB (`corpus/` 228MB + `indice/` 406MB + manifest 22MB) | CC-BY-4.0 |

El comprimido contendrá `LICENSE`, `corpus_manifest.json`, `corpus/` con los
documentos procesados e `indice/` con `index.faiss`, `bm25.pkl` y
`chunks.jsonl`. **34 379 documentos / 99 067 fragmentos**, 100% trazables a su
norma de origen (`traceability_check.py`).

Construido con `python -m src.ingest.ingest_raw_sources` a partir de
`data/corpus/` (volcado del equipo: scraping de secretariasenado.gov.co +
metadatos de relatoría de la Corte Constitucional + providencias de la Corte
Suprema ya trozadas) — **no** con `build_corpus.py`/`fetch.py` como preveía el
plan original, porque las fuentes que terminó usando el equipo llegaron ya
parseadas en JSON, no como HTML/PDF crudo. `build_corpus.py` sigue en el repo
para cualquier fuente que sí llegue cruda. Detalle de las decisiones de
ingesta (qué se descartó y por qué) en `CORPUS.md` secciones 1 y 3.

## Arquitectura

| Componente              | Elección                                                                                                      | Motivo                                                                            |
|-------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Encoder                 | `intfloat/multilingual-e5-large`                                                                              | Mismo encoder que `scripts/evaluate.py` usa para RAGAS; un solo stack.            |
| Decoder                 | **Qwen3-8B** GGUF (Q4_K_M) vía `llama-cpp-python` (candidatos alternos en `config.LLM_CANDIDATES`)             | Mejor de 6 decoders ≤8B medidos en Turing (31.65 vs 22.10 de Llama sin reranker). CUDA en Turing (canónico) / Metal en Mac (auxiliar); GBNF para JSON forzado. |
| Segmentación            | Regex `ARTÍCULO\s+\d+` (códigos/leyes); ventana deslizante ~250 palabras con solapamiento (jurisprudencia)    | Ver `Ejemplo de entrega/CORPUS.md` de los organizadores.                          |
| Recuperación            | Híbrida: BM25 (`bm25s`) top-30 ∪ denso (`faiss.IndexFlatIP`) top-30, fusión RRF (k=60)                        | No requiere calibrar escalas entre BM25 y coseno.                                 |
| Reordenamiento          | `BAAI/bge-reranker-v2-m3`, **activo por defecto** (`USE_RERANKER=0` lo apaga)                                 | Midió mejora con Qwen3 (33.19 vs 31.65); umbrales de abstención recalibrados a su escala sigmoide. |
| Mecanismo de abstención | Umbral sobre el score de recuperación (reranker: 0.10 / 0.02; RRF: 0.014 / 0.006) (`src/retrieve/sufficiency.py`), nunca autoevaluación del LLM | Evita una fuente extra de no-determinismo.                                        |

Detalle completo de decisiones y justificación en `SPEC.md` (repo de la
hackathon, no versionado aquí).

## Reproducción

Un único comando reconstruye el índice (si hace falta) y genera la entrega.

```bash
pip install -r requirements.txt
./run.sh sample      # o: python src/main.py --split sample --rebuild-index
```

Variables de entorno relevantes (`src/config.py`):

| Variable         | Para qué                                                      | Default                                    |
|------------------|---------------------------------------------------------------|--------------------------------------------|
| `LLM_MODEL_NAME` | candidato de `config.LLM_CANDIDATES`                          | `qwen3-8b`                                 |
| `LLM_MODEL_PATH` | ruta al `.gguf` del decoder (override absoluto)               | —                                          |
| `RAG_BACKEND`    | `turing` (canónico) \| `mac` (auxiliar), override de la autodetección | autodetectado                      |
| `USE_RERANKER`   | `1` activa el reranker, `0` lo apaga                          | `1`                                        |
| `FINAL_TOP_K` / `FUSED_TOP_K` | pasajes al prompt / candidatos antes del rerank  | `6` / `20`                                 |
| `RAW_CORPUS_DIR` | carpeta compartida del equipo con el corpus crudo (OneDrive)  | ver `src/config.py`                        |

Requisitos de hardware: GPU CUDA (Turing, canónica) para el decoder — Apple Silicon (Metal) solo como auxiliar;
~6 GB libres para el GGUF cuantizado; CPU alcanza para ingesta e indexación.

Tiempo sobre las 50 preguntas de muestra en Turing (RTX 4090, Qwen3-8B + reranker): 7.64 s/ítem
(`python -m src.validate.latency_check`), ~2.1 h extrapolado a las 992 preguntas contra un
presupuesto de 6 h.

## Corpus y agente: cómo se construyen

1. **Corpus → índice vectorial** (`src/ingest/`, `src/index/`): el flujo *diseñado* para fuentes HTML/PDF crudas es
   `fetch.py` → `parse_html.py` /
   `parse_pdf.py` → `segment.py` → `normalize.py` → `build_corpus.py`
   (orquesta todo y escribe `corpus/<doc_id>.txt` + `corpus_manifest.json`).
   El flujo que **realmente** construyó el corpus actual es distinto:
   `src/ingest/ingest_raw_sources.py` normaliza `data/corpus/` (JSON
   ya pre-parseado o pre-troceado por fuente — ver `## Corpus e índice`)
   directamente al mismo `corpus/<doc_id>.txt` + manifest + `chunks.jsonl`,
   descartando en el camino los fragmentos no trazables a su norma de origen.
   `index/embed.py` + `build_faiss.py` + `build_bm25.py` construyen el índice
   denso y disperso desde `indice/chunks.jsonl` en cualquiera de los dos
   casos — ese contrato (`chunks.jsonl` con al menos `doc_id`/`texto`) es lo
   que realmente los desacopla de cómo se llegó ahí.
2. **Sistema agéntico** (`src/retrieve/`, `src/generate/`, `src/pipeline/`):
   una pasada de recuperación híbrida + fusión RRF, decisión de suficiencia
   determinista, y como máximo una llamada de generación con salida forzada
   a JSON por gramática GBNF (`src/generate/grammars/`). `answer_one.py`
   ensambla la respuesta final; `run_batch.py` corre el banco completo con
   escritura incremental resiliente a fallos.

## Resultados sobre las preguntas de muestra

Medido en Turing (GPU) con `python scripts/evaluate.py --submission out/<run>.jsonl --split sample`,
configuración activa (Qwen3-8B + reranker, top-6):

| Componente                                               |                                               Puntos | Posibles |
|----------------------------------------------------------|-----------------------------------------------------:|---------:|
| Exactitud en cerradas (11/15 correctas)                  |                                                14.67 |       20 |
| Calidad de citación (índice 0.551)                       |                                                11.02 |       20 |
| Abstención calibrada                                     |                                                 6.51 |       10 |
| Corrección texto libre (RAGAS, correctness 0.469)        |                                                14.08 |       30 |
| **Total automático (sin RAGAS / con RAGAS)**             |                                  **32.20 / 46.28** | **50 / 80** |

Evolución (sin RAGAS): 26.61 (Mac, Llama, 10-01) → 22.10 (Turing, Llama) → 31.65 (Qwen3) → 33.19 (Qwen3 +
reranker) → 32.20 con pool ampliado y prompt conciso, que sube RAGAS 0.418 → 0.469 (total 45.73 → 46.28/80; `SPEC.md` 12). Comparación de 6 decoders ≤8B (Llama-3.1, Mistral-7B-v0.3, Qwen2.5-7B, Aya-Expanse-8B,
Qwen3-8B): ver `SPEC.md` 11.2. Con solo 15 cerradas, ±1 pregunta = ±1.33 pts — tomar las diferencias
pequeñas con cautela. Historial con fecha en `CORPUS.md`.

Validaciones en Turing: `determinism_check` 0 divergencias (5 ítems, procesos frescos);
`latency_check` 7.64 s/ítem (2.10 h para 992, margen 3.90 h).

## Interfaz gráfica

```bash
streamlit run interfaz/app.py
```

## Validaciones

```bash
python -m src.validate.run_all          # <2 min: trazabilidad, manifest, fuga
python -m src.validate.run_all --full   # + determinismo y latencia (requiere LLM cargado)
```

## Próximos pasos

Lista completa en [`SPEC.md` sección 11.6](../ai-week-hackathon-2026/SPEC.md#11-turing-como-máquina-canónica-y-resultados-2026-10-02):

1. RAGAS tiene presupuesto limitado (20 USD): usar `src/validate/ragas_dev.py` solo en finalistas.
2. Chequeo auxiliar de determinismo en el Mac (ya no bloqueante).
3. Señal de vigencia (norma vigente vs. derogada) y Estatuto Orgánico del Sistema Financiero.
4. Mejorar recuperación (recall@20 ≈ 0.79) y reintentar el razonamiento previo en `multiple_choice`.
5. Informe técnico, video ≤5 min, interfaz probada de punta a punta.

**Decoder intercambiable:** `config.LLM_CANDIDATES` registra `llama-3.1-8b-instruct`, `llama-3.1-8b-chat`,
`qwen3-8b` (activo), `qwen2.5-7b`, `mistral-7b-v0.3` y `aya-expanse-8b` (CC-BY-NC, no apto para entrega
sin confirmar licencia). `src/main.py --model-name <x>` o `LLM_MODEL_NAME=<x>` alternan para comparar;
la corrida que produce `submissions.jsonl` usa siempre un único candidato fijo.

## Limitaciones conocidas

1. El Estatuto Orgánico del Sistema Financiero llegó con solo 26 artículos en el scraping (el real
   tiene muchos más); si el tiempo alcanza, vale la pena re-scrapear esa fuente específica.
2. Dos fuentes quedan casi completamente fuera del índice (reglamento CNE, directivas
   presidenciales) porque `citations.py` —oficial, sin modificar— no reconoce esos tipos de norma
   como citables; volumen bajo (15 fragmentos), probablemente no vale la pena salvo que
   `seed_targets.json` los priorice.
3. Sin señal de "norma vigente vs. derogada": cuando dos fragmentos sobre la misma materia citan
   normas distintas (una vigente, una superada), la recuperación no distingue cuál preferir —
   causó uno de los fallos diagnosticados en `SPEC.md` 10.2.
4. Los selectores de `src/ingest/parse_html.py`/`build_corpus.py` (el flujo original para fuentes
   HTML/PDF crudas) siguen sin verificar contra una descarga real — no se necesitaron esta vez
   porque las fuentes usadas llegaron ya parseadas en JSON (ver `## Corpus e índice`), pero
   quedarían como riesgo abierto si el equipo necesita ingerir algo nuevo por esa vía.
5. `determinism_check.py` y `latency_check.py` pasan en Turing (ver `SPEC.md` 11.5); falta el chequeo
   auxiliar de determinismo cruzado en el Mac y la medición de RAGAS (30 pts).
6. Las diferencias de puntaje entre configuraciones se miden sobre 50 ítems (15 cerradas): ±1 pregunta
   cerrada = ±1.33 pts, así que hay ruido; validar cambios finos también con
   `python -m src.validate.retrieval_eval`.
