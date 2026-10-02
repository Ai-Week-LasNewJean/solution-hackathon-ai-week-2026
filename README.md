# Las NewJeans — Hackathon 2026

**Universidad de los Andes**

Sistema de respuesta a preguntas de derecho colombiano con un decoder abierto
≤8B parametros, un encoder abierto y un corpus juridico propio. Arquitectura
completa documentada en [`SPEC.md`](../ai-week-hackathon-2026/SPEC.md) del
repositorio de la hackathon (no se versiona aqui).

> **Estado (2026-10-01):** pipeline completo corriendo de punta a punta contra
> el corpus real — 34 379 documentos / 99 067 fragmentos indexados, decoder
> descargado, corrida completa de las 50 preguntas de muestra puntuada por el
> evaluador oficial. Detalle de estado y próximos pasos en
> [`SPEC.md` sección 10](../ai-week-hackathon-2026/SPEC.md#10-estado-actual-2026-10-01-y-próximos-pasos)
> del repo de la hackathon. Todavía no corren `determinism_check.py` ni
> `latency_check.py` en el Mac, ni nada en Turing — eso es lo más urgente
> antes de congelar nada, no las mejoras de precisión de abajo.

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
| Decoder                 | Llama-3.1-8B-Instruct GGUF (Q4_K_M) vía `llama-cpp-python`                                                    | Un binario, Metal en Mac / CUDA en Turing; soporte GBNF maduro para JSON forzado. |
| Segmentación            | Regex `ARTÍCULO\s+\d+` (códigos/leyes); ventana deslizante ~250 palabras con solapamiento (jurisprudencia)    | Ver `Ejemplo de entrega/CORPUS.md` de los organizadores.                          |
| Recuperación            | Híbrida: BM25 (`bm25s`) top-30 ∪ denso (`faiss.IndexFlatIP`) top-30, fusión RRF (k=60)                        | No requiere calibrar escalas entre BM25 y coseno.                                 |
| Reordenamiento          | `BAAI/bge-reranker-v2-m3`, opcional tras `USE_RERANKER=1`                                                     | Se activa solo si mide mejora contra `evaluate.py --split sample`.                |
| Mecanismo de abstención | Umbral sobre el score de recuperación fusionado (`src/retrieve/sufficiency.py`), nunca autoevaluación del LLM | Evita una fuente extra de no-determinismo.                                        |

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
| `LLM_MODEL_PATH` | ruta al `.gguf` del decoder                                   | `models/llama-3.1-8b-instruct-q4_k_m.gguf` |
| `RAG_BACKEND`    | `mac`\|`turing`, sobreescribe la autodetección por plataforma | autodetectado                              |
| `USE_RERANKER`   | `1` activa el reranker opcional                               | `0`                                        |
| `RAW_CORPUS_DIR` | carpeta compartida del equipo con el corpus crudo (OneDrive)  | ver `src/config.py`                        |

Requisitos de hardware: Apple Silicon (Metal) o GPU CUDA para el decoder;
~6 GB libres para el GGUF cuantizado; CPU alcanza para ingesta e indexación.

Tiempo estimado sobre las 50 preguntas de muestra: ~22s/pregunta medido en un
ítem individual en el Mac (presupuesto del sábado: ~21.8s/pregunta para 992
preguntas en 6h) — la corrida completa de las 50 muestras tomó el lote
completo sin problemas, pero falta la medición formal con
`python -m src.validate.latency_check` (ver `## Limitaciones conocidas`).

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

Primera corrida real contra el corpus e índice definitivos (2026-10-01), sin ajustar todavía ningún
umbral ni prompt:

| Componente                                               |                                               Puntos | Posibles |
|----------------------------------------------------------|-----------------------------------------------------:|---------:|
| Exactitud en cerradas (10/15 correctas)                  |                                                13.33 |       20 |
| Calidad de citación (índice 0.367, 0 citas sin respaldo) |                                                 7.35 |       20 |
| Abstención calibrada                                     |                                                 5.93 |       10 |
| Corrección texto libre (RAGAS)                           | pendiente (`--ragas`, requiere `OPENROUTER_API_KEY`) |       30 |
| **Total automático**                                     |                                            **26.61** |   **50** |

Diagnóstico de los 5 ítems de opción múltiple no acertados (dos causas distintas de generación, una
de retrieval, una de cobertura del corpus, una de presupuesto de tokens) y la lista priorizada de
próximos pasos están en
[`SPEC.md` secciones 10.2–10.4](../ai-week-hackathon-2026/SPEC.md#10-estado-actual-2026-10-01-y-próximos-pasos).

Se actualiza corriendo `python scripts/evaluate.py --submission out/sample.jsonl --split sample`.
Evolución con fecha en `CORPUS.md`.

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

Lista completa y priorizada (bloqueantes de entrega primero, luego mejoras de precisión, cobertura
de corpus, puntaje sin medir, e interfaz/entregables) en
[`SPEC.md` sección 10.4](../ai-week-hackathon-2026/SPEC.md#10-estado-actual-2026-10-01-y-próximos-pasos).
Los tres más urgentes ahora mismo:

1. `determinism_check.py` y `latency_check.py` en el Mac — ninguno de los dos ha corrido todavía
   contra el estado real (corpus + índice + decoder), y son requisitos duros, no mejoras.
2. Reordenar el esquema JSON de generación (razonamiento antes de la respuesta final, no al revés) —
   causa directa de 2 de los 4 fallos de opción múltiple diagnosticados esta corrida.
3. Correr `evaluate.py --split sample --ragas` apenas haya `OPENROUTER_API_KEY`: 30 de los 50 puntos
   automáticos siguen sin medirse.

También en evaluación (no implementado): poder cargar distintos decoders GGUF ≤8B intercambiables
para compararlos empíricamente contra `evaluate.py --split sample`, sin perder la regla de "Mac
canónico" ni el requisito de determinismo en la entrega final — propuesta y tradeoffs en
[
`SPEC.md` sección 10.5](../ai-week-hackathon-2026/SPEC.md#105-decoder-intercambiable--propuesta-en-evaluación-no-implementada).

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
5. `determinism_check.py` y `latency_check.py` no han corrido contra el estado real del sistema (corpus + índice +
   decoder) — ver `## Próximos pasos`.
