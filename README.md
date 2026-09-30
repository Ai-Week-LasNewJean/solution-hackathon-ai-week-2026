# Las NewJeans — Hackathon 2026

**Universidad de los Andes**

Sistema de respuesta a preguntas de derecho colombiano con un decoder abierto
≤8B parametros, un encoder abierto y un corpus juridico propio. Arquitectura
completa documentada en [`SPEC.md`](../ai-week-hackathon-2026/SPEC.md) del
repositorio de la hackathon (no se versiona aqui).

> **Estado:** scaffolding del pipeline. El corpus se esta construyendo en la
> carpeta compartida de OneDrive del equipo; esta rama trae el codigo de
> ingesta/indexacion y el sistema agentico listos para correr en cuanto el
> corpus tenga contenido. Ver `## Corpus e indice` abajo.

## Corpus e índice

<!-- OBLIGATORIO antes de la entrega del sabado. El jurado descarga desde
     aqui. Verificar el enlace desde una sesion privada del navegador antes
     de las 15:00. -->

| Recurso | Enlace | Tamaño | Licencia |
|---|---|---|---|
| Corpus procesado e índice vectorial | `<URL pendiente>` | | CC-BY-4.0 |

El comprimido contendrá `LICENSE`, `corpus_manifest.json`, `corpus/` con los
documentos procesados e `indice/` con `index.faiss`, `bm25.pkl` y
`chunks.jsonl`. Mientras se construye, el corpus vive en la carpeta
compartida de OneDrive (ver `RAW_CORPUS_DIR` en `src/config.py`).

## Arquitectura

| Componente | Elección | Motivo |
|---|---|---|
| Encoder | `intfloat/multilingual-e5-large` | Mismo encoder que `scripts/evaluate.py` usa para RAGAS; un solo stack. |
| Decoder | Llama-3.1-8B-Instruct GGUF (Q4_K_M) vía `llama-cpp-python` | Un binario, Metal en Mac / CUDA en Turing; soporte GBNF maduro para JSON forzado. |
| Segmentación | Regex `ARTÍCULO\s+\d+` (códigos/leyes); ventana deslizante ~250 palabras con solapamiento (jurisprudencia) | Ver `Ejemplo de entrega/CORPUS.md` de los organizadores. |
| Recuperación | Híbrida: BM25 (`bm25s`) top-30 ∪ denso (`faiss.IndexFlatIP`) top-30, fusión RRF (k=60) | No requiere calibrar escalas entre BM25 y coseno. |
| Reordenamiento | `BAAI/bge-reranker-v2-m3`, opcional tras `USE_RERANKER=1` | Se activa solo si mide mejora contra `evaluate.py --split sample`. |
| Mecanismo de abstención | Umbral sobre el score de recuperación fusionado (`src/retrieve/sufficiency.py`), nunca autoevaluación del LLM | Evita una fuente extra de no-determinismo. |

Detalle completo de decisiones y justificación en `SPEC.md` (repo de la
hackathon, no versionado aquí).

## Reproducción

Un único comando reconstruye el índice (si hace falta) y genera la entrega.

```bash
pip install -r requirements.txt
./run.sh sample      # o: python src/main.py --split sample --rebuild-index
```

Variables de entorno relevantes (`src/config.py`):

| Variable | Para qué | Default |
|---|---|---|
| `LLM_MODEL_PATH` | ruta al `.gguf` del decoder | `models/llama-3.1-8b-instruct-q4_k_m.gguf` |
| `RAG_BACKEND` | `mac`\|`turing`, sobreescribe la autodetección por plataforma | autodetectado |
| `USE_RERANKER` | `1` activa el reranker opcional | `0` |
| `RAW_CORPUS_DIR` | carpeta compartida del equipo con el corpus crudo (OneDrive) | ver `src/config.py` |

Requisitos de hardware: Apple Silicon (Metal) o GPU CUDA para el decoder;
~6 GB libres para el GGUF cuantizado; CPU alcanza para ingesta e indexación.

Tiempo estimado sobre las 50 preguntas de muestra: pendiente de medir con
`python -m src.validate.latency_check` una vez el corpus y el modelo estén
disponibles.

## Corpus y agente: cómo se construyen

1. **Corpus → índice vectorial** (`src/ingest/`, `src/index/`):
   `fetch.py` descarga las fuentes de `data/seed_targets.json` (o se lee lo
   que el equipo ya recolectó en `RAW_CORPUS_DIR`); `parse_html.py` /
   `parse_pdf.py` extraen texto; `segment.py` lo corta por artículo o por
   ventana deslizante; `normalize.py` limpia y antepone el nombre de la
   norma; `build_corpus.py` orquesta todo eso y escribe `corpus/<doc_id>.txt`
   + `corpus_manifest.json`. `index/embed.py` + `build_faiss.py` +
   `build_bm25.py` construyen el índice denso y disperso desde
   `indice/chunks.jsonl`.
2. **Sistema agéntico** (`src/retrieve/`, `src/generate/`, `src/pipeline/`):
   una pasada de recuperación híbrida + fusión RRF, decisión de suficiencia
   determinista, y como máximo una llamada de generación con salida forzada
   a JSON por gramática GBNF (`src/generate/grammars/`). `answer_one.py`
   ensambla la respuesta final; `run_batch.py` corre el banco completo con
   escritura incremental resiliente a fallos.

## Resultados sobre las preguntas de muestra

| Componente | Puntos | Posibles |
|---|---:|---:|
| Exactitud en cerradas | pendiente | 20 |
| Calidad de citación | pendiente | 20 |
| Abstención calibrada | pendiente | 10 |

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

## Limitaciones conocidas

1. El corpus aún no tiene contenido: el scaffolding de ingesta/indexación no
   se ha corrido contra las fuentes reales.
2. Los selectores de `src/ingest/parse_html.py` están escritos a partir de la
   estructura pública conocida de los sitios fuente, sin verificación contra
   una descarga real (sin acceso de red a `*.gov.co` desde el entorno donde
   se escribió este scaffolding).
3. El GGUF del decoder no está descargado todavía; `src/generate/llm.py`
   falla con un mensaje explícito hasta que `LLM_MODEL_PATH` exista.
