"""Configuracion central: rutas, modelos, hiperparametros de recuperacion y
generacion, y el selector de backend mac|turing (SPEC.md secciones 2 y 3).

Todo lo que otro modulo necesite parametrizar vive aqui, para que
determinism_check.py y latency_check.py puedan registrar la configuracion
exacta usada en cada corrida.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

# --- Rutas -------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
SCHEMA = ROOT / "schema"

CORPUS_DIR = ROOT / "corpus"                     # corpus/<doc_id>.txt, congelado antes de entregar
CORPUS_MANIFEST = ROOT / "corpus_manifest.json"
INDEX_DIR = ROOT / "indice"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"          # texto + doc_id + offsets + metadatos
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# Fuente cruda del corpus mientras el equipo lo construye. Configurable via
# env var porque cada miembro del equipo sincroniza OneDrive en una ruta
# distinta; el pipeline de indexacion solo necesita leer de aqui, nunca
# escribir (la carpeta es compartida).
RAW_CORPUS_DIR = Path(os.environ.get(
    "RAW_CORPUS_DIR",
    Path.home() / "Library/CloudStorage/OneDrive-Universidaddelosandes/2026-20/ai-week/corpus",
))

# --- Backend mac|turing (SPEC.md seccion 2) -----------------------------
# El Mac es la fuente de verdad canonica para toda generacion que termine en
# submissions.jsonl desde el jueves. Turing solo se usa para iterar rapido.


def detect_backend() -> str:
    """Backend de inferencia segun el hardware detectado, salvo override explicito."""
    override = os.environ.get("RAG_BACKEND")
    if override:
        return override
    return "mac" if platform.system() == "Darwin" else "turing"


BACKEND = detect_backend()
IS_CANONICAL = BACKEND == "mac"  # solo el Mac puede producir submissions.jsonl final

# --- Encoder de embeddings ----------------------------------------------
# Mismo encoder que scripts/evaluate.py (JUEZ_ENCODER) para el componente de
# similitud de RAGAS: reutilizarlo evita un segundo stack.

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# --- Segmentacion ---------------------------------------------------------

SEGMENT_WINDOW_WORDS = 250
SEGMENT_OVERLAP_SENTENCES = 2

# --- Recuperacion ----------------------------------------------------------

K_BM25 = 30
K_DENSE = 30
RRF_K = 60
FUSED_TOP_K = 20           # candidatos tras fusion, antes de (opcional) rerank
FINAL_TOP_K = 6            # pasajes que llegan al prompt de generacion
MAX_PASAJES_EVIDENCIA = 10  # debe igualar evaluate.MAX_PASAJES_EVIDENCIA

USE_RERANKER = os.environ.get("USE_RERANKER", "0") == "1"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Umbral de suficiencia: se ajusta empiricamente barriendo contra
# scripts/evaluate.py --split sample (ver src/retrieve/sufficiency.py). El
# valor inicial es un punto de partida conservador, no una medicion.
#
# Escala del score: es RRF, no similitud coseno. Con RRF_K=60 y como mucho 2
# listas (bm25 + denso), el score maximo posible es 2/(RRF_K+1) = 0.0328
# (el pasaje queda de primero en ambas listas); aparecer de primero en una
# sola lista da 1/(RRF_K+1) = 0.0164. Los umbrales de abajo estan en esa
# escala -- si RRF_K cambia, hay que recalibrarlos.
SUFFICIENCY_SCORE_THRESHOLD = 0.014   # ~top-1 en al menos una de las dos listas
RETRY_SCORE_THRESHOLD = 0.006          # ~top-5 en al menos una de las dos listas

# --- Decoder / generacion --------------------------------------------------
# Decoder intercambiable (SPEC.md 10.5): registro de candidatos <=8B probados
# (nombre corto -> archivo GGUF en models/), para comparar A/B contra
# evaluate.py --split sample con el mismo criterio empirico que el reranker
# opcional. "Intercambiable" es comodidad de desarrollo/demo, NUNCA una
# caracteristica de la corrida final: el candidato que termine en
# submissions.jsonl se fija en LLM_MODEL_NAME (o LLM_MODEL_PATH) antes de
# congelar, y debe ser el mismo en todas las 992 preguntas, validado por
# determinism_check.py en el Mac exacto de la entrega.
LLM_CANDIDATES: dict[str, dict] = {
    # chat_template=None: prompt crudo tal cual (comportamiento original,
    # validado -- nunca tocar para este candidato). "chatml": generate()
    # envuelve el prompt con marcadores <|im_start|>/<|im_end|> antes de
    # pasarlo al completion crudo de llama.cpp -- Qwen3-Instruct se entreno
    # casi exclusivamente en ChatML y sin turno explicito sigue
    # instrucciones peor. La gramatica GBNF por formato fuerza "{" como
    # primer caracter en cualquier caso, asi que el modo "thinking" de Qwen3
    # (bloque <think>...</think> antes de responder) no puede filtrarse al
    # output -- el grammar lo prohibe desde el primer token.
    "llama-3.1-8b-instruct": {"filename": "llama-3.1-8b-instruct-q4_k_m.gguf", "chat_template": None},
    "qwen3-8b": {"filename": "Qwen3-8B-Q4_K_M.gguf", "chat_template": "chatml"},
}

# Candidato activo: llama-3.1-8b-instruct (NO qwen3-8b). Se promovio Qwen3 a
# activo el 2026-10-02 tras un A/B contra evaluate.py --split sample que
# mide claramente mejor (cerradas empatadas 10/15, pero citacion 0.5 vs
# 0.367, abstencion 6.86 vs 5.81, total 30.19 vs 26.49, ver CORPUS.md
# seccion 4) -- pero una segunda medicion de latency_check.py sobre el
# estado final mostro que el margen real es mas delgado y mas ruidoso de lo
# que la primera corrida sugeria: 0.25h (no 0.50h) contra el presupuesto de
# 6h para las 992 preguntas del sabado, con un item truncado por limite de
# tokens en esa misma corrida. Con ~2.5 dias de desarrollo restantes y sin
# margen para resolver un desborde de presupuesto en vivo el sabado, se
# revirtio a Llama-3.1-8B-Instruct (margen medido 1.09h) como el candidato
# mas seguro por ahora. Qwen3 sigue disponible via LLM_MODEL_NAME=qwen3-8b /
# --model-name qwen3-8b para seguir afinandolo (p.ej. ajustar sus propios
# MAX_TOKENS_BY_FORMAT) antes de reconsiderar promoverlo otra vez -- NO
# volver a promoverlo sin remedir latency_check.py en el Mac exacto que se
# usara el sabado, idealmente sobre una corrida mas sostenida que 50 items.
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama-3.1-8b-instruct")


def llm_model_path(name: str | None = None) -> Path:
    """Resuelve un nombre corto de LLM_CANDIDATES a su ruta GGUF en models/.
    LLM_MODEL_PATH (env) tiene prioridad absoluta sobre el registro, para no
    romper el uso anterior de apuntar a un GGUF arbitrario fuera del
    registro."""
    override = os.environ.get("LLM_MODEL_PATH")
    if override:
        return Path(override)
    name = name or LLM_MODEL_NAME
    if name not in LLM_CANDIDATES:
        raise KeyError(f"modelo desconocido: {name!r}. candidatos: {sorted(LLM_CANDIDATES)}")
    return ROOT / "models" / LLM_CANDIDATES[name]["filename"]


def llm_chat_template(name: str | None = None) -> str | None:
    """Plantilla de chat a aplicar antes de generar (ver LLM_CANDIDATES).
    None (incl. cuando LLM_MODEL_PATH hace override fuera del registro) ->
    prompt crudo sin envolver, el comportamiento original."""
    if os.environ.get("LLM_MODEL_PATH"):
        return None
    name = name or LLM_MODEL_NAME
    return LLM_CANDIDATES.get(name, {}).get("chat_template")


LLM_MODEL_PATH = llm_model_path()  # ruta resuelta del candidato activo (compat con el uso previo)
LLM_N_CTX = 8192
LLM_N_GPU_LAYERS = -1      # todas las capas en GPU/Metal; 0 fuerza CPU
LLM_TEMPERATURE = 0.0
LLM_SEED = 0               # fijo, exigido por determinism_check.py

MAX_TOKENS_BY_FORMAT = {
    # semi_open: <=150 palabras; open_ended.analisis: 5-8 oraciones (schema).
    # Margenes generosos en tokens (no palabras) para dejar espacio a JSON/gramatica.
    # multiple_choice subido de 320 a 700 (SPEC.md 10.2 #4, 10.4 #6): el item
    # 128 enumera varios tipos de entidad en la justificacion y se trunco a
    # mitad de generacion con 320 *y* con 480 (probado empiricamente); 650
    # fue el primer valor que lo completo, 700 deja margen. NOTA: el reorden
    # de llaves de 10.4 #5 (respuesta_correcta al final) se probo por
    # separado y se revirtio -- bajo este mismo item_id resulto en una
    # regresion neta (6/15 vs 10/15 en cerradas sobre sample_50, ver
    # CORPUS.md seccion 4): reordenar las llaves del few-shot/gramatica
    # cambia la secuencia de tokens del prompt lo suficiente para que la
    # decodificacion greedy diverja en contenido, no solo en la posicion de
    # la letra final. El budget de tokens es ortogonal a eso y se mantiene.
    "multiple_choice": 700,
    "semi_open": 420,
    "open_ended": 900,
}

# --- Latencia ---------------------------------------------------------------
# Presupuesto del sabado: ~6h para 992 preguntas ~= 21.8 seg/pregunta.

N_PREGUNTAS_SABADO = 992
PRESUPUESTO_SABADO_SEGUNDOS = 6 * 60 * 60
PRESUPUESTO_POR_PREGUNTA_SEGUNDOS = PRESUPUESTO_SABADO_SEGUNDOS / N_PREGUNTAS_SABADO
