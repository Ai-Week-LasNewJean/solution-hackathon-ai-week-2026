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

LLM_MODEL_PATH = Path(os.environ.get(
    "LLM_MODEL_PATH", str(ROOT / "models" / "llama-3.1-8b-instruct-q4_k_m.gguf")))
LLM_N_CTX = 8192
LLM_N_GPU_LAYERS = -1      # todas las capas en GPU/Metal; 0 fuerza CPU
LLM_TEMPERATURE = 0.0
LLM_SEED = 0               # fijo, exigido por determinism_check.py

MAX_TOKENS_BY_FORMAT = {
    # semi_open: <=150 palabras; open_ended.analisis: 5-8 oraciones (schema).
    # Margenes generosos en tokens (no palabras) para dejar espacio a JSON/gramatica.
    "multiple_choice": 320,
    "semi_open": 420,
    "open_ended": 900,
}

# --- Latencia ---------------------------------------------------------------
# Presupuesto del sabado: ~6h para 992 preguntas ~= 21.8 seg/pregunta.

N_PREGUNTAS_SABADO = 992
PRESUPUESTO_SABADO_SEGUNDOS = 6 * 60 * 60
PRESUPUESTO_POR_PREGUNTA_SEGUNDOS = PRESUPUESTO_SABADO_SEGUNDOS / N_PREGUNTAS_SABADO
