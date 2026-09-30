"""Embeddings con intfloat/multilingual-e5-large (SPEC.md seccion 3).

Mismo encoder que scripts/evaluate.py usa para RAGAS (JUEZ_ENCODER): un solo
stack, validado por los organizadores. e5 exige el prefijo query:/passage: y
vectores normalizados L2 para que el producto punto (IndexFlatIP) equivalga a
similitud coseno.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Vectores normalizados L2 para fragmentos del corpus (prefijo passage:)."""
    prefixed = [config.E5_PASSAGE_PREFIX + t for t in texts]
    return _model().encode(prefixed, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    """Vector normalizado L2 para una consulta (prefijo query:)."""
    return _model().encode([config.E5_QUERY_PREFIX + text], normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)[0]
