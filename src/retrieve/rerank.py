"""Reranker opcional (SPEC.md seccion 3): construir el pipeline SIN esto
primero; activarlo solo si `evaluate.py --split sample` muestra una mejora
medida con config.USE_RERANKER=1 (o env var USE_RERANKER=1).
"""
from __future__ import annotations

from functools import lru_cache

from src import config
from src.retrieve.hybrid import Passage


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, candidates: list[Passage], top_k: int = config.FINAL_TOP_K) -> list[Passage]:
    """Reordena `candidates` por relevancia cruzada query-pasaje y recorta a
    `top_k`. No-op (solo recorta) si config.USE_RERANKER es False."""
    if not config.USE_RERANKER or not candidates:
        return candidates[:top_k]

    pairs = [(query, c.texto) for c in candidates]
    scores = _model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [Passage(doc_id=c.doc_id, texto=c.texto, inicio=c.inicio, fin=c.fin, score=float(s))
            for c, s in ranked[:top_k]]
