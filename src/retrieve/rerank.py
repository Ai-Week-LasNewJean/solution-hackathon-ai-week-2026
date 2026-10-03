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

    return CrossEncoder(config.RERANKER_MODEL, max_length=config.RERANKER_MAX_LEN)


def _cap_per_doc(ranked: list[Passage], top_k: int) -> list[Passage]:
    """Recorta a `top_k` con a lo sumo config.MAX_PER_DOC fragmentos por doc_id
    (0 = sin tope), conservando el orden. Si el tope deja menos de `top_k`,
    completa con los descartados en su orden original."""
    if config.MAX_PER_DOC <= 0:
        return ranked[:top_k]
    picked: list[Passage] = []
    spare: list[Passage] = []
    per_doc: dict[str, int] = {}
    for p in ranked:
        if per_doc.get(p.doc_id, 0) < config.MAX_PER_DOC:
            per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
            picked.append(p)
        else:
            spare.append(p)
        if len(picked) == top_k:
            return picked
    return picked + spare[:top_k - len(picked)]


def rerank(query: str, candidates: list[Passage], top_k: int = config.FINAL_TOP_K) -> list[Passage]:
    """Reordena `candidates` por relevancia cruzada query-pasaje y recorta a
    `top_k`. No-op (solo recorta) si config.USE_RERANKER es False."""
    if not config.USE_RERANKER or not candidates:
        return _cap_per_doc(candidates, top_k)

    pairs = [(query, c.texto) for c in candidates]
    scores = _model().predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return _cap_per_doc([Passage(doc_id=c.doc_id, texto=c.texto, inicio=c.inicio, fin=c.fin,
                                 score=float(s)) for c, s in ranked], top_k)
