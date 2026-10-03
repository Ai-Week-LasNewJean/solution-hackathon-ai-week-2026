"""Recuperacion hibrida: BM25 top-k union denso top-k, fundidos por Reciprocal
Rank Fusion (SPEC.md seccion 3 y 5). RRF no requiere calibrar escalas entre
BM25 y similitud coseno -- solo usa el rango de cada lista.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from src import config
from src.index import build_bm25, build_faiss
from src.index.embed import embed_query


@dataclass
class Passage:
    doc_id: str
    texto: str
    inicio: int | None
    fin: int | None
    score: float          # score de fusion RRF (no comparable entre corridas distintas)


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    return [json.loads(line) for line in
            config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _faiss_index():
    return build_faiss.load(config.FAISS_INDEX_PATH)


@lru_cache(maxsize=1)
def _bm25_index():
    return build_bm25.load(config.BM25_INDEX_PATH)


def _rrf_fuse(ranked_lists: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """ranked_lists: cada una es una lista de indices de chunk en orden
    descendente de relevancia. Devuelve [(indice_chunk, score_rrf), ...]
    ordenado descendente."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def retrieve(query: str, k_bm25: int = config.K_BM25, k_dense: int = config.K_DENSE,
             top_k: int = config.FUSED_TOP_K) -> list[Passage]:
    """BM25 top-k_bm25 union denso top-k_dense, fusionados por RRF, recortados
    a `top_k`. Requiere que el indice (config.CHUNKS_PATH, FAISS_INDEX_PATH,
    BM25_INDEX_PATH) ya este construido."""
    chunks = _chunks()

    bm25_hits = _bm25_index().search(query, k_bm25)
    bm25_ranked = [idx for idx, _ in bm25_hits]

    qvec = embed_query(query)
    dense_hits = build_faiss.search(_faiss_index(), qvec, k_dense)
    dense_ranked = [idx for idx, _ in dense_hits]

    fused = _rrf_fuse([bm25_ranked, dense_ranked])[:top_k]
    passages = []
    for idx, score in fused:
        c = chunks[idx]
        passages.append(Passage(doc_id=c["doc_id"], texto=c["texto"],
                                 inicio=c.get("inicio"), fin=c.get("fin"), score=score))
    return passages
