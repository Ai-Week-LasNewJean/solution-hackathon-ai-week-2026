"""Indice disperso BM25 (SPEC.md seccion 3). `bm25s` (NumPy puro, sin JVM) con
`rank_bm25` como fallback si `bm25s` no esta disponible.

La tokenizacion reutiliza citations.norm() (minusculas, sin acentos, espacios
colapsados) para que BM25 y el matching de citas normalicen exactamente
igual -- evita que una cita se pierda en la recuperacion por una diferencia
de acentuacion que citations.py si tolera.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.index.bm25_types import BM25Index, tokenize

__all__ = ["BM25Index", "tokenize", "build", "save", "load"]


def build(chunks_path: Path) -> BM25Index:
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_tokens = [tokenize(c["texto"]) for c in chunks]
    doc_ids = [str(i) for i in range(len(chunks))]

    try:
        import bm25s

        model = bm25s.BM25()
        model.index(bm25s.tokenize([" ".join(t) for t in corpus_tokens], show_progress=False))
        return BM25Index(engine="bm25s", model=model, doc_ids=doc_ids)
    except ImportError:
        from rank_bm25 import BM25Okapi

        return BM25Index(engine="rank_bm25", model=BM25Okapi(corpus_tokens), doc_ids=doc_ids)


def save(index: BM25Index, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(index, fh)


def load(index_path: Path) -> BM25Index:
    with index_path.open("rb") as fh:
        return pickle.load(fh)


def main() -> int:
    import argparse

    from src import config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    ap.add_argument("--out", type=Path, default=config.BM25_INDEX_PATH)
    args = ap.parse_args()
    save(build(args.chunks), args.out)
    print(f"indice bm25 escrito en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
