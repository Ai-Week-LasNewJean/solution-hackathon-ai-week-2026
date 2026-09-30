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
import re
from dataclasses import dataclass
from pathlib import Path

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(citations.norm(text))


@dataclass
class BM25Index:
    engine: str            # "bm25s" | "rank_bm25"
    model: object
    doc_ids: list[str]     # posicion i -> id del fragmento en chunks.jsonl (str(indice))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if self.engine == "bm25s":
            import bm25s

            results, scores = self.model.retrieve(bm25s.tokenize([" ".join(tokens)], show_progress=False),
                                                   k=min(k, len(self.doc_ids)), show_progress=False)
            return [(int(idx), float(sc)) for idx, sc in zip(results[0], scores[0])]
        scores = self.model.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in ranked]


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
