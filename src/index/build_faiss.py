"""Indice vectorial denso: faiss IndexFlatIP (SPEC.md seccion 3).

Busqueda exacta, sin aleatoriedad de entrenamiento (a diferencia de
IVF/HNSW) y trivialmente reconstruible desde chunks.jsonl -- mas importante
que la velocidad a esta escala (miles de fragmentos, no millones).
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src import config
from src.index.embed import embed_passages


def build(chunks_path: Path, out_path: Path, batch_size: int = 64) -> faiss.Index:
    """Construye el indice a partir de chunks.jsonl (una fila por fragmento,
    con al menos la llave "texto") y lo serializa en out_path."""
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError(f"{chunks_path} no tiene fragmentos")

    vectors = np.empty((len(chunks), config.EMBEDDING_DIM), dtype="float32")
    for start in range(0, len(chunks), batch_size):
        batch = [c["texto"] for c in chunks[start:start + batch_size]]
        vectors[start:start + len(batch)] = embed_passages(batch)

    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(vectors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    return index


def load(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def search(index: faiss.Index, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Devuelve [(posicion_en_chunks, score), ...] en orden descendente de score."""
    scores, ids = index.search(query_vector.reshape(1, -1).astype("float32"), k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    ap.add_argument("--out", type=Path, default=config.FAISS_INDEX_PATH)
    args = ap.parse_args()
    build(args.chunks, args.out)
    print(f"indice faiss escrito en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
