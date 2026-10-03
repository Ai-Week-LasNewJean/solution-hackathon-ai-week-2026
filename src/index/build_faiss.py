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


def _read_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(chunks_path: Path, out_path: Path, batch_size: int = 64, reuse: Path | None = None) -> faiss.Index:
    """Construye el indice a partir de chunks.jsonl (una fila por fragmento,
    con al menos la llave "texto") y lo serializa en out_path.

    reuse: carpeta de un indice anterior (chunks.jsonl + index.faiss). Los
    fragmentos con texto identico toman su vector de alli y solo se embeben
    los nuevos -- mismo encoder, asi que el resultado equivale a reconstruir
    todo (salvo ruido de punto flotante por lote), en minutos y sin GPU."""
    chunks = _read_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"{chunks_path} no tiene fragmentos")

    vectors = np.empty((len(chunks), config.EMBEDDING_DIM), dtype="float32")
    todo = list(range(len(chunks)))
    if reuse is not None:
        old_index = load(reuse / "index.faiss")
        old_pos = {c["texto"]: i for i, c in enumerate(_read_chunks(reuse / "chunks.jsonl"))}
        todo = []
        for i, c in enumerate(chunks):
            j = old_pos.get(c["texto"])
            if j is None:
                todo.append(i)
            else:
                vectors[i] = old_index.reconstruct(j)
        print(f"[faiss] reutilizados {len(chunks) - len(todo)} vectores de {reuse}; embebiendo {len(todo)} nuevos")
    for start in range(0, len(todo), batch_size):
        ids = todo[start:start + batch_size]
        vectors[ids] = embed_passages([chunks[i]["texto"] for i in ids])

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
    ap.add_argument("--reuse", type=Path, default=None,
                    help="carpeta de un indice previo: reutiliza vectores de fragmentos con texto identico")
    args = ap.parse_args()
    build(args.chunks, args.out, reuse=args.reuse)
    print(f"indice faiss escrito en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
