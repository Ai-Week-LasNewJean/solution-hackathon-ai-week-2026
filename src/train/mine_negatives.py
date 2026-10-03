"""Etapa 2: negativos dificiles para cada pregunta sintetica, con la misma
recuperacion hibrida del pipeline (BM25 + denso + RRF, sin reranker).

Se descartan como negativos los candidatos que probablemente tambien
responden la pregunta (falsos negativos): otros fragmentos de la misma
providencia, y cualquier fragmento que cite el mismo articulo que el positivo.

Reanudable: una linea por pregunta en data/train/pairs.jsonl (solo llaves,
el texto se resuelve contra chunks.jsonl al entrenar).

    python -m src.train.mine_negatives [--negs 7]
"""
from __future__ import annotations

import argparse
import time

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations

from src import config
from src.retrieve.hybrid import retrieve
from src.train.common import (PAIRS_PATH, SYNTH_PATH, append_jsonl, chunk_key, chunks_by_key,
                              read_jsonl)

POOL = 30


def _articles(texto: str) -> set[tuple]:
    return citations.article_level(citations.extract(texto[:600]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negs", type=int, default=7)
    args = ap.parse_args()
    chunks = chunks_by_key()
    done = {r["key"] for r in read_jsonl(PAIRS_PATH)}
    todo = [r for r in read_jsonl(SYNTH_PATH) if r.get("query") and r["key"] not in done
            and r["key"] in chunks]
    print(f"{len(done)} ya minadas; faltan {len(todo)}", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        pos = chunks[r["key"]]
        pos_is_sent = pos["doc_id"].startswith("sentencia")
        pos_arts = _articles(pos["texto"])
        cands = retrieve(r["query"], top_k=POOL)
        rank, negs = None, []
        for j, p in enumerate(cands):
            k = chunk_key(p.doc_id, p.texto)
            if k == r["key"]:
                rank = j
                continue
            if pos_is_sent and p.doc_id == pos["doc_id"]:
                continue
            if pos_arts and pos_arts & _articles(p.texto):
                continue
            if len(negs) < args.negs:
                negs.append(k)
        append_jsonl(PAIRS_PATH, {"key": r["key"], "query": r["query"], "pos_rank": rank,
                                  "negs": negs})
        if i % 200 == 0:
            print(f"{len(done) + i}  {i / (time.time() - t0):.1f}/s", flush=True)
    rows = read_jsonl(PAIRS_PATH)
    found = [x["pos_rank"] for x in rows if x["pos_rank"] is not None]
    print(f"positivo en el pool de {POOL}: {len(found)}/{len(rows)}; en top-6: "
          f"{sum(r < 6 for r in found)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
