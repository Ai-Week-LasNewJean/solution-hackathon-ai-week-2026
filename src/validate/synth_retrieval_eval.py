"""Recuperacion sobre las preguntas sinteticas de data/train/synth_queries.jsonl
(Qwen3-8B, una por fragmento, SPEC.md 15.3) -- un segundo banco, mas grande que
las 41 normas de sample_50, para comparar variantes de corpus/indice.

Cada pregunta se liga a su fragmento de origen por la llave de src/train/common.py
calculada sobre el indice congelado (indice/chunks.jsonl), y de ahi a
(doc_id, articulo). Acierto@k = algun pasaje del top-k es ese mismo articulo
(o, para providencias sin articulo, el mismo doc_id) en el indice evaluado
(RAG_INDEX_DIR), aunque su texto haya cambiado (p.ej. con CHUNK_HEADINGS).

Caveat: las preguntas comparten vocabulario con el texto del que salieron (sin
encabezados), asi que el sesgo favorece al indice base.

    RAG_INDEX_DIR=... python -m src.validate.synth_retrieval_eval [--rerank] [--n 600]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

from src import config
from src.retrieve import query_expand, rerank
from src.retrieve.hybrid import _chunks, retrieve
from src.train.common import SYNTH_PATH, chunk_key, read_jsonl

FROZEN_CHUNKS = config.ROOT / "indice" / "chunks.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 6, 10])
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    config.USE_RERANKER = args.rerank

    origin = {}
    with FROZEN_CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            origin[chunk_key(c["doc_id"], c["texto"])] = (c["doc_id"], c.get("articulo"))
    by_text = {c["texto"]: (c["doc_id"], c.get("articulo")) for c in _chunks()}
    present = set(by_text.values())

    qs = [q for q in read_jsonl(SYNTH_PATH) if q.get("query") and q["key"] in origin and origin[q["key"]] in present]
    step = max(1, len(qs) // args.n)
    qs = qs[::step][:args.n]

    hits = collections.defaultdict(list)
    for q in qs:
        target = origin[q["key"]]
        query = query_expand.expand(q["query"])
        cands = retrieve(query)
        if args.rerank:
            cands = rerank.rerank(query, cands, top_k=max(args.ks))
        got = [by_text.get(p.texto) for p in cands]
        kind = "providencia" if target[0].startswith("sentencia") else "norma"
        for k in args.ks:
            ok = float(target in got[:k])
            for g in ("all", q["style"], kind):
                hits[(g, k)].append(ok)
    groups = sorted({g for g, _ in hits}, key=lambda g: (g != "all", g))
    print(f"n={len(qs)}  rerank={args.rerank}  index={config.INDEX_DIR}")
    for g in groups:
        row = "  ".join(f"@{k} {statistics.mean(hits[(g, k)]):.3f}" for k in args.ks)
        print(f"{g:12s} (n={len(hits[(g, args.ks[0])]):4d})  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
