"""Evaluacion rapida de recuperacion (sin LLM): para cada item de
sample_50 mide si las normas del `legal_basis` de referencia aparecen en los
top-k pasajes recuperados (misma extraccion que usa evaluate.py:
citations.extract/bodies). Sirve para iterar rapido sobre recuperacion,
reranker y umbrales sin pagar la generacion.

    python -m src.validate.retrieval_eval [--rerank] [--k 6 10]
"""
from __future__ import annotations

import argparse
import json
import statistics

import citations

from src import config
from src.pipeline.answer_one import _build_query
from src.retrieve import query_expand, rerank
from src.retrieve.hybrid import retrieve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 10, 20])
    ap.add_argument("--fused", type=int, default=config.FUSED_TOP_K)
    args = ap.parse_args()
    config.USE_RERANKER = args.rerank

    items = [json.loads(l) for l in (config.DATA / "sample_50.jsonl").read_text().splitlines() if l.strip()]
    hits = {k: [] for k in args.ks}
    for it in items:
        ref = citations.bodies(citations.extract(it.get("legal_basis") or ""))
        if not ref:
            continue
        q = query_expand.expand(_build_query(it))
        cands = retrieve(q, top_k=args.fused)
        if args.rerank:
            cands = rerank.rerank(q, cands, top_k=max(args.ks))
        for k in args.ks:
            got = set()
            for p in cands[:k]:
                got |= citations.bodies(citations.extract(p.texto))
            hits[k].append(len(ref & got) / len(ref))
    for k in args.ks:
        print(f"recall@{k}: {statistics.mean(hits[k]):.3f} (n={len(hits[k])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
