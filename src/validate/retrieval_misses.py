"""Lista, por item de sample_50, las normas de referencia que NO aparecen en
el top-k (con reranker) y los doc_id que si se recuperaron: diagnostico de
huecos de recuperacion vs. huecos de corpus.

    python -m src.validate.retrieval_misses [--k 10]
"""
from __future__ import annotations

import argparse
import json

import citations

from src import config
from src.pipeline.answer_one import _build_query
from src.retrieve import query_expand
from src.retrieve.hybrid import retrieve
from src.retrieve import rerank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    items = [json.loads(l) for l in (config.DATA / "sample_50.jsonl").read_text().splitlines() if l.strip()]
    for it in items:
        ref = citations.bodies(citations.extract(it.get("legal_basis") or ""))
        if not ref:
            continue
        q = query_expand.expand(_build_query(it))
        ps = rerank.rerank(q, retrieve(q, top_k=config.FUSED_TOP_K), top_k=args.k)
        got = set()
        for p in ps:
            got |= citations.bodies(citations.extract(p.texto))
        missed = ref - got
        if missed:
            print(f"{it['id']} [{it['area'][:20]}] legal_basis={it['legal_basis']!r}\n   miss={sorted(missed)}"
                  f"\n   docs={[p.doc_id for p in ps]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
