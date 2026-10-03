"""Etapa 4: recall de normas de referencia en sample_50 (misma medida que
src.validate.retrieval_eval --rerank) para el reranker base y cada checkpoint
del ajuste fino. sample_50 nunca se usa para entrenar: es el test.

Reanudable: cada modelo evaluado se agrega a data/train/eval_log.jsonl y no
se vuelve a evaluar.

    python -m src.train.eval_checkpoints
"""
from __future__ import annotations

import json
import statistics

import src  # noqa: F401
import citations

from src import config
from src.pipeline.answer_one import _build_query
from src.retrieve import query_expand, rerank
from src.retrieve.hybrid import retrieve
from src.train.common import EVAL_LOG_PATH, MODEL_DIR, append_jsonl, read_jsonl
from src.train.train_reranker import BASE_MODEL

KS = (3, 6, 10)


def _recall(items, cands_by_id) -> dict:
    hits = {k: [] for k in KS}
    for it in items:
        ref = citations.bodies(citations.extract(it.get("legal_basis") or ""))
        q = query_expand.expand(_build_query(it))
        ranked = rerank.rerank(q, cands_by_id[it["id"]], top_k=max(KS))
        for k in KS:
            got = set()
            for p in ranked[:k]:
                got |= citations.bodies(citations.extract(p.texto))
            hits[k].append(len(ref & got) / len(ref))
    return {f"recall@{k}": round(statistics.mean(v), 3) for k, v in hits.items()}


def main() -> int:
    items = [json.loads(l) for l in (config.DATA / "sample_50.jsonl").read_text().splitlines() if l.strip()]
    items = [it for it in items if citations.extract(it.get("legal_basis") or "")]
    # candidatos de la recuperacion hibrida: no dependen del reranker, se calculan una vez
    cands = {it["id"]: retrieve(query_expand.expand(_build_query(it)), top_k=config.FUSED_TOP_K)
             for it in items}
    done = {r["model"] for r in read_jsonl(EVAL_LOG_PATH)}
    models = [BASE_MODEL] + sorted((str(p) for p in MODEL_DIR.glob("checkpoint-*")),
                                   key=lambda s: int(s.rsplit("-", 1)[1]))
    if (MODEL_DIR / "final").exists():
        models.append(str(MODEL_DIR / "final"))
    config.USE_RERANKER = True
    for m in models:
        if m in done:
            continue
        config.RERANKER_MODEL = m
        rerank._model.cache_clear()
        row = {"model": m, **_recall(items, cands)}
        append_jsonl(EVAL_LOG_PATH, row)
        print(row, flush=True)
    for r in read_jsonl(EVAL_LOG_PATH):
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
