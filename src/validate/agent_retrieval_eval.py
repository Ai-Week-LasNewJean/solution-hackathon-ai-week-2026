"""Evaluacion del agente filtro sin generar respuestas: para cada item de
sample_50 con normas de referencia compara
  - reranker: recall de normas en el top-k del reranker
  - agente:   recall de normas en los pasajes elegidos (lo que cita cite_builder)
              y en el top-k tras reordenar (lo que ve el decoder)
y cuantas normas se citarian (n_cit, menos = menos relleno).

    python -m src.validate.agent_retrieval_eval [--show 15] [--keep 4]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import citations

from src import config
from src.pipeline.answer_one import _build_query, _retrieve_ranked
from src.retrieve import filter_agent


def _bodies(ps) -> set:
    out = set()
    for p in ps:
        out |= citations.bodies(citations.extract(p.texto))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=config.FILTER_SHOW)
    ap.add_argument("--keep", type=int, default=config.FILTER_KEEP)
    ap.add_argument("--chars", type=int, default=config.FILTER_CHARS)
    args = ap.parse_args()
    config.AGENT_FILTER = True
    config.FILTER_SHOW, config.FILTER_KEEP, config.FILTER_CHARS = args.show, args.keep, args.chars

    items = [json.loads(l) for l in (config.DATA / "sample_50.jsonl").read_text().splitlines() if l.strip()]
    m = {k: [] for k in ("rr@2", "rr@3", "rr@6", "ag_chosen", "ag@6", "n_chosen", "n_cit_ag", "n_cit_rr3", "rr@10", "ag@10", "u2", "n_u2", "u3", "n_u3")}
    t_agent = []
    for it in items:
        ref = citations.bodies(citations.extract(it.get("legal_basis") or ""))
        if not ref:
            continue
        ranked = _retrieve_ranked(_build_query(it))
        t0 = time.monotonic()
        chosen = filter_agent.select(it, ranked)
        t_agent.append(time.monotonic() - t0)
        reord = filter_agent.reorder(ranked, chosen)
        sel = [ranked[i] for i in chosen] or ranked[:2]
        r = lambda ps: len(ref & _bodies(ps)) / len(ref)
        m["rr@2"].append(r(ranked[:2])); m["rr@3"].append(r(ranked[:3])); m["rr@6"].append(r(ranked[:6]))
        m["ag_chosen"].append(r(sel)); m["ag@6"].append(r(reord[:6]))
        m["n_chosen"].append(len(chosen)); m["n_cit_ag"].append(len(_bodies(sel)))
        m["n_cit_rr3"].append(len(_bodies(ranked[:3])))
        m["rr@10"].append(r(ranked[:10])); m["ag@10"].append(r(reord[:10]))
        for n in (2, 3):
            u = ranked[:n] + [ranked[i] for i in chosen[:n] if i >= n]
            m[f"u{n}"].append(r(u)); m[f"n_u{n}"].append(len(_bodies(u)))
        print(it["id"], "chosen", [c + 1 for c in chosen], "hit", round(m["ag_chosen"][-1], 2),
              "rr@3", round(m["rr@3"][-1], 2), flush=True)
    print(json.dumps({k: round(statistics.mean(v), 3) for k, v in m.items()}
                     | {"n": len(m["rr@6"]), "agent_s": round(statistics.mean(t_agent), 2),
                        "show": args.show, "keep": args.keep, "chars": args.chars}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
