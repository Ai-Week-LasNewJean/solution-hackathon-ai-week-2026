"""Que items cambian de contexto al cambiar de indice (SPEC.md de la solucion, seccion 4).

El decoder es greedy y determinista y su prompt depende solo del item y de la
lista final de pasajes (doc_id + texto, en orden; ver prompts/_common.py) y la
abstencion solo del score del mejor pasaje. Si con el indice nuevo la
recuperacion final de un item es identica a la de una corrida previa, la
respuesta tambien lo es: solo hay que regenerar los items cuyo contexto cambia.

    # 1) recuperacion final con el indice activo (RAG_INDEX_DIR) -> jsonl
    python -m src.validate.retrieval_diff dump --split test --out out/retr_F.jsonl
    # 2) comparar contra una corrida e2e hecha con otro indice
    python -m src.validate.retrieval_diff compare --retr out/retr_F.jsonl \
        --run out/final_test.jsonl --ids-out out/rerun_ids.txt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src import config


def dump(split: str, out: Path) -> int:
    from src.pipeline.answer_one import final_retrieval

    src_path = config.DATA / ("sample_50.jsonl" if split == "sample" else "test_992.jsonl")
    items = [json.loads(l) for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if os.environ.get("RAG_IDS_FILE"):   # mismo filtro que run_batch
        keep = {int(x) for x in Path(os.environ["RAG_IDS_FILE"]).read_text().split()}
        items = [it for it in items if it["id"] in keep]
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    with out.open("a", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            ranked, decision = final_retrieval(item)
            f.write(json.dumps({"id": item["id"], "decision": decision,
                                "pasajes": [[p.doc_id, p.texto, round(float(p.score), 4)] for p in ranked]},
                               ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"[retrieval_diff] {n}/{len(items)}", flush=True)
    return 0


def compare(retr: Path, run: Path, ids_out: Path | None) -> int:
    new = {r["id"]: r for r in map(json.loads, retr.read_text(encoding="utf-8").splitlines()) if r}
    old = {r["id"]: r for r in map(json.loads, run.read_text(encoding="utf-8").splitlines()) if r}
    same, rerun, reasons = [], [], {"abstencion_previa": 0, "contexto_distinto": 0, "sin_corrida_previa": 0}
    for i, r in new.items():
        o = old.get(i)
        if o is None:
            rerun.append(i); reasons["sin_corrida_previa"] += 1
            continue
        if o.get("abstencion") or not o.get("pasajes_recuperados"):
            rerun.append(i); reasons["abstencion_previa"] += 1   # sin pasajes no se puede comparar
            continue
        prev = [[p["doc_id"], p["texto"], round(float(p["score"]), 4)] for p in o["pasajes_recuperados"]]
        if r["decision"] == "abstain" or [p[:2] for p in prev] != [p[:2] for p in r["pasajes"]] \
                or any(abs(a[2] - b[2]) > 1e-3 for a, b in zip(prev, r["pasajes"])):
            rerun.append(i); reasons["contexto_distinto"] += 1
        else:
            same.append(i)
    print(f"[retrieval_diff] identicos {len(same)} / {len(new)}; a regenerar {len(rerun)} {reasons}")
    if ids_out:
        ids_out.write_text("\n".join(map(str, sorted(rerun))) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--split", default="test"); d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare"); c.add_argument("--retr", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True); c.add_argument("--ids-out", type=Path)
    a = ap.parse_args()
    return dump(a.split, a.out) if a.cmd == "dump" else compare(a.retr, a.run, a.ids_out)


if __name__ == "__main__":
    raise SystemExit(main())
