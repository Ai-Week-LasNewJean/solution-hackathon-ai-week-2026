"""Ensambla la entrega final a partir de dos corridas (SPEC.md de la solucion, seccion 4).

`--base`: corrida e2e completa con el indice anterior. `--rerun`: corrida e2e
con el indice NUEVO solo sobre los items cuyo contexto cambio
(src/validate/retrieval_diff.py). Para el resto, la recuperacion final con el
indice nuevo es identica (mismos pasajes, mismos scores) y el decoder es
determinista, asi que la linea de --base es exactamente la que produciria el
sistema nuevo. Cada id sale de --rerun si esta en --ids, si no de --base.

    python -m src.pipeline.splice --base out/final_test.jsonl --rerun out/rerun_F.jsonl \
        --ids out/rerun_ids.txt --out out/submission_F.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config


def _rows(p: Path) -> dict[int, dict]:
    return {r["id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--rerun", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    order = [json.loads(l)["id"] for l in (config.DATA / "test_992.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base, rerun = _rows(a.base), _rows(a.rerun)
    ids = {int(x) for x in a.ids.read_text().split()}
    missing = [i for i in order if (i in ids and i not in rerun) or (i not in ids and i not in base)]
    if missing:
        raise SystemExit(f"faltan {len(missing)} items, p.ej. {missing[:10]}")
    with a.out.open("w", encoding="utf-8", newline="\n") as f:
        for i in order:
            f.write(json.dumps(rerun[i] if i in ids else base[i], ensure_ascii=False) + "\n")
    print(f"[splice] {len(order)} items -> {a.out} ({len(ids)} regenerados con el indice nuevo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
