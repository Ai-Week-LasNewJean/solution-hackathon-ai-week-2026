"""Fusiona las salidas por shard en submissions.jsonl y valida (sabado 2026-10-03).

    python -m src.validate.merge_shards out/final_s1.jsonl out/final_s0.jsonl out/final_s0_rev.jsonl

- Descarta filas con "error" (fallos del proceso, no respuestas).
- Si un id aparece en mas de un archivo (las dos maquinas se cruzaron), compara las filas sin
  latencia_ms: deben ser identicas (chequeo de reproducibilidad entre maquinas); gana la primera.
- Escribe submissions.jsonl en el orden de data/test_992.jsonl y corre validate() del evaluador
  y el schema.
"""
from __future__ import annotations

import json
import sys

import jsonschema

import src  # noqa: F401  (registra scripts/ en sys.path)
import evaluate

from src import config


def main(paths: list[str]) -> int:
    order = [json.loads(l)["id"] for l in (config.DATA / "test_992.jsonl").read_text().splitlines() if l.strip()]
    rows: dict[int, dict] = {}
    overlap = mismatch = 0
    for p in paths:
        try:
            lines = open(p, encoding="utf-8").read().splitlines()
        except FileNotFoundError:
            print(f"(no existe {p})")
            continue
        n = 0
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            n += 1
            if r["id"] in rows:
                overlap += 1
                a = {k: v for k, v in rows[r["id"]].items() if k != "latencia_ms"}
                b = {k: v for k, v in r.items() if k != "latencia_ms"}
                if a != b:
                    mismatch += 1
                    print(f"  DIVERGE id {r['id']} entre archivos")
                continue
            rows[r["id"]] = r
        print(f"{p}: {n} filas validas")
    missing = [i for i in order if i not in rows]
    print(f"cubiertos {len(rows)}/{len(order)}; faltan {len(missing)} {missing[:20]}")
    print(f"solapados {overlap}, divergentes {mismatch}")

    out = config.ROOT / "submissions.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i in order:
            if i in rows:
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    subs = [rows[i] for i in order if i in rows]
    probs = evaluate.validate(subs, set(order))
    schema = json.loads((config.SCHEMA / "submission.schema.json").read_text())
    bad = 0
    for r in subs:
        try:
            jsonschema.validate(r, schema)
        except jsonschema.ValidationError:
            bad += 1
    print(f"validate(): {len(probs)} problemas {probs[:5]}; schema: {bad} filas invalidas")
    print(f"abstenciones {sum(bool(r.get('abstencion')) for r in subs)}; escrito {out}")
    return 0 if not missing and not probs and not bad and not mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
