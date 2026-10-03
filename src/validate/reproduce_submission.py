"""Re-ejecuta items de la entrega en un proceso nuevo y compara contra las lineas entregadas
(lo mismo que hara el jurado en la verificacion en vivo; SPEC.md de la solucion, seccion 7).

Compara cada linea completa salvo `latencia_ms`. Usa el indice activo (por defecto `indice/`).

    python -m src.validate.reproduce_submission --submission out/submission_final.jsonl --ids 3 721 1106
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    a = ap.parse_args()

    from src.pipeline.answer_one import answer

    items = {json.loads(l)["id"]: json.loads(l) for l in (config.DATA / "test_992.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    sub = {r["id"]: r for r in map(json.loads, a.submission.read_text(encoding="utf-8").splitlines()) if r}
    bad = 0
    for i in a.ids:
        got = answer(items[i])
        want = dict(sub[i])
        got.pop("latencia_ms", None)
        want.pop("latencia_ms", None)
        same = json.dumps(got, sort_keys=True, ensure_ascii=False) == json.dumps(want, sort_keys=True, ensure_ascii=False)
        bad += not same
        print(f"[reproduce] id={i} {'IDENTICO' if same else 'DIFIERE'}", flush=True)
        if not same:
            for k in sorted(set(got) | set(want)):
                if got.get(k) != want.get(k):
                    print(f"    {k}: entregado={str(want.get(k))[:120]!r} | ahora={str(got.get(k))[:120]!r}")
    print(f"[reproduce] {len(a.ids) - bad}/{len(a.ids)} identicos con {config.INDEX_DIR}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
