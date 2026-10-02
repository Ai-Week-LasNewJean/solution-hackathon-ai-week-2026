"""Evaluacion rapida solo de las cerradas de sample_50 (15 items, ~2 min):
corre el pipeline completo (answer()) y reporta aciertos por item.

    python -m src.validate.mc_eval [--model-name qwen3-8b] [--tag x]
"""
from __future__ import annotations

import argparse
import json
import time

from src import config
from src.pipeline.answer_one import answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    items = [json.loads(l) for l in (config.DATA / "sample_50.jsonl").read_text().splitlines() if l.strip()]
    items = [i for i in items if i["formato"] == "multiple_choice"]
    ok, rows, t0 = 0, [], time.time()
    for it in items:
        r = answer(it, args.model_name)
        good = (not r["abstencion"]) and r.get("respuesta_correcta") == it["respuesta_correcta"]
        ok += good
        rows.append((it["id"], good, r["abstencion"], r.get("respuesta_correcta"), it["respuesta_correcta"]))
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("MC", f"{ok}/{len(items)}", f"{time.time()-t0:.0f}s", "wrong:", [r[0] for r in rows if not r[1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
