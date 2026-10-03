"""Proxy GRATIS (sin API) de la correccion de texto libre: similitud coseno
(e5-large, el mismo encoder del juez) entre `respuesta` del sistema y
`respuesta_esperada`, mas cobertura de normas citadas. Sirve para iterar sin
gastar el presupuesto de RAGAS (20 USD); confirmar solo configuraciones
finalistas con ragas_dev.

    python -m src.validate.freetext_proxy out/X.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import src  # noqa: F401
import evaluate

from src import config
from src.index.embed import embed_passages


def main(path: str) -> int:
    key = {json.loads(l)["id"]: json.loads(l) for l in (config.DATA / "sample_50.jsonl").read_text().splitlines() if l.strip()}
    subs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    ids, ans, ref = [], [], []
    for s in subs:
        if s["formato"] == "multiple_choice":
            continue
        ids.append(s["id"]); ans.append(evaluate.ragas_text(s) if not s.get("abstencion") else "")
        ref.append(key[s["id"]]["respuesta_esperada"])
    a, r = embed_passages(ans), embed_passages(ref)
    sims = [float(x @ y) if t else 0.0 for x, y, t in zip(a, r, ans)]
    print(f"proxy_similitud_media={statistics.mean(sims):.4f} n={len(sims)} palabras_media="
          f"{statistics.mean(len(t.split()) for t in ans):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
