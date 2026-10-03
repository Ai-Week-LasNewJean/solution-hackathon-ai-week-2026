"""Corre el pipeline solo sobre un formato de sample_50 (iteracion rapida).

La configuracion se pasa por variables de entorno (src/config.py), igual que
en la corrida completa:

    MC_VOTES=3 python -m src.experiments.run_subset multiple_choice out/mc_votes3.jsonl

Imprime la exactitud en cerradas (si el formato es multiple_choice) y deja la
salida en el .jsonl indicado, lista para cite_sim.py o para fusionarse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.pipeline.answer_one import answer

ROOT = Path(__file__).resolve().parents[2]
FLAWED_IDS = {374}


def main() -> None:
    formato, out = sys.argv[1], Path(sys.argv[2])
    items = [json.loads(l) for l in open(ROOT / "data" / "sample_50.jsonl", encoding="utf-8")]
    items = [it for it in items if it["formato"] == formato]
    ok = n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            pregunta = {k: it[k] for k in ("id", "formato", "pregunta", "opciones", "area", "tema")
                        if k in it}
            s = answer(pregunta)
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            if formato == "multiple_choice" and it["id"] not in FLAWED_IDS:
                n += 1
                ok += (not s.get("abstencion")) and s.get("respuesta_correcta") == it["respuesta_correcta"]
    if n:
        print(f"{out.name}: cerradas {ok}/{n} = {ok / n:.3f} -> {20 * ok / n:.2f} pts")


if __name__ == "__main__":
    main()
