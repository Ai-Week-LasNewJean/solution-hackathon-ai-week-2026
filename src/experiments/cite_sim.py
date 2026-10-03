"""Simulador offline de post-procesos de citacion (sin LLM, sin API).

Toma una entrega ya generada (out/*.jsonl), le aplica el post-proceso
deterministico de citas del pipeline (citation_check.add_evidence_citations)
con distintos k y la puntua con el evaluador oficial (solo las partes
deterministas: cerradas, citacion y abstencion). Mide ideas de citacion en
segundos en vez de re-generar las 50 preguntas.

    python -m src.experiments.cite_sim out/sample_base.jsonl [otra.jsonl ...]
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import src  # noqa: F401  (registra scripts/ en sys.path)
import evaluate
from common import read_jsonl

from src.generate.citation_check import add_evidence_citations


def con_citas(s: dict, k: int) -> dict:
    if s.get("abstencion") or k <= 0:
        return s
    return add_evidence_citations(s["formato"], s, evaluate.answer_text(s),
                                  s.get("pasajes_recuperados") or [], k)


def puntuar(subs: list[dict]) -> dict:
    """Mismo armado de la clave que evaluate.main() para el split sample."""
    key = {r["id"]: r for r in read_jsonl(evaluate.DATA / "sample_50.jsonl")}
    by_id = {s["id"]: s for s in subs}
    closed = [q for q, r in key.items()
              if r["formato"] == "multiple_choice" and q not in evaluate.FLAWED_IDS]
    c = evaluate.score_closed(by_id, key, closed)
    ci = evaluate.score_citations(by_id, key)
    a = evaluate.score_abstention(by_id, key, set(closed))
    return {"cerradas": c["puntos"], "citas": ci["puntos"], "abst": a["puntos"],
            "total50": round(c["puntos"] + ci["puntos"] + a["puntos"], 2),
            "recall": ci["recall_citas_ponderado"], "sin_resp": ci["tasa_sin_respaldo"]}


def main() -> None:
    print(f"{'entrega':<28}{'k':>3}{'total/50':>9}{'cerr':>7}{'citas':>7}{'abst':>7}"
          f"{'recall':>8}{'sin_resp':>9}")
    for ruta in sys.argv[1:]:
        subs = read_jsonl(Path(ruta))
        for k in (0, 1, 3, 6, 10):
            r = puntuar([con_citas(copy.deepcopy(s), k) for s in subs])
            print(f"{Path(ruta).stem:<28}{k:>3}{r['total50']:>9}{r['cerradas']:>7}{r['citas']:>7}"
                  f"{r['abst']:>7}{r['recall']:>8}{r['sin_resp']:>9}")


if __name__ == "__main__":
    main()
