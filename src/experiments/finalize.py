"""Arma la entrega final a partir de una corrida cruda, sin volver a generar.

Aplica el completado de citas (citation_check.add_evidence_citations) con el
k elegido y valida la forma contra el split con evaluate.validate:

    python -m src.experiments.finalize out/test_raw.jsonl out/submissions.jsonl --k 3 --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import src  # noqa: F401  (registra scripts/ en sys.path)
import evaluate
from common import read_jsonl

from src.experiments.cite_sim import con_citas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=0, help="pasajes de los que se completan citas (0 = no tocar)")
    ap.add_argument("--split", choices=("sample", "test"), default="test")
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="otras corridas crudas del mismo split (p. ej. el proceso en reversa) a fusionar")
    ap.add_argument("--fill-missing", action="store_true",
                    help="los items sin respuesta van como abstencion (0.5 en abstencion en vez de 0)")
    args = ap.parse_args()

    items = read_jsonl(evaluate.DATA / ("sample_50.jsonl" if args.split == "sample" else "test_992.jsonl"))
    ids = {r["id"] for r in items}
    # fusion por id: gana la primera fila sin excepcion
    by_id: dict[int, dict] = {}
    for ruta in [args.raw, *args.extra]:
        if not ruta.exists():
            continue
        for s in read_jsonl(ruta):
            if s.get("id") in ids and ("error" not in s or s["id"] not in by_id):
                if s["id"] not in by_id or "error" in by_id[s["id"]]:
                    by_id[s["id"]] = s
    generados = len(by_id)
    if args.fill_missing:
        for it in items:
            by_id.setdefault(it["id"], {"id": it["id"], "formato": it["formato"],
                                        "abstencion": True, "pasajes_recuperados": []})
    subs = [con_citas(by_id[it["id"]], args.k) for it in items if it["id"] in by_id]
    problems = evaluate.validate(subs, ids)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for s in subs:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    errores = sum("error" in s for s in subs)
    print(f"{args.out}: {generados} generados de {len(ids)}, {len(subs)} escritos, k={args.k}, problemas de forma: {len(problems)}, "
          f"items con excepcion: {errores}, abstenciones: {sum(bool(s.get('abstencion')) for s in subs)}")
    for p in problems[:20]:
        print("  -", p)


if __name__ == "__main__":
    main()
