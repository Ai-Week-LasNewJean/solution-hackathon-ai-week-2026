"""Corrida por lotes, resiliente a fallos, con escritura JSONL incremental
(SPEC.md seccion 4 y plan del sabado): un fallo en un item no debe perder el
trabajo ya hecho en los 991 restantes.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import src  # noqa: F401
from common import read_jsonl  # scripts/common.py, oficial

from src import config
from src.pipeline.answer_one import answer


def _already_done(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    return {row["id"] for row in read_jsonl(out_path) if isinstance(row.get("id"), int)}


def main(split: str, out_path: Path, resume: bool = True) -> int:
    """split: "sample" (data/sample_50.jsonl) o "test" (data/test_992.jsonl,
    entregado el sabado). Escribe una linea por item tan pronto se resuelve,
    para poder reanudar tras una interrupcion sin repetir trabajo ya hecho."""
    src_path = config.DATA / ("sample_50.jsonl" if split == "sample" else "test_992.jsonl")
    items = read_jsonl(src_path)

    done_ids = _already_done(out_path) if resume else set()
    pending = [it for it in items if it["id"] not in done_ids]
    print(f"[run_batch] {len(items)} items, {len(done_ids)} ya resueltos, "
          f"{len(pending)} pendientes", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8", newline="\n") as fh:
        for i, item in enumerate(pending, start=1):
            try:
                result = answer(item)
            except Exception:  # noqa: BLE001  -- un item roto no debe tumbar el lote
                traceback.print_exc()
                result = {"id": item["id"], "formato": item["formato"],
                          "abstencion": True, "pasajes_recuperados": [],
                          "error": "excepcion no controlada, ver stderr"}
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(pending):
                print(f"[run_batch] {i}/{len(pending)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--out", type=Path, default=config.ROOT / "out" / "dev.jsonl")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(args.split, args.out, resume=not args.no_resume))
