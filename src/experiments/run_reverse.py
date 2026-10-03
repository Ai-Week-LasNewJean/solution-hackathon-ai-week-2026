"""Segundo trabajador para el test: recorre los items en orden inverso y se
salta los que la corrida principal ya resolvio sin excepcion (relee su archivo
antes de cada item). Reintenta los items con excepcion de la principal.
Las dos salidas se fusionan con src/experiments/finalize.py --extra.

    python -m src.experiments.run_reverse out/test_raw.jsonl out/test_rev.jsonl
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import src  # noqa: F401  (registra scripts/ en sys.path)
from common import read_jsonl

from src import config
from src.pipeline.answer_one import answer


def _ok_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:   # linea a medio escribir por el otro proceso
                continue
            if "error" not in row:
                ids.add(row["id"])
    return ids


def main() -> None:
    principal, out = Path(sys.argv[1]), Path(sys.argv[2])
    items = read_jsonl(config.DATA / "test_992.jsonl")[::-1]
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        for item in items:
            if item["id"] in _ok_ids(principal) | _ok_ids(out):
                continue
            try:
                result = answer(item)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                continue
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
