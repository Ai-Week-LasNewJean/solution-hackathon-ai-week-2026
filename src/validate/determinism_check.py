"""Reejecuta N items dos veces en procesos frescos y compara byte a byte
(SPEC.md seccion 6). Corre en Turing (maquina canonica desde el cambio de
instrucciones del 2026-10-02). El Mac solo sirve como chequeo auxiliar cruzado.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src import config

RUN_ONE_SNIPPET = """
import json, sys
from pathlib import Path
from src.pipeline.answer_one import answer
from common import read_jsonl
item = [r for r in read_jsonl(Path(sys.argv[1])) if r["id"] == int(sys.argv[2])][0]
print(json.dumps(answer(item), ensure_ascii=False, sort_keys=True))
"""


def _run_fresh_process(items_path: Path, item_id: int) -> str:
    """Corre answer_one.answer() para un solo item en un proceso de Python
    nuevo (no reutiliza el modelo ya cargado en este proceso), para que la
    comparacion sea honesta sobre el arranque real."""
    result = subprocess.run(
        [sys.executable, "-c", RUN_ONE_SNIPPET, str(items_path), str(item_id)],
        cwd=config.ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()[-1]


def check(items_path: Path, item_ids: list[int]) -> dict:
    if not config.IS_CANONICAL:
        return {"ok": False, "motivo": "determinism_check solo corre en Turing, la maquina canonica (config.BACKEND != 'turing')"}

    divergencias = []
    for item_id in item_ids:
        first = _run_fresh_process(items_path, item_id)
        second = _run_fresh_process(items_path, item_id)
        first_obj, second_obj = json.loads(first), json.loads(second)
        # latencia_ms varia por definicion; se excluye de la comparacion byte a byte.
        first_obj.pop("latencia_ms", None)
        second_obj.pop("latencia_ms", None)
        if first_obj != second_obj:
            divergencias.append({"id": item_id, "corrida_1": first_obj, "corrida_2": second_obj})

    return {"n_items": len(item_ids), "divergencias": divergencias, "ok": not divergencias}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=Path, default=config.DATA / "sample_50.jsonl")
    ap.add_argument("--ids", type=int, nargs="+", required=True,
                     help="3-5 ids de muestra, por ejemplo: --ids 51 58 60")
    args = ap.parse_args()
    report = check(args.items, args.ids)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
