"""Mide sample_50.jsonl en el hardware real y extrapola a 992 items, contra
el presupuesto de ~6h del sabado (SPEC.md seccion 6)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src import config
from src.pipeline.answer_one import answer

from common import read_jsonl  # scripts/common.py, oficial


def check(items_path: Path = config.DATA / "sample_50.jsonl") -> dict:
    items = read_jsonl(items_path)
    latencias_ms = []
    for item in items:
        t0 = time.monotonic()
        answer(item)
        latencias_ms.append((time.monotonic() - t0) * 1000)

    n = len(latencias_ms)
    promedio_seg = (sum(latencias_ms) / n / 1000) if n else 0.0
    estimado_992_seg = promedio_seg * config.N_PREGUNTAS_SABADO
    margen = config.PRESUPUESTO_SABADO_SEGUNDOS - estimado_992_seg

    return {
        "n_items": n,
        "backend": config.BACKEND,
        "promedio_seg_por_item": round(promedio_seg, 2),
        "presupuesto_seg_por_item": round(config.PRESUPUESTO_POR_PREGUNTA_SEGUNDOS, 2),
        "estimado_992_horas": round(estimado_992_seg / 3600, 2),
        "presupuesto_horas": round(config.PRESUPUESTO_SABADO_SEGUNDOS / 3600, 2),
        "margen_horas": round(margen / 3600, 2),
        "ok": margen > 0,
    }


def main() -> int:
    report = check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
