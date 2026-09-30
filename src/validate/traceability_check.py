"""Cada fragmento del indice debe indicar su norma de origen (SPEC.md seccion
6): requisito de correccion mas consecuente del reto, porque sin el la
citacion de la respuesta nunca liga con el pasaje recuperado y el respaldo de
citas cae a cero aunque la recuperacion haya sido correcta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import src  # noqa: F401
import citations

from src import config

MIN_TRACEABLE_RATIO = 0.98


def check(chunks_path: Path = config.CHUNKS_PATH) -> dict:
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    total = len(chunks)
    no_trazables = [c for c in chunks if not citations.extract(c["texto"][:200])]
    ratio = 1 - len(no_trazables) / total if total else 0.0
    return {
        "total_fragmentos": total,
        "no_trazables": len(no_trazables),
        "ratio_trazable": round(ratio, 4),
        "umbral": MIN_TRACEABLE_RATIO,
        "ok": ratio >= MIN_TRACEABLE_RATIO,
        "ejemplos_no_trazables": [c.get("doc_id") for c in no_trazables[:10]],
    }


def main() -> int:
    report = check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
