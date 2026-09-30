"""Agregador unico de validaciones, objetivo <2 min, gate antes de cada
checkpoint (SPEC.md seccion 6): correo del viernes 17:00 y entrega del
sabado 15:00.

No corre determinism_check.py ni latency_check.py por defecto (necesitan el
LLM cargado y toman minutos, no segundos) -- pasar --full para incluirlos.
"""
from __future__ import annotations

import json
import sys

from src import config
from src.validate import leakage_check, manifest_check, traceability_check


def run(full: bool = False) -> dict:
    results: dict = {}

    if config.CHUNKS_PATH.exists():
        results["traceability_check"] = traceability_check.check()
    else:
        results["traceability_check"] = {"ok": False, "motivo": f"no existe {config.CHUNKS_PATH}"}

    if config.CORPUS_MANIFEST.exists():
        results["manifest_check"] = manifest_check.check()
    else:
        results["manifest_check"] = {"ok": False, "motivo": f"no existe {config.CORPUS_MANIFEST}"}

    if config.CORPUS_DIR.exists() and any(config.CORPUS_DIR.glob("*.txt")):
        results["leakage_check"] = leakage_check.check(config.DATA / "sample_50.jsonl")
    else:
        results["leakage_check"] = {"ok": False, "motivo": f"no hay .txt en {config.CORPUS_DIR}"}

    if full:
        from src.validate import determinism_check, latency_check

        sample_ids = [51, 58, 60]  # ver data/sample_50.jsonl
        results["determinism_check"] = determinism_check.check(config.DATA / "sample_50.jsonl", sample_ids)
        results["latency_check"] = latency_check.check()

    results["ok"] = all(r.get("ok") for r in results.values())
    return results


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                     help="incluye determinism_check y latency_check (requieren el LLM cargado)")
    args = ap.parse_args()
    report = run(full=args.full)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
