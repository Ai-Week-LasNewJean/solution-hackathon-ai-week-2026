"""Envoltorio de desarrollo sobre scripts/evaluate.py --ragas (oficial, sin
modificar): misma metrica y mismo juez, pero con RunConfig de RAGAS mas
tolerante (timeout largo, menos concurrencia). Motivo: desde Turing las
llamadas del juez (GLM-5.3-flash via OpenRouter) tardan ~25-30 s cada una
dentro de RAGAS y con la concurrencia por defecto (16 workers, 180 s) casi
todos los jobs terminan en TimeoutError y computan como cero.

    python -m src.validate.ragas_dev --submission out/X.jsonl --split sample
Los argumentos se pasan tal cual a evaluate.py (agrega --ragas solo).
"""
from __future__ import annotations

import sys

import src  # noqa: F401  (registra scripts/ en sys.path)
import evaluate  # scripts/evaluate.py
import ragas
from ragas.run_config import RunConfig

MIN_REMAINING_USD = 8.0  # presupuesto total de RAGAS del equipo: 20 USD; no gastar por debajo de esto


def _key_info() -> dict:
    import os
    import requests
    from evaluate import cargar_env
    cargar_env()
    r = requests.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=20)
    return r.json()["data"]


_orig = ragas.evaluate


def _patched(*args, **kwargs):
    kwargs.setdefault("run_config", RunConfig(timeout=900, max_workers=6, max_retries=1))
    return _orig(*args, **kwargs)


ragas.evaluate = _patched

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--ragas" not in argv:
        argv.append("--ragas")
    before = _key_info()
    rem = before.get("limit_remaining")
    if rem is not None and rem < MIN_REMAINING_USD:
        sys.exit(f"credito restante {rem:.2f} USD < {MIN_REMAINING_USD}: no se corre RAGAS (ver SPEC.md 11)")
    sys.argv = ["evaluate.py"] + argv
    rc = evaluate.main()
    after = _key_info()
    print(f"[ragas_dev] costo de esta corrida: {after['usage'] - before['usage']:.3f} USD; "
          f"restante {after.get('limit_remaining')}", file=sys.stderr)
    raise SystemExit(rc)
