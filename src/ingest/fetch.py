"""Descarga de las fuentes declaradas en data/seed_targets.json.

I/O y CPU puro, sin riesgo de determinismo entre Mac y Turing (SPEC.md tabla
seccion 2) — puede correr en cualquiera de las dos maquinas, o en la maquina
de quien este construyendo el corpus.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

USER_AGENT = "ai-week-hackathon-2026-corpus-builder/1.0 (uso academico, sin fines comerciales)"
TIMEOUT_SECONDS = 30
RETRIES = 3
BACKOFF_SECONDS = 2.0


def fetch_url(url: str, retries: int = RETRIES) -> requests.Response:
    """GET con reintentos y backoff exponencial simple. Lanza la ultima
    excepcion si todos los intentos fallan."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def fetch_all(seed_targets_path: Path, raw_out_dir: Path) -> list[dict]:
    """Descarga cada `donde_buscar` de seed_targets.json a raw_out_dir/<doc_id>.html
    (o .pdf segun content-type), sin sobrescribir lo ya descargado.

    Devuelve una lista de {"doc_id", "norma", "url", "status", "path",
    "fecha_consulta"} para alimentar corpus_manifest.json.
    """
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    targets = json.loads(seed_targets_path.read_text(encoding="utf-8"))["documentos"]

    results: list[dict] = []
    for doc in targets:
        doc_id = _slugify(doc["norma"])
        url = doc.get("donde_buscar")
        if not url:
            continue
        ext = ".pdf" if url.lower().endswith(".pdf") else ".html"
        out_path = raw_out_dir / f"{doc_id}{ext}"

        entry = {
            "doc_id": doc_id, "norma": doc["norma"], "url": url,
            "path": str(out_path), "fecha_consulta": time.strftime("%Y-%m-%d"),
        }
        if out_path.exists():
            entry["status"] = "cached"
            results.append(entry)
            continue
        try:
            resp = fetch_url(url)
            out_path.write_bytes(resp.content)
            entry["status"] = f"ok ({resp.status_code})"
        except requests.RequestException as exc:
            entry["status"] = f"error: {exc}"
        results.append(entry)
    return results


def _slugify(norma: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFD", norma.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s
