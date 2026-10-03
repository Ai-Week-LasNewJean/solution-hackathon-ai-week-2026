"""Utilidades compartidas del ajuste fino del reranker (SPEC.md 15).

Los fragmentos se identifican por una llave estable (sha1 de doc_id + texto)
y no por su posicion en chunks.jsonl: asi los archivos de checkpoint
(data/train/*.jsonl) siguen siendo validos aunque el indice se reconstruya
con fragmentos nuevos.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from src import config

TRAIN_DIR = config.DATA / "train"
SYNTH_PATH = TRAIN_DIR / "synth_queries.jsonl"     # checkpoint etapa 1 (versionado)
PAIRS_PATH = TRAIN_DIR / "pairs.jsonl"             # checkpoint etapa 2 (versionado; solo llaves)
EVAL_LOG_PATH = TRAIN_DIR / "eval_log.jsonl"       # recall de cada checkpoint en sample_50
MODEL_DIR = config.ROOT / "models" / "reranker-ft"  # checkpoints de entrenamiento (no versionado)


def chunk_key(doc_id: str, texto: str) -> str:
    return hashlib.sha1(f"{doc_id}\x00{texto}".encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def chunks_by_key() -> dict[str, dict]:
    out = {}
    with config.CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                c = json.loads(line)
                out[chunk_key(c["doc_id"], c["texto"])] = c
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # ultima linea truncada por un corte: se regenera al reanudar
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
