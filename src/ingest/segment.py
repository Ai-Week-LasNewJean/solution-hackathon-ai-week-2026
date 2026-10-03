"""Segmentacion de texto normativo y de jurisprudencia (SPEC.md seccion 3).

Dos estrategias:
    segment_by_article()   codigos y leyes, por patron "ARTICULO N"
    segment_paragraphs()   jurisprudencia, ventana deslizante con solapamiento
"""
from __future__ import annotations

import re

from src import config

_ARTICLE_RE = re.compile(
    r"(?im)^\s*art[ií]culo\s+(\d+[a-z]?)\b\.?\s*[-.–—]?\s*")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9])")


def segment_by_article(text: str) -> list[dict]:
    """Corta un texto normativo en fragmentos por articulo.

    Devuelve una lista de {"articulo": str, "texto": str, "inicio": int,
    "fin": int}, con offsets de caracter sobre `text` tal como lo exige el
    schema de pasajes_recuperados (inicio/fin). El encabezado "ARTICULO N"
    queda dentro del fragmento (assert_traceable lo necesita).
    """
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []
    chunks: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fragment = text[start:end].strip()
        if not fragment:
            continue
        chunks.append({
            "articulo": m.group(1),
            "texto": fragment,
            "inicio": start,
            "fin": start + len(fragment),
        })
    return chunks


def segment_paragraphs(text: str, target_words: int = config.SEGMENT_WINDOW_WORDS,
                        overlap: int = config.SEGMENT_OVERLAP_SENTENCES) -> list[dict]:
    """Ventana deslizante por oraciones para texto sin estructura de articulos
    (jurisprudencia). `target_words` es el tamano objetivo de cada fragmento y
    `overlap` el numero de oraciones que se repiten entre fragmentos
    contiguos, para no cortar una idea a la mitad del limite.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[dict] = []
    cur: list[str] = []
    cur_words = 0
    cursor = 0  # offset aproximado de inicio del chunk actual en `text`

    def flush(next_start_idx: int) -> None:
        nonlocal cur, cur_words, cursor
        if not cur:
            return
        fragment = " ".join(cur)
        start = text.find(cur[0], cursor)
        start = start if start >= 0 else cursor
        end = start + len(fragment)
        chunks.append({"articulo": None, "texto": fragment, "inicio": start, "fin": end})
        cursor = start

    for sent in sentences:
        cur.append(sent)
        cur_words += len(sent.split())
        if cur_words >= target_words:
            flush(0)
            cur = cur[-overlap:] if overlap else []
            cur_words = sum(len(s.split()) for s in cur)
    if cur and (not chunks or " ".join(cur) != chunks[-1]["texto"]):
        flush(0)
    return chunks
