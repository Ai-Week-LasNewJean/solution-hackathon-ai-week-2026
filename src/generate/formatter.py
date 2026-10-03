"""Parseo de la salida cruda del LLM a un dict por formato (SPEC.md seccion 3).

Con gramatica GBNF la salida ya es JSON valido con las llaves exactas; este
modulo existe igual para el camino sin gramatica (fallback) y para el ajuste
deterministico post-generacion (recorte de palabras) que el SPEC autoriza
explicitamente: "los ajustes deterministas post-generacion son validos".
"""
from __future__ import annotations

import json
import re

REQUIRED_KEYS = {
    "multiple_choice": ("respuesta_correcta", "justificacion", "descarte_opciones"),
    "semi_open": ("respuesta", "palabras_clave", "referencia_legal"),
    "open_ended": ("marco_normativo", "analisis", "jurisprudencia", "conclusion"),
}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class FormatError(ValueError):
    pass


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        raise FormatError(f"no se encontro un objeto JSON en la salida: {raw[:200]!r}")
    return json.loads(m.group(0))


def parse(raw: str, formato: str) -> dict:
    """Convierte la salida cruda del LLM en un dict con las llaves de
    `formato`. Lanza FormatError si faltan llaves requeridas tras el intento
    de extraccion -- quien llama decide si reintenta o abstiene."""
    if formato not in REQUIRED_KEYS:
        raise FormatError(f"formato desconocido: {formato!r}")
    data = _extract_json(raw)
    faltan = [k for k in REQUIRED_KEYS[formato] if k not in data]
    if faltan:
        raise FormatError(f"faltan llaves {faltan} en la salida del formato {formato}")
    return data


def enforce_word_limit(text: str, max_words: int) -> str:
    """Recorte deterministico al limite de palabras del schema (semi_open
    <=150 palabras). Corta por oracion completa cuando es posible, para no
    dejar una frase a medias."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    return truncated[:last_period + 1] if last_period > 0 else truncated + "."
