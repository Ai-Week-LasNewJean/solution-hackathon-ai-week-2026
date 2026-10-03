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


# valor de relleno para llaves que quedaron fuera de un JSON truncado
_RELLENO = {
    "justificacion": "Ver la evidencia citada.",
    "descarte_opciones": {"otras": "La evidencia no respalda las demas opciones."},
    "palabras_clave": ["ver referencia legal"],
    "referencia_legal": "Ver la evidencia citada.",
    "jurisprudencia": "No se identifica jurisprudencia adicional en la evidencia.",
    "conclusion": "Ver el analisis.",
}


def _repair_truncated(raw: str, formato: str) -> dict:
    """JSON cortado por max_tokens (el modelo se extendio en un campo): cierra la
    cadena/objetos abiertos y rellena las llaves requeridas que no alcanzaron a
    salir. Conserva lo ya generado (p. ej. respuesta_correcta)."""
    texto = raw.strip()
    if not texto.startswith("{"):
        raise FormatError(f"no se encontro un objeto JSON en la salida: {raw[:200]!r}")
    texto = texto.rstrip(",: \n")
    for cierre in ('"}', '"}}', '}', '}}', '"]}', ']}', '""}'):
        try:
            data = json.loads(texto + cierre)
            break
        except json.JSONDecodeError:
            continue
    else:
        raise FormatError(f"no se pudo reparar el JSON truncado: {raw[:200]!r}")
    for k in REQUIRED_KEYS[formato]:
        if k not in data or data[k] in ("", None, {}, []):
            data[k] = _RELLENO.get(k, "Ver la evidencia citada.")
    return data


def parse(raw: str, formato: str) -> dict:
    """Convierte la salida cruda del LLM en un dict con las llaves de
    `formato`. Lanza FormatError si faltan llaves requeridas tras el intento
    de extraccion -- quien llama decide si reintenta o abstiene."""
    if formato not in REQUIRED_KEYS:
        raise FormatError(f"formato desconocido: {formato!r}")
    try:
        data = _extract_json(raw)
    except (FormatError, json.JSONDecodeError):
        data = _repair_truncated(raw, formato)
    faltan = [k for k in REQUIRED_KEYS[formato] if k not in data]
    if faltan:
        raise FormatError(f"faltan llaves {faltan} en la salida del formato {formato}")
    return data


# Abreviaturas de codigos que el modelo usa y que citations.extract NO reconoce
# (con puntos intermedios o sin punto final): se expanden al nombre completo para
# que la cita cuente. Solo mayusculas, para no tocar palabras comunes.
_ABREVIATURAS = [
    (r"C\.\s?P\.\s?A\.\s?C\.\s?A\.?|\bC\.?P\.?A\.?C\.?A\b\.?",
     "Código de Procedimiento Administrativo y de lo Contencioso Administrativo"),
    (r"\bC\.\s?G\.\s?P\b\.?", "Código General del Proceso"),
    (r"\bC\.\s?S\.\s?T\b\.?", "Código Sustantivo del Trabajo"),
    (r"\bC\.\s?P\.\s?T\.\s?(?:S\.\s?S\b\.?)?|\bCPTSS\b", "Código Procesal del Trabajo"),
    (r"\bC\.\s?P\.\s?P\b\.?", "Código de Procedimiento Penal"),
]
_ABREVIATURAS = [(re.compile(p), nombre) for p, nombre in _ABREVIATURAS]


def expand_abbreviations(data: dict) -> dict:
    """Expande en todos los campos de texto las abreviaturas de _ABREVIATURAS.
    Ajuste deterministico post-generacion (no cambia el contenido juridico)."""
    def fix(v):
        if isinstance(v, str):
            for pat, nombre in _ABREVIATURAS:
                v = pat.sub(nombre, v)
            return v
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v
    return {k: fix(v) for k, v in data.items()}


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
