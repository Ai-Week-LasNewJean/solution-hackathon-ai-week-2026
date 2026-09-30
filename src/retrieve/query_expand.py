"""Expansion de alias basada en reglas, usando scripts/citations.CODES -- sin
LLM (restriccion dura del reto: ningun modelo cerrado ni generativo puede
tocar la reformulacion de consultas).

Ejemplo: una pregunta que dice "CST" no siempre coincide lexicamente con
fragmentos indexados que llevan el nombre completo "Codigo Sustantivo del
Trabajo" como prefijo (normalize.prefix_with_norm_name). Expandir la consulta
con las variantes conocidas de la misma norma cierra ese hueco sin inventar
contenido.
"""
from __future__ import annotations

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations


def expand(query: str) -> str:
    """Devuelve la consulta original mas, al final, las variantes canonicas de
    cualquier norma que la consulta ya mencione (por su forma abreviada o
    completa). No elimina ni reescribe nada de la consulta original."""
    q_norm = citations.norm(query)
    extras: list[str] = []
    for code, variants in citations.CODES.items():
        if any(v in q_norm for v in variants):
            canonical = max(variants, key=len)
            if canonical not in q_norm:
                extras.append(canonical)
    return query if not extras else query + " " + " ".join(extras)
