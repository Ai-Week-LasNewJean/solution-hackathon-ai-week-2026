"""Utilidades compartidas por los tres prompts por formato.

Los ejemplos few-shot de abajo son deliberadamente estructurales y ficticios
(normas y articulos inventados) -- SPEC.md prohibe copiar el contenido real
de "Ejemplo de entrega" como si fuera real, porque ese contenido no describe
derecho colombiano vigente.
"""
from __future__ import annotations

SYSTEM_PREFIX = (
    "Eres un asistente juridico que responde preguntas de derecho colombiano "
    "usando EXCLUSIVAMENTE los pasajes de evidencia entregados abajo. No "
    "inventes normas, articulos ni jurisprudencia que no aparezcan en la "
    "evidencia. Responde siempre en espanol y en JSON valido, con exactamente "
    "las llaves pedidas."
)


def render_passages(passages: list) -> str:
    """Bloque de evidencia numerado que entra al prompt. `passages` son
    objetos Passage (src.retrieve.hybrid) ya recortados al top final."""
    if not passages:
        return "(sin pasajes recuperados)"
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (doc_id={p.doc_id}) {p.texto}")
    return "\n".join(lines)
