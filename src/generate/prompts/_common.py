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


MAX_EVIDENCE_CHARS = 18000  # ~5-6k tokens de 8192, deja espacio a few-shot + generacion


def render_passages(passages: list) -> str:
    """Bloque de evidencia numerado que entra al prompt. `passages` son
    objetos Passage (src.retrieve.hybrid) ya recortados al top final."""
    if not passages:
        return "(sin pasajes recuperados)"
    # Tope por pasaje: evita desbordar la ventana de contexto (LLM_N_CTX) cuando
    # hay pasajes muy largos o FINAL_TOP_K es alto. Solo recorta lo que entra al
    # prompt; pasajes_recuperados (respaldo de citas) conserva el texto completo.
    cap = max(1200, MAX_EVIDENCE_CHARS // len(passages))
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (doc_id={p.doc_id}) {p.texto[:cap]}")
    return "\n".join(lines)
