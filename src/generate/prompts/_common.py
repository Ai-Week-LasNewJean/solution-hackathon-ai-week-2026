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


# Instruccion extra de citacion (config.PROMPT_CITAS): el evaluador compara la
# norma (tipo + numero + ano, o el nombre del codigo) y no reconoce abreviaturas
# como "C.G.P." -- se pide el nombre completo y todas las normas de la evidencia
# que respaldan la respuesta.
CITAS_EXTRA = (
    " Cita cada norma con su nombre completo, sin abreviaturas (por ejemplo "
    "'articulo 60 del Codigo General del Proceso', 'articulo 5 de la Ley 1480 de "
    "2011', 'Sentencia C-355 de 2006'), y menciona TODAS las normas y sentencias "
    "de la evidencia que respaldan la respuesta, con el numero y el ano exactos "
    "con que aparecen en la evidencia."
)
