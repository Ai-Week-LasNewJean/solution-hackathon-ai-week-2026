"""Prompt para preguntas cerradas (formato multiple_choice)."""
from __future__ import annotations

from src.generate.prompts._common import SYSTEM_PREFIX, render_passages

_FEW_SHOT = """\
Ejemplo estructural (norma y articulo ficticios, solo para mostrar el formato):

Pregunta: ¿Cual es el plazo para interponer el recurso X segun la Ley 9999 de 2099?
Opciones:
A) Tres dias habiles.
B) Cinco dias habiles.
C) Diez dias calendario.
D) Un mes.
Evidencia:
[1] (doc_id=ley_9999_2099) Ley 9999 de 2099. Articulo 12. El recurso X debera \
interponerse dentro de los cinco (5) dias habiles siguientes a la notificacion.

Respuesta JSON:
{"respuesta_correcta": "B", "justificacion": "El articulo 12 de la Ley 9999 de 2099 \
fija el termino en cinco dias habiles.", "descarte_opciones": {"A": "El plazo no es de tres \
dias segun el articulo 12.", "C": "El articulo habla de dias habiles, no calendario.", \
"D": "Un mes excede el termino fijado por la norma."}}
"""


def build(item: dict, passages: list) -> str:
    opciones = item.get("opciones") or {}
    opciones_txt = "\n".join(f"{letra}) {texto}" for letra, texto in opciones.items())
    return (
        f"{SYSTEM_PREFIX}\n\n{_FEW_SHOT}\n"
        f"Pregunta: {item['pregunta']}\n"
        f"Opciones:\n{opciones_txt}\n"
        f"Evidencia:\n{render_passages(passages)}\n\n"
        "Responde solo con el JSON pedido: respuesta_correcta (A|B|C|D), "
        "justificacion (cita la norma/articulo que sustenta la respuesta), "
        "descarte_opciones (por que cada opcion incorrecta no aplica).\n\n"
        "Respuesta JSON:\n"
    )
