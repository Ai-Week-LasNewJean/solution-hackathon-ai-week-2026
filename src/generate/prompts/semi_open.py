"""Prompt para preguntas semi-abiertas (formato semi_open).

"respuesta" es lo unico que ve el juez de RAGAS (evaluate.ragas_text) --
debe ser una respuesta completa y autocontenida por si sola, de 3-5
oraciones y maximo 150 palabras; no delegar el contenido a referencia_legal.
"""
from __future__ import annotations

from src.generate.prompts._common import SYSTEM_PREFIX, render_passages

_FEW_SHOT = """\
Ejemplo estructural (norma ficticia, solo para mostrar el formato):

Pregunta: ¿En que consiste el deber Y segun la Ley 9999 de 2099?
Evidencia:
[1] (doc_id=ley_9999_2099) Ley 9999 de 2099. Articulo 7. Toda persona que ejerza \
la actividad Z debe cumplir el deber Y, consistente en informar a la autoridad \
competente dentro de los diez dias siguientes al hecho.

Respuesta JSON:
{"respuesta": "El deber Y, previsto en el articulo 7 de la Ley 9999 de 2099, obliga a \
quien ejerce la actividad Z a informar a la autoridad competente dentro de los diez \
dias siguientes a la ocurrencia del hecho.", \
"palabras_clave": ["deber Y", "actividad Z", "autoridad competente"], \
"referencia_legal": "Ley 9999 de 2099, articulo 7"}
"""


def build(item: dict, passages: list) -> str:
    return (
        f"{SYSTEM_PREFIX}\n\n{_FEW_SHOT}\n"
        f"Pregunta: {item['pregunta']}\n"
        f"Evidencia:\n{render_passages(passages)}\n\n"
        "Responde solo con el JSON pedido: respuesta (1 a 3 oraciones, maximo "
        "80 palabras: empieza directamente por la respuesta a lo preguntado, "
        "nombra la norma y el articulo que la sustentan y NO agregues datos, "
        "jurisprudencia ni normas que la evidencia no respalde o que la "
        "pregunta no pida), palabras_clave "
        "(lista breve), referencia_legal (norma/articulo citado).\n\n"
        "Respuesta JSON:\n"
    )
