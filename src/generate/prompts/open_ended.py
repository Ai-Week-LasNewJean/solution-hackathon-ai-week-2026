"""Prompt para preguntas abiertas (formato open_ended). Los cuatro campos
cuentan tanto para citas como para RAGAS (evaluate.answer_text / ragas_text)."""
from __future__ import annotations

from src.generate.prompts._common import SYSTEM_PREFIX, render_passages

_FEW_SHOT = """\
Ejemplo estructural (norma y sentencia ficticias, solo para mostrar el formato):

Pregunta: Analice el alcance del principio W en el ordenamiento colombiano.
Evidencia:
[1] (doc_id=ley_9999_2099) Ley 9999 de 2099. Articulo 3. El principio W rige toda \
actuacion de la autoridad V y exige motivacion expresa de sus decisiones.
[2] (doc_id=sentencia_c_0_2099) Sentencia C-0 de 2099. La Corte precisa que el \
principio W no admite excepciones cuando estan en juego derechos fundamentales.

Respuesta JSON:
{"marco_normativo": "El articulo 3 de la Ley 9999 de 2099 consagra el principio W \
como limite a la actuacion de la autoridad V.", "analisis": "El principio W exige \
motivacion expresa de toda decision de la autoridad V, segun el articulo 3 de la \
Ley 9999 de 2099. La Sentencia C-0 de 2099 refuerza este alcance al senalar que no \
admite excepciones frente a derechos fundamentales. En conjunto, la norma y la \
jurisprudencia convergen en un estandar estricto de motivacion. Esto implica que \
cualquier decision que omita dicha motivacion resulta cuestionable.", \
"jurisprudencia": "Sentencia C-0 de 2099: el principio W no admite excepciones \
cuando estan en juego derechos fundamentales.", "conclusion": "El principio W opera \
como un limite estricto, reforzado por la jurisprudencia constitucional."}
"""


def build(item: dict, passages: list) -> str:
    return (
        f"{SYSTEM_PREFIX}\n\n{_FEW_SHOT}\n"
        f"Pregunta: {item['pregunta']}\n"
        f"Evidencia:\n{render_passages(passages)}\n\n"
        "Responde solo con el JSON pedido: marco_normativo, analisis (5 a 8 "
        "oraciones), jurisprudencia, conclusion.\n\n"
        "Respuesta JSON:\n"
    )
