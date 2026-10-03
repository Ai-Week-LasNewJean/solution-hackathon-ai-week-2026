"""Auto-verificacion de citas sin respaldo antes de emitir la respuesta
(SPEC.md seccion 5). Espeja exactamente scripts/evaluate.citas_respaldadas():
mismo MAX_PASAJES_EVIDENCIA, para que lo que el pipeline se autocorrige sea lo
mismo que el evaluador va a penalizar con peso 2x.
"""
from __future__ import annotations

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations

from src import config

assert config.MAX_PASAJES_EVIDENCIA == 10, "debe igualar evaluate.MAX_PASAJES_EVIDENCIA"


def respaldadas(pasajes_recuperados: list[dict]) -> set[tuple]:
    """Citas presentes en los primeros MAX_PASAJES_EVIDENCIA pasajes -- copia
    literal de la logica de evaluate.citas_respaldadas() para que ambos lados
    (generacion y evaluacion) usen exactamente el mismo universo de respaldo."""
    cites: set[tuple] = set()
    for p in (pasajes_recuperados or [])[:config.MAX_PASAJES_EVIDENCIA]:
        cites |= citations.extract(str(p.get("texto") or ""))
    return cites


# campo donde se agregan las normas de la evidencia: ninguno entra al texto que
# juzga RAGAS (evaluate.ragas_text), salvo open_ended, que se deja intacto
CITE_FIELD = {"multiple_choice": "justificacion", "semi_open": "referencia_legal"}


# nombre con articulo gramatical; cada nombre debe seguir siendo reconocido por
# citations.extract (es una de las variantes de CODES / NORM_TYPES, sin acentos)
_CODE_NAMES = {
    "constitucion": "la Constitución Política",
    "codigo_civil": "el Código Civil",
    "codigo_penal": "el Código Penal",
    "codigo_procedimiento_penal": "el Código de Procedimiento Penal",
    "codigo_comercio": "el Código de Comercio",
    "codigo_sustantivo_trabajo": "el Código Sustantivo del Trabajo",
    "codigo_procesal_trabajo": "el Código Procesal del Trabajo",
    "codigo_general_proceso": "el Código General del Proceso",
    "cpaca": "el Código de Procedimiento Administrativo y de lo Contencioso Administrativo",
    "estatuto_tributario": "el Estatuto Tributario",
    "codigo_infancia": "el Código de la Infancia y la Adolescencia",
    "codigo_nacional_policia": "el Código Nacional de Policía",
    "codigo_disciplinario": "el Código General Disciplinario",
    "estatuto_consumidor": "el Estatuto del Consumidor",
    "decision_andina_486": "la Decisión Andina 486",
}
_TYPE_NAMES = {"ley": "la Ley", "decreto": "el Decreto", "acto_legislativo": "el Acto Legislativo",
               "resolucion": "la Resolución", "circular": "la Circular", "acuerdo": "el Acuerdo"}


def _de(nombre: str) -> str:
    """'el Codigo Civil' -> 'del Codigo Civil'; 'la Ley' -> 'de la Ley'."""
    return "del " + nombre[3:] if nombre.startswith("el ") else "de " + nombre


def _cite_text(c: tuple) -> str:
    """Tupla canonica -> texto que citations.extract vuelve a reconocer."""
    body, num, year, art = c
    if body == "jurisprudencia":
        return f"Sentencia {num} de {year}"
    nombre = _CODE_NAMES[body] if num is None else f"{_TYPE_NAMES[body]} {num} de {year}"
    if art:
        return f"Artículo {art} {_de(nombre)}"
    return nombre.split(" ", 1)[1]


def add_evidence_citations(formato: str, data: dict, answer_text: str,
                           pasajes_recuperados: list[dict], k: int) -> dict:
    """Agrega al campo de referencias las normas de los primeros `k` pasajes que
    la respuesta aun no cita. Todas estan respaldadas por construccion (salen de
    la propia evidencia), asi que no pueden generar citas sin respaldo."""
    field = CITE_FIELD.get(formato)
    if not field or k <= 0:
        return data
    ya = citations.bodies(citations.extract(answer_text))
    extra: list[str] = []
    for p in (pasajes_recuperados or [])[:k]:
        for c in sorted(citations.extract(str(p.get("texto") or "")), key=str):
            if (c[0], c[1], c[2]) not in ya:
                ya.add((c[0], c[1], c[2]))
                extra.append(_cite_text(c))
    if extra:
        data = dict(data)
        data[field] = f"{str(data.get(field) or '').rstrip('. ')}. Véase también: {'; '.join(extra)}."
    return data


def unsupported_citations(answer_text: str, pasajes_recuperados: list[dict]) -> set[tuple]:
    """Citas presentes en `answer_text` (a nivel de articulo) que no aparecen
    respaldadas en la evidencia recuperada para este item. Un conjunto no
    vacio dispara, en answer_one.py, una regeneracion correctiva
    determinista o abstencion."""
    got = citations.article_level(citations.extract(answer_text))
    resp = citations.bodies(respaldadas(pasajes_recuperados))
    return {c for c in got if (c[0], c[1], c[2]) not in resp}
