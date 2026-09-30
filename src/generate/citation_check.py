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


def unsupported_citations(answer_text: str, pasajes_recuperados: list[dict]) -> set[tuple]:
    """Citas presentes en `answer_text` (a nivel de articulo) que no aparecen
    respaldadas en la evidencia recuperada para este item. Un conjunto no
    vacio dispara, en answer_one.py, una regeneracion correctiva
    determinista o abstencion."""
    got = citations.article_level(citations.extract(answer_text))
    resp = citations.bodies(respaldadas(pasajes_recuperados))
    return {c for c in got if (c[0], c[1], c[2]) not in resp}
