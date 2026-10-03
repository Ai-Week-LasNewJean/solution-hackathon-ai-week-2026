"""Decision de responder/reintentar/abstenerse, solo con senales de
recuperacion -- nunca preguntandole al LLM que se autoevalue (SPEC.md
seccion 5: la confianza autorreportada es poco confiable y no-determinista).

El umbral (config.SUFFICIENCY_SCORE_THRESHOLD / RETRY_SCORE_THRESHOLD) se
ajusta empiricamente barriendo contra `scripts/evaluate.py --split sample`:
como abstenerse da 0.5 sin importar si la respuesta hubiera sido correcta,
conviene sesgar el umbral hacia responder cuando la recuperacion trajo algo
plausible, y reservar la abstencion para huecos genuinos de cobertura.
"""
from __future__ import annotations

from src import config
from src.retrieve.hybrid import Passage

Decision = str  # "sufficient" | "retry" | "abstain"


def decide(scored: list[Passage]) -> Decision:
    """`scored` es la lista fusionada (ya ordenada por score) de
    retrieve.hybrid.retrieve(). Decide solo a partir de la forma de esa
    lista: si esta vacia, o si el mejor score no alcanza ningun umbral."""
    if not scored:
        return "abstain"
    top_score = scored[0].score
    if top_score >= config.SUFFICIENCY_SCORE_THRESHOLD:
        return "sufficient"
    if top_score >= config.RETRY_SCORE_THRESHOLD:
        return "retry"
    return "abstain"
