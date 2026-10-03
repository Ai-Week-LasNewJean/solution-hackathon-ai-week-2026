"""Agente filtro (LLM): entre los mejores candidatos del reranker elige los
pasajes que de verdad contienen la norma que responde la pregunta.

Cadena de agentes (idea del CEO de Ariel: lo dificil es encontrar la cita
correcta; con la cita correcta la respuesta es trivial):
    recuperacion hibrida (BM25 + denso, RRF)  -> pool de FUSED_TOP_K
    reranker (cross-encoder)                  -> FILTER_SHOW candidatos
    agente filtro (este modulo, Qwen3, temp 0) -> hasta FILTER_KEEP pasajes elegidos
    decoder final                              -> responde solo con la evidencia elegida
    agente de citacion (cite_builder)          -> cita las normas de los pasajes elegidos

Salida forzada por gramatica (grammars/filter.gbnf): {"pasajes": [n, ...]}.
Determinista: misma entrada -> misma lista.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.generate.llm import get_llm
from src.retrieve.hybrid import Passage

GRAMMAR = Path(__file__).resolve().parents[1] / "generate" / "grammars" / "filter.gbnf"


def build_prompt(item: dict, passages: list[Passage], chars: int) -> str:
    q = item["pregunta"].strip()
    if item.get("formato") == "multiple_choice" and item.get("opciones"):
        q += "\nOpciones:\n" + "\n".join(f"{k}) {v}" for k, v in item["opciones"].items())
    ev = "\n".join(f"[{i}] {' '.join(p.texto.split())[:chars]}" for i, p in enumerate(passages, 1))
    return (
        "Eres un abogado colombiano experto en encontrar la fuente legal exacta. "
        "Abajo hay una pregunta y una lista numerada de pasajes (articulos de normas y "
        "providencias), ya ordenada por un buscador de mayor a menor relevancia estimada (el "
        "buscador suele acertar en los primeros, pero a veces la fuente correcta esta mas abajo). Elige los pasajes que contienen la norma, articulo o sentencia "
        "que responde DIRECTAMENTE la pregunta. Ignora pasajes que solo comparten palabras "
        "con la pregunta pero regulan otro tema. Ordenalos del mas al menos util y elige "
        f"como maximo {config.FILTER_KEEP}.\n\n"
        f"Pregunta: {q}\n\nPasajes:\n{ev}\n\n"
        'Responde solo con JSON: {"pasajes": [numeros]}\n'
    )


def select(item: dict, candidates: list[Passage], model_name: str | None = None) -> list[int]:
    """Indices (base 0) en `candidates` de los pasajes elegidos, en orden de
    utilidad. Lista vacia si el agente no elige nada valido."""
    shown = candidates[:config.FILTER_SHOW]
    if not shown:
        return []
    raw = get_llm(model_name).generate(build_prompt(item, shown, config.FILTER_CHARS),
                                       max_tokens=48, grammar_path=GRAMMAR, no_think=True)
    try:
        nums = json.loads(raw)["pasajes"]
    except Exception:  # noqa: BLE001 -- salida rota = sin seleccion, se usa el orden del reranker
        return []
    out: list[int] = []
    for n in nums:
        if isinstance(n, int) and 1 <= n <= len(shown) and (n - 1) not in out:
            out.append(n - 1)
    return out[:config.FILTER_KEEP]


def reorder(candidates: list[Passage], chosen: list[int]) -> list[Passage]:
    """Elegidos primero (en el orden del agente), luego el resto en el orden del reranker."""
    rest = [p for i, p in enumerate(candidates) if i not in chosen]
    return [candidates[i] for i in chosen] + rest
