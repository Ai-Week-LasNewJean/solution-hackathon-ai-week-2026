"""Flujo por pregunta, una pasada por defecto con reintento selectivo
(SPEC.md seccion 5):

pregunta -> construir query -> expansion de alias -> recuperacion hibrida
-> [opcional] rerank -> sufficiency.decide
    "sufficient" -> generar (1 llamada, temp=0, con gramatica)
    "retry"      -> UNA recuperacion ampliada -> redecidir -> generar | abstenerse
    "abstain"    -> saltar la generacion
-> si genero: citation_check.unsupported_citations -> ajuste deterministico
-> ensamblar el dict de entrega (todas las llaves que exige el schema)

Acota el peor caso a 2 pasadas de recuperacion + maximo 1 llamada de
generacion por pregunta, respetando el presupuesto de ~22 seg/pregunta.
"""
from __future__ import annotations

import time
from pathlib import Path

import citations

from src import config
from src.generate import citation_check, formatter
from src.generate.llm import get_llm
from src.generate.prompts import multiple_choice as prompt_multiple_choice
from src.generate.prompts import open_ended as prompt_open_ended
from src.generate.prompts import semi_open as prompt_semi_open
from src.retrieve import query_expand, rerank, sufficiency
from src.retrieve.hybrid import Passage, retrieve

PROMPT_BUILDERS = {
    "multiple_choice": prompt_multiple_choice.build,
    "semi_open": prompt_semi_open.build,
    "open_ended": prompt_open_ended.build,
}

GRAMMAR_PATHS = {
    fmt: Path(__file__).resolve().parents[1] / "generate" / "grammars" / f"{fmt}.gbnf"
    for fmt in PROMPT_BUILDERS
}


def _build_query(item: dict) -> str:
    q = item["pregunta"]
    if item.get("formato") == "multiple_choice" and item.get("opciones"):
        q += " " + " ".join(item["opciones"].values())
    return q


def _to_pasaje_dict(p: Passage) -> dict:
    d = {"doc_id": p.doc_id, "texto": p.texto, "score": p.score}
    if p.inicio is not None:
        d["inicio"] = p.inicio
    if p.fin is not None:
        d["fin"] = p.fin
    return d


def _retrieve_ranked(query: str) -> list[Passage]:
    expanded = query_expand.expand(query)
    candidates = retrieve(expanded, top_k=config.FUSED_TOP_K)
    return rerank.rerank(expanded, candidates, top_k=config.FINAL_TOP_K)


def _generate(item: dict, passages: list[Passage], model_name: str | None = None) -> dict:
    formato = item["formato"]
    prompt = PROMPT_BUILDERS[formato](item, passages)
    raw = get_llm(model_name).generate(
        prompt, max_tokens=config.MAX_TOKENS_BY_FORMAT[formato],
        grammar_path=GRAMMAR_PATHS[formato])
    data = formatter.parse(raw, formato)
    if formato == "semi_open":
        data["respuesta"] = formatter.enforce_word_limit(data["respuesta"], 150)
    return data


def _answer_text_for_check(formato: str, data: dict) -> str:
    """Replica evaluate.answer_text sobre la salida cruda, para correr
    citation_check con la misma senal que usara el evaluador."""
    if formato == "multiple_choice":
        return str(data.get("justificacion") or "")
    if formato == "semi_open":
        return " ".join(str(data.get(k) or "") for k in ("respuesta", "referencia_legal"))
    return " ".join(str(data.get(k) or "")
                     for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion"))


def answer(item: dict, model_name: str | None = None) -> dict:
    """item: una fila de sample_50.jsonl / test_992.jsonl (trae al menos id,
    formato, pregunta, y opciones si es multiple_choice).

    `model_name` (llave de config.LLM_CANDIDATES) alterna el decoder --
    SPEC.md 10.5, solo para comparar/demo en desarrollo. Sin argumento usa
    el candidato activo (config.LLM_MODEL_NAME), el unico que debe producir
    submissions.jsonl.

    Devuelve un dict listo para escribirse como linea de submissions.jsonl,
    validado por schema/submission.schema.json."""
    t0 = time.monotonic()
    query = _build_query(item)

    ranked = _retrieve_ranked(query)
    decision = sufficiency.decide(ranked)

    if decision == "retry":
        # unica recuperacion ampliada: mas candidatos antes de fusionar/recortar
        ranked = retrieve(query_expand.expand(query), k_bm25=config.K_BM25 * 2,
                           k_dense=config.K_DENSE * 2, top_k=config.FUSED_TOP_K)
        ranked = rerank.rerank(query, ranked, top_k=config.FINAL_TOP_K)
        decision = sufficiency.decide(ranked)

    pasajes = [_to_pasaje_dict(p) for p in ranked]

    out: dict = {
        "id": item["id"], "formato": item["formato"],
        "abstencion": decision == "abstain",
        "pasajes_recuperados": [] if decision == "abstain" else pasajes,
    }

    if decision != "abstain":
        data = _generate(item, ranked, model_name)
        answer_text = _answer_text_for_check(item["formato"], data)
        sin_respaldo = citation_check.unsupported_citations(answer_text, pasajes)
        if sin_respaldo:
            # ajuste deterministico: si TODA la evidencia citada carece de
            # respaldo, es mas seguro abstenerse que entregar una cita inventada.
            sin_respaldo_bodies = citations.bodies(sin_respaldo)
            todas_las_citadas = citations.bodies(citations.extract(answer_text))
            if todas_las_citadas and sin_respaldo_bodies >= todas_las_citadas:
                out["abstencion"] = True
                out["pasajes_recuperados"] = []
                out["latencia_ms"] = int((time.monotonic() - t0) * 1000)
                return out
        out.update(data)

    out["latencia_ms"] = int((time.monotonic() - t0) * 1000)
    return out
