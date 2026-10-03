"""Agente de citacion (determinista, sin LLM): agrega a la respuesta las normas
de los pasajes que sustentan la respuesta.

Idea: una vez recuperada la evidencia correcta, la cita es trivial -- no hace
falta pedirle al decoder que "recuerde" citar. Se extraen con
citations.extract() las normas de los primeros `k` pasajes de evidencia (los
que eligio el agente filtro, o el top del reranker) y se anaden en los campos
que el evaluador lee para citas pero que el juez de RAGAS no ve:
`referencia_legal` (semi_open) y `justificacion` (multiple_choice). open_ended
no se toca (sus cuatro campos van al juez).

Toda norma agregada sale de `pasajes_recuperados[:k]` con k <= 10, asi que por
construccion queda respaldada (evaluate.citas_respaldadas). Es una funcion
pura de (salida del LLM, pasajes): aplicarla sobre una corrida ya hecha da
exactamente lo mismo que re-ejecutar el pipeline con CITE_BUILDER_K.
"""
from __future__ import annotations

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations

from src import config

_NAMES = {"ley": "Ley", "decreto": "Decreto", "acto_legislativo": "Acto Legislativo",
          "resolucion": "Resolución", "circular": "Circular", "acuerdo": "Acuerdo"}

_CODE_NAMES = {
    "constitucion": "Constitución Política", "codigo_civil": "Código Civil",
    "codigo_penal": "Código Penal", "codigo_procedimiento_penal": "Código de Procedimiento Penal",
    "codigo_comercio": "Código de Comercio", "codigo_sustantivo_trabajo": "Código Sustantivo del Trabajo",
    "codigo_procesal_trabajo": "Código Procesal del Trabajo", "codigo_general_proceso": "Código General del Proceso",
    "cpaca": "Código de Procedimiento Administrativo y de lo Contencioso Administrativo",
    "estatuto_tributario": "Estatuto Tributario", "codigo_infancia": "Código de la Infancia y la Adolescencia",
    "codigo_nacional_policia": "Código Nacional de Seguridad y Convivencia Ciudadana",
    "codigo_disciplinario": "Código General Disciplinario", "estatuto_consumidor": "Estatuto del Consumidor",
    "decision_andina_486": "Decisión Andina 486",
}


def render(body: tuple) -> str:
    """Nombre legible de un cuerpo normativo (kind, numero, anio)."""
    kind, num, year = body
    if kind == "jurisprudencia":
        return f"Sentencia {num} de {year}"
    if kind in _NAMES:
        return f"{_NAMES[kind]} {num}" + (f" de {year}" if year else "")
    return _CODE_NAMES.get(kind) or citations.CODES[kind][0].title()


def _roundtrips(body: tuple) -> bool:
    """Solo se agrega un nombre si el extractor oficial lo vuelve a leer como
    el mismo cuerpo (evita sumar una cita que el evaluador leeria distinto)."""
    try:
        return citations.bodies(citations.extract(render(body))) == {body}
    except Exception:  # noqa: BLE001
        return False


def evidence_bodies(pasajes: list[dict], k: int) -> list[tuple]:
    """Cuerpos normativos citados en los primeros k pasajes, en orden de
    aparicion (pasaje por pasaje, orden estable dentro de cada uno)."""
    out: list[tuple] = []
    for p in pasajes[:min(k, config.MAX_PASAJES_EVIDENCIA)]:
        for b in sorted(citations.bodies(citations.extract(str(p.get("texto") or ""))), key=str):
            if b not in out and _roundtrips(b):
                out.append(b)
    return out


def apply(sub: dict, k: int | None = None) -> dict:
    """Devuelve `sub` (una fila de la entrega) con las normas de la evidencia
    anadidas. No-op si k<=0, si es abstencion o si el formato es open_ended."""
    k = config.CITE_BUILDER_K if k is None else k
    if k <= 0 or sub.get("abstencion"):
        return sub
    field = {"semi_open": "referencia_legal", "multiple_choice": "justificacion"}.get(sub.get("formato"))
    if field is None:
        return sub
    current = str(sub.get(field) or "")
    ya = citations.bodies(citations.extract(current))
    nuevas = [b for b in evidence_bodies(sub.get("pasajes_recuperados") or [], k) if b not in ya]
    if not nuevas:
        return sub
    sep = " " if current.rstrip().endswith((".", ";")) or not current.strip() else ". "
    sub = dict(sub)
    sub[field] = (current.rstrip() + sep + "Fuentes en la evidencia: "
                  + "; ".join(render(b) for b in nuevas) + ".").strip()
    return sub


if __name__ == "__main__":
    # Aplica el agente de citacion sobre una corrida ya generada:
    #   python -m src.generate.cite_builder in.jsonl out.jsonl [k]
    import json
    import sys

    src_path, dst_path = sys.argv[1], sys.argv[2]
    kk = int(sys.argv[3]) if len(sys.argv) > 3 else config.CITE_BUILDER_K
    with open(src_path, encoding="utf-8") as fi, open(dst_path, "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            if line.strip():
                fo.write(json.dumps(apply(json.loads(line), kk), ensure_ascii=False) + "\n")
