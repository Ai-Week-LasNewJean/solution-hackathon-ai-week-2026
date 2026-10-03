"""Etapa 1: preguntas sinteticas a partir del corpus, con el decoder abierto
(Qwen3-8B). El enunciado solo prohibe modelos cerrados para "datos sinteticos
de apoyo"; el banco de preguntas no se usa (sample_50 queda solo como test).

Reanudable: cada pregunta se agrega a data/train/synth_queries.jsonl apenas
se genera; al relanzar se saltan los fragmentos ya procesados.

    python -m src.train.synth_queries --n 4000
"""
from __future__ import annotations

import argparse
import random
import re
import time

from src.generate.llm import get_llm
from src.train.common import SYNTH_PATH, append_jsonl, chunk_key, chunks_by_key, read_jsonl

SEED = 13
SHARE_SENTENCIAS = 0.3
MIN_WORDS, MAX_PROMPT_WORDS = 40, 350

STYLES = {
    "directa": "Redacta UNA pregunta juridica concreta que este texto responda, como la haria un "
               "estudiante o un abogado. No menciones el numero del articulo.",
    "con_norma": "Redacta UNA pregunta juridica concreta que este texto responda; puedes nombrar el "
                 "codigo, la ley o la corte si es natural, pero no el numero del articulo.",
    "caso": "Redacta UNA pregunta juridica breve planteada como un caso practico (una persona, una "
            "empresa o una entidad en una situacion concreta) que este texto resuelva. No menciones "
            "el numero del articulo.",
}

PROMPT = ("Eres un abogado colombiano que prepara preguntas de examen.\n\nTexto:\n{texto}\n\n"
          "{instr} La pregunta debe entenderse sin ver el texto: no uses expresiones como 'segun el "
          "texto' o 'el articulo citado'. Responde solo con la pregunta, en una linea, entre signos "
          "de interrogacion.")

_LEAK = re.compile(r",?\s*(seg[uú]n|de acuerdo con|conforme a|en)\s+(el|este|dicho)\s+(texto|art[ií]culo)"
                   r"(\s+(proporcionado|citado|anterior|mencionado))?", re.IGNORECASE)


def sample_keys(n: int) -> list[str]:
    """Orden determinista de fragmentos candidatos (70% normas, 30% providencias)."""
    normas, sents = [], []
    for k, c in chunks_by_key().items():
        if len(c["texto"].split()) < MIN_WORDS:
            continue  # articulos vacios/derogados y fragmentos sin contenido
        (sents if c["doc_id"].startswith("sentencia") else normas).append(k)
    rng = random.Random(SEED)
    normas.sort(); sents.sort()
    rng.shuffle(normas); rng.shuffle(sents)
    # intercalado fijo (7 normas, 3 providencias por cada 10): sample_keys(n) es
    # prefijo de sample_keys(N > n), asi que subir --n reanuda sin desperdiciar nada
    keys, i_n, i_s = [], 0, 0
    while len(keys) < n:
        if (len(keys) % 10) < 10 * SHARE_SENTENCIAS and i_s < len(sents):
            keys.append(sents[i_s]); i_s += 1
        else:
            keys.append(normas[i_n]); i_n += 1
    return keys


def clean(q: str) -> str | None:
    q = _LEAK.sub("", q.strip().strip('"').strip())
    if "?" not in q:
        return None
    q = q[:q.rindex("?") + 1]
    if not q.startswith("¿") and "¿" not in q:
        q = "¿" + q
    n = len(q.split())
    return q if 6 <= n <= 70 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    done = {r["key"] for r in read_jsonl(SYNTH_PATH)}
    todo = [k for k in sample_keys(args.n) if k not in done]
    print(f"{len(done)} ya generadas; faltan {len(todo)}", flush=True)
    chunks, llm, t0 = chunks_by_key(), get_llm(), time.time()
    styles = list(STYLES)
    for i, k in enumerate(todo, 1):
        c = chunks[k]
        style = styles[int(k, 16) % len(styles)]
        texto = " ".join(c["texto"].split()[:MAX_PROMPT_WORDS])
        raw = llm.generate(PROMPT.format(texto=texto, instr=STYLES[style]), max_tokens=100,
                           stop=["\n"], no_think=True)
        q = clean(raw)
        append_jsonl(SYNTH_PATH, {"key": k, "doc_id": c["doc_id"], "style": style, "query": q})
        if i % 50 == 0:
            rate = i / (time.time() - t0)
            print(f"{len(done) + i}/{args.n}  {rate:.2f}/s  ETA {(len(todo) - i) / rate / 60:.0f} min",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
