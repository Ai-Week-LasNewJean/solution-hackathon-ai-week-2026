"""Sin fuga del banco de preguntas hacia el corpus (SPEC.md seccion 6):
ningun n-grama largo de pregunta/respuesta esperada debe aparecer literal en
corpus/*.txt. Reverificar apenas llegue test_992.jsonl el sabado -- ahi es
donde una fuga real costaria la descalificacion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import src  # noqa: F401
import citations

from src import config

NGRAM_WORDS = 12  # ventana de palabras para considerar "solapamiento literal"


def _ngrams(text: str, n: int = NGRAM_WORDS) -> set[str]:
    words = citations.norm(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def check(bank_path: Path, corpus_dir: Path = config.CORPUS_DIR) -> dict:
    rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    bank_ngrams: dict[str, set[str]] = {}
    for r in rows:
        campos = [r.get("pregunta", "")]
        campos += [str(v) for v in (r.get("opciones") or {}).values()]
        for k in ("respuesta_esperada", "texto_respuesta_correcta", "legal_basis"):
            if r.get(k):
                campos.append(str(r[k]))
        bank_ngrams[r["id"]] = set().union(*(_ngrams(c) for c in campos)) if campos else set()

    hallazgos: list[dict] = []
    for txt_path in sorted(corpus_dir.glob("*.txt")):
        corpus_ngrams = _ngrams(txt_path.read_text(encoding="utf-8"))
        if not corpus_ngrams:
            continue
        for qid, ngrams in bank_ngrams.items():
            overlap = ngrams & corpus_ngrams
            if overlap:
                hallazgos.append({"doc": txt_path.name, "item_id": qid,
                                   "ejemplo": next(iter(overlap))})

    return {"n_items_banco": len(rows), "n_documentos_corpus": len(list(corpus_dir.glob("*.txt"))),
            "hallazgos": hallazgos, "ok": not hallazgos}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=config.DATA / "sample_50.jsonl")
    args = ap.parse_args()
    report = check(args.bank)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
