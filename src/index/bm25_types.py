"""Tipo BM25Index, separado de build_bm25.py a proposito.

Si la dataclass viviera en build_bm25.py, `python -m src.index.build_bm25`
(la forma de invocacion documentada en su propio main()) ejecuta ese archivo
con __name__ == "__main__", y pickle graba las instancias como pertenecientes
al modulo "__main__" en vez de "src.index.build_bm25". save() funciona sin
error en ese mismo proceso, pero cualquier load() posterior desde otro
proceso (p.ej. src.retrieve.hybrid import build_bm25) revienta con
AttributeError: Can't get attribute 'BM25Index' on <module '__main__'>.
Definir la clase aqui, en un modulo que nunca se ejecuta como script, evita
el problema sin importar como se invoque build_bm25.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BM25Index:
    engine: str            # "bm25s" | "rank_bm25"
    model: object
    doc_ids: list[str]     # posicion i -> id del fragmento en chunks.jsonl (str(indice))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if self.engine == "bm25s":
            import bm25s

            results, scores = self.model.retrieve(bm25s.tokenize([" ".join(tokens)], show_progress=False),
                                                   k=min(k, len(self.doc_ids)), show_progress=False)
            return [(int(idx), float(sc)) for idx, sc in zip(results[0], scores[0])]
        scores = self.model.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in ranked]


def tokenize(text: str) -> list[str]:
    import re

    import src  # noqa: F401  (registra scripts/ en sys.path)
    import citations

    return re.findall(r"[a-z0-9]+", citations.norm(text))
