"""CLI: python src/main.py --split sample|test [--rebuild-index]

Punto de entrada unico de reproducibilidad (SPEC.md seccion 4), tambien
invocado por run.sh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # permite `python src/main.py` (invocacion documentada en SPEC.md y
    # usada por run.sh) ademas de `python -m src.main`: al correr el
    # archivo directamente, Python solo pone src/ en sys.path, no la raiz
    # del repo, y `from src import config` fallaria sin esto.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--out", type=Path, default=None,
                     help="por defecto out/<split>.jsonl")
    ap.add_argument("--rebuild-index", action="store_true",
                     help="reconstruye index.faiss y bm25.pkl desde corpus/ + chunks.jsonl antes de correr")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--model-name", default=None, choices=list(config.LLM_CANDIDATES),
                     help="candidato de config.LLM_CANDIDATES a usar en vez del activo (comparacion A/B, SPEC.md 10.5)")
    args = ap.parse_args(argv)

    if args.rebuild_index:
        from src.index import build_bm25, build_faiss

        print(f"[main] reconstruyendo indice desde {config.CHUNKS_PATH}", file=sys.stderr)
        build_faiss.build(config.CHUNKS_PATH, config.FAISS_INDEX_PATH)
        build_bm25.save(build_bm25.build(config.CHUNKS_PATH), config.BM25_INDEX_PATH)

    from src.pipeline.run_batch import main as run_batch_main

    out_path = args.out or (config.ROOT / "out" / f"{args.split}.jsonl")
    return run_batch_main(args.split, out_path, resume=not args.no_resume, model_name=args.model_name)


if __name__ == "__main__":
    raise SystemExit(main())
