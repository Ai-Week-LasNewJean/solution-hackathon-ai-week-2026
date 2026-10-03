"""Orquestador: fetch -> parse -> segment -> normalize -> corpus/<doc_id>.txt
+ corpus_manifest.json (SPEC.md seccion 4).

Uso:
    python -m src.ingest.build_corpus --seed data/seed_targets.json --raw raw/ --out corpus/

Cada documento de entrada puede venir de dos lugares:
  1. Descargado por fetch.fetch_all() a partir de seed_targets.json (HTML/PDF
     crudos, sin curar).
  2. Ya presente en RAW_CORPUS_DIR (la carpeta de OneDrive compartida por el
     equipo) como texto/HTML/PDF que alguien ya recolecto manualmente.

En ambos casos el resultado es el mismo: un .txt normalizado por doc_id en
corpus/, mas una fila en corpus_manifest.json con sha256 y conteos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src import config
from src.ingest.normalize import assert_traceable, clean_text, prefix_with_norm_name
from src.ingest.parse_html import parse_senado_basedoc, parse_suin
from src.ingest.parse_pdf import extract_text_with_ocr_fallback
from src.ingest.segment import segment_by_article, segment_paragraphs

PARSERS_BY_HOST = {
    "secretariasenado.gov.co": parse_senado_basedoc,
    "suin-juriscol.gov.co": parse_suin,
}


def _parser_for_url(url: str):
    for host, parser in PARSERS_BY_HOST.items():
        if host in url:
            return parser
    return None


def process_document(raw_path: Path, doc_meta: dict, out_dir: Path,
                      strict_traceability: bool = True) -> dict:
    """Procesa un documento crudo (HTML o PDF) ya descargado en `raw_path`.

    Escribe out_dir/<doc_id>.txt (texto normalizado, con cada articulo
    identificable por su encabezado) y devuelve la fila de manifest
    correspondiente. `doc_meta` debe traer al menos doc_id, titulo, fuente,
    url, areas.
    """
    doc_id = doc_meta["doc_id"]
    suffix = raw_path.suffix.lower()

    if suffix == ".pdf":
        raw_text = clean_text(extract_text_with_ocr_fallback(raw_path))
        chunks = segment_by_article(raw_text) or segment_paragraphs(raw_text)
        metodo = ("extraccion de PDF + segmentacion por articulo" if chunks and chunks[0]["articulo"]
                  else "extraccion de PDF + segmentacion por parrafo con solapamiento")
    else:
        html = raw_path.read_text(encoding="utf-8", errors="replace")
        parser = _parser_for_url(doc_meta.get("url", "")) or parse_senado_basedoc
        records = parser(html, doc_meta)
        chunks = [{"articulo": r["articulo"], "texto": r["texto"],
                   "inicio": r["inicio"], "fin": r["fin"]} for r in records]
        metodo = "parser HTML + segmentacion por articulo"

    prefixed = [prefix_with_norm_name(c["texto"], doc_meta) for c in chunks]
    for text in prefixed:
        if strict_traceability:
            assert_traceable(text, doc_meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    doc_text = "\n\n".join(prefixed)
    out_path = out_dir / f"{doc_id}.txt"
    out_path.write_text(doc_text, encoding="utf-8")

    n_articulos = sum(1 for c in chunks if c["articulo"] is not None) or None

    return {
        "doc_id": doc_id,
        "titulo": doc_meta.get("titulo", doc_id),
        "fuente": doc_meta.get("fuente", ""),
        "url": doc_meta.get("url", ""),
        "fecha_consulta": doc_meta.get("fecha_consulta", ""),
        "areas": doc_meta.get("areas", []),
        "n_articulos": n_articulos,
        "n_fragmentos": len(chunks),
        "metodo_ingesta": metodo,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


def build(raw_dir: Path, out_dir: Path, docs_meta: list[dict],
          strict_traceability: bool = True) -> dict:
    """Procesa todos los documentos de `docs_meta` (uno por doc_id) y escribe
    corpus_manifest.json en out_dir.parent. Devuelve el manifest completo."""
    manifest_docs = []
    for doc_meta in docs_meta:
        raw_path = Path(doc_meta["raw_path"])
        if not raw_path.exists():
            print(f"[build_corpus] saltando {doc_meta['doc_id']}: no existe {raw_path}")
            continue
        manifest_docs.append(process_document(raw_path, doc_meta, out_dir, strict_traceability))

    manifest = {
        "equipo": "Las NewJeans",
        "licencia": "CC-BY-4.0",
        "fecha_generacion": __import__("time").strftime("%Y-%m-%d"),
        "enlace_nube": None,
        "encoder": config.EMBEDDING_MODEL,
        "dimension": config.EMBEDDING_DIM,
        "indice": "faiss.IndexFlatIP",
        "n_documentos": len(manifest_docs),
        "n_fragmentos": sum(d["n_fragmentos"] for d in manifest_docs),
        "documentos": manifest_docs,
    }
    (out_dir.parent / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, default=config.DATA / "seed_targets.json")
    ap.add_argument("--raw", type=Path, default=config.ROOT / "raw",
                     help="carpeta con los documentos crudos ya descargados")
    ap.add_argument("--out", type=Path, default=config.CORPUS_DIR)
    ap.add_argument("--no-strict-traceability", action="store_true",
                     help="no abortar si un fragmento no es trazable (solo para depurar)")
    args = ap.parse_args()

    print(
        "Este comando procesa documentos ya presentes en --raw. Para poblar "
        "esa carpeta desde cero use src.ingest.fetch.fetch_all(seed_targets, raw_dir), "
        "o copie ahi lo que el equipo ya recolecto en la carpeta compartida "
        f"de OneDrive ({config.RAW_CORPUS_DIR}).")

    docs_meta = []  # a completar por quien construya el corpus: un dict por doc_id
    build(args.raw, args.out, docs_meta, strict_traceability=not args.no_strict_traceability)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
