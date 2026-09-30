"""Cada doc_id en chunks.jsonl debe existir en corpus_manifest.json, con
campos requeridos y sha256 verificado contra corpus/<doc_id>.txt
(SPEC.md seccion 6)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from src import config

REQUIRED_FIELDS = ("doc_id", "titulo", "fuente", "url", "fecha_consulta",
                    "areas", "n_fragmentos", "metodo_ingesta", "sha256")


def check(manifest_path: Path = config.CORPUS_MANIFEST,
          chunks_path: Path = config.CHUNKS_PATH,
          corpus_dir: Path = config.CORPUS_DIR) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs_by_id = {d["doc_id"]: d for d in manifest["documentos"]}

    problemas: list[str] = []
    for doc_id, doc in docs_by_id.items():
        faltan = [f for f in REQUIRED_FIELDS if doc.get(f) in (None, "")]
        if faltan:
            problemas.append(f"{doc_id}: faltan campos {faltan}")
        txt_path = corpus_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            problemas.append(f"{doc_id}: no existe {txt_path}")
            continue
        real_sha = hashlib.sha256(txt_path.read_bytes()).hexdigest()
        if doc.get("sha256") != real_sha:
            problemas.append(f"{doc_id}: sha256 no coincide (manifest={doc.get('sha256')!r}, "
                             f"real={real_sha!r})")

    chunk_doc_ids = set()
    if chunks_path.exists():
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk_doc_ids.add(json.loads(line)["doc_id"])
    huerfanos = sorted(chunk_doc_ids - set(docs_by_id))
    if huerfanos:
        problemas.append(f"doc_id en chunks.jsonl sin fila en el manifest: {huerfanos}")

    return {"n_documentos": len(docs_by_id), "problemas": problemas, "ok": not problemas}


def main() -> int:
    report = check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
