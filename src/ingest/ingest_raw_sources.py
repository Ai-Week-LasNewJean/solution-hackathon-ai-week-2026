"""Ingesta de las fuentes descargadas en data/corpus/ hacia corpus/<doc_id>.txt
+ corpus_manifest.json + indice/chunks.jsonl (SPEC.md seccion 4).

data/corpus/ es un volcado del equipo con tres formas de texto distintas,
fuera del flujo fetch+parse_html/parse_pdf de build_corpus.py porque ya viene
pre-parseado o pre-troceado:

  1. "Scrapping de secretariasenado.gov.co/*.json": normas ya segmentadas por
     articulo (constitucion, codigos, leyes, decretos, actos legislativos,
     directivas). sentencias.json se descarta: "contenido" viene null en los
     5737 registros -- no hay texto que indexar, solo un grafo de citas.
     targets_manifest.json se descarta: es un indice de scraping, no contenido.

  2. "Providencias de la Corte Constitucional/*.json": metadatos y resumen de
     relatoria (numero, tema, resumen, resuelve), sin el texto integro de la
     providencia. Un chunk por providencia. Los 14 archivos por rango de anios
     se solapan en 1997-1998 (el rango 1992-1998 y el rango 1997-2000
     coinciden ahi) -- se deduplica globalmente por "Numero de la providencia".

  3. "Providencias Corte Suprema/chunks_csj.jsonl": ya viene troceado por el
     equipo en el esquema que build_faiss.py/build_bm25.py esperan (incluye
     "texto", "inicio", "fin"). Se usa tal cual; manifest_csj.json aporta los
     metadatos de manifest. Los .docx/.doc/.pdf del mismo directorio son
     duplicados de los .txt que ya produjeron ese jsonl -- no se tocan.

Uso:
    python -m src.ingest.ingest_raw_sources
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations

from src import config
from src.ingest.normalize import clean_text

RAW = config.DATA / "corpus"
SENADO_DIR = RAW / "Scrapping de secretariasenado.gov.co"
CCONST_DIR = RAW / "Providencias de la Corte Constitucional"
CSJ_DIR = RAW / "Providencias Corte Suprema"

FECHA_DESCARGA = "2026-10-01"

SENADO_FILES = [
    "constitucion.json",
    "codigo_civil.json",
    "codigo_comercio.json",
    "codigo_sustantivo_trabajo.json",
    "codigo_contencioso_administrativo.json",
    "estatuto_organico_sistema_financiero.json",
    "reglamento_cnelectoral.json",
    "decretos.json",
    "actos_legislativos.json",
    "leyes.json",
    "directivas_presidenciales.json",
]
# sentencias.json y targets_manifest.json se descartan a proposito (ver docstring).


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "doc"


def is_traceable(texto: str) -> bool:
    """True si citations.extract() reconoce una norma de origen cerca del
    inicio del fragmento (lo que exige traceability_check.py, SPEC.md
    seccion 6). citations.py es oficial y no se modifica: algunas formas no
    son reconocibles (Autos "A-NNN/AA" de la Corte Constitucional -- la sala
    "a" sola no esta en su lista de prefijos; salas como "stp" tampoco), y
    esos fragmentos simplemente no se incorporan al indice."""
    return bool(citations.extract(texto[:200]))


def _write_doc(doc_id: str, chunk_specs: list[dict], titulo: str, fuente: str, url: str,
               areas: list[str], n_articulos: int | None, metodo_ingesta: str) -> tuple[dict, list[dict]] | None:
    """Escribe corpus/<doc_id>.txt a partir de chunk_specs ([{"articulo","texto"}])
    y arma la fila de manifest + las filas de chunks.jsonl correspondientes.

    Descarta los fragmentos no trazables a su norma de origen (ver
    is_traceable) y, si no queda ninguno, el documento completo (devuelve
    None) -- un documento sin fragmentos trazables no aporta nada al indice
    y solo diluiria la recuperacion."""
    chunk_specs = [c for c in chunk_specs if is_traceable(c["texto"])]
    if not chunk_specs:
        return None
    doc_text = "\n\n".join(c["texto"] for c in chunk_specs)
    out_path = config.CORPUS_DIR / f"{doc_id}.txt"
    out_path.write_text(doc_text, encoding="utf-8")
    manifest_row = {
        "doc_id": doc_id,
        "titulo": titulo,
        "fuente": fuente,
        "url": url,
        "fecha_consulta": FECHA_DESCARGA,
        "areas": areas,
        "n_articulos": n_articulos,
        "n_fragmentos": len(chunk_specs),
        "metodo_ingesta": metodo_ingesta,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    chunk_rows = [
        {"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "articulo": c["articulo"], "texto": c["texto"]}
        for i, c in enumerate(chunk_specs)
    ]
    return manifest_row, chunk_rows


# --- 1. Scrapping de secretariasenado.gov.co --------------------------------

# Para normas numeradas, el nombre tal como lo trae el scraper no siempre es
# trazable ("DECRETO <LEY> 1088 DE 1993" rompe la regex de citations.py por
# los corchetes; algunos registros de leyes.json traen nombre="ACLARACION" en
# vez de "LEY N DE AAAA"). Se reconstruye la forma canonica a partir de
# tipo+numero+anio, que es la que citations._NORM_RE espera, en vez de
# confiar en el campo "nombre" para el prefijo citable.
TIPO_LABEL_NUMBERED = {"ley": "LEY", "decreto": "DECRETO", "acto_legislativo": "ACTO LEGISLATIVO"}

# Para los codigos de nombre fijo, el nombre canonico tiene que calzar con
# alguna variante de citations.CODES (p.ej. "codigo civil"); codigo_civil.json
# y codigo_comercio.json no traen "nombre" y el id ("codigo-civil", con
# guiones) tampoco calza con esa regex -- de ahi el mapeo explicito.
#
# estatuto_organico_sistema_financiero es un caso aparte: ni "nombre" ni
# "numero"/"anio" vienen poblados en la fuente (el scraper los perdio), pero
# su identidad legal es publica y fija -- es el Decreto 663 de 1993 -- y
# scripts/_retrieval_only_test.py (sample_50, item 128) confirmo que sin esa
# forma canonica ("decreto 663 de 1993", que si calza con citations._NORM_RE)
# casi todo el EOSF quedaba fuera del indice por no ser trazable.
TIPO_LABEL_FIXED = {
    "constitucion": "Constitución Política de Colombia",
    "codigo_civil": "Código Civil",
    "codigo_comercio": "Código de Comercio",
    "codigo_sustantivo_trabajo": "Código Sustantivo del Trabajo",
    "codigo_contencioso_administrativo": "Código Contencioso Administrativo",
    "estatuto_organico_sistema_financiero": "DECRETO 663 DE 1993 (Estatuto Orgánico del Sistema Financiero)",
}


def _senado_docs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["documentos"] if "documentos" in data else [data]


def _senado_titulo(doc: dict, doc_id: str) -> str:
    tipo = doc.get("tipo")
    if tipo in TIPO_LABEL_NUMBERED and doc.get("numero") and doc.get("anio"):
        return f"{TIPO_LABEL_NUMBERED[tipo]} {doc['numero']} DE {doc['anio']}"
    if tipo in TIPO_LABEL_FIXED:
        return TIPO_LABEL_FIXED[tipo]
    return doc.get("nombre") or doc_id.replace("-", " ")


def process_senado_file(path: Path) -> tuple[list[dict], list[dict]]:
    manifest_rows, chunk_rows = [], []
    for doc in _senado_docs(path):
        doc_id = doc.get("id") or slugify(doc.get("nombre", path.stem))
        titulo = _senado_titulo(doc, doc_id)
        articulos = doc.get("articulos") or []

        chunk_specs = []
        for art in articulos:
            texto = (art.get("texto") or "").strip()
            if not texto:
                continue  # articulo derogado/sin contenido en la fuente
            etiqueta = art.get("etiqueta") or f"ARTICULO {art.get('numero', '')}"
            chunk_specs.append({
                "articulo": art.get("numero"),
                "texto": clean_text(f"{titulo}. {etiqueta} {texto}"),
            })
        if not chunk_specs and doc.get("introduccion"):
            chunk_specs.append({
                "articulo": None,
                "texto": clean_text(f"{titulo}. {doc['introduccion']}"),
            })
        if not chunk_specs:
            continue

        result = _write_doc(
            doc_id=doc_id,
            chunk_specs=chunk_specs,
            titulo=titulo,
            fuente="Secretaría del Senado de la República de Colombia (scraping)",
            url=f"http://www.secretariasenado.gov.co/senado/basedoc/?q={quote_plus(titulo)}",
            areas=[],
            n_articulos=len(articulos) or None,
            metodo_ingesta="JSON pre-parseado (scraping secretariasenado) + segmentacion por articulo",
        )
        if result is None:
            continue
        row, chunks = result
        manifest_rows.append(row)
        chunk_rows.extend(chunks)
    return manifest_rows, chunk_rows


# --- 2. Providencias de la Corte Constitucional (metadatos de relatoria) ----

_CCONST_NUMERO_PERIOD_RE = re.compile(r"^([A-Za-z]+)\.\s*(\d)")


def _normalize_cconst_numero(numero: str) -> str:
    """"SU.707/96" -> "SU-707/96": algunos registros viejos separan sala y
    numero con un punto en vez de guion, que es lo que citations._SENT_RE
    espera. No cambia nada para los que ya usan guion."""
    return _CCONST_NUMERO_PERIOD_RE.sub(r"\1-\2", numero)


def process_cconst_dir(dir_path: Path) -> tuple[list[dict], list[dict]]:
    manifest_rows, chunk_rows = [], []
    seen_numeros: set[str] = set()
    skipped_dup = skipped_empty = skipped_untraceable = 0

    for path in sorted(dir_path.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data.get("Providencias", []):
            numero_raw = (rec.get("Número de la providencia") or "").strip()
            if not numero_raw:
                skipped_empty += 1
                continue
            if numero_raw in seen_numeros:
                skipped_dup += 1
                continue
            seen_numeros.add(numero_raw)
            numero = _normalize_cconst_numero(numero_raw)

            fecha = rec.get("Fecha de la providencia") or ""
            ponente = rec.get("Magistrado(s) Ponentes") or ""
            tema = rec.get("Tema") or ""
            resumen = rec.get("Resumen") or ""
            resuelve = rec.get("Resuelve/Decisión") or ""

            encabezado = f"Sentencia {numero}, Corte Constitucional de Colombia"
            if fecha:
                encabezado += f" ({fecha})"
            if ponente and ponente != "Sin Información":
                encabezado += f", M.P. {ponente}"
            encabezado += "."

            partes = [encabezado]
            if tema and tema != "Sin Información":
                partes.append(f"Tema: {tema}")
            if resumen and resumen != "Sin Información":
                partes.append(f"Resumen: {resumen}")
            if resuelve and resuelve != "Sin Información":
                partes.append(f"Resuelve: {resuelve}")
            texto = clean_text("\n\n".join(partes))
            if texto == clean_text(encabezado):
                skipped_empty += 1  # sin tema/resumen/resuelve: no aporta nada recuperable
                continue

            doc_id = f"sentencia_cc_{slugify(numero)}"
            result = _write_doc(
                doc_id=doc_id,
                chunk_specs=[{"articulo": None, "texto": texto}],
                titulo=encabezado[:-1],
                fuente="Relatoría de la Corte Constitucional (metadatos de providencia, sin texto íntegro)",
                url="https://www.corteconstitucional.gov.co/relatoria/",
                areas=[],
                n_articulos=None,
                metodo_ingesta="metadatos de relatoria (tema/resumen/resuelve, sin texto integro de la providencia)",
            )
            if result is None:
                skipped_untraceable += 1
                continue
            row, chunks = result
            manifest_rows.append(row)
            chunk_rows.extend(chunks)

    print(f"[cconst] {len(manifest_rows)} providencias, {skipped_dup} duplicadas (rangos solapados), "
          f"{skipped_empty} sin contenido util, {skipped_untraceable} no trazables (Autos y variantes que "
          f"citations.py no reconoce) -- omitidas")
    return manifest_rows, chunk_rows


# --- 3. Providencias Corte Suprema (ya troceadas por el equipo) -------------

def process_csj_dir(dir_path: Path) -> tuple[list[dict], list[dict]]:
    manifest_by_id = {d["doc_id"]: d for d in
                       json.loads((dir_path / "manifest_csj.json").read_text(encoding="utf-8"))}
    raw_chunks = [json.loads(line) for line in
                  (dir_path / "chunks_csj.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    by_doc: dict[str, list[dict]] = {}
    for c in raw_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    manifest_rows, chunk_rows = [], []
    skipped_untraceable = 0
    for doc_id, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda c: c.get("inicio") or 0)
        before = len(doc_chunks)
        doc_chunks = [c for c in doc_chunks if is_traceable(c["texto"])]
        skipped_untraceable += before - len(doc_chunks)
        if not doc_chunks:
            continue
        doc_text = "\n\n".join(c["texto"] for c in doc_chunks)
        out_path = config.CORPUS_DIR / f"{doc_id}.txt"
        out_path.write_text(doc_text, encoding="utf-8")

        base = manifest_by_id.get(doc_id, {})
        manifest_rows.append({
            "doc_id": doc_id,
            "titulo": base.get("titulo", doc_id),
            "fuente": base.get("fuente", "Relatoria de la Corte Suprema de Justicia"),
            "url": base.get("url") or "https://cortesuprema.gov.co/",
            "fecha_consulta": base.get("fecha_consulta", FECHA_DESCARGA),
            "areas": base.get("areas", []),
            "n_articulos": base.get("n_articulos"),
            "n_fragmentos": len(doc_chunks),
            "metodo_ingesta": base.get("metodo_ingesta", "extraccion de PDF/texto + segmentacion por seccion"),
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        })
        chunk_rows.extend(doc_chunks)

    print(f"[csj] {len(manifest_rows)} providencias, {len(chunk_rows)} fragmentos (ya troceados), "
          f"{skipped_untraceable} fragmentos no trazables omitidos (p.ej. sala STP, que citations.py no reconoce)")
    return manifest_rows, chunk_rows


# --- orquestacion ------------------------------------------------------------

def main() -> int:
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_manifest: list[dict] = []
    all_chunks: list[dict] = []

    for fname in SENADO_FILES:
        path = SENADO_DIR / fname
        if not path.exists():
            print(f"[senado] saltando {fname}: no existe")
            continue
        rows, chunks = process_senado_file(path)
        print(f"[senado] {fname}: {len(rows)} documentos, {len(chunks)} fragmentos")
        all_manifest.extend(rows)
        all_chunks.extend(chunks)

    cc_rows, cc_chunks = process_cconst_dir(CCONST_DIR)
    all_manifest.extend(cc_rows)
    all_chunks.extend(cc_chunks)

    csj_rows, csj_chunks = process_csj_dir(CSJ_DIR)
    all_manifest.extend(csj_rows)
    all_chunks.extend(csj_chunks)

    config.CHUNKS_PATH.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks) + "\n", encoding="utf-8")

    manifest = {
        "equipo": "Las NewJeans",
        "licencia": "CC-BY-4.0",
        "fecha_generacion": __import__("time").strftime("%Y-%m-%d"),
        "enlace_nube": None,
        "encoder": config.EMBEDDING_MODEL,
        "dimension": config.EMBEDDING_DIM,
        "indice": "faiss.IndexFlatIP",
        "n_documentos": len(all_manifest),
        "n_fragmentos": len(all_chunks),
        "documentos": all_manifest,
    }
    config.CORPUS_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nTotal: {len(all_manifest)} documentos, {len(all_chunks)} fragmentos")
    print(f"corpus/        -> {config.CORPUS_DIR}")
    print(f"manifest       -> {config.CORPUS_MANIFEST}")
    print(f"chunks.jsonl   -> {config.CHUNKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
