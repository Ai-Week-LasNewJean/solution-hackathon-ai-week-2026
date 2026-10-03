"""Decision 486 de 2000 de la Comision de la Comunidad Andina (Regimen Comun sobre
Propiedad Industrial) -> data/corpus_rescrape/decision_andina_486.json.

La norma no esta en secretariasenado.gov.co (el volcado solo la menciona desde
otras leyes) y la citan 23 items del banco (data/seed_targets.json), la quinta
norma mas citada. Fuente oficial: PDF de la Secretaria General de la CAN,
    https://www.comunidadandina.org/StaticFiles/DocOf/DEC486.pdf
cacheado en data/corpus_rescrape/raw/DEC486.pdf (las re-ejecuciones no tocan la red).

Salida en el mismo esquema que rescrape_senado.py (meta + documentos[0] con
articulos[{id, numero, etiqueta, texto, ubicacion}]); la ingesta la toma con
RESCRAPE_DIR y arma el encabezado citable con TIPO_LABEL_FIXED.

Uso:
    .venv/bin/python -m src.ingest.parse_decision486
"""
from __future__ import annotations

import json
import re
import sys

import pymupdf
import requests

from src.ingest.rescrape_senado import OUT, RAW, UA

URL = "https://www.comunidadandina.org/StaticFiles/DocOf/DEC486.pdf"
PDF = RAW / "DEC486.pdf"

ART_RE = re.compile(r"^Art[íi]culo\s+(\d+)\s*\.?\s*-\s*")
TRANS_RE = re.compile(r"^(PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA)\s*\.?\s*-\s*")
HEAD_RE = re.compile(r"^(T[IÍ]TULO|CAP[IÍ]TULO|SECCI[OÓ]N)\s+[IVXLC\d]+\b")
PAGE_RE = re.compile(r"^-\s*\d+\s*-$")


def _paragraphs(text: str) -> list[str]:
    """Une las lineas del PDF en parrafos (separados por lineas en blanco) y
    pega los literales "a)" con su texto."""
    paras, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if PAGE_RE.match(s):
            continue
        if not s:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    out = []
    for p in paras:
        if out and re.fullmatch(r"[a-z]{1,2}\)|\d+\.", out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(re.sub(r"\s+", " ", p).strip())
    return out


def parse(text: str) -> list[dict]:
    articulos, cur, ubic = [], None, {}
    in_trans = False
    for p in _paragraphs(text):
        m = ART_RE.match(p)
        mt = TRANS_RE.match(p) if in_trans else None
        if m or mt:
            if cur:
                articulos.append(cur)
            if m:
                num, etiqueta, rest = m.group(1), f"ARTÍCULO {m.group(1)}.", p[m.end():]
            else:
                num = f"transitoria-{mt.group(1).lower()}"
                etiqueta, rest = f"DISPOSICIÓN TRANSITORIA {mt.group(1)}.", p[mt.end():]
            cur = {"id": f"art-{num}", "numero": num, "etiqueta": etiqueta, "nombre": "", "tipo": "permanente",
                   "ubicacion": [ubic[k] for k in sorted(ubic)], "_lines": [rest], "referencias": [],
                   "anotaciones": {}}
            continue
        if p.startswith("DISPOSICIONES TRANSITORIAS"):
            in_trans = True
            if cur:
                articulos.append(cur)
                cur = None
            continue
        if p.startswith("Dada en la ciudad de Lima"):
            break
        h = HEAD_RE.match(p)
        if h and len(p) < 120:
            nivel = {"t": 0, "c": 1, "s": 2}[p[0].lower()]
            for k in [k for k in ubic if k >= nivel]:
                del ubic[k]
            ubic[nivel] = p
            if cur:
                articulos.append(cur)
                cur = None
            continue
        if cur is not None:
            cur["_lines"].append(p)
        elif ubic and p.isupper() and len(p) < 150:
            ubic[max(ubic)] += f" {p}"  # nombre del titulo/capitulo en la linea siguiente
    if cur:
        articulos.append(cur)
    for a in articulos:
        a["texto"] = "\n".join(x for x in a.pop("_lines") if x).strip()
    return articulos


def main() -> int:
    if not PDF.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL, headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        PDF.write_bytes(r.content)
    text = "\n".join(page.get_text() for page in pymupdf.open(PDF))
    articulos = parse(text)
    nums = [a["numero"] for a in articulos if a["numero"].isdigit()]
    if nums != [str(i) for i in range(1, 281)]:
        print(f"advertencia: articulos numerados inesperados ({len(nums)})", file=sys.stderr)
    doc = {"id": "decision-andina-486", "tipo": "decision_andina_486", "numero": "486", "anio": 2000,
           "fecha": "2000-09-14", "nombre": "DECISION 486 DE LA COMISION DE LA COMUNIDAD ANDINA",
           "epigrafe": "Régimen Común sobre Propiedad Industrial", "anotaciones": {}, "estructura": [],
           "articulos": articulos, "completo": True, "partes_incluidas": [0], "referencias": [],
           "citada_por": [], "fuente_url": URL}
    data = {"meta": {"fuente": "Secretaría General de la Comunidad Andina (texto oficial en PDF)",
                     "entidad": "decision_andina_486", "total_documentos": 1},
            "documentos": [doc]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision_andina_486.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    print(f"[decision_andina_486] articulos={len(articulos)} ({len(nums)} numerados)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
