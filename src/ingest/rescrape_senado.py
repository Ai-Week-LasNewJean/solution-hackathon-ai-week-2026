"""Re-scrape completo de secretariasenado.gov.co para documentos multi-parte.

El volcado original (data/corpus/Scrapping de secretariasenado.gov.co/*.json)
quedo truncado: solo trae la primera(s) parte(s) de cada documento
(`completo == False`, `partes_incluidas` de 1-3). El sitio sirve cada norma como
  http://www.secretariasenado.gov.co/senado/basedoc/<nombre>.html      (parte 0)
  http://www.secretariasenado.gov.co/senado/basedoc/<nombre>_pr001.html (parte 1) ...
Este script baja TODAS las partes (hasta el primer 404), parsea los articulos
al mismo esquema que los JSON existentes (meta + documentos[0] con id, tipo,
numero, anio, nombre, estructura, articulos[{id, numero, etiqueta, nombre, tipo,
ubicacion, texto, referencias, anotaciones}], completo, partes_incluidas) y
escribe data/corpus_rescrape/<entidad>.json.

Cortesia: <=2 req/s, User-Agent descriptivo, reintentos con backoff, HTML crudo
cacheado en data/corpus_rescrape/raw/ (las re-ejecuciones no tocan la red).
Nota: las "Notas de Vigencia"/"Legislacion Anterior" se cargan por JS en el
sitio (tablas vacias en el HTML), asi que solo se capturan las notas del editor
entre <...> que preceden al texto del articulo (van a anotaciones.notas_inline).

Uso:
    .venv/bin/python -m src.ingest.rescrape_senado               # todas
    .venv/bin/python -m src.ingest.rescrape_senado codigo_civil  # una entidad

Integracion con la ingesta (cambio minimo y retrocompatible en
ingest_raw_sources.py): si la variable de entorno RESCRAPE_DIR apunta a un
directorio con estos JSON (p.ej. data/corpus_rescrape), la ingesta los procesa
ADEMAS de SENADO_FILES y descarta de los archivos truncados todo documento
cuyo `id` coincida con uno re-scrapeado (mismo doc_id => corpus/<doc_id>.txt se
reemplaza, sin duplicados en manifest/chunks). Sin la variable, nada cambia.
    RESCRAPE_DIR=/ruta/data/corpus_rescrape .venv/bin/python -m src.ingest.ingest_raw_sources
Salida: RESCRAPE_OUT (env) o data/corpus_rescrape bajo la raiz del repo.
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "http://www.secretariasenado.gov.co/senado/basedoc/"
UA = "ai-week-uniandes-hackathon-rag/1.0 (academic, non-commercial; contact adrianesvmotta@gmail.com)"
import os
OUT = Path(os.environ.get("RESCRAPE_OUT") or Path(__file__).resolve().parents[2] / "data" / "corpus_rescrape")
RAW = OUT / "raw"
MIN_INTERVAL = 0.6  # s entre requests de red (<2 req/s)
MAX_PARTS = 400

# entidad -> (slug del sitio, id del documento, tipo, numero, anio, nombre)
# El id replica el de los JSON existentes para que la ingesta REEMPLACE
# el corpus/<doc_id>.txt truncado.
ENTITIES = {
    "codigo_civil": ("codigo_civil", "codigo-civil", "codigo_civil", None, None, "CODIGO CIVIL"),
    "codigo_comercio": ("codigo_comercio", "codigo-comercio", "codigo_comercio", None, None, "CODIGO DE COMERCIO"),
    "codigo_sustantivo_trabajo": ("codigo_sustantivo_trabajo", "codigo-sustantivo-trabajo",
                                  "codigo_sustantivo_trabajo", None, None, "CODIGO SUSTANTIVO DEL TRABAJO"),
    "codigo_contencioso_administrativo": ("codigo_contencioso_administrativo", "codigo-contencioso-administrativo",
                                          "codigo_contencioso_administrativo", None, None,
                                          "CODIGO CONTENCIOSO ADMINISTRATIVO"),
    "estatuto_tributario": ("estatuto_tributario", "decreto-624-1989", "decreto", "624", "1989",
                            "ESTATUTO TRIBUTARIO"),
    "ley_1564_2012": ("ley_1564_2012", "ley-1564-2012", "ley", "1564", "2012", "CODIGO GENERAL DEL PROCESO"),
    "ley_100_1993": ("ley_0100_1993", "ley-100-1993", "ley", "100", "1993", "SISTEMA DE SEGURIDAD SOCIAL INTEGRAL"),
    "ley_80_1993": ("ley_0080_1993", "ley-80-1993", "ley", "80", "1993", "ESTATUTO GENERAL DE CONTRATACION"),
    "ley_1437_2011": ("ley_1437_2011", "ley-1437-2011", "ley", "1437", "2011", "CPACA"),
    "ley_599_2000": ("ley_0599_2000", "ley-599-2000", "ley", "599", "2000", "CODIGO PENAL"),
    "ley_906_2004": ("ley_0906_2004", "ley-906-2004", "ley", "906", "2004", "CODIGO DE PROCEDIMIENTO PENAL"),
    "ley_1098_2006": ("ley_1098_2006", "ley-1098-2006", "ley", "1098", "2006", "CODIGO DE LA INFANCIA Y LA ADOLESCENCIA"),
    "ley_1480_2011": ("ley_1480_2011", "ley-1480-2011", "ley", "1480", "2011", "ESTATUTO DEL CONSUMIDOR"),
    "ley_472_1998": ("ley_0472_1998", "ley-472-1998", "ley", "472", "1998", "ACCIONES POPULARES Y DE GRUPO"),
}

_last_req = 0.0
_session = requests.Session()
_session.headers["User-Agent"] = UA


def fetch(slug: str, part: int) -> str | None:
    """HTML de una parte (cache en disco). None si 404 (no hay mas partes)."""
    global _last_req
    name = slug if part == 0 else f"{slug}_pr{part:03d}"
    cache = RAW / f"{name}.html"
    miss = RAW / f"{name}.404"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    if miss.exists():
        return None
    for attempt in range(6):
        wait = MIN_INTERVAL - (time.time() - _last_req)
        if wait > 0:
            time.sleep(wait)
        _last_req = time.time()
        try:
            r = _session.get(f"{BASE}{name}.html", timeout=60)
        except requests.RequestException as e:
            print(f"  retry {name}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            RAW.mkdir(parents=True, exist_ok=True)
            miss.write_text("", encoding="utf-8")
            return None
        if r.status_code == 200:
            text = r.content.decode("latin-1")  # meta charset ISO-8859-1
            RAW.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return text
        print(f"  retry {name}: HTTP {r.status_code}", file=sys.stderr)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"no se pudo bajar {name}")


# --- parseo -----------------------------------------------------------------

ART_RE = re.compile(r"^\W*(?i:ART[IÍ]CULO)\s+(\d+(?:[A-Z]{1,2})?(?:-\d+)?)\s*(?:[oº°](?![A-Za-z]))?\s*\.?\s*")
ART_WORD_RE = re.compile(r"^\W*ART[IÍ]CULOS?\s+[A-ZÁÉÍÓÚÑ]+\b", re.U)  # ARTICULO PRIMERO/ TRANSITORIO
STRUCT_RE = re.compile(r"^(LIBRO|T[IÍ]TULO|CAP[IÍ]TULO|SECCI[OÓ]N|PARTE|SUBSECCI[OÓ]N|SUBT[IÍ]TULO)\b", re.I)
NIVEL = {"libro": 0, "parte": 0, "titulo": 1, "subtitulo": 2, "capitulo": 3, "seccion": 4, "subseccion": 5}
EDITOR_NOTE_RE = re.compile(r"^\s*(<[^<>]*>)\s*")


def _strip_acc(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\{([^{}]*)\}", r"\1", s)  # {empleador} -> empleador (marca del editor)
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


def parse_part(text: str) -> list[tuple[str, str, object]]:
    """Lista de eventos en orden: ('art', numero, (etiqueta, nombre, cuerpo))
    | ('struct', etiqueta, nombre) | ('para', texto, None) | ('stop', '', None)."""
    soup = BeautifulSoup(text, "lxml")
    logo = soup.find(id="logo_aj")
    if logo:
        logo.decompose()
    for t in soup.find_all("table", id=re.compile(r"^Table\d+$")):
        t.decompose()
    ev = []
    for el in soup.find_all(["p", "table", "ul", "ol"]):
        if el.find_parent(["p", "table", "ul", "ol"]):
            continue
        txt = _clean(el.get_text(" ", strip=False))
        txt = re.sub(r"\s+([.,;:)])", r"\1", txt)
        txt = re.sub(r"\(\s+", "(", txt)
        if not txt or re.fullmatch(r"(Anterior|Siguiente)( \| (Anterior|Siguiente))?", txt):
            continue
        if txt.startswith("-----") or txt.startswith("<Notas de Pie"):
            ev.append(("stop", "", None))
            continue
        m = ART_RE.match(txt)
        if m and el.name == "p":
            num = m.group(1)
            rest = txt[m.end():]
            etiqueta = f"ARTÍCULO {num}" + ("o." if num.isdigit() and "o." in txt[:m.end()] else ".")
            # nombre: el texto del ancla de encabezado, si existe (en mayusculas); si no, vacio
            nombre = ""
            anchor = el.find(["a", "span"])
            if anchor is not None:
                a_txt = _clean(anchor.get_text(" "))
                am = ART_RE.match(a_txt)
                if am and a_txt[am.end():].strip():
                    nombre = a_txt[am.end():].strip()
                    raw_nombre = a_txt[am.end():].strip()
                    if rest.startswith(raw_nombre):
                        rest = rest[len(raw_nombre):].lstrip(". ").lstrip()
            ev.append(("art", num, (etiqueta, nombre.rstrip(". ").strip().strip("<>").strip(), rest)))
            continue
        if ART_WORD_RE.match(txt):
            ev.append(("stop", "", None))
            continue
        if "centrado" in (el.get("class") or []) and STRUCT_RE.match(txt) and len(txt) < 120:
            ev.append(("struct", txt, None))
            continue
        ev.append(("para", txt, None))
    return ev


def build_articles(parts_html: list[str]):
    articulos, estructura, loc = [], [], {}
    cur = None
    last_struct = None  # ultimo nodo estructural sin nombre aun
    seen_ids: dict[str, int] = {}

    def close():
        nonlocal cur
        if cur is not None:
            body = "\n".join(cur.pop("_lines")).strip()
            notas = []
            while True:
                m = EDITOR_NOTE_RE.match(body)
                if not m or len(m.group(1)) > 600:
                    break
                notas.append(html.unescape(m.group(1)))
                body = body[m.end():]
            cur["texto"] = body.strip()
            cur["anotaciones"] = {"notas_inline": [{"texto": n} for n in notas]} if notas else {}
            articulos.append(cur)
        cur = None

    for part in parts_html:
        for kind, a, b in parse_part(part):
            if kind == "art":
                close()
                etiqueta, nombre, rest = b
                n = seen_ids.get(a, 0)
                seen_ids[a] = n + 1
                aid = f"art-{a}" if n == 0 else f"art-{a}-dup{n}"
                cur = {"id": aid, "numero": a, "etiqueta": etiqueta, "nombre": nombre, "tipo": "permanente",
                       "ubicacion": [f"{s['etiqueta']} {s['nombre']}".strip() for s in estructura_stack(loc)],
                       "_lines": [rest] if rest else [], "referencias": []}
                for s in loc.values():
                    s["articulos"].append(aid)
                last_struct = None
            elif kind == "stop":
                close()
                last_struct = None
            elif kind == "struct":
                close()
                nivel = NIVEL.get(_strip_acc(a.split()[0]).rstrip("."), 6)
                for k in [k for k in loc if k >= nivel]:
                    del loc[k]
                m = re.match(r"^(\S+\s+(?:[IVXLCDM]+|\d+|\w+)\b\.?)\s*(.*)$", a)
                label, nm = (m.group(1), m.group(2)) if m else (a, "")
                node = {"nivel": _strip_acc(a.split()[0]), "etiqueta": label, "nombre": nm, "articulos": []}
                loc[nivel] = node
                estructura.append(node)
                last_struct = node
            else:  # para
                if cur is not None:
                    cur["_lines"].append(a)
                elif last_struct is not None and not last_struct["nombre"] and len(a) < 200:
                    last_struct["nombre"] = a
    close()
    return articulos, estructura


def estructura_stack(loc):
    return [loc[k] for k in sorted(loc)]


def scrape(entity: str) -> dict:
    slug, doc_id, tipo, numero, anio, nombre = ENTITIES[entity]
    parts_html, parts = [], []
    p = 0
    while p < MAX_PARTS:
        h = fetch(slug, p)
        if h is None:
            break
        parts_html.append(h)
        parts.append(p)
        p += 1
    if not parts_html:
        raise RuntimeError(f"{entity}: ninguna parte disponible ({slug})")
    articulos, estructura = build_articles(parts_html)
    doc = {"id": doc_id, "tipo": tipo, "numero": numero, "anio": anio, "fecha": None, "nombre": nombre,
           "epigrafe": None, "anotaciones": {}, "estructura": estructura, "articulos": articulos,
           "completo": True, "partes_incluidas": parts, "referencias": [], "citada_por": [],
           "fuente_url": f"{BASE}{slug}.html"}
    return {"meta": {"fuente": "Secretaría del Senado de la República de Colombia - Leyes desde 1992 "
                                "(Avance Jurídico Casa Editorial); re-scrape completo multi-parte",
                     "derechos": "Notas de vigencia, concordancias y demás valores agregados protegidos por "
                                 "derechos de autor; uso no comercial.",
                     "entidad": entity, "total_documentos": 1},
            "documentos": [doc]}


def main(argv: list[str]) -> int:
    ents = argv or list(ENTITIES)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for e in ents:
        try:
            data = scrape(e)
        except Exception as ex:  # noqa: BLE001
            print(f"[{e}] FALLO: {ex}", file=sys.stderr)
            continue
        d = data["documentos"][0]
        (OUT / f"{e}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{e}] partes={len(d['partes_incluidas'])} articulos={len(d['articulos'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
