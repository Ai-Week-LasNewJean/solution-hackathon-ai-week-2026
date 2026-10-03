"""Normas puntuales ausentes del corpus (SPEC.md de la solucion, seccion 2).

El sabado 2026-10-03, con las 992 preguntas ya publicadas, un analisis de
cobertura (solo referencias normativas explicitas en los enunciados) encontro
normas publicas citadas que el corpus no tenia. Este script baja el TEXTO
OFICIAL de cada una (nunca preguntas ni respuestas), lo parte por articulos y
escribe data/corpus_extra/<doc_id>.json con el mismo esquema que
rescrape_senado.py / parse_decision486.py (meta + documentos[0].articulos), de
modo que ingest_raw_sources.py lo incorpora via RESCRAPE_DIR sin cambios.

Fuentes crudas cacheadas en data/corpus_extra/raw/ (las re-ejecuciones no tocan
la red). La Resolucion 368 de 2014 de MinAmbiente solo existe como PDF
escaneado: el texto viene de OCR (src/ingest/ocr_rapid.py, RapidOCR/ONNX en
CPU) y aqui se le repara el espaciado (segmentacion por unigramas con el
vocabulario del propio corpus) y las tildes.

Uso:
    .venv/bin/python -m src.ingest.fetch_extra_norms            # todas
    .venv/bin/python -m src.ingest.fetch_extra_norms --offline  # solo cache
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src import config

OUT_DIR = config.DATA / "corpus_extra"
RAW_DIR = OUT_DIR / "raw"
UA = "LasNewJeans-hackathon-corpus/1.0 (academic; Universidad de los Andes)"
CANC = "https://www.cancilleria.gov.co/sites/default/files/Normograma/docs"

# (doc_id, tipo, numero, anio, nombre, epigrafe, fuente, url, archivo crudo)
EXTRA_NORMS = [
    ("resolucion-368-2014", "resolucion", "368", 2014,
     "RESOLUCIÓN 368 DE 2014 (Ministerio de Ambiente y Desarrollo Sostenible)",
     "Por la cual se asume la competencia del Proyecto \"Recuperación ambiental del relleno sanitario "
     "El Carrasco\" y se toman otras determinaciones",
     "Ministerio de Ambiente y Desarrollo Sostenible (PDF escaneado, OCR)",
     "https://www.minambiente.gov.co/wp-content/uploads/2021/10/Resolucon-0368-de-2014.pdf",
     "resolucion-368-2014.pdf"),
    ("decreto-175-2025", "decreto", "175", 2025, "DECRETO 175 DE 2025",
     "Por el cual se adoptan medidas tributarias en el marco del Estado de Conmoción Interior",
     "Función Pública - Gestor Normativo",
     "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=259629",
     "decreto-175-2025.html"),
    ("decreto-4302-2008", "decreto", "4302", 2008, "DECRETO 4302 DE 2008",
     "Procedimiento para la declaratoria de existencia de razones de interés público (art. 65 Decisión 486)",
     "Superintendencia de Industria y Comercio (Diario Oficial 47.172)",
     "https://www.sic.gov.co/sites/default/files/normatividad/Decreto_4302_2008.pdf",
     "decreto-4302-2008.pdf"),
    ("decreto-1382-2000", "decreto", "1382", 2000, "DECRETO 1382 DE 2000",
     "Por el cual se establecen reglas para el reparto de la acción de tutela",
     "Normograma de la Cancillería", f"{CANC}/decreto_1382_2000.htm", "decreto-1382-2000.html"),
    ("decreto-333-2021", "decreto", "333", 2021, "DECRETO 333 DE 2021",
     "Por el cual se modifican las reglas de reparto de la acción de tutela (Decreto 1069 de 2015)",
     "Normograma de la Cancillería", f"{CANC}/decreto_0333_2021.htm", "decreto-333-2021.html"),
    ("acuerdo-cc-1-2025", "acuerdo", "01", 2025,
     "ACUERDO 01 DE 2025 (Sala Plena de la Corte Constitucional)",
     "Por medio del cual se unifica y actualiza el Reglamento de la Corte Constitucional",
     "Normograma de la Cancillería",
     "https://www.cancilleria.gov.co/normograma/compilacion/docs/acuerdo_cconstitucional_0001_2025.htm",
     "acuerdo-cc-1-2025.html"),
]

ORDINALES = ["PRIMERO", "SEGUNDO", "TERCERO", "CUARTO", "QUINTO", "SEXTO", "SEPTIMO", "OCTAVO",
             "NOVENO", "DECIMO", "UNDECIMO", "DUODECIMO"]
CONSIDERANDO_WORDS = 220  # ventanas del preambulo (e5 trunca ~512 tokens)


def fetch(url: str, path: Path, offline: bool) -> None:
    if path.exists() or offline:
        return
    import requests
    import urllib3

    urllib3.disable_warnings()
    # verify=False: funcionpublica/suin sirven una cadena TLS incompleta
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60, verify=False)
    r.raise_for_status()
    path.write_bytes(r.content)
    time.sleep(0.5)


def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# --- Reparacion de OCR -------------------------------------------------------

@lru_cache(maxsize=1)
def _vocab() -> tuple[dict[str, float], dict[str, str], int]:
    """Unigramas del corpus congelado (normas de derecho publico): costo
    -log p por palabra sin tildes y la forma con tildes mas frecuente."""
    counts: Counter[str] = Counter()
    accented: dict[str, Counter] = {}
    names = ["constitucion", "codigo-contencioso-administrativo", "ley-1437-2011", "ley-99-1993",
             "ley-142-1994", "ley-1333-2009", "codigo-civil", "ley-80-1993", "ley-1757-2015",
             "ley-489-1998", "ley-1450-2011", "ley-1753-2015", "ley-1955-2019"]
    for name in names:
        p = config.CORPUS_DIR / f"{name}.txt"
        if not p.exists():
            continue
        for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", p.read_text(encoding="utf-8")):
            lw = w.lower()
            k = strip_acc(lw)
            counts[k] += 1
            accented.setdefault(k, Counter())[lw] += 1
    total = sum(counts.values())
    cost = {w: math.log(total / c) for w, c in counts.items() if len(w) > 1 or w in "aeouy"}
    best = {k: v.most_common(1)[0][0] for k, v in accented.items()}
    return cost, best, max(len(w) for w in cost)


def _segment(token: str, extra: dict[str, float]) -> str:
    """Parte "rellenosanitarioel" -> "relleno sanitario el" (DP por unigramas).
    Solo acepta la particion si TODAS las piezas son palabras conocidas (del
    corpus o del propio documento); si no, deja el token intacto (nombres
    propios como "Carrasco" no se trocean)."""
    cost, _, maxlen = _vocab()
    low = strip_acc(token.lower())
    if len(low) < 7 or low in cost or low in extra:
        return token
    best = [(0.0, 0)] + [(math.inf, 0)] * len(low)
    for i in range(1, len(low) + 1):
        for j in range(max(0, i - maxlen), i):
            w = low[j:i]
            c = cost.get(w, extra.get(w))
            if c is not None and best[j][0] + c + 3.0 < best[i][0]:   # +3: penaliza trocear de mas
                best[i] = (best[j][0] + c + 3.0, j)
    if best[-1][0] == math.inf:
        return token
    words, i = [], len(low)
    while i > 0:
        j = best[i][1]
        words.append(token[j:i])
        i = j
    return " ".join(reversed(words))


def _accent(word: str) -> str:
    _, best, _ = _vocab()
    form = best.get(strip_acc(word.lower()))
    if not form or form == word.lower():
        return word
    if word.isupper():
        return form.upper()
    return form.capitalize() if word[0].isupper() else form


_PAGE_HEADER = re.compile(
    r"(?im)^.{0,40}Resol\w*\s*No\.?.{0,60}Hoja\s*N\w*\.?\s*\d+.*\n(?:.{0,15}\n)?"
    r"(?:.*(?:determinaciones|Carrasco\W{0,3}y\s*se\s*toman).*\n)?")


def repair_ocr(text: str) -> str:
    text = re.sub(r"<<PAGINA \d+>>\n", "", text)
    text = _PAGE_HEADER.sub("", text)                         # encabezado repetido de cada hoja
    text = re.sub(r"(?<=[a-zA-Z])6(?=n\b)", "ó", text)        # Resoluci6n -> Resolución
    text = re.sub(r"(?<=[a-z]{3})6\b", "ó", text)              # orden6 -> ordenó
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text)          # prorrog0 / s0lo
    # vocabulario propio: palabras que en el documento aparecen sueltas (nombres propios, siglas)
    own = Counter(strip_acc(w.lower()) for w in re.findall(r"(?<![A-Za-z])[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,14}(?![A-Za-z])", text))
    extra = {w: math.log(1e6 / c) for w, c in own.items() if c >= 2}
    cost = _vocab()[0]

    def margin(m: re.Match) -> str:   # la raya del margen escaneado se pega al renglon: "Ique" -> "que"
        w = m.group(2)
        return w if strip_acc(w.lower()) in cost and strip_acc((m.group(1) + w).lower()) not in cost else m.group(0)

    out_lines = []
    for line in text.splitlines():
        line = re.sub(r"^[\s|丨\]\[」「】!:;,.·]*", "", line)
        line = re.sub(r"^([IJl1|])\s?([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)", margin, line)
        line = re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", lambda m: _segment(m.group(0), extra), line)
        line = re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", lambda m: _accent(m.group(0)), line)
        out_lines.append(line)
    return "\n".join(out_lines)


# --- Texto y articulos -----------------------------------------------------

def html_text(path: Path) -> str:
    from bs4 import BeautifulSoup

    raw = path.read_bytes()
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        s = raw.decode("latin-1")
    soup = BeautifulSoup(s, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    t = soup.get_text("\n")
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def pdf_text(path: Path) -> str:
    from src.ingest.parse_pdf import extract_text

    return "\n".join(extract_text(path))


def source_text(raw: Path) -> str:
    if raw.suffix == ".pdf":
        ocr = raw.with_suffix(".ocr.txt")
        if ocr.exists():
            return repair_ocr(ocr.read_text(encoding="utf-8"))
        return pdf_text(raw)
    return html_text(raw)


_ART_RE = re.compile(
    r"(?m)^[ \t]*(?:ART[IÍ]CULO|Art[ií]culo)[ \t]*(\d+[A-Z]?|" + "|".join(ORDINALES) + r")[ \t]*[oº°]?[ \t]*[.:\-]")


def _numero(tok: str) -> str:
    tok = strip_acc(tok.upper())
    return str(ORDINALES.index(tok) + 1) if tok in ORDINALES else tok


def _windows(text: str, n: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def parse_articles(text: str, numero_padre: str) -> list[dict]:
    """Cada ARTICULO a linea propia inicia un articulo; el preambulo (desde el
    epigrafe, incluidos los considerandos) va en ventanas "CONSIDERANDOS".
    Los articulos citados dentro de considerandos no inician linea, y si un
    numero se repite (articulo modificado de otra norma, p.ej. Decreto 333
    que reescribe el 2.2.3.1.2.1) se conserva el primero."""
    # solo encabezados en mayusculas (en el OCR "ARTICULOQUINTO" pierde el espacio);
    # "articulo quinto de la Resolucion ..." en minuscula es una referencia, no un articulo
    text = re.sub(r"ART[IÍ]CULO\s*(" + "|".join(ORDINALES) + r")",
                  lambda m: "\nARTÍCULO " + m.group(1), text)
    text = re.sub(r"(?m)^\s*ART[IíÍ]CU[LI]O\s*(\d+)", r"ARTÍCULO \1", text, flags=re.I)  # OCR: "ARTicuLo 1o"
    marks = list(_ART_RE.finditer(text))
    arts: list[dict] = []
    pre = text[: marks[0].start()] if marks else text
    starts = [i for i in (pre.upper().find(k) for k in ("POR EL CUAL", "POR LA CUAL", "POR MEDIO")) if i >= 0]
    pre = pre[min(starts):] if starts else pre
    for k, w in enumerate(_windows(pre, CONSIDERANDO_WORDS)):
        arts.append({"id": f"pre-{k + 1}", "numero": None, "etiqueta": "CONSIDERANDOS", "nombre": "",
                     "tipo": "permanente", "ubicacion": [], "referencias": [], "anotaciones": {},
                     "texto": w})
    seen: set[str] = set()
    for m, nxt in zip(marks, marks[1:] + [None]):
        num = _numero(m.group(1))
        body = text[m.end(): nxt.start() if nxt else len(text)].strip()
        body = re.sub(r"\s+", " ", body)
        if not body or num in seen:
            continue
        seen.add(num)
        etiqueta = f"ARTÍCULO {num}" + (f" ({strip_acc(m.group(1).upper())})" if not m.group(1)[0].isdigit() else "")
        # articulos largos (reglamento de la Corte): ventanas con la etiqueta repetida
        for k, w in enumerate(_windows(body, 300)):
            arts.append({"id": f"art-{num}" + (f"-{k + 1}" if k else ""), "numero": num,
                         "etiqueta": etiqueta + (" (cont.)" if k else "."), "nombre": "",
                         "tipo": "permanente", "ubicacion": [], "referencias": [], "anotaciones": {},
                         "texto": w})
    return arts


def build(offline: bool) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for doc_id, tipo, numero, anio, nombre, epigrafe, fuente, url, fname in EXTRA_NORMS:
        raw = RAW_DIR / fname
        fetch(url, raw, offline)
        if not raw.exists():
            print(f"[extra] {doc_id}: sin fuente cruda, se omite")
            continue
        arts = parse_articles(source_text(raw), numero)
        # tipo "ley"/"decreto" -> titulo canonico via TIPO_LABEL_NUMBERED; el resto usa "nombre"
        doc = {"id": doc_id, "tipo": tipo, "numero": numero, "anio": anio, "nombre": nombre,
               "epigrafe": epigrafe, "anotaciones": {"fuente": fuente, "url": url},
               "estructura": [], "articulos": arts, "completo": True, "partes_incluidas": [0]}
        out = OUT_DIR / f"{doc_id}.json"
        out.write_text(json.dumps({"meta": {"fuente": fuente, "entidad": doc_id, "url": url,
                                            "total_documentos": 1}, "documentos": [doc]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        n_art = sum(1 for a in arts if a["numero"])
        print(f"[extra] {doc_id}: {n_art} fragmentos de articulos, {len(arts) - n_art} de preambulo -> {out.name}")
        written.append(out)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="no descargar; solo usar data/corpus_extra/raw")
    build(ap.parse_args().offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
