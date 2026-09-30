"""Extraccion de texto de PDF, con OCR selectivo (SPEC.md seccion 3).

OCR se reserva para jurisprudencia escaneada de alto ROI (ver seed_targets.json
/ plan por dias) — nunca se aplica al corpus completo por costo de tiempo.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

MIN_CHARS_PER_PAGE = 40  # por debajo de esto, se asume pagina sin capa de texto


def extract_text(pdf_path: Path) -> list[str]:
    """Texto por pagina usando PyMuPDF. Paginas vacias devuelven "" (needs_ocr
    las detecta despues); no se usa pdfplumber salvo que PyMuPDF falle, por
    velocidad."""
    try:
        with fitz.open(pdf_path) as doc:
            return [page.get_text("text") or "" for page in doc]
    except Exception:
        return _extract_text_pdfplumber(pdf_path)


def _extract_text_pdfplumber(pdf_path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def needs_ocr(page_text: str) -> bool:
    """True si una pagina no trajo suficiente texto extraible (probable escaneo)."""
    return len(page_text.strip()) < MIN_CHARS_PER_PAGE


def ocr_page(pdf_path: Path, page_no: int, lang: str = "spa") -> str:
    """OCR de una sola pagina con Tesseract en espanol. `page_no` es 0-indexado."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path), first_page=page_no + 1, last_page=page_no + 1, dpi=300)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=lang)


def extract_text_with_ocr_fallback(pdf_path: Path, lang: str = "spa") -> str:
    """Texto completo del PDF, recurriendo a OCR pagina por pagina solo donde
    hace falta. Punto de entrada que usa build_corpus.py."""
    pages = extract_text(pdf_path)
    out = []
    for i, page_text in enumerate(pages):
        out.append(ocr_page(pdf_path, i, lang=lang) if needs_ocr(page_text) else page_text)
    return "\n\n".join(out)
