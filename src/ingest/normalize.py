"""Limpieza de texto y trazabilidad de fragmentos (SPEC.md seccion 3 y 6).

assert_traceable() es el guardian de la senal de puntuacion mas consecuente
del reto: un fragmento que no puede ligarse a su norma de origen vale cero en
respaldo de citas aunque la recuperacion haya sido correcta.
"""
from __future__ import annotations

import re
import unicodedata

import src  # noqa: F401  (registra scripts/ en sys.path)
import citations  # scripts/citations.py, oficial, sin modificar

_WS_RE = re.compile(r"[ \t ]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_SOFT_BREAK_RE = re.compile(r"(?<=[a-zaeiouáéíóúñ,])\n(?=[a-z])")


def clean_text(raw: str) -> str:
    """Normaliza un texto crudo (HTML-a-texto o extraccion de PDF).

    - Unifica a UTF-8 / NFC, comillas y guiones.
    - Colapsa espacios horizontales repetidos.
    - Une saltos de linea que partieron una oracion a mitad de palabra
      (tipico de la extraccion de PDF con columnas), conservando los saltos
      de parrafo reales (linea en blanco).
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = _WS_RE.sub(" ", text)
    text = _SOFT_BREAK_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def prefix_with_norm_name(chunk: str, doc_meta: dict) -> str:
    """Antepone el nombre canonico de la norma a un fragmento.

    Decision de diseno documentada en SPEC.md / Ejemplo de entrega/CORPUS.md:
    sin este prefijo, evaluate.citas_respaldadas() no puede ligar una cita de
    la respuesta con el pasaje recuperado, y el respaldo de citas se anula
    aunque la recuperacion haya sido correcta.
    """
    titulo = doc_meta.get("titulo") or doc_meta.get("doc_id", "")
    if norm_prefix_present(chunk):
        return chunk
    return f"{titulo}. {chunk}".strip()


def norm_prefix_present(chunk: str) -> bool:
    """True si el fragmento ya menciona su norma de origen (evita doble prefijo)."""
    return bool(citations.extract(chunk[:200]))


def assert_traceable(chunk: str, doc_meta: dict) -> None:
    """Lanza ValueError si un fragmento no produce ninguna cita normativa
    reconocible cerca de su inicio, tal como lo exige traceability_check.py.
    """
    head = chunk[:200]
    if not citations.extract(head):
        raise ValueError(
            "fragmento no trazable a su norma de origen "
            f"(doc_id={doc_meta.get('doc_id')!r}): {head[:80]!r}...")
