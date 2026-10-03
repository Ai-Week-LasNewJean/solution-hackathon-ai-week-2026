"""OCR de PDFs escaneados con RapidOCR (ONNX, solo CPU) -> <pdf>.ocr.txt.

Se usa para la Resolucion 368 de 2014 de MinAmbiente (22 paginas escaneadas,
sin capa de texto; tesseract no esta instalado en Turing). El detector de
RapidOCR perdia renglones justificados enteros, asi que los renglones se
separan por perfil de proyeccion horizontal (pagina limpia a una columna) y
cada franja se pasa solo por el reconocedor (use_det=False).

Corre en un venv aparte para no tocar el venv CUDA de la solucion:
    uv venv ocrenv --python 3.11
    uv pip install --python ocrenv/bin/python rapidocr-onnxruntime==1.4.4 pymupdf opencv-python-headless
    CUDA_VISIBLE_DEVICES= ocrenv/bin/python src/ingest/ocr_rapid.py data/corpus_extra/raw/resolucion-368-2014.pdf
La reparacion de espaciado/tildes la hace fetch_extra_norms.repair_ocr.
"""
import sys
import unicodedata

import cv2
import numpy as np
import pymupdf as fitz
from rapidocr_onnxruntime import RapidOCR

INK = 140        # gris < INK es tinta
ROW_MIN = 25     # pixeles de tinta por fila para contar como renglon (el borde escaneado aporta ~4)
MIN_H = 8


def page_lines(eng: RapidOCR, page, dpi: int = 200) -> list[str]:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    on = (img < INK).sum(axis=1) > ROW_MIN
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= MIN_H:
                runs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for a, b in runs:
        strip = img[max(0, a - 4): b + 4]
        cols = np.where((strip < INK).sum(axis=0) > 0)[0]
        strip = strip[:, max(0, cols[0] - 6): cols[-1] + 6]
        res, _ = eng(cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR), use_det=False, use_cls=False)
        if res and float(res[0][1]) >= 0.5:
            t = unicodedata.normalize("NFKC", res[0][0]).strip()
            out.append(t.lstrip("」」]|!！I—-_ ").strip() if t[:1] in "」]|!！—" else t)
    return out


def main(pdf: str) -> None:
    eng = RapidOCR()
    pages = []
    for i, page in enumerate(fitz.open(pdf)):
        lines = page_lines(eng, page)
        pages.append(f"<<PAGINA {i + 1}>>\n" + "\n".join(lines))
        print(i + 1, len(lines), file=sys.stderr, flush=True)
    with open(pdf.rsplit(".", 1)[0] + ".ocr.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pages))


if __name__ == "__main__":
    main(sys.argv[1])
