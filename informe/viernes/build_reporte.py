"""Genera REPORTE_AVANCE.pdf (una pagina) desde viernes/reporte.json -> HTML -> PDF (LibreOffice)."""
import json, subprocess, sys, html
from pathlib import Path
here = Path(__file__).parent
d = json.loads((here / "reporte.json").read_text(encoding="utf-8"))
e = html.escape
def tbl(rows, head=None, widths=None):
    h = "".join(f"<th>{e(c)}</th>" for c in head) if head else ""
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{('<tr>'+h+'</tr>') if h else ''}{b}</table>"
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 1.2cm 1.5cm; }}
body {{ font-family: 'Liberation Sans', Arial, sans-serif; font-size: 8.3pt; line-height: 1.22; }}
h1 {{ font-size: 13pt; margin: 0 0 1pt 0; }} h2 {{ font-size: 9.6pt; margin: 6pt 0 2pt 0; border-bottom: 0.6pt solid #888; }}
p {{ margin: 1pt 0; }} table {{ border-collapse: collapse; width: 100%; margin: 1pt 0; }}
td, th {{ border: 0.5pt solid #999; padding: 1pt 3pt; vertical-align: top; text-align: left; }} th {{ background: #e8e8e8; }}
ol {{ margin: 1pt 0 1pt 12pt; padding: 0; }} li {{ margin: 0 0 1pt 0; }}
</style></head><body>
<h1>Reporte de avance — Hackathon 2026</h1>
<p><b>Equipo:</b> {e(d['equipo'])} &nbsp; <b>Integrantes:</b> {e(d['integrantes'])} &nbsp; <b>Fecha de la medición:</b> {e(d['fecha'])}</p>
<h2>1. Puntaje sobre las preguntas de muestra</h2>
<p><code>python scripts/evaluate.py --submission entrega.jsonl --split sample [--ragas]</code> — {e(d['config_medida'])}</p>
{tbl(d['puntaje'], ['Componente','Puntos obtenidos','Puntos posibles'])}
<p>{d['obs_puntaje']}</p>
<h2>2. Estado del corpus</h2>
{tbl(d['corpus'], ['Métrica','Valor'])}
<p><b>Fuentes consultadas:</b> {d['fuentes']}</p>
<h2>3. Arquitectura actual</h2>
{tbl(d['arquitectura'], ['Componente','Elección'])}
<h2>4. Riesgos identificados</h2>
<ol>{''.join(f'<li>{r}</li>' for r in d['riesgos'])}</ol>
</body></html>"""
(here / "REPORTE_AVANCE.html").write_text(page, encoding="utf-8")
subprocess.run(["soffice", "--headless", "--convert-to", "pdf:writer_web_pdf_Export", "--outdir", str(here), str(here / "REPORTE_AVANCE.html")], check=True, capture_output=True)
print((here / "REPORTE_AVANCE.pdf"))
