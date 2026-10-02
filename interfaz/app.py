"""Interfaz grafica Streamlit. Llama a src.pipeline.answer_one.answer()
directamente -- SPEC.md seccion 4. Lanzar con:

    streamlit run interfaz/app.py

Identidad visual tomada de https://en.software-colombia.com/ (paleta,
tipografia Montserrat y logo oficial en public/cropped-logo.png).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import citation_check  # noqa: E402
from src.pipeline.answer_one import _answer_text_for_check, answer  # noqa: E402

import citations  # noqa: E402  (registrado en sys.path por "import src" arriba)

BRAND = {
    "teal": "#09ACC4",
    "teal_dark": "#07839A",
    "orange": "#FF993B",
    "text": "#495057",
    "text_light": "#ADB5BD",
    "heading": "#212529",
    "bg": "#FFFFFF",
    "bg_soft": "#F7F9FA",
    "border": "#E9ECEF",
}

LOGO_PATH = Path(__file__).resolve().parents[1] / "public" / "cropped-logo.png"

st.set_page_config(page_title="RAG Derecho Colombiano", page_icon=str(LOGO_PATH), layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Montserrat', sans-serif;
        color: {BRAND["text"]};
    }}

    .block-container {{
        padding-top: 2rem;
        max-width: 1100px;
    }}

    section[data-testid="stSidebar"] {{
        display: none;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Montserrat', sans-serif;
        color: {BRAND["heading"]};
        font-weight: 700;
    }}

    .sc-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding-bottom: 10px;
        border-bottom: 3px solid {BRAND["teal"]};
        margin-bottom: 28px;
    }}
    .sc-header img {{ height: 56px; }}
    .sc-header .sc-title h1 {{
        margin: 0;
        font-size: 1.7rem;
        color: {BRAND["heading"]};
    }}
    .sc-header .sc-title p {{
        margin: 2px 0 0 0;
        color: {BRAND["teal_dark"]};
        font-weight: 600;
        letter-spacing: .03em;
        text-transform: uppercase;
        font-size: .78rem;
    }}

    div.st-key-question_card, div.st-key-answer_card {{
        background: {BRAND["bg_soft"]};
        border: 1px solid {BRAND["border"]};
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }}
    div.st-key-answer_card {{
        border-left: 4px solid {BRAND["teal"]};
        border-radius: 0 10px 10px 0;
    }}

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {{
        border-radius: 8px !important;
        border: 1px solid {BRAND["border"]} !important;
    }}
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {{
        border-color: {BRAND["teal"]} !important;
        box-shadow: 0 0 0 1px {BRAND["teal"]} !important;
    }}

    div[data-testid="stButton"] button {{
        background: {BRAND["orange"]};
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: .03em;
        text-transform: uppercase;
        font-size: .85rem;
        padding: 0.6rem 1.4rem;
        transition: background .15s ease-in-out;
    }}
    div[data-testid="stButton"] button:hover {{
        background: #E6822C;
        color: #FFFFFF;
    }}
    div[data-testid="stButton"] button:disabled {{
        background: {BRAND["text_light"]};
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {BRAND["border"]};
        border-radius: 8px;
    }}
    div[data-testid="stExpander"] summary {{
        color: {BRAND["teal_dark"]};
        font-weight: 600;
    }}

    .sc-footer {{
        margin-top: 40px;
        padding-top: 14px;
        border-top: 1px solid {BRAND["border"]};
        color: {BRAND["text_light"]};
        font-size: .8rem;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

import base64  # noqa: E402

logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""

st.markdown(
    f"""
    <div class="sc-header">
        {f'<img src="data:image/png;base64,{logo_b64}" alt="Software Colombia" />' if logo_b64 else ''}
        <div class="sc-title">
            <h1>Sistema RAG de derecho colombiano</h1>
            <p>AI Week Hackathon 2026 &mdash; Equipo Las NewJeans</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(key="question_card"):
    st.markdown("### Haz tu pregunta")
    formato = st.selectbox("Formato de la pregunta", ["multiple_choice", "semi_open", "open_ended"])
    pregunta = st.text_area("Pregunta", height=100)

    opciones: dict[str, str] = {}
    if formato == "multiple_choice":
        cols = st.columns(4)
        for letra, col in zip("ABCD", cols):
            opciones[letra] = col.text_input(f"Opcion {letra}")

    responder = st.button("Responder", type="primary", disabled=not pregunta.strip())

if responder:
    item = {"id": 0, "formato": formato, "pregunta": pregunta}
    if formato == "multiple_choice":
        item["opciones"] = {k: v for k, v in opciones.items() if v.strip()}

    with st.spinner("Recuperando evidencia y generando respuesta..."):
        try:
            result = answer(item)
        except FileNotFoundError as exc:
            st.error(f"Falta un recurso local (modelo GGUF o indice): {exc}")
            st.stop()

    if result.get("abstencion"):
        st.warning("El sistema se abstiene: el corpus no trae fundamento suficiente para esta pregunta.")
    else:
        st.subheader("Respuesta")
        with st.container(key="answer_card"):
            for campo in ("respuesta_correcta", "justificacion", "respuesta", "referencia_legal",
                          "marco_normativo", "analisis", "jurisprudencia", "conclusion"):
                if campo in result:
                    st.markdown(f"**{campo}:** {result[campo]}")

    if not result.get("abstencion"):
        answer_text = _answer_text_for_check(result["formato"], result)
        pasajes = result.get("pasajes_recuperados", [])
        all_cites = citations.article_level(citations.extract(answer_text))
        sin_respaldo = citation_check.unsupported_citations(answer_text, pasajes)

        def _fmt_cita(c: tuple) -> str:
            body, numero, anio, articulo = c
            norma = body.replace("_", " ").title()
            if numero:
                norma += f" {numero}" + (f" de {anio}" if anio else "")
            return f"{norma}, Art. {articulo}" if articulo else norma

        if all_cites:
            st.subheader("Citas detectadas en la respuesta")
            badges = []
            for c in sorted(all_cites, key=_fmt_cita):
                respaldada = c not in sin_respaldo
                color = BRAND["teal"] if respaldada else "#E03131"
                etiqueta = "respaldada" if respaldada else "sin respaldo"
                badges.append(
                    f'<span style="display:inline-block;margin:0.2rem;padding:0.2rem 0.6rem;'
                    f'border-radius:999px;background:{color}1A;color:{color};'
                    f'border:1px solid {color};font-size:0.85rem;">'
                    f'{_fmt_cita(c)} &middot; {etiqueta}</span>'
                )
            st.markdown("".join(badges), unsafe_allow_html=True)

    st.subheader(f"Pasajes recuperados ({len(result.get('pasajes_recuperados', []))})")
    for i, p in enumerate(result.get("pasajes_recuperados", []), start=1):
        with st.expander(f"[{i}] {p['doc_id']} (score={p.get('score', 0):.3f})"):
            st.write(p["texto"])

    st.caption(f"Latencia: {result.get('latencia_ms', '?')} ms")

st.markdown(
    '<div class="sc-footer">Interfaz alineada con la identidad visual de '
    '<a href="https://en.software-colombia.com/" target="_blank" style="color:#09ACC4;">Software Colombia</a></div>',
    unsafe_allow_html=True,
)
