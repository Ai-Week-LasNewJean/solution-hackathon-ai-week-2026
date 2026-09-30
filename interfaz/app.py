"""Interfaz grafica Streamlit. Llama a src.pipeline.answer_one.answer()
directamente -- SPEC.md seccion 4. Lanzar con:

    streamlit run interfaz/app.py

TODO (identidad visual de Software Colombia, 10 pts manuales): reemplazar la
paleta de PLACEHOLDER_COLORS por la paleta oficial una vez publicada, y
anadir el logo en st.set_page_config / el encabezado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.answer_one import answer  # noqa: E402

PLACEHOLDER_COLORS = {"primary": "#0B3D91", "accent": "#FFB612"}

st.set_page_config(page_title="RAG Derecho Colombiano", page_icon="⚖️", layout="wide")

st.markdown(
    f"<h1 style='color:{PLACEHOLDER_COLORS['primary']}'>Sistema RAG de derecho colombiano</h1>"
    "<p>AI Week Hackathon 2026 &mdash; Equipo Las NewJeans</p>",
    unsafe_allow_html=True,
)

formato = st.selectbox("Formato de la pregunta", ["multiple_choice", "semi_open", "open_ended"])
pregunta = st.text_area("Pregunta", height=100)

opciones: dict[str, str] = {}
if formato == "multiple_choice":
    cols = st.columns(4)
    for letra, col in zip("ABCD", cols):
        opciones[letra] = col.text_input(f"Opcion {letra}")

if st.button("Responder", type="primary", disabled=not pregunta.strip()):
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
        for campo in ("respuesta_correcta", "justificacion", "respuesta", "referencia_legal",
                      "marco_normativo", "analisis", "jurisprudencia", "conclusion"):
            if campo in result:
                st.markdown(f"**{campo}:** {result[campo]}")

    st.subheader(f"Pasajes recuperados ({len(result.get('pasajes_recuperados', []))})")
    for i, p in enumerate(result.get("pasajes_recuperados", []), start=1):
        with st.expander(f"[{i}] {p['doc_id']} (score={p.get('score', 0):.3f})"):
            st.write(p["texto"])

    st.caption(f"Latencia: {result.get('latencia_ms', '?')} ms")
