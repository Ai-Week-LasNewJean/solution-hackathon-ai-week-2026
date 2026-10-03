"""Parseo HTML de las dos fuentes principales de `data/seed_targets.json`
(SPEC.md seccion 3): Secretaria del Senado (`basedoc`) y SUIN-Juriscol.

NOTA para quien retome esto: los selectores de abajo estan escritos a partir
de la estructura publica conocida de ambos sitios, pero no se pudieron
verificar contra una descarga real en esta sesion (sin acceso de red a
*.gov.co desde este entorno). Antes de correr build_corpus.py contra el
corpus real, bajar una pagina de muestra de cada fuente y confirmar con
`ARTICLE_CONTAINER_HINTS` / `_looks_like_boilerplate()` que el texto
recuperado no incluye menu de navegacion. El regex de segmentacion
(`segment.segment_by_article`) opera sobre texto plano, asi que un selector
de contenedor imperfecto todavia produce buenos resultados mientras no
arrastre el menu.
"""
from __future__ import annotations

from bs4 import BeautifulSoup

from src.ingest.normalize import clean_text
from src.ingest.segment import segment_by_article

# Contenedores candidatos a texto principal, en orden de preferencia. El
# primero que exista y tenga contenido razonable (> 500 caracteres) se usa.
ARTICLE_CONTAINER_HINTS = [
    {"id": "textoBasedoc"},
    {"id": "listado"},
    {"class_": "Section1"},
    {"id": "content"},
    {"class_": "contenido"},
]

_BOILERPLATE_LINES = {
    "inicio", "imprimir", "descargar pdf", "compartir", "buscar", "menu",
    "iniciar sesion", "registrarse", "mapa del sitio", "contactenos",
}


def _looks_like_boilerplate(line: str) -> bool:
    return line.strip().lower() in _BOILERPLATE_LINES or len(line.strip()) < 2


def _extract_main_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    container = None
    for hint in ARTICLE_CONTAINER_HINTS:
        found = soup.find(attrs=hint) if "class_" not in hint else soup.find(class_=hint["class_"])
        if found is not None and len(found.get_text(strip=True)) > 500:
            container = found
            break
    if container is None:
        container = soup.body or soup

    lines = [ln for ln in container.get_text("\n").split("\n") if not _looks_like_boilerplate(ln)]
    return "\n".join(lines)


def parse_senado_basedoc(html: str, doc_meta: dict) -> list[dict]:
    """Codigos y leyes publicados en secretariasenado.gov.co/senado/basedoc.

    Devuelve una lista de ArticleRecord: {"doc_id", "articulo", "texto",
    "inicio", "fin"}, uno por articulo detectado.
    """
    soup = BeautifulSoup(html, "lxml")
    text = clean_text(_extract_main_text(soup))
    return [{"doc_id": doc_meta["doc_id"], **c} for c in segment_by_article(text)]


def parse_suin(html: str, doc_meta: dict) -> list[dict]:
    """Normas publicadas en suin-juriscol.gov.co. Misma estrategia que
    parse_senado_basedoc: el sitio cambia de plantilla con cierta frecuencia,
    asi que la robustez viene del regex de articulo sobre texto plano, no de
    un selector fragil.
    """
    soup = BeautifulSoup(html, "lxml")
    text = clean_text(_extract_main_text(soup))
    return [{"doc_id": doc_meta["doc_id"], **c} for c in segment_by_article(text)]
