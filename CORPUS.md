# Bitácora del corpus — Las NewJeans

> Plantilla del paso 5 del enunciado, adaptada al estado actual: el corpus se
> está construyendo en la carpeta compartida de OneDrive del equipo
> (`src/config.RAW_CORPUS_DIR`); todavía no hay documentos procesados. Las
> secciones 1, 2 y 4 se completan al correr `src/ingest/build_corpus.py` y
> `scripts/evaluate.py` contra un corpus real. La sección 3 (método) ya
> refleja el diseño implementado en `src/ingest/`.

---

## 1. Inventario

Un registro por documento incorporado. Debe coincidir con `corpus_manifest.json`.

| doc_id                                                | Título | Fuente | URL | Fecha de consulta | Artículos | Áreas |
|-------------------------------------------------------|--------|--------|-----|-------------------|----------:|-------|
| _(pendiente — se completa al correr build_corpus.py)_ |        |        |     |                   |           |       |

**Totales**

| Métrica                     | Valor |
|-----------------------------|------:|
| Documentos incorporados     |     0 |
| Artículos indexados         |     0 |
| Fragmentos en el índice     |     0 |
| Tamaño del corpus procesado |     — |

## 2. Criterio de selección

Prioridad de ingesta tomada directamente del orden de `items_del_banco` en
`data/seed_targets.json` (ver `SPEC.md` sección 7, plan por días): Constitución (90) → Código General del Proceso (65) →
CST (37) → Estatuto Tributario (35) →
Decisión Andina 486 (23) → Estatuto del Consumidor (13) → Ley 80/1993 (12) →
Ley 2220/2022 (11) → Decreto 2153/1992 (10) → jurisprudencia de mayor ROI (Sentencia C-355/2006 y siguientes por número
de ítems).

| Área                           | Ítems en el banco | Documentos incorporados | Cobertura estimada |
|--------------------------------|------------------:|------------------------:|--------------------|
| Derecho constitucional         |               134 |                       0 | Sin cobertura aún  |
| Derecho administrativo         |               124 |                       0 | Sin cobertura aún  |
| Derecho penal                  |               123 |                       0 | Sin cobertura aún  |
| Derecho procesal               |               111 |                       0 | Sin cobertura aún  |
| Derecho comercial y sociedades |               104 |                       0 | Sin cobertura aún  |
| Derecho civil                  |               102 |                       0 | Sin cobertura aún  |
| Derecho de familia             |                93 |                       0 | Sin cobertura aún  |
| Derecho tributario             |                92 |                       0 | Sin cobertura aún  |
| Derecho laboral                |                87 |                       0 | Sin cobertura aún  |
| Derecho de los mercados        |                72 |                       0 | Sin cobertura aún  |

Documentos descartados y el motivo del descarte:

<!-- Qué se consideró y no se incorporó, y por qué. -->

## 3. Método de ingesta y limpieza

Implementado en `src/ingest/`; `build_corpus.py` orquesta los pasos 1-6.

1. **Descarga.** `fetch.fetch_url()` / `fetch_all()` contra las URL de
   `data/seed_targets.json`, con reintentos y backoff; o lectura directa de
   lo que el equipo ya dejó en la carpeta compartida de OneDrive (`RAW_CORPUS_DIR`).
2. **Extracción de texto.** `parse_html.parse_senado_basedoc()` /
   `parse_suin()` (BeautifulSoup + lxml) para HTML;
   `parse_pdf.extract_text()` (PyMuPDF, fallback pdfplumber) para PDF con
   capa de texto, `ocr_page()` (Tesseract español) solo donde
   `needs_ocr()` lo detecta.
3. **Normalización.** `normalize.clean_text()`: NFC, comillas/guiones
   unificados, colapso de espacios, unión de saltos de línea que parten una
   oración a mitad de palabra.
4. **Segmentación.** `segment.segment_by_article()` (regex
   `ARTÍCULO\s+\d+`, códigos/leyes) o `segment_paragraphs()` (ventana
   deslizante ~250 palabras, 2 oraciones de solapamiento, jurisprudencia).
5. **Extracción de metadatos.** Cada fragmento lleva `doc_id` y, por
   `normalize.prefix_with_norm_name()`, el nombre completo de la norma como
   encabezado — sin esto, `evaluate.citas_respaldadas()` no puede ligar una
   cita de la respuesta con el pasaje recuperado.
6. **Indexación.** `intfloat/multilingual-e5-large` (prefijos
   `query:`/`passage:`, normalización L2) + `faiss.IndexFlatIP` (denso) +
   `bm25s` (disperso, tokenización compartida con `citations.norm()`).

Problemas encontrados y cómo se resolvieron:

<!-- OCR defectuoso, artículos derogados, numeraciones inconsistentes. -->

**Riesgo abierto.** Los selectores de `parse_html.py` se escribieron a
partir de la estructura pública conocida de secretariasenado.gov.co y
suin-juriscol.gov.co, sin poder verificarlos contra una descarga real en el
entorno donde se escribió este scaffolding (sin acceso de red a `*.gov.co`
desde ahí). Primer paso al retomar: bajar una página de cada fuente y
confirmar que `_extract_main_text()` no arrastra menú de navegación.

## 4. Evolución del puntaje

Puntaje sobre las 50 preguntas de muestra, medido con
`python scripts/evaluate.py --submission out/sample.jsonl --split sample`.

| Fecha | Documentos | Fragmentos | Cerradas /20 | Citación /20 | Abstención /10 | Total /50 | Qué cambió     |
|-------|-----------:|-----------:|-------------:|-------------:|---------------:|----------:|----------------|
|       |            |            |              |              |                |           | Corpus inicial |

Lectura de la curva:

<!-- Qué incorporaciones movieron el puntaje y cuáles no. -->

## 5. Licencia

El corpus se publicará bajo **CC-BY-4.0**. Los textos normativos colombianos
son de dominio público; la licencia cubre el trabajo de procesamiento,
segmentación y extracción de metadatos realizado por el equipo.
