# Bitácora del corpus — Las NewJeans

> Estado tras la primera pasada de `src/ingest/ingest_raw_sources.py` (2026-10-01)
> sobre el volcado de `data/corpus/` (Scrapping de secretariasenado.gov.co +
> Providencias de la Corte Constitucional + Providencias Corte Suprema, ~700MB
> crudos, no versionados -- ver `.gitignore`). `data/corpus/` no viene del flujo
> fetch+parse_html/parse_pdf de `build_corpus.py` (ya trae JSON pre-parseado o
> pre-troceado por fuente); `ingest_raw_sources.py` es el script nuevo que lo
> normaliza al mismo `corpus/<doc_id>.txt` + `corpus_manifest.json` +
> `indice/chunks.jsonl` que el resto del pipeline espera. La sección 3 (método)
> ya reflejaba el diseño implementado en `src/ingest/`; ahora documenta tambien
> la ingesta de estas tres fuentes.

---

## 1. Inventario

34 379 documentos -- demasiados para una tabla en Markdown. El inventario
completo (doc_id, título, fuente, URL, fecha de consulta, artículos, áreas)
vive en `corpus_manifest.json` (versionado, ~21MB); `manifest_check.py`
verifica que cada fila tenga sha256 válido contra `corpus/<doc_id>.txt`.

**Totales**

| Métrica                               |  Valor |
|----------------------------------------|-------:|
| Documentos incorporados                | 34 379 |
| Fragmentos en el índice (chunks.jsonl) | 99 043 |
| — de Secretaría del Senado (scraping)  |  2 627 |
| — de Corte Constitucional (relatoría)  | 31 413 |
| — de Corte Suprema (ya troceados)      |    339 |
| Trazabilidad (`traceability_check.py`) |  100% (umbral 98%) |
| Tamaño del corpus procesado (`corpus/`)|  ~228MB |

Nota sobre Corte Suprema: en `data/corpus/Providencias Corte Suprema/` cada
sentencia trae `.txt` y `.docx` (y a veces `.pdf`) del mismo documento; la
ingesta usa el `chunks_csj.jsonl` ya troceado por el equipo a partir de los
`.txt`, nunca los `.docx`/`.pdf` duplicados.

**Fragmentos descartados por no ser trazables a su norma de origen** (ver
`is_traceable()` en `ingest_raw_sources.py` y sección 3): 17 030 providencias
de la Corte Constitucional (en su mayoría Autos -- `citations.py` no
reconoce la sala "A" sola, solo "au") + 347 fragmentos de Corte Suprema
(sala "STP", tampoco reconocida) + un puñado de códigos sin alias en
`citations.CODES` (Código Contencioso Administrativo pre-2011, Estatuto
Orgánico del Sistema Financiero, Reglamento CNE, directivas presidenciales).
`citations.py` es oficial y no se modifica, así que ese contenido
simplemente no entra al índice: sin cita reconocible nunca puntuaría en
respaldo de citas, y solo diluiría la recuperación.

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

Implementado en `src/ingest/`; `build_corpus.py` orquesta los pasos 1-6 para
documentos descargados como HTML/PDF crudo.

**`data/corpus/` (volcado del equipo, 2026-09-30/10-01) es otro caso:** ya
viene como JSON pre-parseado o pre-troceado por fuente, así que
`src/ingest/ingest_raw_sources.py` (`python -m src.ingest.ingest_raw_sources`)
reemplaza los pasos 1-2 de arriba para esas tres fuentes y conecta con los
pasos 3-6 igual:

- *Scrapping de secretariasenado.gov.co* ya trae los documentos segmentados
  por artículo (`articulos: [{etiqueta, texto, ...}]`); el script arma el
  encabezado citable (`LEY N DE AAAA`, `Código Civil`, etc.) a partir de
  `tipo`+`numero`+`anio` en vez de confiar en el campo `nombre` del scraper
  (que a veces no calza con lo que `citations.py` reconoce -- p.ej.
  `"DECRETO <LEY> 1088 DE 1993"` con corchetes, o `nombre="ACLARACION"` en
  vez del nombre de la ley). `sentencias.json` se descarta: sus 5737
  registros traen `"contenido": null` -- es un grafo de citas, no texto.
- *Providencias de la Corte Constitucional* trae metadatos de relatoría
  (tema/resumen/resuelve), no el texto íntegro de la providencia; un chunk
  por providencia a partir de esos tres campos. Los 14 archivos por rango de
  años se solapan en 1997-1998 -- se deduplica por "Número de la
  providencia" global.
- *Providencias Corte Suprema* ya viene troceada en `chunks_csj.jsonl` en el
  esquema que `build_faiss.py`/`build_bm25.py` esperan; se usa tal cual.

En los tres casos, `is_traceable()` descarta cualquier fragmento donde
`citations.extract()` no reconozca una norma de origen en los primeros 200
caracteres (ver sección 1 para el detalle de qué se perdió y por qué).

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

| Fecha      | Documentos | Fragmentos | Cerradas /20 | Citación /20 | Abstención /10 | Total /50 | Qué cambió                                                                                                   |
|------------|-----------:|-----------:|-------------:|-------------:|---------------:|----------:|---------------------------------------------------------------------------------------------------------------|
| 2026-10-01 |     34 379 |     99 067 |         13.33 |          7.35 |            5.93 |     26.61 | Corpus inicial, primera corrida de punta a punta, ningún umbral/prompt ajustado todavía.                     |
| 2026-10-02 |     34 379 |     99 067 |         13.33 |          7.35 |            5.81 |     26.49 | `MAX_TOKENS_BY_FORMAT["multiple_choice"]` 320→700 (ítem 128 ya no trunca). Reorden de llaves del schema `multiple_choice` (SPEC.md 10.4 #5) probado y **revertido**: ver nota abajo. |

Lectura de la curva:

- **Reorden de llaves de `multiple_choice` (SPEC.md 10.2 #1 / 10.4 #5), probado y revertido.**
  La hipótesis del diagnóstico original (comprometerse con `respuesta_correcta` antes de razonar
  causaba los fallos 290/308) era correcta para el ítem 308 — moverlo después de
  `justificacion`/`descarte_opciones` en la gramática GBNF y el few-shot sí lo corrigió. Pero medido
  contra `evaluate.py --split sample` el cambio completo dio **6/15 en cerradas, no 10/15 o más**: 5
  ítems que antes acertaban (58, 60, 487, 647, 748) pasaron a fallar, con contenido generado
  completamente distinto (otra ley, otro artículo) y no solo la letra reordenada. Con
  `temperatura=0`/greedy, reescribir el orden de llaves en el few-shot y la gramática cambia la
  secuencia de tokens del prompt lo suficiente para que la decodificación diverja desde el primer
  token generado — no es un ajuste "seguro" aunque el razonamiento detrás sea válido. Se revirtió
  `multiple_choice.gbnf` y `prompts/multiple_choice.py` a su forma original
  (`respuesta_correcta` primero). El diagnóstico de 10.2 sigue siendo válido como explicación de la
  causa — la mitigación concreta (reordenar llaves) no es la que funciona; si se retoma, probar algo
  más quirúrgico (p.ej. un campo de razonamiento *separado* antes del JSON final, fuera de la
  gramática forzada) y volver a medir antes de asumir que ayuda.
- **Subir el límite de tokens de `multiple_choice` sí es una mejora neta, aislada de lo anterior.**
  Probado solo (gramática/prompt originales + presupuesto subido): los 15 ítems de opción múltiple
  dieron exactamente el mismo resultado que el baseline (10/15, mismas letras en los 14 que no son el
  128) — el presupuesto de tokens es ortogonal al contenido generado. El ítem 128 seguía truncando
  incluso en 480 tokens; se probó manualmente (320→FAIL, 480→FAIL, 650→OK, 800→OK) y se fijó en 700
  con margen. Ya no lanza `FormatError`/aborta en excepción no controlada, aunque la respuesta en sí
  sigue siendo incorrecta (A en vez de D) — un fallo de conocimiento/recuperación distinto, no de
  formato.

## 5. Licencia

El corpus se publicará bajo **CC-BY-4.0**. Los textos normativos colombianos
son de dominio público; la licencia cubre el trabajo de procesamiento,
segmentación y extracción de metadatos realizado por el equipo.
