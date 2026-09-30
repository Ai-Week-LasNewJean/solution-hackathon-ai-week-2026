# Informe técnico — Las NewJeans

**Hackathon 2026 · Universidad de los Andes**
**Integrantes:**

Máximo 3 páginas al exportar a PDF. Se entrega como `informe/INFORME_TECNICO.pdf`.

> Borrador generado junto con el scaffolding del pipeline (`src/`). Las
> secciones 1-4 describen el diseño ya implementado; las secciones 5-6 se
> completan con números reales una vez el corpus esté construido y el
> decoder corra contra `data/sample_50.jsonl`.

---

## 1. Arquitectura del sistema

Una pregunta entra a `src/pipeline/answer_one.answer()`: se construye la
consulta (pregunta + opciones si es `multiple_choice`), se expande con
alias normativos conocidos (`src/retrieve/query_expand.py`, basado en reglas
sobre `scripts/citations.CODES`, sin LLM), se recupera de forma híbrida (BM25 top-30 ∪ denso top-30, fusión RRF k=60),
opcionalmente se reordena, y
`src/retrieve/sufficiency.decide()` determina si hay evidencia suficiente
para responder, reintentar con una recuperación ampliada, o abstenerse. Si
se responde, el decoder genera una única vez con gramática GBNF forzando el
JSON del formato; `src/generate/citation_check.py` verifica que cada cita de
la respuesta tenga respaldo en la evidencia antes de entregar.

## 2. Selección de encoder y decoder

| Componente | Modelo                                                      | Motivo de la elección                                                                                                               | Alternativas descartadas                |
|------------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| Encoder    | `intfloat/multilingual-e5-large`                            | Mismo encoder que usa `scripts/evaluate.py` para el componente de similitud de RAGAS; un solo stack validado por los organizadores. | —                                       |
| Decoder    | Llama-3.1-8B-Instruct GGUF (Q4_K_M), `llama-cpp-python`     | Un binario compila contra Metal (Mac) y CUDA (Turing); soporte GBNF maduro para forzar JSON.                                        | MLX (sin GBNF maduro), vLLM (solo CUDA) |
| Reranker   | `BAAI/bge-reranker-v2-m3` (opcional, tras `USE_RERANKER=1`) | Se activa solo si mide mejora contra `evaluate.py --split sample`.                                                                  | —                                       |

Configuración de inferencia: `temperatura=0` (greedy, determinista dado
binario+hardware fijos), `n_ctx=8192`, `seed` fijo, tope de tokens por
formato (`src/config.MAX_TOKENS_BY_FORMAT`, derivado de los límites del
schema). Tiempo medio por pregunta: pendiente de medir.

## 3. Estrategia de recuperación

Segmentación por artículo (regex `ARTÍCULO\s+\d+`) para códigos y leyes;
ventana deslizante de ~250 palabras con 2 oraciones de solapamiento para
jurisprudencia. Índice denso `faiss.IndexFlatIP` (búsqueda exacta, sin
aleatoriedad de entrenamiento) e índice disperso `bm25s`, fusionados por
Reciprocal Rank Fusion (k=60) para no tener que calibrar escalas entre BM25
y similitud coseno. Top-30 por rama antes de fundir, top-20 fundido, top-6
final al prompt (con reranker opcional de por medio).

## 4. Verificación de citas y abstención

`src/generate/citation_check.py` espeja exactamente
`scripts/evaluate.citas_respaldadas()`: solo los primeros 10 pasajes
recuperados cuentan como respaldo. Si la respuesta cita una norma a nivel de
artículo que no aparece en esa evidencia, el pipeline aplica un ajuste
determinista (SPEC.md sección 5 lo autoriza explícitamente): si *toda* la
evidencia citada carece de respaldo, el sistema prefiere abstenerse antes
que entregar una cita inventada — la tasa de citas sin respaldo pesa 2× en
contra en el puntaje de citación. La decisión de abstenerse en sí nunca
depende de que el LLM se autoevalúe: solo del score de recuperación fusionado (`src/retrieve/sufficiency.py`), para no
introducir una fuente adicional de
no-determinismo.

## 5. Resultados sobre las preguntas de muestra

| Componente                     |    Puntos | Posibles |
|--------------------------------|----------:|---------:|
| Exactitud en cerradas          | pendiente |       20 |
| Calidad de citación            | pendiente |       20 |
| Abstención calibrada           | pendiente |       10 |
| **Total automático sin RAGAS** | pendiente |   **50** |

Análisis de los errores más frecuentes: pendiente.

## 6. Limitaciones

1. El corpus aún no tiene contenido real indexado: este informe describe el
   diseño del pipeline, no un resultado medido.
2. Los parsers HTML (`src/ingest/parse_html.py`) no se han verificado contra
   una descarga real de las fuentes objetivo.
3. El umbral de suficiencia/abstención (`src/config.SUFFICIENCY_SCORE_THRESHOLD`)
   es un punto de partida sin calibrar empíricamente contra `evaluate.py`.
