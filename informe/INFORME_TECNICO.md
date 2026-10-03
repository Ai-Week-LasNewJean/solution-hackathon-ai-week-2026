# Informe técnico — Las NewJeans

**Hackathon 2026 · Universidad de los Andes**
**Integrantes:**

Máximo 3 páginas al exportar a PDF. Se entrega como `informe/INFORME_TECNICO.pdf`.

---

## 1. Arquitectura del sistema

Partimos de una observación: **lo difícil es encontrar la cita correcta; una vez recuperada, responder es
casi trivial** y no requiere un decoder fuerte. El sistema es una cadena de etapas en la que cada una
refina la evidencia de la anterior antes de llegar a un modelo de 8B que solo redacta:

1. **Recuperador** (`src/retrieve/hybrid.py`): consulta = pregunta (+ opciones en cerradas), expandida
   con alias normativos por reglas (`query_expand.py`, sin LLM). BM25 con stemming español top-100 ∪
   denso `multilingual-e5-large` top-100, fusión Reciprocal Rank Fusion (k=60) → 50 candidatos.
2. **Reordenador** (`rerank.py`): cross-encoder `BAAI/bge-reranker-v2-m3` → top-6 al prompt. Su score
   (sigmoide) decide de forma determinista responder, reintentar con un pool ampliado o abstenerse
   (`sufficiency.py`, umbrales 0.10 / 0.02).
3. **Redactor**: Qwen3-8B, una sola llamada con salida JSON forzada por gramática GBNF por formato.
   `citation_check.py` espeja `evaluate.citas_respaldadas()`: si toda cita de la respuesta carece de
   respaldo en la evidencia, se abstiene en vez de entregar una norma inventada.
4. **Agente de citación** (`src/generate/cite_builder.py`, determinista, sin LLM): extrae con
   `citations.extract` las normas de los 4 primeros pasajes de evidencia y las agrega como "Fuentes en
   la evidencia" en `referencia_legal` (semi-abiertas) o `justificacion` (cerradas). Solo agrega un
   nombre si el extractor oficial lo vuelve a leer como la misma norma. Toda norma agregada está en la
   evidencia entregada: queda respaldada por construcción.

Peor caso: 2 recuperaciones + 1 generación por pregunta. La etapa 4 es una función pura de (salida del
redactor, pasajes), reproducible byte a byte.

## 2. Selección de encoder y decoder

| Componente | Modelo | Motivo | Alternativas medidas |
|---|---|---|---|
| Encoder | `intfloat/multilingual-e5-large` | Mismo encoder que el evaluador usa para RAGAS; recall@10 0.841 | `BAAI/bge-m3`: 0.785 |
| Decoder | **Qwen3-8B** GGUF Q4_K_M, `llama-cpp-python` 0.3.36 (CUDA) | Mejor de 6 decoders ≤8B en `sample_50` (31.65/50 sin reranker) | Llama-3.1-8B 22.10, Mistral-7B 25.44, Qwen2.5-7B 28.20, Aya-Expanse-8B 29.52 (CC-BY-NC) |
| Reranker | `BAAI/bge-reranker-v2-m3` | recall@6 0.724 → 0.764; citación 0.50 → 0.63 | Ajuste fino con 4 000 preguntas sintéticas: recall@3 cae 0.805 → 0.707, no adoptado |

Inferencia: `temperatura=0` (greedy), `seed=0`, `n_ctx=8192`, tope de tokens por formato (700 / 420 /
900), plantilla ChatML. Máquina: RTX 4090 (24 GB). Latencia en las 992 preguntas: **~7.9 s/pregunta**
(~2.2 h, presupuesto 6 h).

## 3. Estrategia de recuperación y corpus

Corpus: 34 380 documentos / **106 237 fragmentos**, 100% trazables a su norma de origen. Códigos y leyes
(secretariasenado.gov.co, reescrapeados completos, incluido el Estatuto Orgánico del Sistema Financiero)
segmentados por artículo con el nombre de la norma como prefijo; relatoría de la Corte Constitucional y
providencias de la Corte Suprema en ventanas de ~250 palabras. Índices exactos y reconstruibles:
`faiss.IndexFlatIP` y `bm25s` (reconstrucción verificada idéntica byte a byte).

| Cambio | recall@6 de normas de referencia |
|---|---:|
| Híbrido RRF, pool 30/30 | 0.695 |
| + reranker | 0.724 |
| + pool 100/100 → 50 | 0.764 |
| + corpus reescrapeado, BM25 con stemming, Estatuto Orgánico | **0.833** (recall@10 0.878) |

Probado y descartado (sin mejora medida): reformulación de consulta con el LLM, anteponer el tema,
vectores extra para fragmentos largos, encabezados de artículo en el fragmento, bono a normas sobre
providencias, penalización de normas derogadas.

## 4. Citación y abstención

El puntaje de citación compara normas (no artículos) contra el fundamento de referencia y penaliza ×2
las citas sin respaldo en los 10 primeros pasajes. El diagnóstico mostró que **la recuperación ya
encontraba la norma, pero el redactor no la citaba**: de 41 normas de referencia en el top-6, solo 29
aparecían en la respuesta. Pedirlo en el prompt subía la citación pero degradaba la respuesta que ve
el juez (RAGAS 0.481 → 0.426). El agente de citación resuelve esto sin tocar el texto que se juzga.

**Agente filtro con LLM (evaluado, no adoptado).** Siguiendo la idea de una cadena de agentes que
filtran la evidencia, Qwen3-8B eligió entre los 10–15 mejores pasajes del reranker los que sustentan la
respuesta (`filter_agent.py`, salida forzada por gramática). Sus elegidos contenían la norma correcta
en el 70% de los casos, frente al 75.6% del top-3 del reranker; de punta a punta 35.47 vs 37.03 /50.
Con un decoder de 8B el cross-encoder ya cumple mejor ese papel.

La abstención depende solo de señales de recuperación, nunca de la autoevaluación del LLM (otra fuente
de no-determinismo). Abstenerse vale 0.5; en `sample_50` los únicos ítems con score < 0.10 fallaron.

## 5. Resultados sobre las preguntas de muestra (`sample_50`, sin RAGAS)

| Componente | Viernes (17:00) | Sin agente de citación | **Entrega** | Posibles |
|---|---:|---:|---:|---:|
| Exactitud en cerradas | 13.33 (10/15) | 14.67 (11/15) | **14.67** (11/15) | 20 |
| Calidad de citación | 11.84 | 12.24 | **16.33** | 20 |
| Abstención calibrada | 6.74 | 7.21 | **8.14** | 10 |
| **Total automático sin RAGAS** | 31.91 | 34.12 | **39.14** | **50** |

RAGAS (`answer_correctness`, juez `z-ai/glm-5.3-flash`) medido el viernes: 0.481 (14.42/30, línea base
0.451). El agente de citación no modifica los campos que lee el juez.

Barrido del número de pasajes citados (k): k=2 → 37.16, k=4 → 37.57, k=6 → 37.97 /50 sobre la entrega
del viernes, con 4.6 / 7.4 / 9.7 normas por respuesta. Elegimos k=4 como equilibrio entre puntaje y
relleno.

**Errores frecuentes.** Las cerradas que fallan siempre son huecos de conocimiento o de corpus, no de
razonamiento: el ítem 528 exige convertir 40 SMLMV a pesos (dato ausente del corpus), el 671 tiene como
fundamento "Doctrina" y el 128 dependía del Estatuto Orgánico. Razonamiento previo, voto por permutación
de opciones y Qwen3 en Q8 dieron 10/15 en todos los casos.

## 6. Limitaciones

1. **Muestra pequeña:** 50 ítems y 15 cerradas; ±1 cerrada = ±1.33 pts. Las decisiones finas se apoyaron
   también en recall de recuperación y en 4 000 preguntas sintéticas.
2. **Techo de recuperación:** el agente de citación solo cita lo que llega al top-4; si la norma no se
   recupera (recall@6 0.833), no se cita. No se aplica en `open_ended`, cuyos cuatro campos van al juez.
3. **Corpus:** la Corte Constitucional está solo como relatoría (sin texto íntegro); faltan leyes
   anteriores a 1992 que secretariasenado no publica y no hay doctrina. No hay señal de vigencia (norma
   vigente vs. derogada).
4. **Determinismo:** greedy a temperatura 0 es determinista para un binario y hardware fijos; la entrega
   y la verificación en vivo usan la misma máquina (RTX 4090, CUDA 13, mismo GGUF).
