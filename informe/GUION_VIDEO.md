# Guion del video (≤ 5 min) — Las NewJeans

Unos 640 palabras habladas (~4:30 a ritmo natural), con 30 s de margen. Tres voces: **Adrian (A)**,
**Laura (L)** y **Andres (B)**. La columna de pantalla dice qué grabar.

---

### 0:00 – 0:25 · Apertura — A

**Pantalla:** interfaz (`streamlit run interfaz/app.py`) con el logo de Software Colombia; cámara de los tres al inicio si se puede.

> Hola, somos Las NewJeans, de la Universidad de los Andes: Adrian Velásquez, Laura Juliana Ferreira y
> Andrés Botero. Construimos un sistema que responde preguntas de derecho colombiano usando solo modelos
> abiertos: un decodificador de ocho mil millones de parámetros, un encoder abierto y un corpus jurídico
> que armamos nosotros.

### 0:25 – 1:05 · El corpus — L

**Pantalla:** `CORPUS.md` y la carpeta `corpus/`; luego un fragmento de `indice/chunks.jsonl` que empiece por "Código Civil. ARTÍCULO…".

> El reto dice que la calidad del corpus pesa más que el modelo, y por ahí empezamos. Reunimos más de
> treinta y cuatro mil documentos: la Constitución, los códigos y las leyes de la Secretaría del Senado,
> que reescrapeamos completos, el Estatuto Orgánico del Sistema Financiero, la relatoría de la Corte
> Constitucional y providencias de la Corte Suprema. Son ciento seis mil fragmentos, cortados por
> artículo, y cada uno lleva el nombre de su norma: el cien por ciento es trazable a su fuente. Eso es
> lo que después permite citar con respaldo.

### 1:05 – 2:05 · Arquitectura: una cadena de agentes — B

**Pantalla:** diagrama simple (pregunta → recuperador → reordenador → redactor → agente de citación → respuesta) o la sección 1 del informe.

> La idea central es que lo difícil es encontrar la cita correcta; una vez la tienes, responder es casi
> trivial. Por eso el sistema es una cadena de etapas, y cada una refina la evidencia de la anterior.
> Primero, un recuperador híbrido: búsqueda léxica BM25 y búsqueda semántica con multilingual-e5, fundidas
> por rangos. Segundo, un reordenador, el cross-encoder bge-reranker, que deja los seis mejores pasajes;
> su puntaje decide si respondemos o nos abstenemos, sin preguntarle nada al modelo. Tercero, Qwen3 de
> ocho mil millones redacta la respuesta en JSON forzado por gramática, a temperatura cero. Y cuarto,
> un agente de citación determinista: toma las normas de los pasajes que sustentan la respuesta y las
> cita. Todo lo que citamos está en la evidencia que entregamos.

### 2:05 – 3:00 · Qué medimos — A

**Pantalla:** tabla de resultados del informe (31.91 → 34.12 → 39.14) y la del barrido de k.

> Medimos cada decisión contra el evaluador oficial. Descubrimos que la recuperación ya encontraba la
> norma correcta, pero el modelo no la citaba: de cuarenta y una normas recuperadas, solo citaba
> veintinueve. Pedírselo en el prompt empeoraba la respuesta. El agente de citación lo resolvió sin
> tocar el texto: en las preguntas de muestra, la calidad de citación pasó de doce a dieciséis puntos
> sobre veinte, y el total automático de treinta y cuatro a treinta y nueve sobre cincuenta, con cero
> citas sin respaldo. También probamos un agente filtro con el propio Qwen y midió peor que el
> reordenador, así que lo dejamos por fuera. Cada decisión salió de una medición, no de una intuición.

### 3:00 – 3:55 · Demo en vivo — L

**Pantalla:** interfaz. Escribir una pregunta semiabierta (por ejemplo: "¿En qué casos procede la acción de grupo?"), mostrar la respuesta, la referencia legal con "Fuentes en la evidencia", la tabla de pasajes recuperados y las etiquetas verdes de respaldo de cada cita.

> Así se ve en la interfaz. Hacemos una pregunta y el sistema responde en unos segundos. Abajo están los
> pasajes que recuperó, con su norma y su puntaje, y cada cita de la respuesta tiene una etiqueta: verde
> si está respaldada por la evidencia. Si el sistema no encuentra fundamento suficiente, se abstiene en
> vez de inventar una norma.

### 3:55 – 4:35 · Reproducibilidad — B

**Pantalla:** terminal con `determinism_check` dando `"ok": true`; `procedencia.json`; `run.sh`.

> Todo corre a temperatura cero y es reproducible. Hoy encontramos que la caché de llama.cpp hacía que una
> respuesta dependiera de la pregunta anterior; la limpiamos en cada llamada y ahora cada respuesta se
> regenera idéntica en un proceso nuevo. El índice está congelado y se reconstruye con un solo script, y
> todo el sistema se instala con un comando.

### 4:35 – 4:55 · Cierre — A

**Pantalla:** los tres, o el logo con el enlace al repositorio.

> En resumen: un corpus trazable, una cadena de agentes que encuentra y cita la norma correcta, y un
> modelo pequeño que solo tiene que redactar. Gracias.

---

**Tips de grabación:** grabar la demo antes (no en vivo) para no depender de la latencia; subir la fuente
de la terminal; cronometrar el ensayo y, si pasa de 4:50, recortar la sección de corpus.
