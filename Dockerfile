# Reproducibilidad desde contenedor limpio (2 pts, SPEC.md checklist).
#
# Backend CPU puro (sin Metal ni CUDA): sirve para probar que el pipeline
# corre de punta a punta desde cero, no para la corrida canonica del sabado
# (esa es siempre el Mac, ver SPEC.md seccion 2). La generacion final que
# termina en submissions.jsonl NO se produce con esta imagen.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    tesseract-ocr tesseract-ocr-spa \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV RAG_BACKEND=turing
ENV LLM_N_GPU_LAYERS=0

ENTRYPOINT ["./run.sh"]
CMD ["sample"]
