#!/usr/bin/env bash
# Punto de entrada de reproducibilidad en un solo comando (SPEC.md seccion 4).
#
#   ./run.sh sample     # corre data/sample_50.jsonl -> out/sample.jsonl
#   ./run.sh test       # corre data/test_992.jsonl  -> out/test.jsonl (sabado)
#
# Variables de entorno relevantes (ver src/config.py):
#   LLM_MODEL_PATH   ruta al GGUF (default: models/llama-3.1-8b-instruct-q4_k_m.gguf)
#   RAG_BACKEND       mac|turing, override del autodetectado por plataforma
#   USE_RERANKER      1 para activar el reranker opcional

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SPLIT="${1:-sample}"

python -m pip install -q -r requirements.txt
python src/main.py --split "$SPLIT" --out "out/${SPLIT}.jsonl"
python scripts/evaluate.py --submission "out/${SPLIT}.jsonl" --split "$SPLIT"
