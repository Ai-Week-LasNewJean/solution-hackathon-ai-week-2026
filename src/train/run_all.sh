#!/usr/bin/env bash
# Ajuste fino del reranker de punta a punta (SPEC.md 15). Cada etapa es
# reanudable: si se corta (o se acaba la noche), relanzar este script continua
# donde quedo. Uso: bash src/train/run_all.sh [N_PREGUNTAS]
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
PY=.venv/bin/python
N=${1:-4000}
$PY -m src.train.synth_queries --n "$N"     # 1. preguntas sinteticas (Qwen3-8B)  -> data/train/synth_queries.jsonl
$PY -m src.train.mine_negatives             # 2. negativos dificiles              -> data/train/pairs.jsonl
$PY -m src.train.train_reranker             # 3. ajuste fino con checkpoints      -> models/reranker-ft/
$PY -m src.train.eval_checkpoints           # 4. recall en sample_50 por checkpoint -> data/train/eval_log.jsonl
