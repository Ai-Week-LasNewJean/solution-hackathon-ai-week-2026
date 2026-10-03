#!/usr/bin/env bash
# Experimento de corpus/indice con checkpoints (reanudable).
#
#   bash src/experiments/run_variant.sh <nombre> <CHUNK_HEADINGS 0|1> [e2e]
#
# Construye corpus + chunks + BM25 + FAISS en $EXP_DIR/<nombre> (por defecto
# ../exp_index/<nombre>, fuera del repo; nunca toca corpus/ ni indice/), corre
# las evaluaciones de recuperacion y, con "e2e", la corrida completa de
# sample_50 + evaluate.py. Cada paso deja un marcador $V/.done_<paso> solo si
# termino bien: si el proceso se corta, volver a correr el mismo comando salta
# los pasos hechos y sigue desde el primero pendiente (e2e ademas reanuda por
# item: src.main retoma $V/sample.jsonl). Para rehacer un paso,
# borrar su marcador.
#
# Al final escribe data/experiments/<nombre>.txt (versionado) y hace commit +
# push (SSH) de ese archivo, para que el resultado quede en GitHub aunque la
# sesion se cierre. Estado de todos los experimentos: data/experiments/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${1:?nombre}; HEAD=${2:?CHUNK_HEADINGS 0|1}; E2E=${3:-}
V=${EXP_DIR:-../exp_index}/$NAME
mkdir -p "$V" data/experiments
PY=.venv/bin/python
export RESCRAPE_DIR=data/corpus_rescrape CHUNK_HEADINGS=$HEAD
export RAG_CORPUS_DIR=$V/corpus RAG_CORPUS_MANIFEST=$V/corpus_manifest.json RAG_INDEX_DIR=$V/indice

step() {  # step <nombre> <comando...>
  local s=$1; shift
  if [[ -f $V/.done_$s ]]; then echo "== $s: hecho (checkpoint), se salta"; return; fi
  echo "== $s: $(date +%H:%M:%S)"
  "$@"
  touch "$V/.done_$s"
}

step ingest      $PY -m src.ingest.ingest_raw_sources
step bm25        $PY -m src.index.build_bm25
step faiss       $PY -m src.index.build_faiss
step eval_sample bash -c "set -o pipefail; $PY -m src.validate.retrieval_eval --rerank --ks 3 6 10 20 2>/dev/null | tee $V/eval_sample.txt"
step eval_synth  bash -c "set -o pipefail; $PY -m src.validate.synth_retrieval_eval --rerank --n 600 2>/dev/null | tee $V/eval_synth.txt"
e2e() {
  # el decoder necesita ~7 GB libres: si otro proceso ocupa la GPU, llama.cpp no crea el
  # contexto y run_batch convierte cada item en abstencion (puntaje falso). Esperar VRAM.
  until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 10000 )); do
    echo "   esperando >=10 GB de VRAM libre..."; sleep 30
  done
  $PY -m src.main --split sample --out "$V/sample.jsonl" >"$V/e2e.log" 2>&1 || return 1
  if grep -q Traceback "$V/e2e.log"; then
    echo "   e2e con errores por item (ver $V/e2e.log): se descarta, sin marcador"
    rm -f "$V/sample.jsonl"; return 1
  fi
  $PY scripts/evaluate.py --submission "$V/sample.jsonl" --split sample | tee "$V/e2e_eval.txt"
}
if [[ $E2E == e2e ]]; then
  step e2e e2e
fi

R=data/experiments/$NAME.txt
{
  echo "# Experimento $NAME ($(date '+%Y-%m-%d %H:%M'), commit $(git rev-parse --short HEAD))"
  echo "RESCRAPE_DIR=$RESCRAPE_DIR CHUNK_HEADINGS=$HEAD  fragmentos=$(wc -l <"$V/indice/chunks.jsonl")"
  echo "sha256 chunks.jsonl: $(sha256sum "$V/indice/chunks.jsonl" | cut -c1-16)"
  echo; echo "## recall de normas de referencia, sample_50 (reranker)"; cat "$V/eval_sample.txt"
  echo; echo "## acierto@k, preguntas sinteticas (reranker)"; cat "$V/eval_synth.txt"
  if [[ -f $V/e2e_eval.txt ]]; then echo; echo "## evaluate.py --split sample"; cat "$V/e2e_eval.txt"; fi
} > "$R"
cat "$R"

# commit + push solo del archivo de resultados (lock: varias variantes en paralelo)
(
  flock 9
  git add "$R"
  git commit -q -m "Experiment $NAME: results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$R" || true
  for i in 1 2 3 4; do
    git push -q git@github.com:Ai-Week-LasNewJean/solution-hackathon-ai-week-2026.git "$(git branch --show-current)" && break
    sleep $((2 ** i))
  done
) 9>"${EXP_DIR:-../exp_index}/.git.lock"
echo "== DONE $NAME"
