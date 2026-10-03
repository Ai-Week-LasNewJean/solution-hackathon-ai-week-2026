#!/usr/bin/env bash
# Cadena del experimento A/B de corpus de la noche del 2026-10-02 (reanudable).
#   A = corpus completado (re-scrape de truncados + Decision 486 + normas faltantes)
#   B = A + encabezados de articulo (CHUNK_HEADINGS=1)
#   base = indice congelado (indice/), solo la corrida e2e de referencia
#
# A y B se empezaron con un script previo sin marcadores (logs en
# $EXP_DIR/A.log y B.log); este script espera a que terminen, convierte su
# avance en checkpoints de run_variant.sh y sigue con la corrida e2e de cada
# variante. Reejecutarlo es seguro: todo paso hecho se salta.
#
#   nohup bash src/experiments/ab_corpus_2026-10-02.sh > ../exp_index/ab.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
X=${EXP_DIR:-../exp_index}

adopt() {  # adopt <variante>: marcadores a partir del log del script previo
  local v=$1 log=$X/$1.log
  [[ -f $log ]] || return 0
  until grep -q "== DONE $v" "$log" || grep -qE "Traceback|out of memory|Killed" "$log"; do sleep 10; done
  grep -q "== DONE $v" "$log" || { echo "[$v] el script previo fallo; run_variant.sh rehara los pasos sin marcador"; return 0; }
  grep -E "^recall@" "$log" > "$X/$v/eval_sample.txt"
  grep -E "^(n=|all|caso|con_norma|directa|norma|providencia) " "$log" > "$X/$v/eval_synth.txt"
  for s in ingest bm25 faiss eval_sample eval_synth; do touch "$X/$v/.done_$s"; done
}

adopt A
adopt B
bash src/experiments/run_variant.sh A 0 e2e
bash src/experiments/run_variant.sh B 1 e2e

# referencia e2e sobre el indice congelado (mismo codigo y prompt, indice/ de 106 237 fragmentos)
B0=$X/base; mkdir -p "$B0"
if [[ ! -f $B0/.done_e2e ]]; then
  .venv/bin/python -m src.main --split sample --out "$B0/sample.jsonl" > "$B0/e2e.log" 2>&1 && \
  .venv/bin/python scripts/evaluate.py --submission "$B0/sample.jsonl" --split sample > "$B0/e2e_eval.txt" && \
  touch "$B0/.done_e2e"
fi
if [[ -f $B0/.done_e2e ]]; then
  { echo "# Referencia: indice congelado indice/ ($(date '+%Y-%m-%d %H:%M'), commit $(git rev-parse --short HEAD))"
    echo; echo "## evaluate.py --split sample"; cat "$B0/e2e_eval.txt"; } > data/experiments/base.txt
  git add data/experiments/base.txt
  git commit -q -m "Experiment base: e2e reference on frozen index" \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- data/experiments/base.txt || true
  for i in 1 2 3 4; do
    git push -q git@github.com:Ai-Week-LasNewJean/solution-hackathon-ai-week-2026.git "$(git branch --show-current)" && break
    sleep $((2 ** i))
  done
fi
echo "== AB DONE"
