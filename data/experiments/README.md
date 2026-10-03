# Experimentos de corpus/índice (checkpoints)

Cada experimento corre con `src/experiments/run_variant.sh <nombre> <CHUNK_HEADINGS> [e2e]`,
construye todo fuera del repo (`../exp_index/<nombre>`, nunca toca `corpus/` ni `indice/`)
y deja un marcador `.done_<paso>` por paso terminado. **Si algo se corta, se vuelve a correr el
mismo comando y sigue desde el primer paso pendiente** (la corrida e2e además reanuda por ítem).
Al terminar, el resultado queda en `data/experiments/<nombre>.txt` y el script hace commit + push.

## Estado (2026-10-02, noche)

| Variante | Qué cambia | recall@3/6/10/20 sample_50 | sintéticas @3/@6 (mismas 595–600) | e2e auto /50 (cerradas · citación · abst.) |
|---|---|---|---|---|
| `base` | Índice congelado (`indice/`, 106 237 fragmentos) | 0.805/0.833/0.878/0.902 | 0.915/0.945 | **34.12** (14.67 · 12.24 · 7.21) |
| `A` | + 267 leyes/decretos truncados completos, + Decisión Andina 486, + 6 normas del banco ausentes (121 886 fragmentos) | 0.805/0.833/0.841/0.902 | 0.901/0.931 | 33.49 (14.67 · 11.84 · 6.98) |
| `B` | A + epígrafe y ubicación en cada artículo (`CHUNK_HEADINGS=1`) | 0.780/0.789/0.829/0.878 | 0.894/0.931 | 32.55 (13.33 · 12.24 · 6.98) |

Lectura: en `sample_50` ninguna variante supera al índice congelado (diferencias dentro del ruido de 50 ítems;
±1 cerrada = 1.33 pts). Los encabezados (B) no ayudan en ninguna métrica: descartados. A cuesta ~1.4 pts de
acierto sintético sobre artículos ya presentes (15k fragmentos más compiten) a cambio de cobertura que
`sample_50` no mide (≥ 60 ítems del banco citan normas antes ausentes o truncadas, según `seed_targets.json`).
Adoptar A es una decisión de riesgo/cobertura, no un resultado medido.

Cadena completa (espera A/B, corre e2e de A, B y base, sube cada resultado):

```bash
nohup bash src/experiments/ab_corpus_2026-10-02.sh > ../exp_index/ab.log 2>&1 &
tail -f ../exp_index/ab.log
```

Prerrequisitos de datos (no versionados, se regeneran con caché en `data/corpus_rescrape/raw/`):

```bash
.venv/bin/python -m src.ingest.rescrape_senado --completar   # leyes_decretos_completos.json (~25 min sin caché)
.venv/bin/python -m src.ingest.parse_decision486              # decision_andina_486.json
```

**Ojo:** las cifras de "preguntas sintéticas" dentro de `A.txt`/`B.txt` vienen de una versión de
`synth_retrieval_eval.py` que muestreaba distinto conjunto de preguntas por índice (no comparables).
Las comparables (mismo conjunto, corregido en el commit siguiente) quedan en `synth_fixed.txt`.

## Para adoptar una variante en la entrega

Nada de esto cambia el índice congelado. Si una variante gana en e2e: regenerar `corpus/`,
`corpus_manifest.json` e `indice/` con la misma configuración (`RESCRAPE_DIR=data/corpus_rescrape`,
`CHUNK_HEADINGS=...`, sin overrides `RAG_*`), poner el default de `CHUNK_HEADINGS` en `src/config.py`,
y repetir `determinism_check.py` y `latency_check.py` (SPEC.md 11.5). Respaldo del índice actual:
`../backup_v3_pre_fullscrape/` (con `SHA256SUMS`).
