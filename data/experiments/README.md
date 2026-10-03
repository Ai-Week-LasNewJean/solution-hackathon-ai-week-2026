# Experimentos de corpus/índice (checkpoints)

Cada experimento corre con `src/experiments/run_variant.sh <nombre> <CHUNK_HEADINGS> [e2e]`,
construye todo fuera del repo (`../exp_index/<nombre>`, nunca toca `corpus/` ni `indice/`)
y deja un marcador `.done_<paso>` por paso terminado. **Si algo se corta, se vuelve a correr el
mismo comando y sigue desde el primer paso pendiente** (la corrida e2e además reanuda por ítem).
Al terminar, el resultado queda en `data/experiments/<nombre>.txt` y el script hace commit + push.

## Estado (2026-10-02, noche)

| Variante | Qué cambia | Resultado |
|---|---|---|
| `base` | Índice congelado (`indice/`, 106 237 fragmentos) | retrieval: recall@3/6/10/20 0.805/0.833/0.878/0.902; sintéticas hit@3/6/10 0.915/0.945/0.955; e2e en `base.txt` |
| `A` | + 267 leyes/decretos truncados en el volcado re-scrapeados completos, + Decisión Andina 486 (PDF oficial CAN), + 6 normas del banco ausentes (121 886 fragmentos) | `A.txt` |
| `B` | A + epígrafe y ubicación (Libro/Título/Capítulo) en cada artículo (`CHUNK_HEADINGS=1`) | `B.txt` |

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

## Para adoptar una variante en la entrega

Nada de esto cambia el índice congelado. Si una variante gana en e2e: regenerar `corpus/`,
`corpus_manifest.json` e `indice/` con la misma configuración (`RESCRAPE_DIR=data/corpus_rescrape`,
`CHUNK_HEADINGS=...`, sin overrides `RAG_*`), poner el default de `CHUNK_HEADINGS` en `src/config.py`,
y repetir `determinism_check.py` y `latency_check.py` (SPEC.md 11.5). Respaldo del índice actual:
`../backup_v3_pre_fullscrape/` (con `SHA256SUMS`).
