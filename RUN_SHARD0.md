# Corrida de la mitad 0 (shard 0) en la segunda máquina — sábado 2026-10-03

Por qué: el lote anterior reutilizaba la caché KV de llama.cpp entre preguntas, así que una respuesta
dependía de la pregunta anterior y **no se reproducía en un proceso fresco** (solo 4/20 coincidían).
El commit `b75b07f` limpia la caché en cada llamada. Hay que regenerar las 992. La máquina principal
corre el shard 1 y luego el shard 0 **en orden inverso**; esta máquina corre el shard 0 **hacia adelante**.
Se encuentran en la mitad y se fusionan.

## 1. Preparar (2 min)

```bash
cd ~/ai-week/solution-hackathon-ai-week-2026      # o donde esté el repo
git fetch origin && git checkout uwu && git pull origin uwu
git log --oneline -1          # debe incluir b75b07f (o posterior)
```

**No correr `uv sync` / `uv add` / `uv lock`** (reemplazan llama-cpp CUDA por la versión CPU).

```bash
uv pip install --python .venv/bin/python --no-deps PyStemmer
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
```

## 2. Verificar (1 min) — si algo no coincide, avisar antes de correr

```bash
nvidia-smi --query-gpu=name,memory.used --format=csv     # 4090 y GPU LIBRE: nada más corriendo
.venv/bin/python -c "import torch, llama_cpp; print(torch.cuda.is_available(), llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
# esperado: True 0.3.36 True
wc -l indice/chunks.jsonl                                # esperado: 106237
sha256sum models/Qwen3-8B-Q4_K_M.gguf | cut -c1-16       # esperado: d98cdcbd03e17ce4
wc -l data/test_992.jsonl                                # esperado: 992
grep -n "self._llama.reset()" src/generate/llm.py        # debe aparecer (cache limpio)
```

**Nunca** correr dos procesos con el modelo a la vez en la misma GPU (no caben y fallan con
`Failed to create llama_context`).

## 3. Correr el shard 0 (~55 min si se completa; se para antes al encontrarse con la otra máquina)

```bash
SHARD=0/2 nohup .venv/bin/python -m src.pipeline.run_batch --split test \
  --out out/final_s0.jsonl --no-resume > out/final_s0.log 2>&1 &
```

Progreso y errores:

```bash
wc -l out/final_s0.jsonl
grep -c '"error"' out/final_s0.jsonl      # debe ser 0; si sube, PARAR y avisar
tail -2 out/final_s0.log
```

Si se corta, relanzar **sin** `--no-resume` (reanuda por ítem), pero antes borrar las filas con
`"error"` si las hubiera:

```bash
grep -v '"error"' out/final_s0.jsonl > /tmp/s0 && mv /tmp/s0 out/final_s0.jsonl
SHARD=0/2 nohup .venv/bin/python -m src.pipeline.run_batch --split test --out out/final_s0.jsonl > out/final_s0.log 2>&1 &
```

## 4. Entregar el resultado (cuando lo pidan, o a las 14:30 aunque no haya terminado)

`out/` está en `.gitignore`, así que se fuerza el archivo en una rama aparte:

```bash
git checkout -b shard0-result
git add -f out/final_s0.jsonl
git commit -m "shard 0 output (forward)"
git push origin shard0-result
```

Un archivo parcial sirve: la máquina principal completa lo que falte desde el otro extremo.
