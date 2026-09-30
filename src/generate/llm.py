"""Motor de inferencia: llama-cpp-python (GGUF), un binario, dos backends
(Metal en Mac, CUDA en Turing) -- SPEC.md secciones 2 y 3.

Riesgo central del reto: con temperatura=0 la decodificacion greedy es
deterministica dado un binario y hardware fijos, pero CUDA y Metal usan
kernels de punto flotante distintos. El Mac debe ser la fuente de verdad
canonica para toda generacion que termine en submissions.jsonl desde el
jueves (config.IS_CANONICAL). Este modulo no impone esa regla por si solo --
quien orqueste la corrida final (pipeline/run_batch.py) debe verificar
config.IS_CANONICAL antes de escribir submissions.jsonl.
"""
from __future__ import annotations

from pathlib import Path

from src import config


class LLM:
    """Envoltorio fino sobre llama_cpp.Llama con generate() determinista."""

    def __init__(self, model_path: Path | None = None, n_ctx: int = config.LLM_N_CTX,
                 n_gpu_layers: int = config.LLM_N_GPU_LAYERS, seed: int = config.LLM_SEED):
        from llama_cpp import Llama

        path = model_path or config.LLM_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"no existe el modelo GGUF en {path}. Descargar un Llama-3.1-8B-Instruct "
                "(o similar, <=8B, abierto) cuantizado Q4_K_M y colocarlo ahi, o pasar "
                "LLM_MODEL_PATH en el entorno.")
        self._llama = Llama(
            model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            seed=seed, verbose=False)
        self.backend = config.BACKEND

    def generate(self, prompt: str, max_tokens: int, grammar_path: Path | None = None) -> str:
        """Genera a temperatura 0 (greedy, determinista dado el binario y
        hardware). `grammar_path` es un archivo .gbnf de src/generate/grammars/;
        sin el, la salida no esta forzada a JSON (usar formatter.parse con
        fallback en ese caso)."""
        grammar = None
        if grammar_path is not None:
            from llama_cpp import LlamaGrammar

            grammar = LlamaGrammar.from_file(str(grammar_path))
        out = self._llama(
            prompt, max_tokens=max_tokens, temperature=config.LLM_TEMPERATURE,
            grammar=grammar)
        return out["choices"][0]["text"]


_singleton: LLM | None = None


def get_llm() -> LLM:
    """Instancia unica y perezosa -- cargar el GGUF una vez por proceso."""
    global _singleton
    if _singleton is None:
        _singleton = LLM()
    return _singleton
