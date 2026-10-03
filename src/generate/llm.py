"""Motor de inferencia: llama-cpp-python (GGUF), un binario, dos backends
(Metal en Mac, CUDA en Turing) -- SPEC.md secciones 2 y 3.

Riesgo central del reto: con temperatura=0 la decodificacion greedy es
deterministica dado un binario y hardware fijos, pero CUDA y Metal usan
kernels de punto flotante distintos. Turing (CUDA) es la fuente de verdad
canonica para toda generacion que termine en submissions.jsonl
(config.IS_CANONICAL); el Mac es solo auxiliar para pruebas de determinismo. Este modulo no impone esa regla por si solo --
quien orqueste la corrida final (pipeline/run_batch.py) debe verificar
config.IS_CANONICAL antes de escribir submissions.jsonl.
"""
from __future__ import annotations

from pathlib import Path

from src import config


class LLM:
    """Envoltorio fino sobre llama_cpp.Llama con generate() determinista."""

    def __init__(self, model_path: Path | None = None, n_ctx: int = config.LLM_N_CTX,
                 n_gpu_layers: int = config.LLM_N_GPU_LAYERS, seed: int = config.LLM_SEED,
                 chat_template: str | None = None):
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
        self.chat_template = chat_template

    def _wrap_prompt(self, prompt: str) -> str:
        """Envuelve `prompt` con los marcadores de turno del chat template
        del candidato activo, SIN pasar por create_chat_completion() --
        seguimos usando completion cruda (mismo mecanismo validado para
        Llama) para no tocar nada del camino ya medido. Solo aplica cuando
        LLM_CANDIDATES[...]['chat_template'] lo pide (ver config.py); para
        Llama (chat_template=None) esto es un no-op, prompt sin modificar."""
        if self.chat_template == "chatml":
            # Qwen3-Instruct se entreno casi exclusivamente en ChatML; sin el
            # turno explicito sigue instrucciones peor. La gramatica GBNF
            # fuerza "{" como primer caracter de cualquier forma, asi que el
            # modo "thinking" de Qwen3 no puede filtrarse al output aunque no
            # se envie enable_thinking=False explicitamente.
            tail = "<think>\n\n</think>\n\n" if getattr(self, "_no_think", False) else ""
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{tail}"
        if self.chat_template == "llama3":
            return ("<|start_header_id|>user<|end_header_id|>\n\n"  # BOS lo antepone llama.cpp
                    f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        if self.chat_template == "mistral":
            return f"[INST] {prompt} [/INST]"  # llama.cpp antepone BOS
        if self.chat_template == "aya":
            return ("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
                    f"{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
        return prompt

    def generate(self, prompt: str, max_tokens: int, grammar_path: Path | None = None,
                 stop: list[str] | None = None, no_think: bool = False) -> str:
        self._no_think = no_think
        """Genera a temperatura 0 (greedy, determinista dado el binario y
        hardware). `grammar_path` es un archivo .gbnf de src/generate/grammars/;
        sin el, la salida no esta forzada a JSON (usar formatter.parse con
        fallback en ese caso)."""
        grammar = None
        if grammar_path is not None:
            from llama_cpp import LlamaGrammar

            grammar = LlamaGrammar.from_file(str(grammar_path))
        # cache limpio por llamada: llama-cpp reutiliza el KV del prefijo comun con el prompt
        # anterior y eso cambia la salida greedy segun que item se proceso antes (2026-10-03:
        # solo 4/20 items del lote coincidian con una re-ejecucion en proceso fresco).
        self._llama.reset()
        out = self._llama(
            self._wrap_prompt(prompt), max_tokens=max_tokens, temperature=config.LLM_TEMPERATURE,
            grammar=grammar, stop=stop)
        return out["choices"][0]["text"]


_cache: dict[Path, LLM] = {}


def get_llm(model_name: str | None = None) -> LLM:
    """Instancia cacheada por ruta GGUF resuelta (SPEC.md 10.5) -- perezosa,
    cargar cada candidato una sola vez por proceso. Sin argumento, usa el
    candidato activo (config.LLM_MODEL_NAME / override LLM_MODEL_PATH), que
    es el unico camino que debe llegar a producir submissions.jsonl. Pasar
    `model_name` (una llave de config.LLM_CANDIDATES) es solo para
    comparar/alternar en desarrollo o demo -- nunca para la corrida final."""
    path = config.llm_model_path(model_name)
    if path not in _cache:
        _cache[path] = LLM(model_path=path, chat_template=config.llm_chat_template(model_name))
    return _cache[path]
