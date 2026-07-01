"""local-engine-router: a single-GPU model switchboard.

A small OpenAI- and Ollama-compatible reverse proxy that fronts a fleet of
local inference engines (vLLM, llama.cpp, SGLang, Ollama, TabbyAPI, and more)
behind one port. It reads the requested model, figures out which engine owns it,
and transparently swaps engines (only one heavy model fits in a memory-
constrained GPU's unified memory at a time) before proxying the request through.
"""

__version__ = "0.6.0"
