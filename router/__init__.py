"""llm-router: a single-GPU model switchboard.

A small OpenAI- and Ollama-compatible reverse proxy that sits in front of the
local ds4 engine and Ollama. It reads the requested model, figures out which
engine owns it, and transparently swaps engines (only one heavy model fits in
the GB10's unified memory at a time) before proxying the request through.
"""

__version__ = "0.2.0"
