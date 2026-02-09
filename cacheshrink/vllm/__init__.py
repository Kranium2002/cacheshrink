"""vLLM integration for cacheshrink compressed KV cache models.

Enables serving cacheshrink MLA models via vLLM with compressed KV cache
blocks. The compressed c_kv is stored in vLLM's paged KV cache and
decompressed on-the-fly during attention computation.

Usage:
    # Save a model for vLLM
    from cacheshrink.vllm import save_for_vllm
    save_for_vllm(model, tokenizer, "./my-mla-model-vllm")

    # Serve with vLLM (plugin auto-registers via entry point)
    # vllm serve ./my-mla-model-vllm
"""

from .plugin import register_cacheshrink_models
from .saving import save_for_vllm

__all__ = [
    "register_cacheshrink_models",
    "save_for_vllm",
]
