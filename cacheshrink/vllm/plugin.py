"""vLLM plugin entry point for cacheshrink model registration.

This module is referenced by the entry point in pyproject.toml:
    [project.entry-points."vllm.general_plugins"]
    cacheshrink = "cacheshrink.vllm.plugin:register_cacheshrink_models"

vLLM calls this function during startup to register custom model architectures.
"""


def register_cacheshrink_models():
    """Register CacheShrinkForCausalLM with vLLM's model registry.

    Uses lazy string import to avoid CUDA re-initialization in forked
    worker processes. The actual model class is only imported when vLLM
    instantiates it.
    """
    from vllm import ModelRegistry

    if "CacheShrinkForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "CacheShrinkForCausalLM",
            "cacheshrink.vllm.model:CacheShrinkForCausalLM",
        )
