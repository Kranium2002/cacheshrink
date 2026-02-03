"""Model handler registry."""

from typing import Type
import torch.nn as nn

from .base import ModelHandler
from .gpt2 import GPT2Handler, GPT2AttentionAdapter
from .llama import LlamaHandler, LlamaAttentionAdapter
from .mistral import MistralHandler, MistralAttentionAdapter
from .qwen import QwenHandler, QwenAttentionAdapter
from ..config import MLAConfig


# Registry mapping model types to handlers
HANDLER_REGISTRY = {
    "gpt2": GPT2Handler,
    "llama": LlamaHandler,
    "mistral": MistralHandler,
    "qwen": QwenHandler,
}

# Registry mapping model types to attention adapters
ADAPTER_REGISTRY = {
    "gpt2": GPT2AttentionAdapter,
    "llama": LlamaAttentionAdapter,
    "mistral": MistralAttentionAdapter,
    "qwen": QwenAttentionAdapter,
}


def get_handler(model: nn.Module, config: MLAConfig) -> ModelHandler:
    """Get the appropriate handler for a model.

    Args:
        model: HuggingFace model
        config: MLA configuration

    Returns:
        Model handler instance

    Raises:
        ValueError: If model type is not supported
    """
    model_type = config.model_type

    if model_type not in HANDLER_REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(HANDLER_REGISTRY.keys())}"
        )

    handler_class = HANDLER_REGISTRY[model_type]
    return handler_class(model, config)


def get_attention_adapter(model_type: str):
    """Get the attention adapter class for a model type.

    Args:
        model_type: Model type string

    Returns:
        Attention adapter class

    Raises:
        ValueError: If model type is not supported
    """
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(ADAPTER_REGISTRY.keys())}"
        )

    return ADAPTER_REGISTRY[model_type]


def register_handler(model_type: str, handler_class: Type[ModelHandler]) -> None:
    """Register a new model handler.

    Args:
        model_type: Model type identifier
        handler_class: Handler class to register
    """
    HANDLER_REGISTRY[model_type] = handler_class


def register_adapter(model_type: str, adapter_class: Type) -> None:
    """Register a new attention adapter.

    Args:
        model_type: Model type identifier
        adapter_class: Adapter class to register
    """
    ADAPTER_REGISTRY[model_type] = adapter_class


__all__ = [
    "ModelHandler",
    "GPT2Handler",
    "GPT2AttentionAdapter",
    "LlamaHandler",
    "LlamaAttentionAdapter",
    "MistralHandler",
    "MistralAttentionAdapter",
    "QwenHandler",
    "QwenAttentionAdapter",
    "get_handler",
    "get_attention_adapter",
    "register_handler",
    "register_adapter",
    "HANDLER_REGISTRY",
    "ADAPTER_REGISTRY",
]
