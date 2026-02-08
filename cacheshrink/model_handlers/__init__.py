"""Model handler registry."""

from typing import Type
import torch.nn as nn

from .base import ModelHandler
from .gpt2 import GPT2Handler, GPT2AttentionAdapter
from .generic import GenericHandler, GenericAttentionAdapter
from ..config import MLAConfig


# Registry mapping model types to handlers
HANDLER_REGISTRY = {
    "gpt2": GPT2Handler,
}

# Registry mapping model types to attention adapters
ADAPTER_REGISTRY = {
    "gpt2": GPT2AttentionAdapter,
}


def get_handler(model: nn.Module, config: MLAConfig) -> ModelHandler:
    """Get the appropriate handler for a model.

    Args:
        model: HuggingFace model
        config: MLA configuration

    Returns:
        Model handler instance. Falls back to GenericHandler for
        unrecognized model types.
    """
    model_type = config.model_type

    handler_class = HANDLER_REGISTRY.get(model_type)
    if handler_class is None:
        # Fall back to GenericHandler for unrecognized model types
        return GenericHandler(model, config)

    return handler_class(model, config)


def get_attention_adapter(model_type: str):
    """Get the attention adapter class for a model type.

    Args:
        model_type: Model type string

    Returns:
        Attention adapter class. Falls back to GenericAttentionAdapter
        for unrecognized model types.
    """
    adapter_class = ADAPTER_REGISTRY.get(model_type)
    if adapter_class is None:
        # Fall back to GenericAttentionAdapter for unrecognized model types
        return GenericAttentionAdapter

    return adapter_class


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
    "GenericHandler",
    "GenericAttentionAdapter",
    "get_handler",
    "get_attention_adapter",
    "register_handler",
    "register_adapter",
    "HANDLER_REGISTRY",
    "ADAPTER_REGISTRY",
]
