"""Handler for Mistral models."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .llama import LlamaHandler, LlamaAttentionAdapter
from ..config import MLAConfig


class MistralHandler(LlamaHandler):
    """Handler for Mistral model family.

    Mistral has the same architecture as LLaMA with:
    - Sliding window attention (handled at mask level, not in weights)
    - GQA with num_key_value_heads < num_attention_heads
    - RoPE with potentially different theta

    We inherit from LlamaHandler since the weight extraction is identical.
    """

    def __init__(self, model: nn.Module, config: MLAConfig):
        # Call grandparent init to avoid LlamaHandler's validation
        # which checks for 'model.layers'
        super(LlamaHandler, self).__init__(model, config)

        # Verify model structure (same as LLaMA)
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Model does not have 'model.layers' attribute. Is this a Mistral model?"
            )

        # Mistral-specific attributes
        self.sliding_window = getattr(model.config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[int]:
        """Get sliding window size if applicable."""
        return self.sliding_window


class MistralAttentionAdapter(LlamaAttentionAdapter):
    """Adapter for Mistral attention.

    Identical to LLaMA adapter since forward signatures are the same.
    Sliding window is handled at the mask level, not in the attention module.
    """

    def __init__(self, mla_attention: nn.Module, config: MLAConfig):
        super().__init__(mla_attention, config)
        # Mistral might have sliding window in config
        self.sliding_window = getattr(config, "sliding_window", None)
