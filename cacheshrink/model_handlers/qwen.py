"""Handler for Qwen models."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .llama import LlamaHandler, LlamaAttentionAdapter
from ..config import MLAConfig


class QwenHandler(LlamaHandler):
    """Handler for Qwen/Qwen2 model family.

    Qwen2 has a similar architecture to LLaMA with:
    - Separate q_proj, k_proj, v_proj, o_proj
    - GQA support
    - RoPE (potentially with different parameters)
    - May have bias in attention (unlike LLaMA)
    - Requires trust_remote_code=True for loading

    We inherit from LlamaHandler since the weight layout is the same.
    """

    def __init__(self, model: nn.Module, config: MLAConfig):
        # Call grandparent init
        super(LlamaHandler, self).__init__(model, config)

        # Verify model structure
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Model does not have 'model.layers' attribute. Is this a Qwen model?"
            )

    def extract_qkv_biases(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract biases - Qwen may have biases unlike LLaMA."""
        attn = self.get_attention_module(layer_idx)

        # Qwen2 typically has bias=True for q, k, v projections
        b_q = attn.q_proj.bias.data.clone() if hasattr(attn.q_proj, 'bias') and attn.q_proj.bias is not None else None
        b_k = attn.k_proj.bias.data.clone() if hasattr(attn.k_proj, 'bias') and attn.k_proj.bias is not None else None
        b_v = attn.v_proj.bias.data.clone() if hasattr(attn.v_proj, 'bias') and attn.v_proj.bias is not None else None
        b_o = attn.o_proj.bias.data.clone() if hasattr(attn.o_proj, 'bias') and attn.o_proj.bias is not None else None

        return b_q, b_k, b_v, b_o


class QwenAttentionAdapter(LlamaAttentionAdapter):
    """Adapter for Qwen attention.

    Similar to LLaMA adapter with potential bias handling.
    """

    def __init__(self, mla_attention: nn.Module, config: MLAConfig):
        super().__init__(mla_attention, config)
