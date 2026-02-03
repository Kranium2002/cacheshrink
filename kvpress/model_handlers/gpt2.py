"""Handler for GPT-2 models."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .base import ModelHandler
from ..config import MLAConfig


class GPT2Handler(ModelHandler):
    """Handler for GPT-2 model family.

    GPT-2 quirks:
    - Uses Conv1D instead of Linear (weight is transposed: [in_features, out_features])
    - Combined c_attn projection for Q, K, V
    - Uses learned position embeddings (not RoPE)
    - Multi-head attention (MHA), not GQA
    """

    def __init__(self, model: nn.Module, config: MLAConfig):
        super().__init__(model, config)
        # Verify model structure
        if not hasattr(model, "transformer"):
            raise ValueError("Model does not have 'transformer' attribute. Is this a GPT-2 model?")

    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get attention module for a layer."""
        return self.model.transformer.h[layer_idx].attn

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get full transformer layer."""
        return self.model.transformer.h[layer_idx]

    def extract_qkv_weights(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract Q, K, V weights from combined c_attn.

        GPT-2 uses Conv1D with weight shape (in_features, out_features).
        c_attn projects to 3 * n_embd (Q, K, V concatenated).
        """
        attn = self.get_attention_module(layer_idx)

        # c_attn.weight shape: (n_embd, 3 * n_embd) in Conv1D format
        # Need to transpose to get (3 * n_embd, n_embd) in Linear format
        c_attn_weight = attn.c_attn.weight.data.T  # (3 * d_model, d_model)

        # Split into Q, K, V
        d_model = self.config.d_model
        W_q = c_attn_weight[:d_model, :]      # (d_model, d_model)
        W_k = c_attn_weight[d_model:2*d_model, :]  # (d_model, d_model)
        W_v = c_attn_weight[2*d_model:, :]    # (d_model, d_model)

        # Output projection (c_proj)
        # c_proj.weight shape: (n_embd, n_embd) in Conv1D format
        W_o = attn.c_proj.weight.data.T  # (d_model, d_model)

        return W_q, W_k, W_v, W_o

    def extract_qkv_biases(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract biases from combined c_attn."""
        attn = self.get_attention_module(layer_idx)

        # c_attn.bias shape: (3 * n_embd,)
        c_attn_bias = attn.c_attn.bias.data if attn.c_attn.bias is not None else None

        if c_attn_bias is not None:
            d_model = self.config.d_model
            b_q = c_attn_bias[:d_model]
            b_k = c_attn_bias[d_model:2*d_model]
            b_v = c_attn_bias[2*d_model:]
        else:
            b_q, b_k, b_v = None, None, None

        # Output projection bias
        b_o = attn.c_proj.bias.data if attn.c_proj.bias is not None else None

        return b_q, b_k, b_v, b_o

    def replace_attention(self, layer_idx: int, mla_attention: nn.Module) -> None:
        """Replace attention with MLA attention."""
        self.model.transformer.h[layer_idx].attn = mla_attention

    def get_attention_attribute_name(self) -> str:
        return "attn"

    def get_layer_prefix(self, layer_idx: int) -> str:
        return f"transformer.h.{layer_idx}.attn"

    def get_embed_tokens(self) -> nn.Module:
        """Get token embedding."""
        return self.model.transformer.wte

    def get_output_layer(self) -> nn.Module:
        """Get LM head."""
        return self.model.lm_head


class GPT2AttentionAdapter(nn.Module):
    """Adapter to make MLA attention compatible with GPT-2's forward signature.

    GPT-2's attention forward has a different signature than LLaMA-style models.
    This adapter wraps MLAAttention to match GPT-2's expectations.
    """

    def __init__(self, mla_attention: nn.Module):
        super().__init__()
        self.mla = mla_attention

        # Copy attributes that GPT-2 code might access
        self.num_heads = mla_attention.n_heads
        self.head_dim = mla_attention.d_head
        self.split_size = mla_attention.d_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with GPT-2 compatible signature.

        GPT-2 uses (key, value) tuple for layer_past, but we store (c_kv,).
        Newer transformers versions use past_key_values instead of layer_past.
        """
        # Handle both old (layer_past) and new (past_key_values) API
        past = layer_past if layer_past is not None else past_key_values

        # Convert GPT-2's past format
        if past is not None:
            # past might be in old format (key, value) or our format (c_kv,)
            if isinstance(past, tuple) and len(past) == 2:
                # Check if it's our format - c_kv would have d_latent as last dim
                if past[0].shape[-1] == self.mla.mla_compression.d_latent:
                    # Our format stored as (c_kv, None) for compatibility
                    past_key_value = (past[0],)
                else:
                    # Old format (key, value) - start fresh
                    past_key_value = None
            elif isinstance(past, tuple) and len(past) == 1:
                # Our format (c_kv,)
                past_key_value = past
            else:
                past_key_value = None
        else:
            past_key_value = None

        # Handle GPT-2's attention mask format
        # GPT-2 uses (batch, 1, 1, seq) with 0 for attend, large negative for mask
        if attention_mask is not None and attention_mask.dim() == 4:
            # Already in GPT-2 format, convert to additive mask
            # GPT-2 mask: 0 = attend, -10000 = mask
            pass  # Keep as is, it's already additive

        # Call MLA attention
        attn_output, new_past, attn_weights = self.mla(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # Format output as GPT-2 expects
        # GPT-2 expects (attn_output, present) where present is (key, value) tuple
        # We return (c_kv, c_kv) to maintain tuple structure (both same for compatibility)
        outputs = (attn_output,)
        if use_cache and new_past is not None:
            # Return as (c_kv, c_kv) tuple for GPT-2 compatibility
            outputs = outputs + ((new_past[0], new_past[0]),)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs

    def check_orthonormality(self):
        """Delegate to MLA attention."""
        return self.mla.check_orthonormality()
