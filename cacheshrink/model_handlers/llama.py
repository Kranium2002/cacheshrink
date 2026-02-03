"""Handler for LLaMA models."""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .base import ModelHandler
from ..config import MLAConfig


class LlamaHandler(ModelHandler):
    """Handler for LLaMA model family.

    LLaMA characteristics:
    - Separate q_proj, k_proj, v_proj, o_proj Linear layers
    - RoPE for position embeddings
    - Supports both MHA and GQA (num_key_value_heads < num_attention_heads)
    - No bias in attention projections (typically)
    - RMSNorm instead of LayerNorm
    """

    def __init__(self, model: nn.Module, config: MLAConfig):
        super().__init__(model, config)
        # Verify model structure
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Model does not have 'model.layers' attribute. Is this a LLaMA model?"
            )

    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get attention module for a layer."""
        return self.model.model.layers[layer_idx].self_attn

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get full transformer layer."""
        return self.model.model.layers[layer_idx]

    def extract_qkv_weights(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract Q, K, V, O weights from separate projections.

        LLaMA uses standard nn.Linear with weight shape (out_features, in_features).
        For GQA: k_proj and v_proj have shape (n_kv_heads * d_head, d_model).
        """
        attn = self.get_attention_module(layer_idx)

        W_q = attn.q_proj.weight.data  # (n_heads * d_head, d_model)
        W_k = attn.k_proj.weight.data  # (n_kv_heads * d_head, d_model)
        W_v = attn.v_proj.weight.data  # (n_kv_heads * d_head, d_model)
        W_o = attn.o_proj.weight.data  # (d_model, n_heads * d_head)

        return W_q, W_k, W_v, W_o

    def extract_qkv_biases(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract biases from separate projections.

        LLaMA typically doesn't use biases, but we handle it if present.
        """
        attn = self.get_attention_module(layer_idx)

        b_q = attn.q_proj.bias.data if attn.q_proj.bias is not None else None
        b_k = attn.k_proj.bias.data if attn.k_proj.bias is not None else None
        b_v = attn.v_proj.bias.data if attn.v_proj.bias is not None else None
        b_o = attn.o_proj.bias.data if attn.o_proj.bias is not None else None

        return b_q, b_k, b_v, b_o

    def replace_attention(self, layer_idx: int, mla_attention: nn.Module) -> None:
        """Replace attention with MLA attention."""
        self.model.model.layers[layer_idx].self_attn = mla_attention

    def get_attention_attribute_name(self) -> str:
        return "self_attn"

    def get_layer_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.self_attn"

    def get_embed_tokens(self) -> nn.Module:
        """Get token embedding."""
        return self.model.model.embed_tokens

    def get_output_layer(self) -> nn.Module:
        """Get LM head."""
        return self.model.lm_head


class LlamaAttentionAdapter(nn.Module):
    """Adapter to make MLA attention compatible with LLaMA's forward signature.

    LLaMA's attention has a specific forward signature with additional arguments
    for cache position, etc. This adapter wraps MLAAttention to match.

    Cache handling:
    - MLA stores compressed c_kv of shape (batch, seq_len, 2*d_latent)
    - DynamicCache expects K and V of shape (batch, n_kv_heads, seq_len, head_dim)
    - We store c_kv as "keys" by reshaping, with dummy "values"
    """

    def __init__(self, mla_attention: nn.Module, config: MLAConfig):
        super().__init__()
        self.mla = mla_attention
        self.config = config

        # Attributes that LLaMA code might access
        self.num_heads = config.n_heads
        self.num_key_value_heads = config.n_kv_heads
        self.head_dim = config.d_head
        self.hidden_size = config.d_model

        # Cache dimensions: c_kv has shape (batch, seq_len, 2*d_latent)
        # We'll reshape to (batch, 1, seq_len, 2*d_latent) to fit DynamicCache
        self._cache_d_latent = config.computed_d_latent

    def _c_kv_to_cache_format(self, c_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert c_kv to cache format.

        c_kv: (batch, seq_len, 2*d_latent)
        Returns: (keys, values) where keys = (batch, 1, seq_len, 2*d_latent), values = dummy
        """
        # Reshape c_kv to look like K: (batch, n_heads=1, seq_len, head_dim=2*d_latent)
        keys = c_kv.unsqueeze(1)  # (batch, 1, seq_len, 2*d_latent)
        # Create dummy values of same shape
        values = torch.zeros_like(keys)
        return keys, values

    def _cache_to_c_kv(self, keys: torch.Tensor) -> torch.Tensor:
        """Convert cache format back to c_kv.

        keys: (batch, 1, seq_len, 2*d_latent)
        Returns: c_kv (batch, seq_len, 2*d_latent)
        """
        return keys.squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,  # Note: plural to match transformers' calling convention
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with LLaMA compatible signature.

        Handles various LLaMA versions and their cache formats.
        """
        layer_idx = self.mla.layer_idx

        # Handle different cache formats
        mla_past = None
        if past_key_values is not None:
            if hasattr(past_key_values, 'layers') and layer_idx < len(past_key_values.layers):
                # New transformers DynamicCache with layers list
                layer = past_key_values.layers[layer_idx]
                if hasattr(layer, 'keys') and layer.keys is not None and layer.keys.numel() > 0:
                    c_kv = self._cache_to_c_kv(layer.keys)
                    mla_past = (c_kv,)
            elif hasattr(past_key_values, 'key_cache'):
                # Older DynamicCache format
                if layer_idx < len(past_key_values.key_cache) and past_key_values.key_cache[layer_idx] is not None:
                    c_kv = self._cache_to_c_kv(past_key_values.key_cache[layer_idx])
                    mla_past = (c_kv,)
            elif isinstance(past_key_values, tuple) and len(past_key_values) >= 1:
                # Direct tuple format (our format)
                expected_cache_dim = 2 * self._cache_d_latent
                if past_key_values[0].shape[-1] == expected_cache_dim:
                    mla_past = past_key_values

        # Call MLA attention
        attn_output, new_past, attn_weights = self.mla(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=mla_past,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Update cache if needed
        present_key_values = None
        if use_cache and new_past is not None:
            c_kv_full = new_past[0]  # MLA returns (c_kv,) where c_kv includes old + new

            if past_key_values is not None and hasattr(past_key_values, 'update'):
                # MLA attention concatenates past_c_kv + new_c_kv internally
                # We only want to update with the NEW portion
                # Determine how many tokens were already in cache
                if mla_past is not None:
                    past_seq_len = mla_past[0].shape[1]
                else:
                    past_seq_len = 0

                # Extract only the new c_kv (the part after past_seq_len)
                c_kv_new = c_kv_full[:, past_seq_len:, :]

                # Update DynamicCache with only the new tokens
                keys, values = self._c_kv_to_cache_format(c_kv_new)
                past_key_values.update(keys, values, layer_idx=layer_idx)
                present_key_values = past_key_values
            else:
                # Return as tuple (legacy format) - c_kv_full is correct here
                present_key_values = new_past

        # Format output based on what LLaMA decoder layer expects
        # Note: The decoder layer ignores the second return value for caching
        # because transformers updates the cache in-place
        if output_attentions:
            return (attn_output, attn_weights, present_key_values)
        else:
            return (attn_output, present_key_values)

    def check_orthonormality(self):
        """Delegate to MLA attention."""
        return self.mla.check_orthonormality()
