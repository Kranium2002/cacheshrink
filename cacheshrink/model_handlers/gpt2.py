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

    In transformers >=5.x, GPT-2 passes a DynamicCache object and expects the
    attention module to call cache.update() internally. The return signature is
    (attn_output, attn_weights). We store the compressed c_kv latent inside
    DynamicCache by splitting c_kv into (c_k, c_v) along the last dimension and
    storing them as separate key and value tensors of shape (batch, 1, seq, d_latent).
    """

    def __init__(self, mla_attention: nn.Module):
        super().__init__()
        self.mla = mla_attention

        # Copy attributes that GPT-2 code might access
        self.num_heads = mla_attention.n_heads
        self.head_dim = mla_attention.d_head
        self.split_size = mla_attention.d_model
        self.layer_idx = mla_attention.layer_idx

        self._cache_d_latent = mla_attention.config.computed_d_latent

    def _c_kv_to_cache_format(self, c_kv: torch.Tensor):
        """Split c_kv into two halves and store as (key, value) in DynamicCache.

        c_kv has shape (batch, seq, 2*d_latent) — first half is c_k, second is c_v.
        We add a head dim to get (batch, 1, seq, d_latent) for each.
        """
        c_k, c_v = c_kv.chunk(2, dim=-1)
        return c_k.unsqueeze(1), c_v.unsqueeze(1)

    def _cache_to_c_kv(self, keys: torch.Tensor, values: torch.Tensor = None) -> torch.Tensor:
        """Reconstruct c_kv from DynamicCache key/value tensors."""
        c_k = keys.squeeze(1)
        if values is not None:
            c_v = values.squeeze(1)
            return torch.cat([c_k, c_v], dim=-1)
        return c_k

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
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with GPT-2 compatible signature.

        Supports both legacy tuple caches and transformers >=5.x DynamicCache.
        """
        # --- Extract c_kv from cache (if any) ---
        mla_past = None
        cache_obj = None  # DynamicCache reference for update

        # Prefer DynamicCache (transformers >=5.x)
        past = layer_past if layer_past is not None else past_key_values
        if past is not None and hasattr(past, "update"):
            # DynamicCache path
            cache_obj = past
            try:
                # DynamicCache stores per-layer; check if our layer has data
                seq_len = 0
                if hasattr(past, "get_seq_length"):
                    try:
                        seq_len = past.get_seq_length(self.layer_idx)
                    except TypeError:
                        seq_len = past.get_seq_length()
                if seq_len > 0:
                    # Retrieve stored key+value tensors and reconstruct c_kv
                    cached_key = None
                    cached_val = None
                    if hasattr(past, "key_cache") and self.layer_idx < len(past.key_cache):
                        cached_key = past.key_cache[self.layer_idx]
                        if hasattr(past, "value_cache") and self.layer_idx < len(past.value_cache):
                            cached_val = past.value_cache[self.layer_idx]
                    elif hasattr(past, "layers") and self.layer_idx < len(past.layers):
                        cached_key = past.layers[self.layer_idx].keys
                        cached_val = past.layers[self.layer_idx].values

                    if cached_key is not None and cached_key.numel() > 0:
                        c_kv = self._cache_to_c_kv(cached_key, cached_val)
                        mla_past = (c_kv,)
            except (IndexError, AttributeError):
                # Cache structure/layout not as expected; treat as no past available
                mla_past = None
        elif past is not None:
            # Legacy tuple path
            if isinstance(past, tuple) and len(past) >= 1:
                t = past[0]
                if isinstance(t, torch.Tensor):
                    expected_dim = 2 * self._cache_d_latent
                    if t.dim() == 3 and t.shape[-1] == expected_dim:
                        mla_past = (t,)
                    elif t.dim() == 4 and t.shape[-1] == expected_dim:
                        mla_past = (self._cache_to_c_kv(t),)

        # Handle GPT-2's attention mask format
        # GPT-2 uses (batch, 1, 1, seq) with 0 for attend, large negative for mask
        if attention_mask is not None and attention_mask.dim() == 4:
            pass  # Keep as is, it's already additive

        # Call MLA attention
        attn_output, new_past, attn_weights = self.mla(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=mla_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # --- Store c_kv back into DynamicCache ---
        if use_cache and new_past is not None and cache_obj is not None:
            c_kv_full = new_past[0]  # (batch, full_seq, dim)
            # Only store the NEW tokens (DynamicCache accumulates)
            past_seq_len = mla_past[0].shape[1] if mla_past is not None else 0
            c_kv_new = c_kv_full[:, past_seq_len:, :]
            keys, values = self._c_kv_to_cache_format(c_kv_new)
            cache_obj.update(keys, values, self.layer_idx, {"cache_position": cache_position})

        # Return (attn_output, attn_weights) — transformers >=5.x signature
        return attn_output, attn_weights

    def check_orthonormality(self):
        """Delegate to MLA attention."""
        return self.mla.check_orthonormality()
