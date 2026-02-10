"""Generic handler for any HuggingFace CausalLM model.

Auto-discovers model structure (layer list, attention modules, projection style)
at init time. Used as a fallback when no specific handler exists for the model type.
"""

from typing import Tuple, Optional
from collections import deque

import torch
import torch.nn as nn

from .base import ModelHandler
from ..config import MLAConfig


class GenericHandler(ModelHandler):
    """Generic handler that auto-discovers model structure.

    Tries known module paths in order, then falls back to heuristic search.
    Caches all discovered attributes at init time for consistent behavior.
    """

    def __init__(self, model: nn.Module, config: MLAConfig):
        super().__init__(model, config)

        # Auto-discover structure
        self._layer_list = self._find_layer_list()
        if self._layer_list is None:
            raise ValueError(
                "Could not find transformer layer list in model. "
                "The model structure is not recognized."
            )

        # Discover attention attribute name from first layer
        sample_layer = self._layer_list[0]
        self._attn_attr = self._find_attn_attr(sample_layer)
        if self._attn_attr is None:
            raise ValueError(
                "Could not find attention module in layer. "
                f"Layer attributes: {[n for n, _ in sample_layer.named_children()]}"
            )

        # Discover projection style from first attention module
        sample_attn = getattr(sample_layer, self._attn_attr)
        self._proj_style, self._proj_info = self._detect_projection_style(sample_attn)

        # Discover embed tokens and output layer
        self._embed_tokens = self._find_embed_tokens()
        self._output_layer = self._find_output_layer()

    def _find_layer_list(self) -> Optional[nn.ModuleList]:
        """Find the transformer layer list by trying known paths, then BFS."""
        known_paths = [
            ("model", "layers"),         # LLaMA, Mistral, Qwen, Gemma, Phi-3
            ("transformer", "h"),        # GPT-2, GPT-J, Falcon
            ("gpt_neox", "layers"),      # Pythia
            ("transformer", "blocks"),   # MPT
            ("model", "decoder", "layers"),  # OPT
        ]

        for path in known_paths:
            obj = self.model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, nn.ModuleList):
                return obj

        # BFS fallback: find any ModuleList with length matching n_layers
        n_layers = self.config.n_layers
        queue = deque([self.model])
        while queue:
            module = queue.popleft()
            for child in module.children():
                if isinstance(child, nn.ModuleList) and len(child) == n_layers:
                    return child
                queue.append(child)

        return None

    def _find_attn_attr(self, layer: nn.Module) -> Optional[str]:
        """Find the attention attribute name on a layer module."""
        # Try known names first
        known_names = ["self_attn", "attn", "attention", "self_attention"]
        for name in known_names:
            if hasattr(layer, name):
                return name

        # Fallback: any child whose name contains "attn" or "attention"
        for name, _ in layer.named_children():
            if "attn" in name.lower() or "attention" in name.lower():
                return name

        return None

    def _detect_projection_style(self, attn: nn.Module) -> Tuple[str, dict]:
        """Detect the projection style of an attention module.

        Returns:
            Tuple of (style, info_dict) where style is one of:
            - "separate": q_proj, k_proj, v_proj, o_proj
            - "combined": c_attn (Conv1D-style) + c_proj
            - "combined_qkv": qkv_proj
        """
        # Check for separate projections (most common: LLaMA, Mistral, Qwen, etc.)
        if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
            # Find output projection
            o_proj_name = None
            for name in ["o_proj", "out_proj", "dense", "c_proj"]:
                if hasattr(attn, name):
                    o_proj_name = name
                    break
            return "separate", {"o_proj": o_proj_name or "o_proj"}

        # Check for combined c_attn (GPT-2 style)
        if hasattr(attn, "c_attn") and hasattr(attn, "c_proj"):
            # Detect if Conv1D (weight shape is (in, out) instead of (out, in))
            w = attn.c_attn.weight.data
            d_model = self.config.d_model
            is_conv1d = (w.shape[0] == d_model and w.shape[1] == 3 * d_model)
            return "combined", {"is_conv1d": is_conv1d}

        # Check for combined qkv_proj
        if hasattr(attn, "qkv_proj"):
            return "combined_qkv", {}

        # Final fallback: look for any combined projection
        for name, mod in attn.named_children():
            if isinstance(mod, nn.Linear):
                if mod.out_features == 3 * self.config.d_model:
                    return "combined", {"attr": name, "is_conv1d": False}

        raise ValueError(
            "Could not detect projection style. "
            f"Attention attributes: {[n for n, _ in attn.named_children()]}"
        )

    def _find_embed_tokens(self) -> Optional[nn.Module]:
        """Find the token embedding module."""
        known_paths = [
            ("model", "embed_tokens"),
            ("transformer", "wte"),
            ("gpt_neox", "embed_in"),
            ("model", "decoder", "embed_tokens"),
        ]

        for path in known_paths:
            obj = self.model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, nn.Embedding):
                return obj

        # Fallback: first Embedding with num_embeddings >= vocab_size
        vocab_size = self.config.vocab_size
        for module in self.model.modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings >= vocab_size:
                return module

        return None

    def _find_output_layer(self) -> Optional[nn.Module]:
        """Find the output/LM head layer."""
        known_names = ["lm_head", "output", "embed_out"]
        for name in known_names:
            if hasattr(self.model, name):
                mod = getattr(self.model, name)
                if isinstance(mod, nn.Linear):
                    return mod

        # Fallback: first top-level Linear with out_features == vocab_size
        vocab_size = self.config.vocab_size
        for name, mod in self.model.named_children():
            if isinstance(mod, nn.Linear) and mod.out_features == vocab_size:
                return mod

        return None

    def extract_qk_norms(self, layer_idx: int):
        attn = self.get_attention_module(layer_idx)
        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        return q_norm, k_norm

    def get_attention_module(self, layer_idx: int) -> nn.Module:
        return getattr(self._layer_list[layer_idx], self._attn_attr)

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        return self._layer_list[layer_idx]

    def extract_qkv_weights(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        attn = self.get_attention_module(layer_idx)

        if self._proj_style == "separate":
            W_q = attn.q_proj.weight.data
            W_k = attn.k_proj.weight.data
            W_v = attn.v_proj.weight.data
            o_name = self._proj_info["o_proj"]
            W_o = getattr(attn, o_name).weight.data
            return W_q, W_k, W_v, W_o

        elif self._proj_style == "combined":
            is_conv1d = self._proj_info.get("is_conv1d", False)
            attr_name = self._proj_info.get("attr", "c_attn")
            c_attn = getattr(attn, attr_name)

            if is_conv1d:
                # Conv1D: weight shape (in_features, out_features) -> transpose
                c_attn_weight = c_attn.weight.data.T
            else:
                c_attn_weight = c_attn.weight.data

            d_q = self.config.n_heads * self.config.d_head
            d_kv = self.config.d_kv

            W_q = c_attn_weight[:d_q, :]
            W_k = c_attn_weight[d_q:d_q + d_kv, :]
            W_v = c_attn_weight[d_q + d_kv:, :]

            c_proj = attn.c_proj
            if is_conv1d:
                W_o = c_proj.weight.data.T
            else:
                W_o = c_proj.weight.data

            return W_q, W_k, W_v, W_o

        elif self._proj_style == "combined_qkv":
            w = attn.qkv_proj.weight.data
            d_q = self.config.n_heads * self.config.d_head
            d_kv = self.config.d_kv

            W_q = w[:d_q, :]
            W_k = w[d_q:d_q + d_kv, :]
            W_v = w[d_q + d_kv:, :]

            # Find output projection
            o_name = None
            for name in ["o_proj", "out_proj", "dense", "c_proj"]:
                if hasattr(attn, name):
                    o_name = name
                    break
            W_o = getattr(attn, o_name).weight.data if o_name else None

            return W_q, W_k, W_v, W_o

        raise ValueError(f"Unknown projection style: {self._proj_style}")

    def extract_qkv_biases(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn = self.get_attention_module(layer_idx)

        if self._proj_style == "separate":
            b_q = attn.q_proj.bias.data if attn.q_proj.bias is not None else None
            b_k = attn.k_proj.bias.data if attn.k_proj.bias is not None else None
            b_v = attn.v_proj.bias.data if attn.v_proj.bias is not None else None
            o_name = self._proj_info["o_proj"]
            o_proj = getattr(attn, o_name)
            b_o = o_proj.bias.data if o_proj.bias is not None else None
            return b_q, b_k, b_v, b_o

        elif self._proj_style == "combined":
            attr_name = self._proj_info.get("attr", "c_attn")
            c_attn = getattr(attn, attr_name)

            c_attn_bias = c_attn.bias.data if c_attn.bias is not None else None
            if c_attn_bias is not None:
                d_q = self.config.n_heads * self.config.d_head
                d_kv = self.config.d_kv
                b_q = c_attn_bias[:d_q]
                b_k = c_attn_bias[d_q:d_q + d_kv]
                b_v = c_attn_bias[d_q + d_kv:]
            else:
                b_q, b_k, b_v = None, None, None

            c_proj = attn.c_proj
            b_o = c_proj.bias.data if c_proj.bias is not None else None
            return b_q, b_k, b_v, b_o

        elif self._proj_style == "combined_qkv":
            bias = attn.qkv_proj.bias.data if attn.qkv_proj.bias is not None else None
            if bias is not None:
                d_q = self.config.n_heads * self.config.d_head
                d_kv = self.config.d_kv
                b_q = bias[:d_q]
                b_k = bias[d_q:d_q + d_kv]
                b_v = bias[d_q + d_kv:]
            else:
                b_q, b_k, b_v = None, None, None

            o_name = None
            for name in ["o_proj", "out_proj", "dense", "c_proj"]:
                if hasattr(attn, name):
                    o_name = name
                    break
            b_o = None
            if o_name:
                o_proj = getattr(attn, o_name)
                b_o = o_proj.bias.data if o_proj.bias is not None else None
            return b_q, b_k, b_v, b_o

        return None, None, None, None

    def replace_attention(self, layer_idx: int, mla_attention: nn.Module) -> None:
        setattr(self._layer_list[layer_idx], self._attn_attr, mla_attention)

    def get_attention_attribute_name(self) -> str:
        return self._attn_attr

    def get_layer_prefix(self, layer_idx: int) -> str:
        # Try to determine the prefix by finding the layer in the model's state dict
        layer_module = self._layer_list[layer_idx]
        for name, mod in self.model.named_modules():
            if mod is layer_module:
                return f"{name}.{self._attn_attr}"
        return f"layer.{layer_idx}.{self._attn_attr}"

    def get_embed_tokens(self) -> nn.Module:
        if self._embed_tokens is None:
            raise ValueError("Could not find token embedding module")
        return self._embed_tokens

    def get_output_layer(self) -> nn.Module:
        if self._output_layer is None:
            raise ValueError("Could not find output/LM head layer")
        return self._output_layer


class GenericAttentionAdapter(nn.Module):
    """Adapter for MLA attention that works with any model's forward signature.

    Handles DynamicCache conversion, position_ids, and different cache formats.
    """

    def __init__(self, mla_attention: nn.Module, config: MLAConfig):
        super().__init__()
        self.mla = mla_attention
        self.config = config

        self.num_heads = config.n_heads
        self.num_key_value_heads = config.n_kv_heads
        self.head_dim = config.d_head
        self.hidden_size = config.d_model

        self._cache_d_latent = config.computed_d_latent

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass compatible with transformers 5.x HF model signatures.

        Returns (attn_output, attn_weights) matching the signature expected by
        decoder layers in transformers 5.x. The DynamicCache is updated in-place
        via cache.update(), not returned in the output tuple.
        """
        layer_idx = self.mla.layer_idx

        # Handle different cache formats
        mla_past = None
        cache_obj = None  # DynamicCache reference for in-place update

        if past_key_values is not None:
            if hasattr(past_key_values, 'update'):
                # DynamicCache path (transformers 5.x)
                cache_obj = past_key_values
                if hasattr(past_key_values, 'layers') and layer_idx < len(past_key_values.layers):
                    layer = past_key_values.layers[layer_idx]
                    if hasattr(layer, 'keys') and layer.keys is not None and layer.keys.numel() > 0:
                        c_kv = self._cache_to_c_kv(layer.keys, layer.values)
                        mla_past = (c_kv,)
            elif isinstance(past_key_values, tuple) and len(past_key_values) >= 1:
                expected_cache_dim = 2 * self._cache_d_latent
                if past_key_values[0].shape[-1] == expected_cache_dim:
                    mla_past = past_key_values

        attn_output, new_past, attn_weights = self.mla(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=mla_past,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        # Store c_kv back into DynamicCache in-place
        if new_past is not None and cache_obj is not None:
            c_kv_full = new_past[0]
            # Only store the NEW tokens (DynamicCache accumulates)
            past_seq_len = mla_past[0].shape[1] if mla_past is not None else 0
            c_kv_new = c_kv_full[:, past_seq_len:, :]
            keys, values = self._c_kv_to_cache_format(c_kv_new)
            cache_obj.update(keys, values, layer_idx=layer_idx)

        # Return (attn_output, attn_weights) — transformers 5.x signature
        return attn_output, attn_weights

    def check_orthonormality(self):
        return self.mla.check_orthonormality()
