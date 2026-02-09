"""Tests for model handlers."""

import pytest
import torch
import torch.nn as nn

from cacheshrink.model_handlers import (
    get_handler,
    get_attention_adapter,
    GPT2Handler,
    HANDLER_REGISTRY,
    ADAPTER_REGISTRY,
)


class TestHandlerRegistry:
    """Tests for handler registry."""

    def test_all_handlers_registered(self):
        """Test that all expected handlers are in registry."""
        assert "gpt2" in HANDLER_REGISTRY

    def test_all_adapters_registered(self):
        """Test that all expected adapters are in registry."""
        assert "gpt2" in ADAPTER_REGISTRY

    def test_get_attention_adapter(self):
        """Test getting attention adapter by model type."""
        from cacheshrink.model_handlers.gpt2 import GPT2AttentionAdapter

        assert get_attention_adapter("gpt2") == GPT2AttentionAdapter

    def test_get_attention_adapter_unknown_falls_back_to_generic(self):
        """Test getting adapter for unknown model type falls back to GenericAttentionAdapter."""
        from cacheshrink.model_handlers import GenericAttentionAdapter
        assert get_attention_adapter("invalid_model") is GenericAttentionAdapter


class TestGPT2Handler:
    """Tests for GPT-2 handler."""

    def test_extract_qkv_weights(self, gpt2_config, minimal_gpt2_attention):
        """Test weight extraction from GPT-2 attention."""
        # Create a minimal model structure
        class FakeModel:
            class Transformer:
                def __init__(self, attn):
                    self.h = [type('Layer', (), {'attn': attn})()]
            def __init__(self, attn):
                self.transformer = self.Transformer(attn)

        model = FakeModel(minimal_gpt2_attention)
        handler = GPT2Handler(model, gpt2_config)

        W_q, W_k, W_v, W_o = handler.extract_qkv_weights(0)

        # Check shapes (GPT-2 style: d_model for all)
        assert W_q.shape == (gpt2_config.d_model, gpt2_config.d_model)
        assert W_k.shape == (gpt2_config.d_model, gpt2_config.d_model)
        assert W_v.shape == (gpt2_config.d_model, gpt2_config.d_model)
        assert W_o.shape == (gpt2_config.d_model, gpt2_config.d_model)

    def test_get_attention_attribute_name(self, gpt2_config):
        """Test attention attribute name."""
        class FakeModel:
            class Transformer:
                h = []
            transformer = Transformer()

        model = FakeModel()
        handler = GPT2Handler(model, gpt2_config)

        assert handler.get_attention_attribute_name() == "attn"


class TestGPT2AttentionAdapter:
    """Tests for GPT-2 attention adapter."""

    def test_forward_signature(self, gpt2_config, device, dtype):
        """Test that adapter has correct forward signature."""
        from cacheshrink.attention import MLAAttention
        from cacheshrink.model_handlers.gpt2 import GPT2AttentionAdapter

        mla_attn = MLAAttention(gpt2_config).to(device, dtype)
        adapter = GPT2AttentionAdapter(mla_attn)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        # GPT-2 style forward call
        outputs = adapter(
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        )

        assert outputs[0].shape == hidden_states.shape

    def test_forward_with_cache(self, gpt2_config, device, dtype):
        """Test adapter with caching via DynamicCache (transformers >=5.x)."""
        from transformers.cache_utils import DynamicCache
        from cacheshrink.attention import MLAAttention
        from cacheshrink.model_handlers.gpt2 import GPT2AttentionAdapter

        mla_attn = MLAAttention(gpt2_config).to(device, dtype)
        adapter = GPT2AttentionAdapter(mla_attn)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        # First pass with DynamicCache
        cache = DynamicCache()
        outputs = adapter(hidden_states, past_key_values=cache, use_cache=True)

        # Returns (attn_output, attn_weights) — cache is updated in-place
        assert len(outputs) == 2  # (output, attn_weights)
        assert outputs[0].shape == hidden_states.shape
        assert cache.get_seq_length() == seq_len
