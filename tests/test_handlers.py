"""Tests for model handlers."""

import pytest
import torch
import torch.nn as nn

from kvpress.model_handlers import (
    get_handler,
    get_attention_adapter,
    GPT2Handler,
    LlamaHandler,
    MistralHandler,
    QwenHandler,
    HANDLER_REGISTRY,
    ADAPTER_REGISTRY,
)


class TestHandlerRegistry:
    """Tests for handler registry."""

    def test_all_handlers_registered(self):
        """Test that all expected handlers are in registry."""
        assert "gpt2" in HANDLER_REGISTRY
        assert "llama" in HANDLER_REGISTRY
        assert "mistral" in HANDLER_REGISTRY
        assert "qwen" in HANDLER_REGISTRY

    def test_all_adapters_registered(self):
        """Test that all expected adapters are in registry."""
        assert "gpt2" in ADAPTER_REGISTRY
        assert "llama" in ADAPTER_REGISTRY
        assert "mistral" in ADAPTER_REGISTRY
        assert "qwen" in ADAPTER_REGISTRY

    def test_get_attention_adapter(self):
        """Test getting attention adapter by model type."""
        from kvpress.model_handlers.gpt2 import GPT2AttentionAdapter
        from kvpress.model_handlers.llama import LlamaAttentionAdapter

        assert get_attention_adapter("gpt2") == GPT2AttentionAdapter
        assert get_attention_adapter("llama") == LlamaAttentionAdapter

    def test_get_attention_adapter_invalid(self):
        """Test getting adapter for invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            get_attention_adapter("invalid_model")


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


class TestLlamaHandler:
    """Tests for LLaMA handler."""

    def test_extract_qkv_weights(self, llama_config, minimal_llama_attention):
        """Test weight extraction from LLaMA attention."""
        # Create a minimal model structure
        class FakeModel:
            class Model:
                def __init__(self, attn):
                    self.layers = [type('Layer', (), {'self_attn': attn})()]
            def __init__(self, attn):
                self.model = self.Model(attn)

        model = FakeModel(minimal_llama_attention)
        handler = LlamaHandler(model, llama_config)

        W_q, W_k, W_v, W_o = handler.extract_qkv_weights(0)

        # Check shapes (LLaMA style with GQA)
        expected_q_shape = (llama_config.n_heads * llama_config.d_head, llama_config.d_model)
        expected_kv_shape = (llama_config.n_kv_heads * llama_config.d_head, llama_config.d_model)
        expected_o_shape = (llama_config.d_model, llama_config.n_heads * llama_config.d_head)

        assert W_q.shape == expected_q_shape
        assert W_k.shape == expected_kv_shape
        assert W_v.shape == expected_kv_shape
        assert W_o.shape == expected_o_shape

    def test_get_attention_attribute_name(self, llama_config):
        """Test attention attribute name."""
        class FakeModel:
            class Model:
                layers = []
            model = Model()

        model = FakeModel()
        handler = LlamaHandler(model, llama_config)

        assert handler.get_attention_attribute_name() == "self_attn"


class TestMistralHandler:
    """Tests for Mistral handler."""

    def test_inheritance(self):
        """Test that MistralHandler inherits from LlamaHandler."""
        assert issubclass(MistralHandler, LlamaHandler)


class TestQwenHandler:
    """Tests for Qwen handler."""

    def test_inheritance(self):
        """Test that QwenHandler inherits from LlamaHandler."""
        assert issubclass(QwenHandler, LlamaHandler)


class TestGPT2AttentionAdapter:
    """Tests for GPT-2 attention adapter."""

    def test_forward_signature(self, gpt2_config, device, dtype):
        """Test that adapter has correct forward signature."""
        from kvpress.attention import MLAAttention
        from kvpress.model_handlers.gpt2 import GPT2AttentionAdapter

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
        """Test adapter with caching."""
        from kvpress.attention import MLAAttention
        from kvpress.model_handlers.gpt2 import GPT2AttentionAdapter

        mla_attn = MLAAttention(gpt2_config).to(device, dtype)
        adapter = GPT2AttentionAdapter(mla_attn)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        # First pass with cache
        outputs = adapter(hidden_states, use_cache=True)

        assert len(outputs) == 2  # (output, past)
        assert outputs[1] is not None


class TestLlamaAttentionAdapter:
    """Tests for LLaMA attention adapter."""

    def test_forward_signature(self, llama_config, device, dtype):
        """Test that adapter has correct forward signature."""
        from kvpress.attention import MLAAttention
        from kvpress.model_handlers.llama import LlamaAttentionAdapter

        mla_attn = MLAAttention(llama_config).to(device, dtype)
        adapter = LlamaAttentionAdapter(mla_attn, llama_config)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, llama_config.d_model, device=device, dtype=dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # LLaMA style forward call
        outputs = adapter(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

        assert outputs[0].shape == hidden_states.shape
