"""Tests for GenericHandler and GenericAttentionAdapter."""

import warnings
import pytest
import torch
import torch.nn as nn

from cacheshrink.config import MLAConfig
from cacheshrink.model_handlers import (
    get_handler,
    get_attention_adapter,
    GenericHandler,
    GenericAttentionAdapter,
)
from cacheshrink.attention import MLAAttention


# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


class MockLlamaAttention(nn.Module):
    """LLaMA-style attention with separate q/k/v/o projections."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_head):
        super().__init__()
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)


class MockLlamaLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_head):
        super().__init__()
        self.self_attn = MockLlamaAttention(d_model, n_heads, n_kv_heads, d_head)
        self.mlp = nn.Linear(d_model, d_model)


class MockLlamaModel(nn.Module):
    """Mock model with model.model.layers[i].self_attn pattern."""

    def __init__(self, d_model=64, n_heads=4, n_kv_heads=4, d_head=16, n_layers=2, vocab_size=1000):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLlamaLayer(d_model, n_heads, n_kv_heads, d_head)
            for _ in range(n_layers)
        ])
        self.model.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


class MockGPT2Attention(nn.Module):
    """GPT-2 style attention with combined c_attn (Conv1D-like)."""

    def __init__(self, d_model):
        super().__init__()
        # Conv1D style: weight shape is (in_features, out_features)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        # Transpose weights to simulate Conv1D layout
        with torch.no_grad():
            self.c_attn.weight.data = self.c_attn.weight.data.T.contiguous()
            self.c_proj.weight.data = self.c_proj.weight.data.T.contiguous()


class MockGPT2Layer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = MockGPT2Attention(d_model)
        self.mlp = nn.Linear(d_model, d_model)


class MockGPT2Model(nn.Module):
    """Mock model with model.transformer.h[i].attn pattern."""

    def __init__(self, d_model=64, n_layers=2, vocab_size=1000):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([
            MockGPT2Layer(d_model) for _ in range(n_layers)
        ])
        self.transformer.wte = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llama_mock_model():
    return MockLlamaModel()


@pytest.fixture
def gpt2_mock_model():
    return MockGPT2Model()


# ---------------------------------------------------------------------------
# Tests: GenericHandler auto-discovery
# ---------------------------------------------------------------------------


class TestGenericHandlerDiscovery:
    """Test auto-discovery of model structure."""

    def test_finds_llama_style_layers(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        assert handler._layer_list is not None
        assert len(handler._layer_list) == 2

    def test_finds_gpt2_style_layers(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        assert handler._layer_list is not None
        assert len(handler._layer_list) == 2

    def test_finds_llama_attn_attr(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        assert handler._attn_attr == "self_attn"

    def test_finds_gpt2_attn_attr(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        assert handler._attn_attr == "attn"

    def test_detects_separate_projections(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        assert handler._proj_style == "separate"

    def test_detects_combined_projections(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        assert handler._proj_style == "combined"


# ---------------------------------------------------------------------------
# Tests: weight/bias extraction
# ---------------------------------------------------------------------------


class TestGenericHandlerWeights:
    """Test weight and bias extraction."""

    def test_llama_style_weight_extraction(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        W_q, W_k, W_v, W_o = handler.extract_qkv_weights(0)

        assert W_q.shape == (64, 64)  # n_heads*d_head x d_model
        assert W_k.shape == (64, 64)  # n_kv_heads*d_head x d_model
        assert W_v.shape == (64, 64)
        assert W_o.shape == (64, 64)

    def test_llama_style_no_bias(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        b_q, b_k, b_v, b_o = handler.extract_qkv_biases(0)

        assert b_q is None
        assert b_k is None
        assert b_v is None
        assert b_o is None

    def test_gpt2_style_weight_extraction(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        W_q, W_k, W_v, W_o = handler.extract_qkv_weights(0)

        assert W_q.shape == (64, 64)
        assert W_k.shape == (64, 64)
        assert W_v.shape == (64, 64)
        assert W_o.shape == (64, 64)

    def test_gpt2_style_has_bias(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        b_q, b_k, b_v, b_o = handler.extract_qkv_biases(0)

        # Linear layers have bias by default
        assert b_q is not None
        assert b_k is not None
        assert b_v is not None
        assert b_o is not None


# ---------------------------------------------------------------------------
# Tests: replace_attention
# ---------------------------------------------------------------------------


class TestGenericHandlerReplace:
    """Test attention replacement."""

    def test_replace_llama_style(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)

        dummy = nn.Linear(10, 10)
        handler.replace_attention(0, dummy)

        assert handler.get_attention_module(0) is dummy

    def test_replace_gpt2_style(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)

        dummy = nn.Linear(10, 10)
        handler.replace_attention(0, dummy)

        assert handler.get_attention_module(0) is dummy


# ---------------------------------------------------------------------------
# Tests: embed_tokens / output_layer
# ---------------------------------------------------------------------------


class TestGenericHandlerModules:
    """Test embed_tokens and output_layer discovery."""

    def test_llama_embed_tokens(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        embed = handler.get_embed_tokens()
        assert isinstance(embed, nn.Embedding)
        assert embed.num_embeddings == 1000

    def test_llama_output_layer(self, llama_mock_model, generic_config):
        handler = GenericHandler(llama_mock_model, generic_config)
        lm_head = handler.get_output_layer()
        assert isinstance(lm_head, nn.Linear)
        assert lm_head.out_features == 1000

    def test_gpt2_embed_tokens(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        embed = handler.get_embed_tokens()
        assert isinstance(embed, nn.Embedding)
        assert embed.num_embeddings == 1000

    def test_gpt2_output_layer(self, gpt2_mock_model, generic_config_no_rope):
        handler = GenericHandler(gpt2_mock_model, generic_config_no_rope)
        lm_head = handler.get_output_layer()
        assert isinstance(lm_head, nn.Linear)
        assert lm_head.out_features == 1000


# ---------------------------------------------------------------------------
# Tests: get_handler() fallback
# ---------------------------------------------------------------------------


class TestGetHandlerFallback:
    """Test that get_handler falls back to GenericHandler for unknown types."""

    def test_get_handler_returns_generic_for_unknown_type(self, llama_mock_model, generic_config):
        handler = get_handler(llama_mock_model, generic_config)
        assert isinstance(handler, GenericHandler)

    def test_get_attention_adapter_returns_generic_for_unknown_type(self):
        adapter_cls = get_attention_adapter("generic")
        assert adapter_cls is GenericAttentionAdapter

    def test_get_attention_adapter_returns_generic_for_totally_unknown(self):
        adapter_cls = get_attention_adapter("phi3_whatever")
        assert adapter_cls is GenericAttentionAdapter


# ---------------------------------------------------------------------------
# Tests: _detect_model_type returns "generic" for unknown
# ---------------------------------------------------------------------------


class TestDetectModelType:
    """Test that _detect_model_type returns 'generic' for unknown model types."""

    def test_unknown_type_returns_generic(self):
        class FakeConfig:
            model_type = "stablelm"
            architectures = ["StableLmForCausalLM"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = MLAConfig._detect_model_type(FakeConfig())
            assert result == "generic"
            assert len(w) == 1
            assert "generic" in str(w[0].message).lower()

    def test_known_type_routes_to_generic(self):
        """LLaMA, Mistral, Qwen now route through GenericHandler."""
        for model_type in ("llama", "mistral", "qwen2"):
            class FakeConfig:
                pass
            FakeConfig.model_type = model_type

            result = MLAConfig._detect_model_type(FakeConfig())
            assert result == "generic"


# ---------------------------------------------------------------------------
# Tests: _extract_config for generic
# ---------------------------------------------------------------------------


class TestExtractConfigGeneric:
    """Test _extract_config with generic model type."""

    def test_extract_config_standard_attributes(self):
        class FakeConfig:
            model_type = "phi"
            num_attention_heads = 32
            num_key_value_heads = 8
            hidden_size = 2048
            num_hidden_layers = 24
            max_position_embeddings = 4096
            vocab_size = 50000
            attention_bias = False
            rms_norm_eps = 1e-6
            rope_theta = 500000.0
            rope_scaling = None

        config = MLAConfig._extract_config(FakeConfig(), "generic")

        assert config["n_heads"] == 32
        assert config["n_kv_heads"] == 8
        assert config["d_model"] == 2048
        assert config["d_head"] == 64
        assert config["n_layers"] == 24
        assert config["max_position_embeddings"] == 4096
        assert config["vocab_size"] == 50000
        assert config["use_bias"] is False
        assert config["rope_theta"] == 500000.0
        assert config["extra_config"]["uses_rope"] is True

    def test_extract_config_gpt2_style_attributes(self):
        class FakeConfig:
            model_type = "gptj"
            n_head = 16
            n_embd = 1024
            n_layer = 12
            n_positions = 2048
            vocab_size = 50257
            bias = True
            layer_norm_epsilon = 1e-5

        config = MLAConfig._extract_config(FakeConfig(), "generic")

        assert config["n_heads"] == 16
        assert config["n_kv_heads"] == 16  # MHA fallback
        assert config["d_model"] == 1024
        assert config["n_layers"] == 12
        assert config["extra_config"]["uses_rope"] is False  # n_positions without rope_theta


# ---------------------------------------------------------------------------
# Tests: GenericAttentionAdapter forward pass
# ---------------------------------------------------------------------------


class TestGenericAttentionAdapter:
    """Test GenericAttentionAdapter forward pass."""

    def test_forward_with_rope(self, generic_config):
        mla = MLAAttention(generic_config, layer_idx=0)
        adapter = GenericAttentionAdapter(mla, generic_config)

        batch_size, seq_len = 2, 8
        hidden = torch.randn(batch_size, seq_len, generic_config.d_model)

        output = adapter(hidden)
        assert output[0].shape == (batch_size, seq_len, generic_config.d_model)

    def test_forward_without_rope(self, generic_config_no_rope):
        mla = MLAAttention(generic_config_no_rope, layer_idx=0)
        adapter = GenericAttentionAdapter(mla, generic_config_no_rope)

        batch_size, seq_len = 2, 8
        hidden = torch.randn(batch_size, seq_len, generic_config_no_rope.d_model)

        output = adapter(hidden)
        assert output[0].shape == (batch_size, seq_len, generic_config_no_rope.d_model)

    def test_forward_with_use_cache(self, generic_config):
        mla = MLAAttention(generic_config, layer_idx=0)
        adapter = GenericAttentionAdapter(mla, generic_config)

        batch_size, seq_len = 1, 4
        hidden = torch.randn(batch_size, seq_len, generic_config.d_model)

        output = adapter(hidden, use_cache=True)
        assert output[0].shape == (batch_size, seq_len, generic_config.d_model)
        # Second element should be the cache
        assert output[1] is not None

    def test_check_orthonormality(self, generic_config):
        mla = MLAAttention(generic_config, layer_idx=0)
        adapter = GenericAttentionAdapter(mla, generic_config)
        errors = adapter.check_orthonormality()
        assert "W_uk" in errors
        assert "W_uv" in errors


# ---------------------------------------------------------------------------
# Tests: BFS fallback for unusual structures
# ---------------------------------------------------------------------------


class TestBFSFallback:
    """Test BFS fallback for models with unusual structure."""

    def test_bfs_finds_layers(self, generic_config):
        """Test that BFS finds layers nested in an unusual path."""
        class UnusualModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.encoder = nn.Module()
                self.backbone.encoder.layers = nn.ModuleList([
                    MockLlamaLayer(64, 4, 4, 16) for _ in range(2)
                ])
                self.lm_head = nn.Linear(64, 1000, bias=False)

        model = UnusualModel()
        handler = GenericHandler(model, generic_config)
        assert handler._layer_list is not None
        assert len(handler._layer_list) == 2

    def test_fails_when_no_layers_found(self, generic_config):
        """Test that handler raises when no layer list is found."""
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

        model = EmptyModel()
        with pytest.raises(ValueError, match="Could not find transformer layer list"):
            GenericHandler(model, generic_config)
