"""Shared test fixtures for grassmann-mla tests."""

import pytest
import torch
import torch.nn as nn

from cacheshrink.config import MLAConfig


@pytest.fixture
def device():
    """Get test device (CPU for CI, CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float32


@pytest.fixture
def gpt2_config():
    """Create a minimal GPT-2 style MLAConfig for testing."""
    return MLAConfig(
        model_name="test-gpt2",
        model_type="gpt2",
        n_heads=4,
        n_kv_heads=4,
        d_model=64,
        d_head=16,
        n_layers=2,
        compression_ratio=4.0,
        max_position_embeddings=128,
        vocab_size=1000,
        use_bias=True,
        layer_norm_eps=1e-5,
    )


@pytest.fixture
def llama_config():
    """Create a minimal LLaMA style MLAConfig for testing."""
    return MLAConfig(
        model_name="test-llama",
        model_type="llama",
        n_heads=8,
        n_kv_heads=2,  # GQA: 4 query heads per KV head
        d_model=64,
        d_head=8,
        n_layers=2,
        compression_ratio=4.0,
        max_position_embeddings=128,
        rope_theta=10000.0,
        vocab_size=1000,
        use_bias=False,
        layer_norm_eps=1e-5,
    )


@pytest.fixture
def sample_hidden_states(gpt2_config, device, dtype):
    """Generate sample hidden states for testing."""
    batch_size = 2
    seq_len = 16
    return torch.randn(
        batch_size, seq_len, gpt2_config.d_model,
        device=device, dtype=dtype
    )


@pytest.fixture
def sample_kv_weights(gpt2_config, device, dtype):
    """Generate sample K and V projection weights."""
    d_model = gpt2_config.d_model
    d_kv = gpt2_config.d_kv
    W_k = torch.randn(d_kv, d_model, device=device, dtype=dtype)
    W_v = torch.randn(d_kv, d_model, device=device, dtype=dtype)
    return W_k, W_v


@pytest.fixture
def sample_calibration_data(gpt2_config, device, dtype):
    """Generate sample calibration data."""
    n_samples = 100
    return torch.randn(
        n_samples, gpt2_config.d_model,
        device=device, dtype=dtype
    )


class MinimalGPT2Attention(nn.Module):
    """Minimal GPT-2 style attention for testing."""

    def __init__(self, config: MLAConfig):
        super().__init__()
        self.n_head = config.n_heads
        self.n_embd = config.d_model

        # GPT-2 uses Conv1D but we'll use Linear for simplicity
        # Just need the weight shapes to match
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        # Transpose weights to match Conv1D layout
        with torch.no_grad():
            self.c_attn.weight.data = self.c_attn.weight.data.T.contiguous()
            self.c_proj.weight.data = self.c_proj.weight.data.T.contiguous()


class MinimalLlamaAttention(nn.Module):
    """Minimal LLaMA style attention for testing."""

    def __init__(self, config: MLAConfig):
        super().__init__()
        self.num_heads = config.n_heads
        self.num_key_value_heads = config.n_kv_heads
        self.head_dim = config.d_head
        self.hidden_size = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)


@pytest.fixture
def minimal_gpt2_attention(gpt2_config):
    """Create minimal GPT-2 attention module."""
    return MinimalGPT2Attention(gpt2_config)


@pytest.fixture
def minimal_llama_attention(llama_config):
    """Create minimal LLaMA attention module."""
    return MinimalLlamaAttention(llama_config)
