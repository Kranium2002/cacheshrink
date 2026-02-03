"""Tests for MLAAttention module."""

import pytest
import torch

from cacheshrink.attention import (
    MLAAttention,
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    MLACache,
)


class TestRotaryEmbedding:
    """Tests for RotaryEmbedding."""

    def test_initialization(self, device):
        """Test RoPE initialization."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048).to(device)

        assert rope.dim == 64
        assert rope.max_position_embeddings == 2048
        assert rope.inv_freq.shape == (32,)  # dim // 2
        assert rope.cos_cached.shape == (2048, 64)
        assert rope.sin_cached.shape == (2048, 64)

    def test_forward_without_position_ids(self, device, dtype):
        """Test RoPE forward without explicit position IDs."""
        rope = RotaryEmbedding(dim=64).to(device)

        x = torch.randn(2, 8, 16, 64, device=device, dtype=dtype)  # batch, heads, seq, dim
        cos, sin = rope(x, seq_len=16)

        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    def test_forward_with_position_ids(self, device, dtype):
        """Test RoPE forward with explicit position IDs."""
        rope = RotaryEmbedding(dim=64).to(device)

        x = torch.randn(2, 8, 16, 64, device=device, dtype=dtype)
        position_ids = torch.arange(16, device=device).unsqueeze(0).expand(2, -1)
        cos, sin = rope(x, position_ids=position_ids)

        assert cos.shape == (2, 16, 64)
        assert sin.shape == (2, 16, 64)

    def test_cache_extension(self, device, dtype):
        """Test that cache extends for longer sequences."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=128).to(device)

        assert rope.max_seq_len_cached == 128

        x = torch.randn(1, 1, 256, 64, device=device, dtype=dtype)
        cos, sin = rope(x, seq_len=256)

        assert rope.max_seq_len_cached >= 256


class TestRotateHalf:
    """Tests for rotate_half function."""

    def test_rotate_half_shape(self, device, dtype):
        """Test rotate_half preserves shape."""
        x = torch.randn(2, 8, 16, 64, device=device, dtype=dtype)
        rotated = rotate_half(x)
        assert rotated.shape == x.shape

    def test_rotate_half_values(self, device, dtype):
        """Test rotate_half computes correct values."""
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype, device=device)
        rotated = rotate_half(x)
        expected = torch.tensor([[-5, -6, -7, -8, 1, 2, 3, 4]], dtype=dtype, device=device)
        assert torch.allclose(rotated, expected)


class TestApplyRotaryPosEmb:
    """Tests for apply_rotary_pos_emb function."""

    def test_apply_rope(self, device, dtype):
        """Test RoPE application."""
        batch, n_heads, seq_len, head_dim = 2, 8, 16, 64
        n_kv_heads = 2

        q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, n_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        cos = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
        sin = torch.randn(seq_len, head_dim, device=device, dtype=dtype)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape


class TestMLAAttention:
    """Tests for MLAAttention module."""

    def test_initialization(self, gpt2_config, device, dtype):
        """Test attention initialization."""
        attn = MLAAttention(gpt2_config).to(device, dtype)

        assert attn.n_heads == gpt2_config.n_heads
        assert attn.n_kv_heads == gpt2_config.n_kv_heads
        assert attn.d_head == gpt2_config.d_head
        assert attn.d_model == gpt2_config.d_model

        # Check projections exist
        assert attn.q_proj is not None
        assert attn.o_proj is not None
        assert attn.mla_compression is not None

    def test_forward_basic(self, gpt2_config, device, dtype):
        """Test basic forward pass."""
        attn = MLAAttention(gpt2_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        output, past_kv, attn_weights = attn(
            hidden_states,
            use_cache=False,
            output_attentions=False,
        )

        assert output.shape == hidden_states.shape
        assert past_kv is None
        assert attn_weights is None

    def test_forward_with_cache(self, gpt2_config, device, dtype):
        """Test forward pass with KV caching."""
        attn = MLAAttention(gpt2_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        # First pass - create cache
        output1, past_kv, _ = attn(
            hidden_states,
            use_cache=True,
        )

        assert past_kv is not None
        assert len(past_kv) == 1  # (c_kv,)
        # Cache stores [c_k, c_v] concatenated, so size is 2*d_latent
        assert past_kv[0].shape == (batch_size, seq_len, 2 * gpt2_config.computed_d_latent)

        # Second pass - use cache
        new_hidden = torch.randn(batch_size, 1, gpt2_config.d_model, device=device, dtype=dtype)
        output2, new_past_kv, _ = attn(
            new_hidden,
            past_key_value=past_kv,
            use_cache=True,
        )

        assert output2.shape == (batch_size, 1, gpt2_config.d_model)
        assert new_past_kv[0].shape == (batch_size, seq_len + 1, 2 * gpt2_config.computed_d_latent)

    def test_forward_with_attention_mask(self, gpt2_config, device, dtype):
        """Test forward pass with attention mask."""
        attn = MLAAttention(gpt2_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        # Create causal mask (additive, -inf for masked positions)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        output, _, _ = attn(
            hidden_states,
            attention_mask=mask,
        )

        assert output.shape == hidden_states.shape

    def test_forward_with_output_attentions(self, gpt2_config, device, dtype):
        """Test forward pass returning attention weights."""
        attn = MLAAttention(gpt2_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, gpt2_config.d_model, device=device, dtype=dtype)

        output, _, attn_weights = attn(
            hidden_states,
            output_attentions=True,
        )

        assert attn_weights is not None
        expected_shape = (batch_size, gpt2_config.n_heads, seq_len, seq_len)
        assert attn_weights.shape == expected_shape

    def test_check_orthonormality(self, gpt2_config, device, dtype):
        """Test orthonormality checking."""
        attn = MLAAttention(gpt2_config).to(device, dtype)

        errors = attn.check_orthonormality()

        assert "W_uk" in errors
        assert "W_uv" in errors
        assert errors["W_uk"][0] < 1e-5
        assert errors["W_uv"][0] < 1e-5


class TestMLAAttentionGQA:
    """Tests for MLAAttention with GQA."""

    def test_gqa_forward(self, llama_config, device, dtype):
        """Test forward pass with GQA config."""
        attn = MLAAttention(llama_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, llama_config.d_model, device=device, dtype=dtype)

        output, _, _ = attn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_gqa_n_rep(self, llama_config, device, dtype):
        """Test GQA repetition factor."""
        attn = MLAAttention(llama_config).to(device, dtype)

        assert attn.n_rep == llama_config.n_rep
        assert attn.n_rep == llama_config.n_heads // llama_config.n_kv_heads


class TestMLACache:
    """Tests for MLACache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = MLACache()
        assert len(cache.cache) == 0

    def test_update(self, device, dtype):
        """Test cache update."""
        cache = MLACache()

        c_kv = torch.randn(2, 16, 32, device=device, dtype=dtype)
        result = cache.update(0, c_kv)

        assert result.shape == c_kv.shape
        assert cache.get_seq_length(0) == 16

    def test_update_append(self, device, dtype):
        """Test cache update appends correctly."""
        cache = MLACache()

        c_kv1 = torch.randn(2, 16, 32, device=device, dtype=dtype)
        c_kv2 = torch.randn(2, 8, 32, device=device, dtype=dtype)

        cache.update(0, c_kv1)
        result = cache.update(0, c_kv2)

        assert result.shape == (2, 24, 32)
        assert cache.get_seq_length(0) == 24

    def test_get(self, device, dtype):
        """Test cache retrieval."""
        cache = MLACache()

        c_kv = torch.randn(2, 16, 32, device=device, dtype=dtype)
        cache.update(0, c_kv)

        retrieved = cache.get(0)
        assert torch.equal(retrieved, c_kv)

        # Non-existent layer returns None
        assert cache.get(1) is None

    def test_clear(self, device, dtype):
        """Test cache clearing."""
        cache = MLACache()

        c_kv = torch.randn(2, 16, 32, device=device, dtype=dtype)
        cache.update(0, c_kv)
        cache.update(1, c_kv)

        cache.clear()

        assert cache.get(0) is None
        assert cache.get(1) is None

    def test_memory_usage(self, device, dtype):
        """Test memory usage calculation."""
        cache = MLACache()

        # float32 = 4 bytes per element
        c_kv = torch.randn(2, 16, 32, device=device, dtype=torch.float32)
        cache.update(0, c_kv)

        expected_bytes = 2 * 16 * 32 * 4
        assert cache.get_memory_usage() == expected_bytes
