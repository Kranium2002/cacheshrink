"""Tests for MLACompression module."""

import pytest
import torch

from cacheshrink.compression import MLACompression
from cacheshrink.utils import check_orthonormality


class TestMLACompression:
    """Tests for MLACompression module."""

    def test_initialization(self, gpt2_config, device, dtype):
        """Test compression module initialization."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        assert compression.d_model == gpt2_config.d_model
        assert compression.d_latent == gpt2_config.computed_d_latent
        assert compression.d_kv == gpt2_config.d_kv

        # Check W_down_k and W_down_v shapes (separate compression for K and V)
        assert compression.W_down_k.weight.shape == (gpt2_config.computed_d_latent, gpt2_config.d_model)
        assert compression.W_down_v.weight.shape == (gpt2_config.computed_d_latent, gpt2_config.d_model)

        # Check W_uk, W_uv shapes - stored as (d_kv, d_latent) for geoopt Stiefel
        assert compression.W_uk.shape == (gpt2_config.d_kv, gpt2_config.computed_d_latent)
        assert compression.W_uv.shape == (gpt2_config.d_kv, gpt2_config.computed_d_latent)

    def test_orthonormality_at_init(self, gpt2_config, device, dtype):
        """Test that W_uk and W_uv have orthonormal columns at initialization."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        errors = compression.check_orthonormality()

        # Should be very close to orthonormal (< 1e-5 error)
        assert errors["W_uk"][0] < 1e-5, f"W_uk max error: {errors['W_uk'][0]}"
        assert errors["W_uv"][0] < 1e-5, f"W_uv max error: {errors['W_uv'][0]}"

    def test_compress_shape(self, gpt2_config, sample_hidden_states, device, dtype):
        """Test compression output shape."""
        compression = MLACompression(gpt2_config).to(device, dtype)
        sample_hidden_states = sample_hidden_states.to(device, dtype)

        c_kv = compression.compress(sample_hidden_states)

        batch_size, seq_len, _ = sample_hidden_states.shape
        # c_kv stores [c_k, c_v] concatenated, so size is 2*d_latent
        expected_shape = (batch_size, seq_len, 2 * gpt2_config.computed_d_latent)
        assert c_kv.shape == expected_shape

    def test_decompress_k_shape(self, gpt2_config, device, dtype):
        """Test K decompression output shape."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        # c_kv has shape (batch, seq, 2*d_latent) - stores [c_k, c_v]
        c_kv = torch.randn(batch_size, seq_len, 2 * gpt2_config.computed_d_latent, device=device, dtype=dtype)

        K = compression.decompress_k(c_kv)

        expected_shape = (batch_size, seq_len, gpt2_config.d_kv)
        assert K.shape == expected_shape

    def test_decompress_v_shape(self, gpt2_config, device, dtype):
        """Test V decompression output shape."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        # c_kv has shape (batch, seq, 2*d_latent) - stores [c_k, c_v]
        c_kv = torch.randn(batch_size, seq_len, 2 * gpt2_config.computed_d_latent, device=device, dtype=dtype)

        V = compression.decompress_v(c_kv)

        expected_shape = (batch_size, seq_len, gpt2_config.d_kv)
        assert V.shape == expected_shape

    def test_forward_returns_correct_shapes(self, gpt2_config, sample_hidden_states, device, dtype):
        """Test full forward pass returns correct shapes."""
        compression = MLACompression(gpt2_config).to(device, dtype)
        sample_hidden_states = sample_hidden_states.to(device, dtype)

        c_kv, K, V = compression(sample_hidden_states)

        batch_size, seq_len, _ = sample_hidden_states.shape

        # c_kv stores [c_k, c_v] concatenated, so size is 2*d_latent
        assert c_kv.shape == (batch_size, seq_len, 2 * gpt2_config.computed_d_latent)
        assert K.shape == (batch_size, seq_len, gpt2_config.d_kv)
        assert V.shape == (batch_size, seq_len, gpt2_config.d_kv)

    def test_init_from_weights(self, gpt2_config, device, dtype):
        """Test initialization from pre-computed weights."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        # Create some weights
        d_latent = gpt2_config.computed_d_latent
        d_kv = gpt2_config.d_kv
        d_model = gpt2_config.d_model

        # Separate W_down for K and V
        W_down_k = torch.randn(d_latent, d_model, device=device, dtype=dtype)
        W_down_v = torch.randn(d_latent, d_model, device=device, dtype=dtype)
        # W_uk and W_uv have shape (d_kv, d_latent) - orthonormal columns
        W_uk = torch.randn(d_kv, d_latent, device=device, dtype=dtype)
        W_uv = torch.randn(d_kv, d_latent, device=device, dtype=dtype)

        compression.init_from_weights(W_down_k, W_down_v, W_uk, W_uv)

        # Check shapes
        assert compression.W_down_k.weight.shape == W_down_k.shape
        assert compression.W_down_v.weight.shape == W_down_v.shape

        # Note: init_from_weights doesn't enforce orthonormality,
        # it just copies the weights as-is

    def test_project_to_manifold(self, gpt2_config, device, dtype):
        """Test manifold projection."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        # Corrupt orthonormality slightly
        with torch.no_grad():
            compression.W_uk.data += 0.01 * torch.randn_like(compression.W_uk)
            compression.W_uv.data += 0.01 * torch.randn_like(compression.W_uv)

        # Check it's no longer orthonormal
        errors_before = compression.check_orthonormality()
        assert errors_before["W_uk"][0] > 1e-5 or errors_before["W_uv"][0] > 1e-5

        # Project back to manifold
        compression.project_to_manifold()

        # Should be orthonormal again
        errors_after = compression.check_orthonormality()
        assert errors_after["W_uk"][0] < 1e-5
        assert errors_after["W_uv"][0] < 1e-5

    def test_get_euclidean_params(self, gpt2_config, device, dtype):
        """Test getting Euclidean parameters."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        euclidean_params = compression.get_euclidean_params()

        # Now returns [W_down_k.weight, W_down_v.weight]
        assert len(euclidean_params) == 2
        assert euclidean_params[0] is compression.W_down_k.weight
        assert euclidean_params[1] is compression.W_down_v.weight

    def test_get_manifold_params(self, gpt2_config, device, dtype):
        """Test getting manifold parameters."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        manifold_params = compression.get_manifold_params()

        assert len(manifold_params) == 2
        # Check by id since tensor comparison is ambiguous
        param_ids = [id(p) for p in manifold_params]
        assert id(compression.W_uk) in param_ids
        assert id(compression.W_uv) in param_ids


class TestMLACompressionGQA:
    """Tests for MLACompression with GQA configs."""

    def test_gqa_dimensions(self, llama_config, device, dtype):
        """Test dimensions for GQA config."""
        compression = MLACompression(llama_config).to(device, dtype)

        # GQA has fewer KV heads
        expected_d_kv = llama_config.n_kv_heads * llama_config.d_head
        assert compression.d_kv == expected_d_kv

        # W_uk and W_uv are stored as (d_kv, d_latent) - d_kv rows, d_latent columns
        assert compression.W_uk.shape[0] == expected_d_kv
        assert compression.W_uv.shape[0] == expected_d_kv
        assert compression.W_uk.shape[1] == llama_config.computed_d_latent
        assert compression.W_uv.shape[1] == llama_config.computed_d_latent

    def test_gqa_forward(self, llama_config, device, dtype):
        """Test forward pass with GQA config."""
        compression = MLACompression(llama_config).to(device, dtype)

        batch_size, seq_len = 2, 16
        h = torch.randn(batch_size, seq_len, llama_config.d_model, device=device, dtype=dtype)

        c_kv, K, V = compression(h)

        # Check shapes
        expected_d_kv = llama_config.n_kv_heads * llama_config.d_head
        assert K.shape == (batch_size, seq_len, expected_d_kv)
        assert V.shape == (batch_size, seq_len, expected_d_kv)
