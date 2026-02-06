"""Tests for MLACompression module."""

import pytest
import torch

from cacheshrink.compression import MLACompression
from cacheshrink.improved_compression import JointKVCompression, DecoupledRoPECompression
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


class TestMLACompressionReconstructionLoss:
    """Tests for reconstruction loss methods in MLACompression."""

    def test_has_original_weights_false_by_default(self, gpt2_config, device, dtype):
        """Test that has_original_weights returns False by default."""
        compression = MLACompression(gpt2_config).to(device, dtype)
        assert compression.has_original_weights() is False

    def test_store_original_weights(self, gpt2_config, device, dtype):
        """Test that store_original_weights correctly stores weights as buffers."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        # Create dummy original weights
        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)

        compression.store_original_weights(W_k, W_v)

        # Check that weights are stored
        assert compression.has_original_weights() is True
        assert compression.W_k_original is not None
        assert compression.W_v_original is not None
        assert compression.W_k_original.shape == W_k.shape
        assert compression.W_v_original.shape == W_v.shape

        # Check that stored weights are detached copies (not same tensor)
        assert not compression.W_k_original.requires_grad
        assert not compression.W_v_original.requires_grad

    def test_compute_reconstruction_loss_raises_without_weights(self, gpt2_config, device, dtype):
        """Test that compute_reconstruction_loss raises ValueError when weights not stored."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        h = torch.randn(2, 16, gpt2_config.d_model, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original weights not stored"):
            compression.compute_reconstruction_loss(h)

    def test_compute_reconstruction_loss_returns_mse(self, gpt2_config, device, dtype):
        """Test that compute_reconstruction_loss computes MSE correctly."""
        compression = MLACompression(gpt2_config).to(device, dtype)

        # Create and store original weights
        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        compression.store_original_weights(W_k, W_v)

        # Initialize compression from these weights for a fair test
        d_latent = gpt2_config.computed_d_latent
        W_down_k = torch.randn(d_latent, gpt2_config.d_model, device=device, dtype=dtype)
        W_down_v = torch.randn(d_latent, gpt2_config.d_model, device=device, dtype=dtype)
        W_uk = torch.randn(gpt2_config.d_kv, d_latent, device=device, dtype=dtype)
        W_uv = torch.randn(gpt2_config.d_kv, d_latent, device=device, dtype=dtype)
        compression.init_from_weights(W_down_k, W_down_v, W_uk, W_uv)

        h = torch.randn(2, 16, gpt2_config.d_model, device=device, dtype=dtype)

        k_loss, v_loss = compression.compute_reconstruction_loss(h)

        # Losses should be positive scalars
        assert k_loss.ndim == 0  # scalar
        assert v_loss.ndim == 0  # scalar
        assert k_loss.item() >= 0
        assert v_loss.item() >= 0

    def test_compute_reconstruction_loss_gradients_flow(self, gpt2_config, device):
        """Test that gradients flow through reconstruction loss computation."""
        # Use float32 for gradient computation
        compression = MLACompression(gpt2_config).to(device, torch.float32)

        # Store original weights
        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device)
        compression.store_original_weights(W_k, W_v)

        h = torch.randn(2, 16, gpt2_config.d_model, device=device, requires_grad=True)

        k_loss, v_loss = compression.compute_reconstruction_loss(h)
        total_loss = k_loss + v_loss
        total_loss.backward()

        # Check that gradients exist for compression parameters
        assert compression.W_down_k.weight.grad is not None
        assert compression.W_down_v.weight.grad is not None


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


class TestJointKVCompressionReconstructionLoss:
    """Tests for reconstruction loss methods in JointKVCompression."""

    def test_has_original_weights_false_by_default(self, gpt2_config, device, dtype):
        """Test that has_original_weights returns False by default."""
        compression = JointKVCompression(gpt2_config).to(device, dtype)
        assert compression.has_original_weights() is False

    def test_store_original_weights(self, gpt2_config, device, dtype):
        """Test that store_original_weights correctly stores weights as buffers."""
        compression = JointKVCompression(gpt2_config).to(device, dtype)

        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)

        compression.store_original_weights(W_k, W_v)

        assert compression.has_original_weights() is True
        assert compression.W_k_original is not None
        assert compression.W_v_original is not None
        assert not compression.W_k_original.requires_grad
        assert not compression.W_v_original.requires_grad

    def test_compute_reconstruction_loss_raises_without_weights(self, gpt2_config, device, dtype):
        """Test that compute_reconstruction_loss raises ValueError when weights not stored."""
        compression = JointKVCompression(gpt2_config).to(device, dtype)

        h = torch.randn(2, 16, gpt2_config.d_model, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original weights not stored"):
            compression.compute_reconstruction_loss(h)

    def test_compute_reconstruction_loss_returns_mse(self, gpt2_config, device, dtype):
        """Test that compute_reconstruction_loss computes MSE correctly."""
        compression = JointKVCompression(gpt2_config).to(device, dtype)

        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        compression.store_original_weights(W_k, W_v)

        h = torch.randn(2, 16, gpt2_config.d_model, device=device, dtype=dtype)

        k_loss, v_loss = compression.compute_reconstruction_loss(h)

        assert k_loss.ndim == 0
        assert v_loss.ndim == 0
        assert k_loss.item() >= 0
        assert v_loss.item() >= 0


class TestDecoupledRoPECompressionReconstructionLoss:
    """Tests for reconstruction loss methods in DecoupledRoPECompression."""

    @pytest.fixture
    def decoupled_compression(self, gpt2_config, device, dtype):
        """Create a DecoupledRoPECompression instance."""
        # d_rope must be smaller than d_kv to leave room for d_content
        d_rope = min(16, gpt2_config.d_kv // 2)
        return DecoupledRoPECompression(
            d_model=gpt2_config.d_model,
            d_kv=gpt2_config.d_kv,
            d_latent=gpt2_config.computed_d_latent,
            d_rope=d_rope,
        ).to(device, dtype)

    def test_has_original_weights_false_by_default(self, decoupled_compression):
        """Test that has_original_weights returns False by default."""
        assert decoupled_compression.has_original_weights() is False

    def test_store_original_weights(self, decoupled_compression, gpt2_config, device, dtype):
        """Test that store_original_weights correctly stores weights as buffers."""
        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)

        decoupled_compression.store_original_weights(W_k, W_v)

        assert decoupled_compression.has_original_weights() is True
        assert decoupled_compression.W_k_original is not None
        assert decoupled_compression.W_v_original is not None
        assert not decoupled_compression.W_k_original.requires_grad
        assert not decoupled_compression.W_v_original.requires_grad

    def test_compute_reconstruction_loss_raises_without_weights(self, decoupled_compression, gpt2_config, device, dtype):
        """Test that compute_reconstruction_loss raises ValueError when weights not stored."""
        h = torch.randn(2, 16, gpt2_config.d_model, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original weights not stored"):
            decoupled_compression.compute_reconstruction_loss(h)

    def test_compute_reconstruction_loss_returns_mse(self, decoupled_compression, gpt2_config, device, dtype):
        """Test that compute_reconstruction_loss computes MSE correctly."""
        W_k = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(gpt2_config.d_kv, gpt2_config.d_model, device=device, dtype=dtype)
        decoupled_compression.store_original_weights(W_k, W_v)

        h = torch.randn(2, 16, gpt2_config.d_model, device=device, dtype=dtype)

        k_loss, v_loss = decoupled_compression.compute_reconstruction_loss(h)

        assert k_loss.ndim == 0
        assert v_loss.ndim == 0
        assert k_loss.item() >= 0
        assert v_loss.item() >= 0
