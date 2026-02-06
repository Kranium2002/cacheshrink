"""Tests for xKV cross-layer compression modules."""

import pytest
import torch

from cacheshrink.config import MLAConfig
from cacheshrink.xkv_compression import XKVCompression, XKVCompressionGroup
from cacheshrink.utils import check_orthonormality


@pytest.fixture
def xkv_config():
    """Create a GQA-style MLAConfig for xKV testing."""
    return MLAConfig(
        model_name="test-xkv",
        model_type="llama",
        n_heads=32,
        n_kv_heads=8,  # GQA
        d_model=512,
        d_head=64,
        n_layers=8,
        compression_ratio=4.0,
        max_position_embeddings=128,
        vocab_size=1000,
        use_bias=False,
        layer_norm_eps=1e-5,
        use_cross_layer=True,
        cross_layer_group_size=4,
    )


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float32


class TestXKVCompression:
    """Tests for XKVCompression module."""

    def test_initialization(self, xkv_config, device, dtype):
        """Test XKVCompression initialization."""
        d_latent = xkv_config.computed_d_latent
        group_layers = [0, 1, 2, 3]

        comp = XKVCompression(xkv_config, layer_idx=0, group_layers=group_layers, d_latent=d_latent)
        comp = comp.to(device, dtype)

        assert comp.d_model == xkv_config.d_model
        assert comp.d_latent == d_latent
        assert comp.d_kv == xkv_config.d_kv
        assert comp.layer_idx == 0
        assert comp.group_layers == group_layers

    def test_compress_shape(self, xkv_config, device, dtype):
        """Test compression output shape."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        batch_size, seq_len = 2, 16
        h = torch.randn(batch_size, seq_len, xkv_config.d_model, device=device, dtype=dtype)

        c = comp.compress(h)

        # c should have shape (batch, seq, 2 * d_latent)
        assert c.shape == (batch_size, seq_len, 2 * d_latent)

    def test_decompress_k_shape(self, xkv_config, device, dtype):
        """Test K decompression output shape."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        batch_size, seq_len = 2, 16
        c = torch.randn(batch_size, seq_len, 2 * d_latent, device=device, dtype=dtype)

        K = comp.decompress_k(c)

        assert K.shape == (batch_size, seq_len, xkv_config.d_kv)

    def test_decompress_v_shape(self, xkv_config, device, dtype):
        """Test V decompression output shape."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        batch_size, seq_len = 2, 16
        c = torch.randn(batch_size, seq_len, 2 * d_latent, device=device, dtype=dtype)

        V = comp.decompress_v(c)

        assert V.shape == (batch_size, seq_len, xkv_config.d_kv)

    def test_forward(self, xkv_config, device, dtype):
        """Test full forward pass."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        batch_size, seq_len = 2, 16
        h = torch.randn(batch_size, seq_len, xkv_config.d_model, device=device, dtype=dtype)

        c, K, V = comp(h)

        assert c.shape == (batch_size, seq_len, 2 * d_latent)
        assert K.shape == (batch_size, seq_len, xkv_config.d_kv)
        assert V.shape == (batch_size, seq_len, xkv_config.d_kv)

    def test_orthonormality_at_init(self, xkv_config, device, dtype):
        """Test W_uk and W_uv have orthonormal columns at initialization."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        errors = comp.check_orthonormality()

        # Should be very close to orthonormal
        assert errors["W_uk"][0] < 1e-5, f"W_uk max error: {errors['W_uk'][0]}"
        assert errors["W_uv"][0] < 1e-5, f"W_uv max error: {errors['W_uv'][0]}"

    def test_project_to_manifold(self, xkv_config, device, dtype):
        """Test manifold projection restores orthonormality."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        # Corrupt orthonormality
        with torch.no_grad():
            comp.W_uk.data += 0.01 * torch.randn_like(comp.W_uk)
            comp.W_uv.data += 0.01 * torch.randn_like(comp.W_uv)

        # Check corruption
        errors_before = comp.check_orthonormality()
        assert errors_before["W_uk"][0] > 1e-5 or errors_before["W_uv"][0] > 1e-5

        # Project back
        comp.project_to_manifold()

        # Should be orthonormal again (use slightly relaxed tolerance for numerical stability)
        errors_after = comp.check_orthonormality()
        assert errors_after["W_uk"][0] < 1e-4
        assert errors_after["W_uv"][0] < 1e-4

    def test_store_original_weights(self, xkv_config, device, dtype):
        """Test storing original weights for reconstruction loss."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        assert comp.has_original_weights() is False

        W_k = torch.randn(xkv_config.d_kv, xkv_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(xkv_config.d_kv, xkv_config.d_model, device=device, dtype=dtype)

        comp.store_original_weights(W_k, W_v)

        assert comp.has_original_weights() is True
        assert comp.W_k_original is not None
        assert comp.W_v_original is not None

    def test_compute_reconstruction_loss(self, xkv_config, device, dtype):
        """Test reconstruction loss computation."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        # Store original weights
        W_k = torch.randn(xkv_config.d_kv, xkv_config.d_model, device=device, dtype=dtype)
        W_v = torch.randn(xkv_config.d_kv, xkv_config.d_model, device=device, dtype=dtype)
        comp.store_original_weights(W_k, W_v)

        h = torch.randn(2, 16, xkv_config.d_model, device=device, dtype=dtype)

        k_loss, v_loss = comp.compute_reconstruction_loss(h)

        assert k_loss.ndim == 0  # scalar
        assert v_loss.ndim == 0
        assert k_loss.item() >= 0
        assert v_loss.item() >= 0

    def test_reconstruction_loss_raises_without_weights(self, xkv_config, device, dtype):
        """Test that reconstruction loss raises if weights not stored."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        h = torch.randn(2, 16, xkv_config.d_model, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="Original weights not stored"):
            comp.compute_reconstruction_loss(h)

    def test_get_euclidean_params(self, xkv_config, device, dtype):
        """Test getting Euclidean parameters."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        params = comp.get_euclidean_params()

        assert len(params) == 2
        assert params[0] is comp.W_down_k.weight
        assert params[1] is comp.W_down_v.weight

    def test_get_manifold_params(self, xkv_config, device, dtype):
        """Test getting manifold parameters."""
        d_latent = xkv_config.computed_d_latent
        comp = XKVCompression(xkv_config, 0, [0, 1, 2, 3], d_latent).to(device, dtype)

        params = comp.get_manifold_params()

        assert len(params) == 2
        param_ids = [id(p) for p in params]
        assert id(comp.W_uk) in param_ids
        assert id(comp.W_uv) in param_ids


class TestXKVCompressionGroup:
    """Tests for XKVCompressionGroup module."""

    def test_initialization(self, xkv_config, device, dtype):
        """Test group initialization."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        assert group.d_latent == d_latent
        assert group.layer_indices == layer_indices
        assert len(group.layer_compressions) == 4

    def test_shared_decompression_matrices(self, xkv_config, device, dtype):
        """Test that layers in a group share W_uk, W_uv."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        comp0 = group.get_compression(0)
        comp1 = group.get_compression(1)
        comp2 = group.get_compression(2)
        comp3 = group.get_compression(3)

        # All should share the exact same W_uk and W_uv tensors
        assert comp0.W_uk is comp1.W_uk
        assert comp1.W_uk is comp2.W_uk
        assert comp2.W_uk is comp3.W_uk

        assert comp0.W_uv is comp1.W_uv
        assert comp1.W_uv is comp2.W_uv
        assert comp2.W_uv is comp3.W_uv

        # They should also be the same as the group's shared matrices
        assert comp0.W_uk is group.shared_W_uk
        assert comp0.W_uv is group.shared_W_uv

    def test_per_layer_compression_matrices(self, xkv_config, device, dtype):
        """Test that W_down is different per layer."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        comp0 = group.get_compression(0)
        comp1 = group.get_compression(1)

        # W_down_k and W_down_v should be different modules
        assert comp0.W_down_k is not comp1.W_down_k
        assert comp0.W_down_v is not comp1.W_down_v

    def test_init_shared_basis(self, xkv_config, device, dtype):
        """Test initializing shared basis matrices."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        # Create new basis matrices
        W_uk = torch.randn(xkv_config.d_kv, d_latent, device=device, dtype=dtype)
        W_uv = torch.randn(xkv_config.d_kv, d_latent, device=device, dtype=dtype)

        group.init_shared_basis(W_uk, W_uv)

        # Check orthonormality after initialization (use relaxed tolerance for numerical stability)
        errors = group.check_orthonormality()
        assert errors["W_uk"][0] < 1e-4
        assert errors["W_uv"][0] < 1e-4

    def test_get_compression_invalid_layer(self, xkv_config, device, dtype):
        """Test that accessing invalid layer raises KeyError."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        with pytest.raises(KeyError):
            group.get_compression(5)  # Not in group

    def test_get_euclidean_params(self, xkv_config, device, dtype):
        """Test getting all Euclidean parameters from group."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        params = group.get_euclidean_params()

        # Should have 2 params per layer (W_down_k, W_down_v)
        assert len(params) == 4 * 2

    def test_get_manifold_params(self, xkv_config, device, dtype):
        """Test getting shared manifold parameters (only once)."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        params = group.get_manifold_params()

        # Should only have 2 params (shared_W_uk, shared_W_uv), not per-layer
        assert len(params) == 2

    def test_compress_decompress_per_layer(self, xkv_config, device, dtype):
        """Test compression/decompression for each layer in group."""
        d_latent = xkv_config.computed_d_latent
        layer_indices = [0, 1, 2, 3]

        group = XKVCompressionGroup(xkv_config, layer_indices, d_latent).to(device, dtype)

        batch_size, seq_len = 2, 16
        h = torch.randn(batch_size, seq_len, xkv_config.d_model, device=device, dtype=dtype)

        for layer_idx in layer_indices:
            comp = group.get_compression(layer_idx)
            c, K, V = comp(h)

            assert c.shape == (batch_size, seq_len, 2 * d_latent)
            assert K.shape == (batch_size, seq_len, xkv_config.d_kv)
            assert V.shape == (batch_size, seq_len, xkv_config.d_kv)


class TestMLAConfigXKV:
    """Tests for MLAConfig xKV-related properties."""

    def test_n_groups_with_cross_layer(self, xkv_config):
        """Test n_groups property when use_cross_layer=True."""
        # 8 layers, group_size=4 -> 2 groups
        assert xkv_config.n_groups == 2

    def test_n_groups_without_cross_layer(self, xkv_config):
        """Test n_groups when use_cross_layer=False."""
        xkv_config.use_cross_layer = False
        # Without cross-layer, each layer is its own "group"
        assert xkv_config.n_groups == xkv_config.n_layers

    def test_get_layer_group(self, xkv_config):
        """Test get_layer_group returns correct group index."""
        # Group size 4, so layers 0-3 -> group 0, layers 4-7 -> group 1
        assert xkv_config.get_layer_group(0) == 0
        assert xkv_config.get_layer_group(3) == 0
        assert xkv_config.get_layer_group(4) == 1
        assert xkv_config.get_layer_group(7) == 1

    def test_get_group_layers(self, xkv_config):
        """Test get_group_layers returns correct layer indices."""
        assert xkv_config.get_group_layers(0) == [0, 1, 2, 3]
        assert xkv_config.get_group_layers(1) == [4, 5, 6, 7]

    def test_get_group_layers_partial_group(self):
        """Test get_group_layers with non-divisible layer count."""
        config = MLAConfig(
            model_name="test",
            model_type="llama",
            n_heads=32,
            n_kv_heads=8,
            d_model=512,
            d_head=64,
            n_layers=10,  # Not divisible by 4
            compression_ratio=4.0,
            use_cross_layer=True,
            cross_layer_group_size=4,
        )

        # 10 layers / 4 = 2 full groups + 1 partial
        assert config.n_groups == 3
        assert config.get_group_layers(0) == [0, 1, 2, 3]
        assert config.get_group_layers(1) == [4, 5, 6, 7]
        assert config.get_group_layers(2) == [8, 9]  # Partial group
