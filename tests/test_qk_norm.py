"""Tests for QK norm support (Qwen3-style qk_norm=True)."""

import torch

from cacheshrink.attention import RMSNorm, MLAAttention
from cacheshrink.config import MLAConfig


class TestRMSNorm:
    """Test the RMSNorm implementation."""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(32, eps=1e-6)
        x = torch.randn(1, 5, 32)
        out = norm(x)
        # After RMSNorm with unit weights, the RMS of each vector should be ~1
        rms = torch.sqrt(out.float().pow(2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_dtype_preservation_float32(self):
        norm = RMSNorm(16)
        x = torch.randn(1, 3, 16, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_dtype_preservation_float16(self):
        norm = RMSNorm(16)
        x = torch.randn(1, 3, 16, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_dtype_preservation_bfloat16(self):
        norm = RMSNorm(16)
        x = torch.randn(1, 3, 16, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_learnable_weight(self):
        norm = RMSNorm(8)
        assert norm.weight.shape == (8,)
        assert norm.weight.requires_grad

    def test_weight_scaling(self):
        norm = RMSNorm(4, eps=1e-6)
        norm.weight.data = torch.tensor([2.0, 2.0, 2.0, 2.0])
        x = torch.randn(1, 1, 4)
        out_scaled = norm(x)

        norm2 = RMSNorm(4, eps=1e-6)
        # norm2 has unit weights
        out_unit = norm2(x)

        assert torch.allclose(out_scaled, 2.0 * out_unit, atol=1e-5)


class TestMLAAttentionWithQKNorm:
    """Test MLAAttention with QK norm support."""

    def _make_config(self, qk_norm=False):
        return MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            compression_ratio=2.0,
            max_position_embeddings=128,
            vocab_size=100,
            use_bias=False,
            qk_norm=qk_norm,
            extra_config={"uses_rope": True},
        )

    def test_default_norms_are_none(self):
        config = self._make_config(qk_norm=False)
        attn = MLAAttention(config, layer_idx=0)
        assert attn.q_norm is None
        assert attn.k_norm is None

    def test_forward_without_norms(self):
        config = self._make_config(qk_norm=False)
        attn = MLAAttention(config, layer_idx=0)
        x = torch.randn(1, 8, 64)
        out, _, _ = attn(x)
        assert out.shape == (1, 8, 64)

    def test_forward_with_norms_attached(self):
        config = self._make_config(qk_norm=False)
        attn = MLAAttention(config, layer_idx=0)

        # Attach norms manually (simulating what converter does)
        # Norms operate on per-head dim (d_head), matching Qwen3's pattern
        attn.q_norm = RMSNorm(config.d_head)
        attn.k_norm = RMSNorm(config.d_head)

        x = torch.randn(1, 8, 64)
        out, _, _ = attn(x)
        assert out.shape == (1, 8, 64)

    def test_output_differs_with_vs_without_norm(self):
        config = self._make_config(qk_norm=False)

        # Without norms
        attn_no_norm = MLAAttention(config, layer_idx=0)

        # With norms (use non-unit weights to ensure difference)
        attn_with_norm = MLAAttention(config, layer_idx=0)
        # Copy weights to match
        attn_with_norm.load_state_dict(attn_no_norm.state_dict())

        attn_with_norm.q_norm = RMSNorm(config.d_head)
        attn_with_norm.k_norm = RMSNorm(config.d_head)
        # Use weight != 1 to guarantee different output
        attn_with_norm.q_norm.weight.data.fill_(0.5)
        attn_with_norm.k_norm.weight.data.fill_(0.5)

        x = torch.randn(1, 8, 64)
        torch.manual_seed(42)
        out_no_norm, _, _ = attn_no_norm(x)
        torch.manual_seed(42)
        out_with_norm, _, _ = attn_with_norm(x)

        assert not torch.allclose(out_no_norm, out_with_norm, atol=1e-5)

    def test_caching_works_with_norms(self):
        config = self._make_config(qk_norm=False)
        attn = MLAAttention(config, layer_idx=0)

        attn.q_norm = RMSNorm(config.d_head)
        attn.k_norm = RMSNorm(config.d_head)

        # First pass
        x1 = torch.randn(1, 4, 64)
        out1, past_kv, _ = attn(x1, use_cache=True)
        assert out1.shape == (1, 4, 64)
        assert past_kv is not None

        # Second pass with cache
        x2 = torch.randn(1, 2, 64)
        out2, past_kv2, _ = attn(x2, past_key_value=past_kv, use_cache=True)
        assert out2.shape == (1, 2, 64)


class TestMLAConfigQKNorm:
    """Test MLAConfig qk_norm field."""

    def test_default_is_false(self):
        config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
        )
        assert config.qk_norm is False

    def test_set_to_true(self):
        config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            qk_norm=True,
        )
        assert config.qk_norm is True

    def test_serialization_roundtrip(self):
        config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            qk_norm=True,
        )
        config_dict = config.to_dict()
        assert config_dict["qk_norm"] is True

        restored = MLAConfig.from_dict(config_dict)
        assert restored.qk_norm is True

    def test_backward_compat_missing_qk_norm(self):
        """Old configs without qk_norm should default to False."""
        config_dict = {
            "model_name": "test",
            "model_type": "generic",
            "n_heads": 4,
            "n_kv_heads": 4,
            "d_model": 64,
            "d_head": 16,
            "n_layers": 2,
        }
        config = MLAConfig.from_dict(config_dict)
        assert config.qk_norm is False

    def test_save_load_roundtrip(self, tmp_path):
        config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            qk_norm=True,
        )
        path = str(tmp_path / "config.json")
        config.save(path)
        loaded = MLAConfig.load(path)
        assert loaded.qk_norm is True


class TestTrainerQKNormFreezing:
    """Test that QK norm parameters stay frozen during training setup."""

    def test_norm_params_frozen(self):
        config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            compression_ratio=2.0,
            max_position_embeddings=128,
            vocab_size=100,
            use_bias=False,
            qk_norm=True,
            extra_config={"uses_rope": True},
        )

        # Build a minimal model with MLA attention + norms
        attn = MLAAttention(config, layer_idx=0)
        attn.q_norm = RMSNorm(config.d_head)
        attn.k_norm = RMSNorm(config.d_head)

        # Simulate trainer's _freeze_non_mla_params logic
        for name, param in attn.named_parameters():
            if "mla_compression" in name or "mla." in name:
                if "q_norm" in name or "k_norm" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False

        # Verify norm params are frozen
        for name, param in attn.named_parameters():
            if "q_norm" in name or "k_norm" in name:
                assert not param.requires_grad, f"{name} should be frozen"

        # Verify compression params are trainable
        for name, param in attn.named_parameters():
            if "mla_compression" in name:
                assert param.requires_grad, f"{name} should be trainable"
