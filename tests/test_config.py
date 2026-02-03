"""Tests for MLAConfig."""

import pytest
import json
import tempfile
import os

from kvpress.config import MLAConfig


class TestMLAConfig:
    """Tests for MLAConfig dataclass."""

    def test_basic_creation(self):
        """Test basic config creation."""
        config = MLAConfig(
            model_name="test-model",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
        )

        assert config.model_name == "test-model"
        assert config.model_type == "gpt2"
        assert config.n_heads == 12
        assert config.n_kv_heads == 12
        assert config.d_model == 768
        assert config.d_head == 64
        assert config.n_layers == 12

    def test_d_kv_property(self):
        """Test d_kv computation."""
        # MHA case
        config = MLAConfig(
            model_name="test",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
        )
        assert config.d_kv == 12 * 64  # 768

        # GQA case
        config_gqa = MLAConfig(
            model_name="test",
            model_type="llama",
            n_heads=32,
            n_kv_heads=8,
            d_model=4096,
            d_head=128,
            n_layers=32,
        )
        assert config_gqa.d_kv == 8 * 128  # 1024

    def test_computed_d_latent(self):
        """Test d_latent computation from compression ratio."""
        config = MLAConfig(
            model_name="test",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
            compression_ratio=4.0,
        )
        # With separate K/V compression: d_latent = d_kv / compression_ratio = 768 / 4 = 192
        # Cache stores c_k (d_latent) + c_v (d_latent) = 2*d_latent
        # Original stores K (d_kv) + V (d_kv) = 2*d_kv
        # Compression ratio = 2*d_kv / 2*d_latent = d_kv / d_latent
        assert config.computed_d_latent == 192

    def test_computed_d_latent_with_override(self):
        """Test explicit d_latent override."""
        config = MLAConfig(
            model_name="test",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
            compression_ratio=4.0,
            d_latent=256,  # Override
        )
        assert config.computed_d_latent == 256

    def test_is_gqa_property(self):
        """Test GQA detection."""
        # MHA
        config_mha = MLAConfig(
            model_name="test",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
        )
        assert not config_mha.is_gqa

        # GQA
        config_gqa = MLAConfig(
            model_name="test",
            model_type="llama",
            n_heads=32,
            n_kv_heads=8,
            d_model=4096,
            d_head=128,
            n_layers=32,
        )
        assert config_gqa.is_gqa

    def test_n_rep_property(self):
        """Test n_rep computation for GQA."""
        config = MLAConfig(
            model_name="test",
            model_type="llama",
            n_heads=32,
            n_kv_heads=8,
            d_model=4096,
            d_head=128,
            n_layers=32,
        )
        assert config.n_rep == 4  # 32 / 8

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = MLAConfig(
            model_name="test-model",
            model_type="llama",
            n_heads=32,
            n_kv_heads=8,
            d_model=4096,
            d_head=128,
            n_layers=32,
            compression_ratio=8.0,
            rope_theta=500000.0,
        )

        config_dict = original.to_dict()
        restored = MLAConfig.from_dict(config_dict)

        assert restored.model_name == original.model_name
        assert restored.model_type == original.model_type
        assert restored.n_heads == original.n_heads
        assert restored.n_kv_heads == original.n_kv_heads
        assert restored.compression_ratio == original.compression_ratio
        assert restored.rope_theta == original.rope_theta

    def test_save_and_load(self):
        """Test file save/load."""
        config = MLAConfig(
            model_name="test-model",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
            compression_ratio=4.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.save(path)

            # Verify file exists and is valid JSON
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["model_name"] == "test-model"

            # Load and verify
            loaded = MLAConfig.load(path)
            assert loaded.model_name == config.model_name
            assert loaded.compression_ratio == config.compression_ratio

    def test_repr(self):
        """Test string representation."""
        config = MLAConfig(
            model_name="test",
            model_type="gpt2",
            n_heads=12,
            n_kv_heads=12,
            d_model=768,
            d_head=64,
            n_layers=12,
        )
        repr_str = repr(config)
        assert "MLAConfig" in repr_str
        assert "gpt2" in repr_str


class TestMLAConfigFromPretrained:
    """Tests for MLAConfig.from_pretrained (requires transformers)."""

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers required"
    )
    def test_from_pretrained_gpt2(self):
        """Test loading config from GPT-2."""
        config = MLAConfig.from_pretrained("gpt2", compression_ratio=4.0)

        assert config.model_type == "gpt2"
        assert config.n_heads == 12
        assert config.n_kv_heads == 12  # GPT-2 uses MHA
        assert config.d_model == 768
        assert config.d_head == 64
        assert config.n_layers == 12
        assert not config.is_gqa
